#![forbid(unsafe_code)]

use std::{error::Error, path::PathBuf, time::Instant};

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};
use ft_core::ExecutionMode;
use ft_nn::{CrossEntropyLoss, LossModule, Module, named_parameters};
use ft_optim::{AdamW, Optimizer};
use set_vision_data::{AtlasCollection, BoardBatch, BoardKind, RenderedBoardCollection, Rng};
use set_vision_policy::{
    ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT, CARD_PIXELS, DeployedVisualSetPolicy,
    RELATION_CLASS_COUNT,
};

const BOARD_CARDS: usize = 12;
const TRIPLE_COUNT: usize = 220;
const TEMPERATURE: f64 = 0.1;
const MARGIN: f64 = 0.25;
const ATTRIBUTE_LOSS_WEIGHT: f64 = 4.0;
const RELATION_LOSS_WEIGHT: f64 = 1.0;
const CANDIDATE_POOL: usize = 8;
const DEFAULT_STEPS: usize = 1_000;
const DEFAULT_DATA_SEED: u64 = 590_011;
const DEFAULT_LEARNING_RATE: f64 = 8.0e-5;

struct Config {
    atlas_dir: PathBuf,
    input_checkpoint: PathBuf,
    output_root: PathBuf,
    steps: usize,
    data_seed: u64,
    learning_rate: f64,
    dense_triple_loss: bool,
    decision_head_only: bool,
    shape_fill_decision_only: bool,
    rendered_boards_stdin: bool,
}

impl Config {
    fn from_args() -> Result<Self, Box<dyn Error>> {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("vision crate has no repository parent")?
            .to_path_buf();
        let mut config = Self {
            atlas_dir: repository.join("vision-artifacts/board-v8-training-atlases"),
            input_checkpoint: repository.join("vision-artifacts/active-v10/rate-0/set-vision.ftsv"),
            output_root: repository.join("vision-artifacts/active-v23"),
            steps: DEFAULT_STEPS,
            data_seed: DEFAULT_DATA_SEED,
            learning_rate: DEFAULT_LEARNING_RATE,
            dense_triple_loss: false,
            decision_head_only: false,
            shape_fill_decision_only: false,
            rendered_boards_stdin: false,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--atlas-dir" => {
                    config.atlas_dir =
                        PathBuf::from(arguments.next().ok_or("--atlas-dir requires a path")?);
                }
                "--input-checkpoint" => {
                    config.input_checkpoint = PathBuf::from(
                        arguments
                            .next()
                            .ok_or("--input-checkpoint requires a path")?,
                    );
                }
                "--output-root" => {
                    config.output_root =
                        PathBuf::from(arguments.next().ok_or("--output-root requires a path")?);
                }
                "--steps" => {
                    config.steps = arguments
                        .next()
                        .ok_or("--steps requires a value")?
                        .parse::<usize>()?;
                }
                "--data-seed" => {
                    config.data_seed = arguments
                        .next()
                        .ok_or("--data-seed requires a value")?
                        .parse::<u64>()?;
                }
                "--learning-rate" => {
                    config.learning_rate = arguments
                        .next()
                        .ok_or("--learning-rate requires a value")?
                        .parse::<f64>()?;
                }
                "--dense-triple-loss" => config.dense_triple_loss = true,
                "--decision-head-only" => config.decision_head_only = true,
                "--shape-fill-decision-only" => config.shape_fill_decision_only = true,
                "--rendered-boards-stdin" => config.rendered_boards_stdin = true,
                unknown => return Err(format!("unknown argument: {unknown}").into()),
            }
        }
        if config.steps == 0 {
            return Err("steps must be positive".into());
        }
        if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive".into());
        }
        if config.decision_head_only && config.shape_fill_decision_only {
            return Err("parameter-scope flags are mutually exclusive".into());
        }
        Ok(config)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParameterScope {
    All,
    DecisionHead,
    ShapeFillDecision,
}

impl Config {
    fn parameter_scope(&self) -> ParameterScope {
        if self.decision_head_only {
            ParameterScope::DecisionHead
        } else if self.shape_fill_decision_only {
            ParameterScope::ShapeFillDecision
        } else {
            ParameterScope::All
        }
    }
}

enum TrainingBoards {
    Atlas(AtlasCollection),
    Rendered(RenderedBoardCollection),
}

impl TrainingBoards {
    fn make_board_of_kind(
        &self,
        rng: &mut Rng,
        kind: BoardKind,
    ) -> Result<BoardBatch, Box<dyn Error>> {
        match self {
            Self::Atlas(atlases) => Ok(atlases.make_training_board_of_kind(rng, BOARD_CARDS, kind)),
            Self::Rendered(boards) => Ok(boards.make_board_of_kind(rng, kind)?),
        }
    }

    fn description(&self) -> String {
        match self {
            Self::Atlas(atlases) => format!("atlas collection ({} atlases)", atlases.len()),
            Self::Rendered(boards) => {
                format!("in-memory browser collection ({} boards)", boards.len())
            }
        }
    }
}

fn load_training_boards(config: &Config) -> Result<TrainingBoards, Box<dyn Error>> {
    if config.rendered_boards_stdin {
        return Ok(TrainingBoards::Rendered(
            RenderedBoardCollection::read_framed(std::io::stdin().lock())?,
        ));
    }
    Ok(TrainingBoards::Atlas(AtlasCollection::load(
        &config.atlas_dir,
    )?))
}

fn board_kind_for_step(step: usize) -> BoardKind {
    if step % 2 == 1 {
        BoardKind::HasSet
    } else {
        BoardKind::NoSet
    }
}

fn masked_smooth_max(
    session: &mut FrankenTorchSession,
    logits: TensorNodeId,
    targets: &[f64],
    include_positive: bool,
) -> Result<TensorNodeId, AutogradError> {
    let offsets = targets
        .iter()
        .map(|&target| {
            let is_positive = target >= 0.5;
            if is_positive == include_positive {
                0.0
            } else {
                -100.0
            }
        })
        .collect::<Vec<_>>();
    let offsets = session.tensor_variable(offsets, vec![targets.len()], false)?;
    let masked = session.tensor_add(logits, offsets)?;
    let scaled = session.tensor_mul_scalar(masked, 1.0 / TEMPERATURE)?;
    let maximum = session.tensor_logsumexp(scaled, 0)?;
    session.tensor_mul_scalar(maximum, TEMPERATURE)
}

fn soft_threshold(
    session: &mut FrankenTorchSession,
    value: TensorNodeId,
) -> Result<TensorNodeId, AutogradError> {
    let scaled = session.tensor_mul_scalar(value, 1.0 / TEMPERATURE)?;
    let penalty = session.tensor_softplus(scaled)?;
    session.tensor_mul_scalar(penalty, TEMPERATURE)
}

fn board_action_loss(
    session: &mut FrankenTorchSession,
    set_logits: TensorNodeId,
    targets: &[f64],
    kind: BoardKind,
) -> Result<TensorNodeId, AutogradError> {
    let logits = session.tensor_reshape(set_logits, vec![targets.len()])?;
    let invalid_max = masked_smooth_max(session, logits, targets, false)?;
    let mut loss = soft_threshold(session, invalid_max)?;
    if kind == BoardKind::HasSet {
        let valid_max = masked_smooth_max(session, logits, targets, true)?;
        let negative_valid = session.tensor_neg(valid_max)?;
        let valid_threshold = soft_threshold(session, negative_valid)?;
        loss = session.tensor_add(loss, valid_threshold)?;

        let rank_delta = session.tensor_sub(invalid_max, valid_max)?;
        let margin = session.tensor_variable(vec![MARGIN], vec![1], false)?;
        let rank_delta = session.tensor_add(rank_delta, margin)?;
        let ranking = soft_threshold(session, rank_delta)?;
        loss = session.tensor_add(loss, ranking)?;
    }
    Ok(loss)
}

fn masked_mean(
    session: &mut FrankenTorchSession,
    values: TensorNodeId,
    mask: Vec<f64>,
) -> Result<TensorNodeId, Box<dyn Error>> {
    let count = mask.iter().filter(|&&value| value > 0.5).count();
    if count == 0 {
        return Err("masked mean requires at least one selected value".into());
    }
    let mask = session.tensor_variable(mask, vec![TRIPLE_COUNT], false)?;
    let selected = session.tensor_mul(values, mask)?;
    let total = session.tensor_sum_dim(selected, 0)?;
    let count = f64::from(u32::try_from(count)?);
    Ok(session.tensor_mul_scalar(total, 1.0 / count)?)
}

fn balanced_dense_triple_loss(
    session: &mut FrankenTorchSession,
    set_logits: TensorNodeId,
    targets: &[f64],
    kind: BoardKind,
) -> Result<TensorNodeId, Box<dyn Error>> {
    let logits = session.tensor_reshape(set_logits, vec![TRIPLE_COUNT])?;
    let negative_losses = session.tensor_softplus(logits)?;
    let negative_mask = targets
        .iter()
        .map(|&target| f64::from(target < 0.5))
        .collect::<Vec<_>>();
    let negative_mean = masked_mean(session, negative_losses, negative_mask)?;
    if kind == BoardKind::NoSet {
        return Ok(negative_mean);
    }
    let negative_logits = session.tensor_neg(logits)?;
    let positive_losses = session.tensor_softplus(negative_logits)?;
    let positive_mask = targets
        .iter()
        .map(|&target| f64::from(target >= 0.5))
        .collect::<Vec<_>>();
    let positive_mean = masked_mean(session, positive_losses, positive_mask)?;
    let balanced = session.tensor_add(negative_mean, positive_mean)?;
    Ok(session.tensor_mul_scalar(balanced, 0.5)?)
}

fn smooth_max_values(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values = values.collect::<Vec<_>>();
    let maximum = values.iter().copied().reduce(f64::max)?;
    let exponential_sum = values
        .iter()
        .map(|value| ((value - maximum) / TEMPERATURE).exp())
        .sum::<f64>();
    Some(maximum + TEMPERATURE * exponential_sum.ln())
}

fn soft_threshold_value(value: f64) -> f64 {
    let scaled = value / TEMPERATURE;
    TEMPERATURE * (scaled.max(0.0) + (-scaled.abs()).exp().ln_1p())
}

fn board_hardness(scores: &[f64], targets: &[f64], kind: BoardKind) -> Result<f64, &'static str> {
    let invalid_max = smooth_max_values(
        scores
            .iter()
            .zip(targets)
            .filter_map(|(&score, &target)| (target < 0.5).then_some(score)),
    )
    .ok_or("board has no invalid triples")?;
    let mut hardness = soft_threshold_value(invalid_max);
    if kind == BoardKind::HasSet {
        let valid_max = smooth_max_values(
            scores
                .iter()
                .zip(targets)
                .filter_map(|(&score, &target)| (target >= 0.5).then_some(score)),
        )
        .ok_or("Has-SET board has no valid triple")?;
        hardness += soft_threshold_value(-valid_max);
        hardness += soft_threshold_value(MARGIN + invalid_max - valid_max);
    }
    Ok(hardness)
}

fn mine_hard_board(
    session: &mut FrankenTorchSession,
    policy: &DeployedVisualSetPolicy,
    boards: &TrainingBoards,
    rng: &mut Rng,
    kind: BoardKind,
    graph_boundary: usize,
) -> Result<(BoardBatch, f64), Box<dyn Error>> {
    let mut hardest = None;
    for _ in 0..CANDIDATE_POOL {
        let board = boards.make_board_of_kind(rng, kind)?;
        let scores = session.with_no_grad(|session| {
            policy.score_board(session, board.pixels.clone(), BOARD_CARDS)
        })?;
        let hardness = board_hardness(&scores.scores, &board.set_targets, kind)?;
        if hardest
            .as_ref()
            .is_none_or(|(best_hardness, _)| hardness > *best_hardness)
        {
            hardest = Some((hardness, board));
        }
    }
    session.truncate_autograd_graph(graph_boundary);
    hardest
        .map(|(hardness, board)| (board, hardness))
        .ok_or_else(|| "hard-board pool was empty".into())
}

fn save_snapshot(
    session: &FrankenTorchSession,
    policy: &DeployedVisualSetPolicy,
    output_root: &std::path::Path,
    step: usize,
) -> Result<(), Box<dyn Error>> {
    let output_dir = output_root.join(format!("step-{step}"));
    std::fs::create_dir_all(&output_dir)?;
    let checkpoint = output_dir.join("set-vision.ftsv");
    let state = policy.state_dict(session)?;
    ft_serialize::save_state_dict(&state, &checkpoint)?;
    println!("snapshot: step={step} path={}", checkpoint.display());
    Ok(())
}

fn print_config(config: &Config) {
    println!("input_checkpoint: {}", config.input_checkpoint.display());
    println!("atlas_dir: {}", config.atlas_dir.display());
    println!("steps: {}", config.steps);
    println!("data_seed: {}", config.data_seed);
    println!("learning_rate: {}", config.learning_rate);
    println!("temperature: {TEMPERATURE}");
    println!("margin: {MARGIN}");
    println!("candidate_pool: {CANDIDATE_POOL}");
    println!("dense_triple_loss: {}", config.dense_triple_loss);
    println!("decision_head_only: {}", config.decision_head_only);
    println!(
        "shape_fill_decision_only: {}",
        config.shape_fill_decision_only
    );
    println!("rendered_boards_stdin: {}", config.rendered_boards_stdin);
}

fn print_training_summary(config: &Config, started: Instant) -> Result<(), Box<dyn Error>> {
    let seconds = started.elapsed().as_secs_f64();
    println!("training_seconds: {seconds:.3}");
    println!(
        "steps_per_second: {:.3}",
        f64::from(u32::try_from(config.steps)?) / seconds
    );
    Ok(())
}

fn configure_trainable_parameters(
    session: &mut FrankenTorchSession,
    policy: &DeployedVisualSetPolicy,
    scope: ParameterScope,
) -> Result<(Vec<TensorNodeId>, TensorNodeId), Box<dyn Error>> {
    let mut trainable_parameters = Vec::new();
    for (name, parameter) in named_parameters(policy, "") {
        if parameter_is_trainable(&name, scope) {
            trainable_parameters.push(parameter);
        } else {
            session.tensor_requires_grad_(parameter, false)?;
        }
    }
    let gradient_probe = *trainable_parameters
        .first()
        .ok_or("board fine-tune has no trainable parameters")?;
    Ok((trainable_parameters, gradient_probe))
}

fn parameter_is_trainable(name: &str, scope: ParameterScope) -> bool {
    let is_decision_head =
        name.starts_with("attribute_relation_") || name.starts_with("set_relation_");
    match scope {
        ParameterScope::All => true,
        ParameterScope::DecisionHead => is_decision_head,
        ParameterScope::ShapeFillDecision => {
            is_decision_head
                || name.starts_with("attribute_head_2.")
                || name.starts_with("attribute_head_3.")
                || name.starts_with("fill_texture_head.")
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::from_args()?;
    let training_boards = load_training_boards(&config)?;
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let policy = DeployedVisualSetPolicy::new(&mut session)?;
    let state = ft_serialize::load_state_dict(&config.input_checkpoint)?;
    policy.load_state_dict(&mut session, &state, true)?;
    let (trainable_parameters, gradient_probe_parameter) =
        configure_trainable_parameters(&mut session, &policy, config.parameter_scope())?;
    let trainable_tensor_count = trainable_parameters.len();
    let graph_boundary = session.autograd_graph_node_count();
    let mut optimizer = AdamW::new(trainable_parameters, config.learning_rate).weight_decay(1.0e-4);
    let mut rng = Rng::new(config.data_seed);
    let started = Instant::now();

    print_config(&config);
    println!("training_boards: {}", training_boards.description());
    println!("trainable_tensors: {trainable_tensor_count}");

    for step in 1..=config.steps {
        let kind = board_kind_for_step(step);
        let (board, mined_hardness) = mine_hard_board(
            &mut session,
            &policy,
            &training_boards,
            &mut rng,
            kind,
            graph_boundary,
        )?;
        let input = session.tensor_variable(board.pixels, vec![BOARD_CARDS, CARD_PIXELS], false)?;
        let (set_logits, attribute_logits, relation_logits) =
            policy.forward_board_with_relations(&mut session, input, BOARD_CARDS)?;
        let board_loss = board_action_loss(&mut session, set_logits, &board.set_targets, kind)?;
        let dense_loss = if config.dense_triple_loss {
            Some(balanced_dense_triple_loss(
                &mut session,
                set_logits,
                &board.set_targets,
                kind,
            )?)
        } else {
            None
        };

        let attribute_logits = session.tensor_reshape(
            attribute_logits,
            vec![BOARD_CARDS * ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        )?;
        let attribute_targets = session.tensor_variable(
            board.attribute_targets,
            vec![BOARD_CARDS * ATTRIBUTE_COUNT],
            false,
        )?;
        let attribute_loss =
            CrossEntropyLoss.forward(&mut session, attribute_logits, attribute_targets)?;
        let attribute_loss = session.tensor_mul_scalar(attribute_loss, ATTRIBUTE_LOSS_WEIGHT)?;

        let relation_logits = session.tensor_reshape(
            relation_logits,
            vec![TRIPLE_COUNT * ATTRIBUTE_COUNT, RELATION_CLASS_COUNT],
        )?;
        let relation_targets = session.tensor_variable(
            board.relation_targets,
            vec![TRIPLE_COUNT * ATTRIBUTE_COUNT],
            false,
        )?;
        let relation_loss =
            CrossEntropyLoss.forward(&mut session, relation_logits, relation_targets)?;
        let relation_loss = session.tensor_mul_scalar(relation_loss, RELATION_LOSS_WEIGHT)?;

        let auxiliary_loss = session.tensor_add(attribute_loss, relation_loss)?;
        let board_supervision = if let Some(dense_loss) = dense_loss {
            session.tensor_add(board_loss, dense_loss)?
        } else {
            board_loss
        };
        let loss = session.tensor_add(board_supervision, auxiliary_loss)?;
        let loss_value = session.tensor_values(loss)?[0];
        let board_loss_value = session.tensor_values(board_loss)?[0];
        if !loss_value.is_finite() || !board_loss_value.is_finite() {
            return Err(format!("non-finite loss at step {step}").into());
        }
        let report = session.tensor_backward(loss)?;
        let first_gradient = session
            .tensor_accumulated_gradient(gradient_probe_parameter)?
            .ok_or("missing trainable-parameter gradient")?;
        if first_gradient.iter().any(|gradient| !gradient.is_finite()) {
            return Err(format!("non-finite shared-weight gradient at step {step}").into());
        }
        optimizer.step(&mut session, &report)?;
        optimizer.zero_grad(&mut session)?;
        session.truncate_autograd_graph(graph_boundary);

        if step == 1 || step % 25 == 0 || step == config.steps {
            println!(
                "step {step:>4}/{} kind={kind:?} loss={loss_value:.6} board_loss={board_loss_value:.6} mined_hardness={mined_hardness:.6}",
                config.steps
            );
        }
        if step == 250 || step == config.steps {
            save_snapshot(&session, &policy, &config.output_root, step)?;
        }
    }

    print_training_summary(&config, started)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_dense_loss_gives_classes_equal_weight_and_finite_gradients()
    -> Result<(), Box<dyn Error>> {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits =
            session.tensor_variable(vec![0.0; TRIPLE_COUNT], vec![TRIPLE_COUNT, 1], true)?;
        let mut targets = vec![0.0; TRIPLE_COUNT];
        targets[0] = 1.0;
        let loss = balanced_dense_triple_loss(&mut session, logits, &targets, BoardKind::HasSet)?;
        let value = session.tensor_values(loss)?[0];
        assert!((value - 2.0_f64.ln()).abs() < 1.0e-12);
        let _report = session.tensor_backward(loss)?;
        let gradient = session
            .tensor_accumulated_gradient(logits)?
            .ok_or("missing dense-loss input gradient")?;
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(gradient[0] < 0.0);
        assert!(gradient[1] > 0.0);
        Ok(())
    }

    #[test]
    fn no_set_dense_loss_is_the_mean_negative_bce() -> Result<(), Box<dyn Error>> {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits =
            session.tensor_variable(vec![0.0; TRIPLE_COUNT], vec![TRIPLE_COUNT, 1], false)?;
        let targets = vec![0.0; TRIPLE_COUNT];
        let loss = balanced_dense_triple_loss(&mut session, logits, &targets, BoardKind::NoSet)?;
        assert!((session.tensor_values(loss)?[0] - 2.0_f64.ln()).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn parameter_scopes_freeze_unrelated_specialists() {
        assert!(parameter_is_trainable(
            "set_relation_out.weight",
            ParameterScope::DecisionHead
        ));
        assert!(!parameter_is_trainable(
            "attribute_head_2.classifier.weight",
            ParameterScope::DecisionHead
        ));
        assert!(parameter_is_trainable(
            "attribute_head_2.classifier.weight",
            ParameterScope::ShapeFillDecision
        ));
        assert!(parameter_is_trainable(
            "fill_texture_head.classifier.bias",
            ParameterScope::ShapeFillDecision
        ));
        assert!(!parameter_is_trainable(
            "attribute_head_0.classifier.weight",
            ParameterScope::ShapeFillDecision
        ));
        assert!(!parameter_is_trainable(
            "shared_convolution.weight",
            ParameterScope::ShapeFillDecision
        ));
    }
}
