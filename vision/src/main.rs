#![forbid(unsafe_code)]

mod config;

use config::TrainConfig;
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_nn::{BCEWithLogitsLoss, CrossEntropyLoss, LossModule, Module};
use ft_optim::{AdamW, Optimizer};
use set_vision_data::{AtlasCollection, Rng, decode_card};
use set_vision_policy::{BoardDecision, BoardScores, CARD_PIXELS, VisualSetPolicy};
use std::io::Write;

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(config) = TrainConfig::from_args().map_err(std::io::Error::other)? else {
        println!("{}", TrainConfig::usage());
        return Ok(());
    };
    std::fs::create_dir_all(&config.output_dir)?;
    // Evaluation-only runs do not need the training corpus. Avoid loading it so
    // a shard process holds only its validation atlas in memory.
    let atlases = if config.eval_only && !config.probe_only {
        None
    } else {
        Some(AtlasCollection::load(&config.atlas_dir)?)
    };
    let validation_atlases = AtlasCollection::load(&config.validation_atlas_dir)?;
    let mut rng = Rng::new(config.seed);

    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let model = VisualSetPolicy::new(&mut session)?;
    if config.eval_only || config.probe_only {
        let checkpoint = config.output_dir.join("set-vision.ftsv");
        let state = ft_serialize::load_state_dict(&checkpoint)?;
        model.load_state_dict(&mut session, &state, true)?;
        println!("checkpoint: {}", checkpoint.display());
    }
    let graph_boundary = session.autograd_graph_node_count();
    let mut ensemble = if let Some(output_dir) = config.ensemble_output_dir.as_ref() {
        let mut ensemble_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ensemble_model = VisualSetPolicy::new(&mut ensemble_session)?;
        let checkpoint = output_dir.join("set-vision.ftsv");
        let state = ft_serialize::load_state_dict(&checkpoint)?;
        ensemble_model.load_state_dict(&mut ensemble_session, &state, true)?;
        let graph_boundary = ensemble_session.autograd_graph_node_count();
        println!("ensemble_checkpoint: {}", checkpoint.display());
        let mut third = None;
        if let Some(third_output_dir) = config.ensemble_third_output_dir.as_ref() {
            let mut third_session = FrankenTorchSession::new(ExecutionMode::Strict);
            let third_model = VisualSetPolicy::new(&mut third_session)?;
            let third_checkpoint = third_output_dir.join("set-vision.ftsv");
            let third_state = ft_serialize::load_state_dict(&third_checkpoint)?;
            third_model.load_state_dict(&mut third_session, &third_state, true)?;
            let third_graph_boundary = third_session.autograd_graph_node_count();
            println!("ensemble_third_checkpoint: {}", third_checkpoint.display());
            third = Some(EnsembleMember {
                session: third_session,
                model: third_model,
                graph_boundary: third_graph_boundary,
            });
        }
        let mut fourth = None;
        if let Some(fourth_output_dir) = config.ensemble_fourth_output_dir.as_ref() {
            let mut fourth_session = FrankenTorchSession::new(ExecutionMode::Strict);
            let fourth_model = VisualSetPolicy::new(&mut fourth_session)?;
            let fourth_checkpoint = fourth_output_dir.join("set-vision.ftsv");
            let fourth_state = ft_serialize::load_state_dict(&fourth_checkpoint)?;
            fourth_model.load_state_dict(&mut fourth_session, &fourth_state, true)?;
            let fourth_graph_boundary = fourth_session.autograd_graph_node_count();
            println!(
                "ensemble_fourth_checkpoint: {}",
                fourth_checkpoint.display()
            );
            fourth = Some(EnsembleMember {
                session: fourth_session,
                model: fourth_model,
                graph_boundary: fourth_graph_boundary,
            });
        }
        let weights = if let Some(weights) = config.ensemble_four_weights {
            weights
        } else if let Some(weights) = config.ensemble_weights {
            [weights[0], weights[1], weights[2], 0.0]
        } else {
            [
                config.ensemble_primary_weight,
                1.0 - config.ensemble_primary_weight,
                0.0,
                0.0,
            ]
        };
        if fourth.is_some() {
            println!(
                "ensemble_weights: {:.17},{:.17},{:.17},{:.17}",
                weights[0], weights[1], weights[2], weights[3]
            );
        } else if third.is_some() {
            println!(
                "ensemble_weights: {:.17},{:.17},{:.17}",
                weights[0], weights[1], weights[2]
            );
        } else {
            println!(
                "ensemble_primary_weight: {:.17}",
                config.ensemble_primary_weight
            );
        }
        Some(EnsemblePolicy {
            session: ensemble_session,
            model: ensemble_model,
            graph_boundary,
            third,
            fourth,
            weights,
        })
    } else {
        None
    };

    if config.probe_only {
        let atlases = atlases.as_ref().ok_or("probe requires an atlas corpus")?;
        let mut probe_rng = Rng::new(99005);
        let batch = atlases.make_batch(&mut probe_rng, 64, false);
        let input = session.tensor_variable(batch.pixels, vec![64, 3, CARD_PIXELS], false)?;
        let set_targets = session.tensor_variable(batch.set_targets, vec![64, 1], false)?;
        let attribute_targets =
            session.tensor_variable(batch.attribute_targets, vec![64 * 3 * 4], false)?;
        let (set_logits, attribute_logits) = model.forward_with_attributes(&mut session, input)?;
        let attribute_logits = session.tensor_reshape(attribute_logits, vec![64 * 3 * 4, 3])?;
        let set_loss = BCEWithLogitsLoss.forward(&mut session, set_logits, set_targets)?;
        let attribute_loss =
            CrossEntropyLoss.forward(&mut session, attribute_logits, attribute_targets)?;
        let attribute_loss =
            session.tensor_mul_scalar(attribute_loss, config.attribute_loss_weight)?;
        let loss = session.tensor_add(set_loss, attribute_loss)?;
        println!("probe_loss: {:.9}", session.tensor_values(loss)?[0]);
        let _report = session.tensor_backward(loss)?;
        let first_parameter = model.parameters()[0];
        let first_gradient = session
            .tensor_accumulated_gradient(first_parameter)?
            .ok_or("missing shared weight gradient")?;
        println!("probe_shared_weight_gradient_0: {:?}", &first_gradient[..8]);
        return Ok(());
    }

    if !config.eval_only {
        let atlases = atlases
            .as_ref()
            .ok_or("training requires an atlas corpus")?;
        let mut optimizer =
            AdamW::new(model.parameters(), config.learning_rate).weight_decay(1.0e-4);
        println!(
            "training {} steps × {} triples from {}",
            config.steps,
            config.batch_size,
            config.atlas_dir.display()
        );
        for step in 1..=config.steps {
            let progress =
                f64::from(u32::try_from(step - 1)?) / f64::from(u32::try_from(config.steps)?);
            let cosine = 1.0_f64.midpoint((std::f64::consts::PI * progress).cos());
            optimizer.set_lr(config.learning_rate * (0.05 + 0.95 * cosine));
            let batch = atlases.make_batch(&mut rng, config.batch_size, config.augment);
            let input = session.tensor_variable(
                batch.pixels,
                vec![config.batch_size, 3, CARD_PIXELS],
                false,
            )?;
            let set_targets =
                session.tensor_variable(batch.set_targets, vec![config.batch_size, 1], false)?;
            let attribute_targets = session.tensor_variable(
                batch.attribute_targets,
                vec![config.batch_size * 3 * 4],
                false,
            )?;
            let (set_logits, attribute_logits) =
                model.forward_with_attributes(&mut session, input)?;
            let attribute_logits =
                session.tensor_reshape(attribute_logits, vec![config.batch_size * 3 * 4, 3])?;
            let set_loss = BCEWithLogitsLoss.forward(&mut session, set_logits, set_targets)?;
            let attribute_loss =
                CrossEntropyLoss.forward(&mut session, attribute_logits, attribute_targets)?;
            let attribute_loss =
                session.tensor_mul_scalar(attribute_loss, config.attribute_loss_weight)?;
            let loss = session.tensor_add(set_loss, attribute_loss)?;
            let loss_value = session.tensor_values(loss)?[0];
            let report = session.tensor_backward(loss)?;
            optimizer.step(&mut session, &report)?;
            optimizer.zero_grad(&mut session)?;
            session.truncate_autograd_graph(graph_boundary);

            if step == 1 || step == config.steps || step % 100 == 0 {
                println!("step {step:>4}/{}  loss {loss_value:.5}", config.steps);
            }
        }
    }

    let evaluation = evaluate(
        &mut session,
        &model,
        ensemble.as_mut(),
        &validation_atlases,
        graph_boundary,
        config.eval_batches,
        config.eval_boards,
        config.triple_seed,
        config.board_seed,
    )?;
    println!("triple_seed: {}", config.triple_seed);
    println!("board_seed: {}", config.board_seed);
    println!("triple_accuracy: {:.6}", evaluation.triple_accuracy);
    println!(
        "triple_correct: {}/{}",
        evaluation.triple_correct, evaluation.triple_total
    );
    println!("attribute_accuracy: {:.6}", evaluation.attribute_accuracy);
    println!(
        "attribute_correct: {}/{}",
        evaluation.attribute_correct, evaluation.attribute_total
    );
    println!(
        "attribute_accuracy_by_kind: count={:.6},colour={:.6},shape={:.6},fill={:.6}",
        evaluation.attribute_accuracy_by_kind[0],
        evaluation.attribute_accuracy_by_kind[1],
        evaluation.attribute_accuracy_by_kind[2],
        evaluation.attribute_accuracy_by_kind[3]
    );
    println!("board_accuracy: {:.6}", evaluation.board_accuracy);
    println!("board_failures: {}", evaluation.board_failures);
    for atlas in &evaluation.atlases {
        println!(
            "atlas: {} attribute_accuracy={:.6} board_accuracy={:.6} boards={}",
            atlas.name, atlas.attribute_accuracy, atlas.board_accuracy, atlas.boards
        );
    }
    println!("evaluation_seconds: {:.6}", evaluation.seconds);
    println!("evaluation_protocol: board-multiraster");
    let abstention_path = config.output_dir.join("abstention-samples.tsv");
    write_abstention_samples(&abstention_path, &evaluation.abstention_samples)?;
    println!("abstention_samples: {}", abstention_path.display());
    if !config.eval_only {
        let state = model.state_dict(&session)?;
        let checkpoint = config.output_dir.join("set-vision.ftsv");
        ft_serialize::save_state_dict(&state, &checkpoint)?;
        println!("checkpoint: {}", checkpoint.display());
    }
    Ok(())
}

struct EnsembleMember {
    session: FrankenTorchSession,
    model: VisualSetPolicy,
    graph_boundary: usize,
}

struct EnsemblePolicy {
    session: FrankenTorchSession,
    model: VisualSetPolicy,
    graph_boundary: usize,
    third: Option<EnsembleMember>,
    fourth: Option<EnsembleMember>,
    weights: [f64; 4],
}

#[derive(Debug)]
struct Evaluation {
    triple_accuracy: f64,
    attribute_accuracy: f64,
    attribute_accuracy_by_kind: [f64; 4],
    board_accuracy: f64,
    board_failures: usize,
    triple_correct: usize,
    triple_total: usize,
    attribute_correct: usize,
    attribute_total: usize,
    seconds: f64,
    atlases: Vec<AtlasEvaluation>,
    abstention_samples: Vec<AbstentionSample>,
}

#[derive(Debug)]
struct AbstentionSample {
    atlas: String,
    correct: bool,
    decision: BoardDecision,
}

#[derive(Debug)]
struct AtlasEvaluation {
    name: String,
    attribute_accuracy: f64,
    board_accuracy: f64,
    boards: usize,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn evaluate(
    session: &mut FrankenTorchSession,
    model: &VisualSetPolicy,
    mut ensemble: Option<&mut EnsemblePolicy>,
    atlases: &AtlasCollection,
    graph_boundary: usize,
    batches: usize,
    boards: usize,
    triple_seed: u64,
    board_seed: u64,
) -> Result<Evaluation, Box<dyn std::error::Error>> {
    const BATCH_SIZE: usize = 64;
    const BOARD_CARDS: usize = 12;
    let started = std::time::Instant::now();
    let mut triple_correct = 0_usize;
    let mut attribute_correct = 0_usize;
    let mut attribute_correct_by_kind = [0_usize; 4];
    let mut atlas_attribute_correct = vec![0_usize; atlases.len()];
    let mut atlas_attribute_total = vec![0_usize; atlases.len()];
    let mut atlas_board_correct = vec![0_usize; atlases.len()];
    let mut atlas_board_total = vec![0_usize; atlases.len()];
    let mut abstention_samples = Vec::with_capacity(boards);
    let mut triple_rng = Rng::new(triple_seed);
    let mut board_rng = Rng::new(board_seed);
    session.no_grad_enter();
    if let Some(ensemble) = ensemble.as_deref_mut() {
        ensemble.session.no_grad_enter();
        if let Some(third) = ensemble.third.as_mut() {
            third.session.no_grad_enter();
        }
        if let Some(fourth) = ensemble.fourth.as_mut() {
            fourth.session.no_grad_enter();
        }
    }
    for _ in 0..batches {
        let batch = atlases.make_batch(&mut triple_rng, BATCH_SIZE, false);
        let (set_values, attribute_values) = if let Some(ensemble) = ensemble.as_deref_mut() {
            if ensemble.fourth.is_some() {
                let input = session.tensor_variable(
                    batch.pixels.clone(),
                    vec![BATCH_SIZE, 3, CARD_PIXELS],
                    false,
                )?;
                let (set_logits, attribute_logits) =
                    model.forward_with_attributes(session, input)?;
                let ensemble_input = ensemble.session.tensor_variable(
                    batch.pixels.clone(),
                    vec![BATCH_SIZE, 3, CARD_PIXELS],
                    false,
                )?;
                let (ensemble_set_logits, ensemble_attribute_logits) = ensemble
                    .model
                    .forward_with_attributes(&mut ensemble.session, ensemble_input)?;
                let third = ensemble
                    .third
                    .as_mut()
                    .ok_or("third ensemble model disappeared")?;
                let third_input = third.session.tensor_variable(
                    batch.pixels.clone(),
                    vec![BATCH_SIZE, 3, CARD_PIXELS],
                    false,
                )?;
                let (third_set_logits, third_attribute_logits) = third
                    .model
                    .forward_with_attributes(&mut third.session, third_input)?;
                let fourth = ensemble
                    .fourth
                    .as_mut()
                    .ok_or("fourth ensemble model disappeared")?;
                let fourth_input = fourth.session.tensor_variable(
                    batch.pixels,
                    vec![BATCH_SIZE, 3, CARD_PIXELS],
                    false,
                )?;
                let (fourth_set_logits, fourth_attribute_logits) = fourth
                    .model
                    .forward_with_attributes(&mut fourth.session, fourth_input)?;
                let primary_set_values = session.tensor_values(set_logits)?;
                let ensemble_set_values = ensemble.session.tensor_values(ensemble_set_logits)?;
                let third_set_values = third.session.tensor_values(third_set_logits)?;
                let fourth_set_values = fourth.session.tensor_values(fourth_set_logits)?;
                let primary_attribute_values = session.tensor_values(attribute_logits)?;
                let ensemble_attribute_values =
                    ensemble.session.tensor_values(ensemble_attribute_logits)?;
                let third_attribute_values = third.session.tensor_values(third_attribute_logits)?;
                let fourth_attribute_values =
                    fourth.session.tensor_values(fourth_attribute_logits)?;
                let set_values = weighted_average_four(
                    &primary_set_values,
                    &ensemble_set_values,
                    &third_set_values,
                    &fourth_set_values,
                    ensemble.weights,
                )?;
                let attribute_values = weighted_average_four(
                    &primary_attribute_values,
                    &ensemble_attribute_values,
                    &third_attribute_values,
                    &fourth_attribute_values,
                    ensemble.weights,
                )?;
                (set_values, attribute_values)
            } else if ensemble.third.is_some() {
                let input = session.tensor_variable(
                    batch.pixels.clone(),
                    vec![BATCH_SIZE, 3, CARD_PIXELS],
                    false,
                )?;
                let (set_logits, attribute_logits) =
                    model.forward_with_attributes(session, input)?;
                let ensemble_input = ensemble.session.tensor_variable(
                    batch.pixels.clone(),
                    vec![BATCH_SIZE, 3, CARD_PIXELS],
                    false,
                )?;
                let (ensemble_set_logits, ensemble_attribute_logits) = ensemble
                    .model
                    .forward_with_attributes(&mut ensemble.session, ensemble_input)?;
                let third = ensemble
                    .third
                    .as_mut()
                    .ok_or("third ensemble model disappeared")?;
                let third_input = third.session.tensor_variable(
                    batch.pixels,
                    vec![BATCH_SIZE, 3, CARD_PIXELS],
                    false,
                )?;
                let (third_set_logits, third_attribute_logits) = third
                    .model
                    .forward_with_attributes(&mut third.session, third_input)?;
                let primary_set_values = session.tensor_values(set_logits)?;
                let ensemble_set_values = ensemble.session.tensor_values(ensemble_set_logits)?;
                let third_set_values = third.session.tensor_values(third_set_logits)?;
                let primary_attribute_values = session.tensor_values(attribute_logits)?;
                let ensemble_attribute_values =
                    ensemble.session.tensor_values(ensemble_attribute_logits)?;
                let third_attribute_values = third.session.tensor_values(third_attribute_logits)?;
                let set_values = weighted_average_three(
                    &primary_set_values,
                    &ensemble_set_values,
                    &third_set_values,
                    [
                        ensemble.weights[0],
                        ensemble.weights[1],
                        ensemble.weights[2],
                    ],
                )?;
                let attribute_values = weighted_average_three(
                    &primary_attribute_values,
                    &ensemble_attribute_values,
                    &third_attribute_values,
                    [
                        ensemble.weights[0],
                        ensemble.weights[1],
                        ensemble.weights[2],
                    ],
                )?;
                (set_values, attribute_values)
            } else {
                let input = session.tensor_variable(
                    batch.pixels.clone(),
                    vec![BATCH_SIZE, 3, CARD_PIXELS],
                    false,
                )?;
                let (set_logits, attribute_logits) =
                    model.forward_with_attributes(session, input)?;
                let ensemble_input = ensemble.session.tensor_variable(
                    batch.pixels,
                    vec![BATCH_SIZE, 3, CARD_PIXELS],
                    false,
                )?;
                let (ensemble_set_logits, ensemble_attribute_logits) = ensemble
                    .model
                    .forward_with_attributes(&mut ensemble.session, ensemble_input)?;
                let set_values = average_logits(
                    &session.tensor_values(set_logits)?,
                    &ensemble.session.tensor_values(ensemble_set_logits)?,
                    ensemble.weights[0],
                )?;
                let attribute_values = average_logits(
                    &session.tensor_values(attribute_logits)?,
                    &ensemble.session.tensor_values(ensemble_attribute_logits)?,
                    ensemble.weights[0],
                )?;
                (set_values, attribute_values)
            }
        } else {
            let input =
                session.tensor_variable(batch.pixels, vec![BATCH_SIZE, 3, CARD_PIXELS], false)?;
            let (set_logits, attribute_logits) = model.forward_with_attributes(session, input)?;
            (
                session.tensor_values(set_logits)?,
                session.tensor_values(attribute_logits)?,
            )
        };
        for (logit, target) in set_values.iter().zip(batch.set_targets) {
            triple_correct += usize::from((*logit >= 0.0) == (target >= 0.5));
        }
        for (attribute_index, (logits, target)) in attribute_values
            .as_chunks::<3>()
            .0
            .iter()
            .zip(batch.attribute_targets)
            .enumerate()
        {
            let predicted = logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map_or(usize::MAX, |(index, _)| index);
            let correct = predicted == attribute_target_index(target);
            attribute_correct += usize::from(correct);
            attribute_correct_by_kind[attribute_index % 4] += usize::from(correct);
            let atlas_index = batch.atlas_indices[attribute_index / 12];
            atlas_attribute_correct[atlas_index] += usize::from(correct);
            atlas_attribute_total[atlas_index] += 1;
        }
        session.truncate_autograd_graph(graph_boundary);
        if let Some(ensemble) = ensemble.as_deref_mut() {
            ensemble
                .session
                .truncate_autograd_graph(ensemble.graph_boundary);
            if let Some(third) = ensemble.third.as_mut() {
                third.session.truncate_autograd_graph(third.graph_boundary);
            }
            if let Some(fourth) = ensemble.fourth.as_mut() {
                fourth
                    .session
                    .truncate_autograd_graph(fourth.graph_boundary);
            }
        }
    }
    let mut board_failures = 0_usize;
    for _ in 0..boards {
        let board = atlases.make_board(&mut board_rng, BOARD_CARDS);
        let output = if let Some(ensemble) = ensemble.as_deref_mut() {
            if ensemble.fourth.is_some() {
                let primary = model.score_board(session, board.pixels.clone(), BOARD_CARDS)?;
                let secondary = ensemble.model.score_board(
                    &mut ensemble.session,
                    board.pixels.clone(),
                    BOARD_CARDS,
                )?;
                let third = ensemble
                    .third
                    .as_mut()
                    .ok_or("third ensemble model disappeared")?;
                let tertiary = third.model.score_board(
                    &mut third.session,
                    board.pixels.clone(),
                    BOARD_CARDS,
                )?;
                let fourth = ensemble
                    .fourth
                    .as_mut()
                    .ok_or("fourth ensemble model disappeared")?;
                let quaternary =
                    fourth
                        .model
                        .score_board(&mut fourth.session, board.pixels, BOARD_CARDS)?;
                average_board_scores_four(
                    &primary,
                    &secondary,
                    &tertiary,
                    &quaternary,
                    ensemble.weights,
                )?
            } else if ensemble.third.is_some() {
                let primary = model.score_board(session, board.pixels.clone(), BOARD_CARDS)?;
                let secondary = ensemble.model.score_board(
                    &mut ensemble.session,
                    board.pixels.clone(),
                    BOARD_CARDS,
                )?;
                let third = ensemble
                    .third
                    .as_mut()
                    .ok_or("third ensemble model disappeared")?;
                let tertiary =
                    third
                        .model
                        .score_board(&mut third.session, board.pixels, BOARD_CARDS)?;
                average_board_scores_three(
                    &primary,
                    &secondary,
                    &tertiary,
                    [
                        ensemble.weights[0],
                        ensemble.weights[1],
                        ensemble.weights[2],
                    ],
                )?
            } else {
                let primary = model.score_board(session, board.pixels.clone(), BOARD_CARDS)?;
                let secondary =
                    ensemble
                        .model
                        .score_board(&mut ensemble.session, board.pixels, BOARD_CARDS)?;
                average_board_scores(&primary, &secondary, ensemble.weights[0])?
            }
        } else {
            model.score_board(session, board.pixels, BOARD_CARDS)?
        };
        let decision = output
            .best_decision()
            .ok_or("policy produced no finite board decision")?;
        let best = output
            .triples
            .iter()
            .position(|triple| *triple == decision.triple)
            .ok_or("policy decision was absent from its triple list")?;
        let correct = board.set_targets.get(best).copied() == Some(1.0);
        abstention_samples.push(AbstentionSample {
            atlas: atlases.name(board.atlas_index).to_string(),
            correct,
            decision,
        });
        if !correct {
            let best_valid = output
                .scores
                .iter()
                .enumerate()
                .filter(|(index, _)| {
                    board
                        .set_targets
                        .get(*index)
                        .is_some_and(|target| *target > 0.5)
                })
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map_or(usize::MAX, |(index, _)| index);
            let selected = output.triples.get(best).copied();
            let selected_ids = selected.map(|triple| triple.map(|position| board.ids[position]));
            let predicted = selected.map(|triple| {
                triple.map(|position| {
                    predicted_attributes(
                        &output.attribute_logits[position * 12..(position + 1) * 12],
                    )
                })
            });
            let best_valid_positions = output.triples.get(best_valid).copied();
            let best_valid_ids =
                best_valid_positions.map(|triple| triple.map(|position| board.ids[position]));
            println!(
                "board_failure: atlas={} selected={:?} actual={:?} predicted={predicted:?} score={:.6} best_valid={:?} best_valid_score={:.6}",
                atlases.name(board.atlas_index),
                selected_ids,
                selected_ids.map(|triple| triple.map(decode_card)),
                output
                    .scores
                    .get(best)
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY),
                best_valid_ids,
                output
                    .scores
                    .get(best_valid)
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY),
            );
        }
        board_failures += usize::from(!correct);
        atlas_board_correct[board.atlas_index] += usize::from(correct);
        atlas_board_total[board.atlas_index] += 1;
        session.truncate_autograd_graph(graph_boundary);
        if let Some(ensemble) = ensemble.as_deref_mut() {
            ensemble
                .session
                .truncate_autograd_graph(ensemble.graph_boundary);
            if let Some(third) = ensemble.third.as_mut() {
                third.session.truncate_autograd_graph(third.graph_boundary);
            }
            if let Some(fourth) = ensemble.fourth.as_mut() {
                fourth
                    .session
                    .truncate_autograd_graph(fourth.graph_boundary);
            }
        }
    }
    session.no_grad_exit();
    if let Some(ensemble) = ensemble {
        ensemble.session.no_grad_exit();
        if let Some(third) = ensemble.third.as_mut() {
            third.session.no_grad_exit();
        }
        if let Some(fourth) = ensemble.fourth.as_mut() {
            fourth.session.no_grad_exit();
        }
    }
    let triple_total = batches * BATCH_SIZE;
    let attribute_total = triple_total * 3 * 4;
    Ok(Evaluation {
        triple_accuracy: ratio(triple_correct, triple_total)?,
        attribute_accuracy: ratio(attribute_correct, attribute_total)?,
        attribute_accuracy_by_kind: std::array::from_fn(|index| {
            ratio(attribute_correct_by_kind[index], attribute_total / 4).unwrap_or(0.0)
        }),
        board_accuracy: ratio(boards - board_failures, boards)?,
        board_failures,
        triple_correct,
        triple_total,
        attribute_correct,
        attribute_total,
        seconds: started.elapsed().as_secs_f64(),
        abstention_samples,
        atlases: (0..atlases.len())
            .map(|index| {
                Ok(AtlasEvaluation {
                    name: atlases.name(index).to_string(),
                    attribute_accuracy: ratio(
                        atlas_attribute_correct[index],
                        atlas_attribute_total[index],
                    )?,
                    board_accuracy: ratio(atlas_board_correct[index], atlas_board_total[index])?,
                    boards: atlas_board_total[index],
                })
            })
            .collect::<Result<Vec<_>, std::num::TryFromIntError>>()?,
    })
}

fn average_logits(
    first: &[f64],
    second: &[f64],
    primary_weight: f64,
) -> Result<Vec<f64>, std::io::Error> {
    validate_primary_weight(primary_weight)?;
    if first.len() != second.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "ensemble logit length mismatch: {} versus {}",
                first.len(),
                second.len()
            ),
        ));
    }
    let mut averaged = Vec::with_capacity(first.len());
    for (&left, &right) in first.iter().zip(second) {
        if !left.is_finite() || !right.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ensemble logits must be finite",
            ));
        }
        let value = if primary_weight.to_bits() == 0.5_f64.to_bits() {
            left.midpoint(right)
        } else {
            primary_weight * left + (1.0 - primary_weight) * right
        };
        if !value.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ensemble logit average must be finite",
            ));
        }
        averaged.push(value);
    }
    Ok(averaged)
}

fn average_board_scores(
    first: &BoardScores,
    second: &BoardScores,
    primary_weight: f64,
) -> Result<BoardScores, std::io::Error> {
    if first.triples != second.triples {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ensemble triple ordering mismatch",
        ));
    }
    Ok(BoardScores {
        triples: first.triples.clone(),
        scores: average_logits(&first.scores, &second.scores, primary_weight)?,
        attribute_logits: average_logits(
            &first.attribute_logits,
            &second.attribute_logits,
            primary_weight,
        )?,
    })
}

fn weighted_average_three(
    first: &[f64],
    second: &[f64],
    third: &[f64],
    weights: [f64; 3],
) -> Result<Vec<f64>, std::io::Error> {
    validate_three_weights(weights)?;
    if first.len() != second.len() || first.len() != third.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "three-model ensemble logit lengths differ: {}, {}, {}",
                first.len(),
                second.len(),
                third.len()
            ),
        ));
    }
    let mut averaged = Vec::with_capacity(first.len());
    for ((&left, &middle), &right) in first.iter().zip(second).zip(third) {
        if !left.is_finite() || !middle.is_finite() || !right.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ensemble logits must be finite",
            ));
        }
        let value = weights[0] * left + weights[1] * middle + weights[2] * right;
        if !value.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ensemble logit average must be finite",
            ));
        }
        averaged.push(value);
    }
    Ok(averaged)
}

fn average_board_scores_three(
    first: &BoardScores,
    second: &BoardScores,
    third: &BoardScores,
    weights: [f64; 3],
) -> Result<BoardScores, std::io::Error> {
    if first.triples != second.triples || first.triples != third.triples {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "three-model ensemble triple ordering mismatch",
        ));
    }
    Ok(BoardScores {
        triples: first.triples.clone(),
        scores: weighted_average_three(&first.scores, &second.scores, &third.scores, weights)?,
        attribute_logits: weighted_average_three(
            &first.attribute_logits,
            &second.attribute_logits,
            &third.attribute_logits,
            weights,
        )?,
    })
}

fn weighted_average_four(
    first: &[f64],
    second: &[f64],
    third: &[f64],
    fourth: &[f64],
    weights: [f64; 4],
) -> Result<Vec<f64>, std::io::Error> {
    validate_four_weights(weights)?;
    if first.len() != second.len() || first.len() != third.len() || first.len() != fourth.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "four-model ensemble logit lengths differ: {}, {}, {}, {}",
                first.len(),
                second.len(),
                third.len(),
                fourth.len()
            ),
        ));
    }
    let mut averaged = Vec::with_capacity(first.len());
    for (((&one, &two), &three), &four) in first.iter().zip(second).zip(third).zip(fourth) {
        if !one.is_finite() || !two.is_finite() || !three.is_finite() || !four.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ensemble logits must be finite",
            ));
        }
        let value = weights[0] * one + weights[1] * two + weights[2] * three + weights[3] * four;
        if !value.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ensemble logit average must be finite",
            ));
        }
        averaged.push(value);
    }
    Ok(averaged)
}

fn average_board_scores_four(
    first: &BoardScores,
    second: &BoardScores,
    third: &BoardScores,
    fourth: &BoardScores,
    weights: [f64; 4],
) -> Result<BoardScores, std::io::Error> {
    if first.triples != second.triples
        || first.triples != third.triples
        || first.triples != fourth.triples
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "four-model ensemble triple ordering mismatch",
        ));
    }
    Ok(BoardScores {
        triples: first.triples.clone(),
        scores: weighted_average_four(
            &first.scores,
            &second.scores,
            &third.scores,
            &fourth.scores,
            weights,
        )?,
        attribute_logits: weighted_average_four(
            &first.attribute_logits,
            &second.attribute_logits,
            &third.attribute_logits,
            &fourth.attribute_logits,
            weights,
        )?,
    })
}

fn validate_primary_weight(primary_weight: f64) -> Result<(), std::io::Error> {
    if !(primary_weight.is_finite() && 0.0 < primary_weight && primary_weight < 1.0) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "ensemble primary weight must be finite and strictly between 0 and 1",
        ));
    }
    Ok(())
}

fn validate_three_weights(weights: [f64; 3]) -> Result<(), std::io::Error> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "three ensemble weights must be finite and positive",
        ));
    }
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || (total - 1.0).abs() > 1.0e-9 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "three ensemble weights must sum to one",
        ));
    }
    Ok(())
}

fn validate_four_weights(weights: [f64; 4]) -> Result<(), std::io::Error> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "four ensemble weights must be finite and positive",
        ));
    }
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || (total - 1.0).abs() > 1.0e-9 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "four ensemble weights must sum to one",
        ));
    }
    Ok(())
}

fn write_abstention_samples(
    path: &std::path::Path,
    samples: &[AbstentionSample],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    writeln!(
        writer,
        "atlas\tcorrect\tscore\tscore_margin\tmean_attribute_entropy\tmin_attribute_margin"
    )?;
    for sample in samples {
        writeln!(
            writer,
            "{}\t{}\t{:.12}\t{:.12}\t{:.12}\t{:.12}",
            sample.atlas,
            u8::from(sample.correct),
            sample.decision.score,
            sample.decision.score_margin,
            sample.decision.mean_attribute_entropy,
            sample.decision.min_attribute_margin,
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn ratio(numerator: usize, denominator: usize) -> Result<f64, std::num::TryFromIntError> {
    Ok(f64::from(u32::try_from(numerator)?) / f64::from(u32::try_from(denominator)?))
}

fn attribute_target_index(target: f64) -> usize {
    if target < 0.5 {
        0
    } else if target < 1.5 {
        1
    } else {
        2
    }
}

fn predicted_attributes(values: &[f64]) -> [usize; 4] {
    std::array::from_fn(|attribute| {
        values[attribute * 3..(attribute + 1) * 3]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map_or(usize::MAX, |(index, _)| index)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        BoardScores, average_board_scores, average_board_scores_four, average_board_scores_three,
        average_logits, weighted_average_four, weighted_average_three,
    };

    #[test]
    fn averages_logits_componentwise_without_rounding_shortcuts() {
        let averaged = average_logits(&[1.0, 3.0, -5.0], &[5.0, 7.0, 9.0], 0.5).ok();
        assert_eq!(averaged, Some(vec![3.0, 5.0, 2.0]));
    }

    #[test]
    fn non_default_primary_weight_uses_primary_and_secondary_coefficients() {
        let averaged = average_logits(&[1.0, 3.0, -5.0], &[5.0, 7.0, 9.0], 0.25).ok();
        assert_eq!(averaged, Some(vec![4.0, 6.0, 5.5]));
    }

    #[test]
    fn three_model_weighted_average_uses_all_three_logits() {
        let averaged =
            weighted_average_three(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], [0.25, 0.25, 0.5]).ok();
        assert_eq!(averaged, Some(vec![3.5, 4.5]));
    }

    #[test]
    fn four_model_weighted_average_uses_all_four_logits() {
        let averaged = weighted_average_four(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            [0.1, 0.2, 0.3, 0.4],
        )
        .ok();
        assert_eq!(averaged, Some(vec![5.0, 6.0]));
    }

    #[test]
    fn averages_board_scores_and_attribute_logits_componentwise() {
        let first = BoardScores {
            triples: vec![[0, 1, 2]],
            scores: vec![1.0, 3.0],
            attribute_logits: vec![1.0, 3.0, 5.0],
        };
        let second = BoardScores {
            triples: first.triples.clone(),
            scores: vec![5.0, 7.0],
            attribute_logits: vec![5.0, 7.0, 9.0],
        };
        let averaged = average_board_scores(&first, &second, 0.5).ok();
        assert_eq!(
            averaged.as_ref().map(|scores| scores.scores.clone()),
            Some(vec![3.0, 5.0])
        );
        assert_eq!(
            averaged.map(|scores| scores.attribute_logits),
            Some(vec![3.0, 5.0, 7.0])
        );
    }

    #[test]
    fn board_average_requires_matching_triple_order_and_shapes() {
        let first = BoardScores {
            triples: vec![[0, 1, 2], [0, 1, 3]],
            scores: vec![1.0, 2.0],
            attribute_logits: vec![1.0, 2.0, 3.0],
        };
        let second = BoardScores {
            triples: vec![[0, 1, 3], [0, 1, 2]],
            scores: vec![3.0, 4.0],
            attribute_logits: vec![4.0, 5.0, 6.0],
        };
        assert!(average_board_scores(&first, &second, 0.5).is_err());

        let second = BoardScores {
            triples: first.triples.clone(),
            scores: vec![3.0],
            attribute_logits: first.attribute_logits.clone(),
        };
        assert!(average_board_scores(&first, &second, 0.5).is_err());
    }

    #[test]
    fn board_average_rejects_non_finite_logits() {
        let first = BoardScores {
            triples: vec![[0, 1, 2]],
            scores: vec![f64::NAN],
            attribute_logits: vec![1.0],
        };
        let second = BoardScores {
            triples: first.triples.clone(),
            scores: vec![1.0],
            attribute_logits: first.attribute_logits.clone(),
        };
        assert!(average_board_scores(&first, &second, 0.5).is_err());
    }

    #[test]
    fn three_model_board_average_combines_scores_and_attributes() {
        let first = BoardScores {
            triples: vec![[0, 1, 2]],
            scores: vec![1.0, 3.0],
            attribute_logits: vec![1.0, 3.0, 5.0],
        };
        let second = BoardScores {
            triples: first.triples.clone(),
            scores: vec![3.0, 5.0],
            attribute_logits: vec![3.0, 5.0, 7.0],
        };
        let third = BoardScores {
            triples: first.triples.clone(),
            scores: vec![5.0, 7.0],
            attribute_logits: vec![5.0, 7.0, 9.0],
        };
        let averaged = average_board_scores_three(&first, &second, &third, [0.25, 0.25, 0.5]).ok();
        assert_eq!(
            averaged.as_ref().map(|scores| scores.scores.clone()),
            Some(vec![3.5, 5.5])
        );
        assert_eq!(
            averaged.map(|scores| scores.attribute_logits),
            Some(vec![3.5, 5.5, 7.5])
        );
    }

    #[test]
    fn four_model_board_average_combines_scores_and_attributes() {
        let first = BoardScores {
            triples: vec![[0, 1, 2]],
            scores: vec![1.0, 3.0],
            attribute_logits: vec![1.0, 3.0, 5.0],
        };
        let second = BoardScores {
            triples: first.triples.clone(),
            scores: vec![3.0, 5.0],
            attribute_logits: vec![3.0, 5.0, 7.0],
        };
        let third = BoardScores {
            triples: first.triples.clone(),
            scores: vec![5.0, 7.0],
            attribute_logits: vec![5.0, 7.0, 9.0],
        };
        let fourth = BoardScores {
            triples: first.triples.clone(),
            scores: vec![7.0, 9.0],
            attribute_logits: vec![7.0, 9.0, 11.0],
        };
        let averaged =
            average_board_scores_four(&first, &second, &third, &fourth, [0.1, 0.2, 0.3, 0.4]).ok();
        assert_eq!(
            averaged.as_ref().map(|scores| scores.scores.clone()),
            Some(vec![5.0, 7.0])
        );
        assert_eq!(
            averaged.map(|scores| scores.attribute_logits),
            Some(vec![5.0, 7.0, 9.0])
        );
    }

    #[test]
    fn averaging_rejects_invalid_primary_weight() {
        assert!(average_logits(&[1.0], &[2.0], 0.0).is_err());
        assert!(average_logits(&[1.0], &[2.0], 1.0).is_err());
        assert!(average_logits(&[1.0], &[2.0], f64::NAN).is_err());
    }
}
