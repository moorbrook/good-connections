#![forbid(unsafe_code)]

mod config;

use config::TrainConfig;
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_nn::{BCEWithLogitsLoss, CrossEntropyLoss, LossModule, Module};
use ft_optim::{AdamW, Optimizer};
use set_vision_data::{AtlasCollection, Rng, decode_card};
use set_vision_policy::{BoardDecision, CARD_PIXELS, VisualSetPolicy};
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
            let batch = atlases.make_training_batch(&mut rng, config.batch_size, config.augment);
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
    for _ in 0..batches {
        let batch = atlases.make_batch(&mut triple_rng, BATCH_SIZE, false);
        let input =
            session.tensor_variable(batch.pixels, vec![BATCH_SIZE, 3, CARD_PIXELS], false)?;
        let (set_logits, attribute_logits) = model.forward_with_attributes(session, input)?;
        for (logit, target) in session
            .tensor_values(set_logits)?
            .iter()
            .zip(batch.set_targets)
        {
            triple_correct += usize::from((*logit >= 0.0) == (target >= 0.5));
        }
        for (attribute_index, (logits, target)) in session
            .tensor_values(attribute_logits)?
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
    }
    let mut board_failures = 0_usize;
    for _ in 0..boards {
        let board = atlases.make_board(&mut board_rng, BOARD_CARDS);
        let output = model.score_board(session, board.pixels, BOARD_CARDS)?;
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
    }
    session.no_grad_exit();
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
