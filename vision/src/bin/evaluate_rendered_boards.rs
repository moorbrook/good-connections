use std::{env, error::Error, path::PathBuf};

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_nn::Module;
use set_vision_data::{BoardKind, RenderedBoardCollection, decode_card};
use set_vision_policy::{BoardScores, DeployedVisualSetPolicy, RawBoardAction};

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_options()?;
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let model = DeployedVisualSetPolicy::new(&mut session)?;
    let state = ft_serialize::load_state_dict(&config.checkpoint)?;
    model.load_state_dict(&mut session, &state, true)?;
    let graph_boundary = session.autograd_graph_node_count();
    session.no_grad_enter();

    let started = std::time::Instant::now();
    let mut correct = 0_usize;
    let mut has_set_correct = 0_usize;
    let mut has_set_total = 0_usize;
    let mut no_set_correct = 0_usize;
    let mut no_set_total = 0_usize;
    let board_count =
        RenderedBoardCollection::visit_framed(std::io::stdin().lock(), |index, board| {
            let pixels = board.cards.iter().flatten().copied().collect::<Vec<_>>();
            let scores = model.score_board(&mut session, pixels, board.cards.len())?;
            let decision = scores
                .best_raw_decision()
                .ok_or("model returned no board decision")?;
            let action_correct = match decision.action {
                RawBoardAction::NoSet => board.kind == BoardKind::NoSet,
                RawBoardAction::Set(positions) => {
                    let ids = positions.map(|position| board.ids[position]);
                    is_set(ids)
                }
            };
            correct += usize::from(action_correct);
            match board.kind {
                BoardKind::HasSet => {
                    has_set_total += 1;
                    has_set_correct += usize::from(action_correct);
                }
                BoardKind::NoSet => {
                    no_set_total += 1;
                    no_set_correct += usize::from(action_correct);
                }
            }
            write_evidence(
                index + config.index_offset,
                board.kind,
                board.ids,
                &scores,
                action_correct,
            )?;
            session.truncate_autograd_graph(graph_boundary);
            Ok(())
        })?;
    eprintln!(
        "rendered_board_summary boards={} raw_accuracy={:.8} has_set_accuracy={:.8} no_set_accuracy={:.8} seconds={:.3}",
        board_count,
        ratio(correct, board_count)?,
        ratio(has_set_correct, has_set_total)?,
        ratio(no_set_correct, no_set_total)?,
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}

struct Options {
    checkpoint: PathBuf,
    index_offset: usize,
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let mut checkpoint = None;
    let mut index_offset = 0;
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(
                    arguments.next().ok_or("--checkpoint requires a path")?,
                ));
            }
            "--index-offset" => {
                index_offset = arguments
                    .next()
                    .ok_or("--index-offset requires a value")?
                    .parse()?;
            }
            _ => {
                return Err(
                    "usage: evaluate_rendered_boards --checkpoint PATH [--index-offset N]".into(),
                );
            }
        }
    }
    Ok(Options {
        checkpoint: checkpoint.ok_or("--checkpoint is required")?,
        index_offset,
    })
}

fn write_evidence(
    index: usize,
    kind: BoardKind,
    ids: &[usize],
    scores: &BoardScores,
    correct: bool,
) -> Result<(), Box<dyn Error>> {
    let raw = scores
        .best_raw_decision()
        .ok_or("model returned no board decision")?;
    let values = &scores.scores;
    let mut ranked = values.clone();
    ranked.sort_by(|left, right| right.total_cmp(left));
    let value_count = f64::from(u32::try_from(values.len())?);
    let mean = values.iter().sum::<f64>() / value_count;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / value_count;
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let logsumexp = maximum
        + values
            .iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln();
    let positive_count = u32::try_from(values.iter().filter(|&&value| value >= 0.0).count())?;
    let positive_fraction = f64::from(positive_count) / value_count;
    let kind = match kind {
        BoardKind::HasSet => "has-set",
        BoardKind::NoSet => "no-set",
    };
    let action_is_set = matches!(raw.action, RawBoardAction::Set(_));
    let positions = raw.strongest_triple.triple;
    print!(
        "{{\"version\":1,\"index\":{index},\"kind\":\"{kind}\",\"boardIds\":[{}],\"correct\":{correct},\"actionIsSet\":{action_is_set},\"strongestPositions\":[{},{},{}],\"score\":{:.17},\"scoreMargin\":{:.17},\"attributeEntropy\":{:.17},\"attributeMargin\":{:.17},\"scoreMean\":{mean:.17},\"scoreStddev\":{:.17},\"scoreLogsumexp\":{logsumexp:.17},\"positiveScoreFraction\":{positive_fraction:.17},\"sortedTripleScores\":[",
        ids.iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
        positions[0],
        positions[1],
        positions[2],
        raw.strongest_triple.score,
        raw.strongest_triple.score_margin,
        raw.strongest_triple.mean_attribute_entropy,
        raw.strongest_triple.min_attribute_margin,
        variance.sqrt(),
    );
    for (score_index, score) in ranked.iter().enumerate() {
        if score_index > 0 {
            print!(",");
        }
        print!("{score:.17}");
    }
    println!("]}}");
    Ok(())
}

fn is_set(ids: [usize; 3]) -> bool {
    let cards = ids.map(decode_card);
    (0..4)
        .all(|attribute| (cards[0][attribute] + cards[1][attribute] + cards[2][attribute]) % 3 == 0)
}

fn ratio(numerator: usize, denominator: usize) -> Result<f64, std::num::TryFromIntError> {
    if denominator == 0 {
        Ok(0.0)
    } else {
        Ok(f64::from(u32::try_from(numerator)?) / f64::from(u32::try_from(denominator)?))
    }
}

#[cfg(test)]
mod tests {
    use super::{is_set, ratio};

    #[test]
    fn scores_set_rules_without_selecting_candidates() {
        assert!(is_set([0, 40, 80]));
        assert!(!is_set([0, 1, 3]));
    }

    #[test]
    fn empty_ratio_is_zero() {
        assert_eq!(
            ratio(0, 0).unwrap_or_else(|error| unreachable!("zero counts convert: {error}")),
            0.0,
        );
    }
}
