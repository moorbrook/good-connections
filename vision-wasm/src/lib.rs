#![deny(unsafe_op_in_unsafe_fn)]

// `no_mangle` is an unsafe attribute in Rust 2024 because symbol collisions can
// be unsound. These names are the crate's deliberately tiny browser ABI;
// there are no unsafe blocks or raw-memory interfaces behind them.

use std::cell::RefCell;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_nn::Module;
#[cfg(feature = "ensemble-four")]
use set_vision_policy::BoardScores;
use set_vision_policy::{
    ATTRIBUTE_LOGITS, ATTRIBUTE_VALUE_COUNT, CARD_HEIGHT, CARD_PIXELS, CARD_WIDTH, CHANNELS,
    DeployedVisualSetPolicy, RawBoardAction,
};

const MIN_CARDS: usize = 3;
const MAX_CARDS: usize = 32;
const EMBEDDED_CHECKPOINT: &[u8] = include_bytes!("../../vision-artifacts/model/set-vision.ftsv");
#[cfg(feature = "ensemble-four")]
const EMBEDDED_CHECKPOINT_2: &[u8] =
    include_bytes!("../../vision-artifacts/model/set-vision-ensemble-2.ftsv");
#[cfg(feature = "ensemble-four")]
const EMBEDDED_CHECKPOINT_3: &[u8] =
    include_bytes!("../../vision-artifacts/model/set-vision-ensemble-3.ftsv");
#[cfg(feature = "ensemble-four")]
const EMBEDDED_CHECKPOINT_4: &[u8] =
    include_bytes!("../../vision-artifacts/model/set-vision-ensemble-4.ftsv");

#[cfg(feature = "ensemble-four")]
struct EnsembleMember {
    session: FrankenTorchSession,
    policy: DeployedVisualSetPolicy,
    graph_boundary: usize,
}

#[cfg(feature = "ensemble-four")]
impl EnsembleMember {
    fn new(checkpoint: &[u8]) -> Result<Self, String> {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let policy =
            DeployedVisualSetPolicy::new(&mut session).map_err(|error| error.to_string())?;
        let state = ft_serialize::load_state_dict_from_bytes(checkpoint)
            .map_err(|error| error.to_string())?;
        policy
            .load_state_dict(&mut session, &state, true)
            .map_err(|error| error.to_string())?;
        session.no_grad_enter();
        let graph_boundary = session.autograd_graph_node_count();
        Ok(Self {
            session,
            policy,
            graph_boundary,
        })
    }
}

struct Runtime {
    session: FrankenTorchSession,
    policy: DeployedVisualSetPolicy,
    graph_boundary: usize,
    #[cfg(feature = "ensemble-four")]
    ensemble: Vec<EnsembleMember>,
    cards: Vec<f64>,
    card_count: usize,
    last_score: f32,
    last_score_margin: f32,
    last_attribute_entropy: f32,
    last_attribute_margin: f32,
    last_action_is_set: bool,
    last_score_mean: f32,
    last_score_stddev: f32,
    last_score_logsumexp: f32,
    last_positive_score_fraction: f32,
    last_attribute_probabilities: Vec<f32>,
    last_triple_scores: Vec<f32>,
}

impl Runtime {
    fn new() -> Result<Self, String> {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let policy =
            DeployedVisualSetPolicy::new(&mut session).map_err(|error| error.to_string())?;
        let state = ft_serialize::load_state_dict_from_bytes(EMBEDDED_CHECKPOINT)
            .map_err(|error| error.to_string())?;
        policy
            .load_state_dict(&mut session, &state, true)
            .map_err(|error| error.to_string())?;
        session.no_grad_enter();
        let graph_boundary = session.autograd_graph_node_count();
        #[cfg(feature = "ensemble-four")]
        let ensemble = [
            EMBEDDED_CHECKPOINT_2,
            EMBEDDED_CHECKPOINT_3,
            EMBEDDED_CHECKPOINT_4,
        ]
        .into_iter()
        .map(EnsembleMember::new)
        .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            session,
            policy,
            graph_boundary,
            #[cfg(feature = "ensemble-four")]
            ensemble,
            cards: Vec::new(),
            card_count: 0,
            last_score: f32::NEG_INFINITY,
            last_score_margin: f32::NEG_INFINITY,
            last_attribute_entropy: f32::INFINITY,
            last_attribute_margin: f32::NEG_INFINITY,
            last_action_is_set: false,
            last_score_mean: f32::NAN,
            last_score_stddev: f32::NAN,
            last_score_logsumexp: f32::NAN,
            last_positive_score_fraction: f32::NAN,
            last_attribute_probabilities: Vec::new(),
            last_triple_scores: Vec::new(),
        })
    }

    fn reset_cards(&mut self, card_count: usize) -> bool {
        if !(MIN_CARDS..=MAX_CARDS).contains(&card_count) {
            return false;
        }
        self.cards = vec![0.0; card_count * CARD_PIXELS];
        self.card_count = card_count;
        self.last_score = f32::NEG_INFINITY;
        self.last_score_margin = f32::NEG_INFINITY;
        self.last_attribute_entropy = f32::INFINITY;
        self.last_attribute_margin = f32::NEG_INFINITY;
        self.last_action_is_set = false;
        self.last_score_mean = f32::NAN;
        self.last_score_stddev = f32::NAN;
        self.last_score_logsumexp = f32::NAN;
        self.last_positive_score_fraction = f32::NAN;
        self.last_attribute_probabilities.clear();
        self.last_triple_scores.clear();
        true
    }

    fn best_triple(&mut self) -> Result<u32, String> {
        if self.card_count < MIN_CARDS {
            return Err("too few cards".to_string());
        }
        let output = self
            .policy
            .score_board(&mut self.session, self.cards.clone(), self.card_count)
            .map_err(|error| error.to_string())?;
        self.session.truncate_autograd_graph(self.graph_boundary);
        #[cfg(feature = "ensemble-four")]
        let output = {
            let mut output = output;
            let divisor = f64::from(
                u32::try_from(self.ensemble.len() + 1).map_err(|error| error.to_string())?,
            );
            for member in &mut self.ensemble {
                let next = member
                    .policy
                    .score_board(&mut member.session, self.cards.clone(), self.card_count)
                    .map_err(|error| error.to_string())?;
                member
                    .session
                    .truncate_autograd_graph(member.graph_boundary);
                accumulate_board_scores(&mut output, next)?;
            }
            scale_board_scores(&mut output, 1.0 / divisor);
            output
        };

        let score_summary = summarize_scores(&output.scores)
            .ok_or_else(|| "policy produced no finite score summary".to_string())?;
        self.last_attribute_probabilities =
            attribute_probabilities(&output.attribute_logits, self.card_count)
                .ok_or_else(|| "policy produced invalid attribute logits".to_string())?;
        #[allow(clippy::cast_possible_truncation)]
        {
            self.last_triple_scores = output.scores.iter().map(|&score| score as f32).collect();
        }

        let Some(raw_decision) = output.best_raw_decision() else {
            return Err("policy produced no finite score".to_string());
        };
        let decision = raw_decision.strongest_triple;
        let [first, second, third] = decision.triple;
        #[allow(clippy::cast_possible_truncation)]
        {
            self.last_score = decision.score as f32;
            self.last_score_margin = decision.score_margin as f32;
            self.last_attribute_entropy = decision.mean_attribute_entropy as f32;
            self.last_attribute_margin = decision.min_attribute_margin as f32;
            self.last_score_mean = score_summary.mean as f32;
            self.last_score_stddev = score_summary.stddev as f32;
            self.last_score_logsumexp = score_summary.logsumexp as f32;
            self.last_positive_score_fraction = score_summary.positive_fraction as f32;
        }
        self.last_action_is_set = matches!(raw_decision.action, RawBoardAction::Set(_));
        let first = u32::try_from(first).map_err(|error| error.to_string())?;
        let second = u32::try_from(second).map_err(|error| error.to_string())?;
        let third = u32::try_from(third).map_err(|error| error.to_string())?;
        Ok((first << 16) | (second << 8) | third)
    }
}

#[cfg(feature = "ensemble-four")]
fn accumulate_board_scores(total: &mut BoardScores, next: BoardScores) -> Result<(), String> {
    if total.triples != next.triples || total.relation_class_count != next.relation_class_count {
        return Err("ensemble topology or triple ordering mismatch".to_string());
    }
    accumulate_values(&mut total.scores, next.scores)?;
    accumulate_values(&mut total.attribute_logits, next.attribute_logits)?;
    accumulate_values(&mut total.relation_logits, next.relation_logits)
}

#[cfg(feature = "ensemble-four")]
fn accumulate_values(total: &mut [f64], next: Vec<f64>) -> Result<(), String> {
    if total.len() != next.len() {
        return Err("ensemble logit length mismatch".to_string());
    }
    for (total, next) in total.iter_mut().zip(next) {
        *total += next;
    }
    Ok(())
}

#[cfg(feature = "ensemble-four")]
fn scale_board_scores(scores: &mut BoardScores, scale: f64) {
    for values in [
        &mut scores.scores,
        &mut scores.attribute_logits,
        &mut scores.relation_logits,
    ] {
        for value in values {
            *value *= scale;
        }
    }
}

fn attribute_probabilities(logits: &[f64], card_count: usize) -> Option<Vec<f32>> {
    if logits.len() != card_count.checked_mul(ATTRIBUTE_LOGITS)? {
        return None;
    }
    let (attributes, remainder) = logits.as_chunks::<ATTRIBUTE_VALUE_COUNT>();
    if !remainder.is_empty() {
        return None;
    }
    let mut probabilities = Vec::with_capacity(logits.len());
    for attribute in attributes {
        if attribute.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let maximum = attribute.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exponentials = attribute.map(|value| (value - maximum).exp());
        let total = exponentials.iter().sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        probabilities.extend(exponentials.map(|value| (value / total) as f32));
    }
    Some(probabilities)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoreSummary {
    mean: f64,
    stddev: f64,
    logsumexp: f64,
    positive_fraction: f64,
}

fn summarize_scores(scores: &[f64]) -> Option<ScoreSummary> {
    let finite = scores
        .iter()
        .copied()
        .filter(|score| score.is_finite())
        .collect::<Vec<_>>();
    let count = finite.len();
    if count == 0 {
        return None;
    }
    let count_f64 = f64::from(u32::try_from(count).ok()?);
    let mean = finite.iter().sum::<f64>() / count_f64;
    let variance = finite
        .iter()
        .map(|score| (score - mean).powi(2))
        .sum::<f64>()
        / count_f64;
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let logsumexp = maximum
        + finite
            .iter()
            .map(|score| (score - maximum).exp())
            .sum::<f64>()
            .ln();
    let positive_count = finite.iter().filter(|&&score| score >= 0.0).count();
    let positive_fraction = f64::from(u32::try_from(positive_count).ok()?) / count_f64;
    Some(ScoreSummary {
        mean,
        stddev: variance.sqrt(),
        logsumexp,
        positive_fraction,
    })
}

thread_local! {
    // Browser WASM is single-threaded. Thread-local ownership avoids pretending
    // FrankenTorch sessions are cross-thread state and requires no unsafe code.
    static RUNTIME: RefCell<Option<Result<Runtime, String>>> = const { RefCell::new(None) };
}

fn with_runtime<T>(action: impl FnOnce(&mut Runtime) -> T) -> Option<T> {
    RUNTIME.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Runtime::new());
        }
        slot.as_mut()?.as_mut().ok().map(action)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn model_ready() -> u32 {
    u32::from(with_runtime(|_| ()).is_some())
}

#[unsafe(no_mangle)]
pub extern "C" fn card_width() -> u32 {
    u32::try_from(CARD_WIDTH).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn card_height() -> u32 {
    u32::try_from(CARD_HEIGHT).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn card_channels() -> u32 {
    u32::try_from(CHANNELS).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn reset_cards(card_count: u32) -> u32 {
    let Ok(card_count) = usize::try_from(card_count) else {
        return 0;
    };
    u32::from(with_runtime(|runtime| runtime.reset_cards(card_count)).unwrap_or(false))
}

#[unsafe(no_mangle)]
pub extern "C" fn set_pixel(index: u32, value: f32) -> u32 {
    let Ok(index) = usize::try_from(index) else {
        return 0;
    };
    with_runtime(|runtime| {
        let Some(pixel) = runtime.cards.get_mut(index) else {
            return 0;
        };
        *pixel = f64::from(value.clamp(0.0, 1.0));
        1
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn best_triple() -> u32 {
    with_runtime(|runtime| runtime.best_triple().unwrap_or(u32::MAX)).unwrap_or(u32::MAX)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_score() -> f32 {
    with_runtime(|runtime| runtime.last_score).unwrap_or(f32::NEG_INFINITY)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_score_margin() -> f32 {
    with_runtime(|runtime| runtime.last_score_margin).unwrap_or(f32::NEG_INFINITY)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_attribute_entropy() -> f32 {
    with_runtime(|runtime| runtime.last_attribute_entropy).unwrap_or(f32::INFINITY)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_attribute_margin() -> f32 {
    with_runtime(|runtime| runtime.last_attribute_margin).unwrap_or(f32::NEG_INFINITY)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_action_is_set() -> u32 {
    u32::from(with_runtime(|runtime| runtime.last_action_is_set).unwrap_or(false))
}

#[unsafe(no_mangle)]
pub extern "C" fn last_score_mean() -> f32 {
    with_runtime(|runtime| runtime.last_score_mean).unwrap_or(f32::NAN)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_score_stddev() -> f32 {
    with_runtime(|runtime| runtime.last_score_stddev).unwrap_or(f32::NAN)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_score_logsumexp() -> f32 {
    with_runtime(|runtime| runtime.last_score_logsumexp).unwrap_or(f32::NAN)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_positive_score_fraction() -> f32 {
    with_runtime(|runtime| runtime.last_positive_score_fraction).unwrap_or(f32::NAN)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_attribute_probability(index: u32) -> f32 {
    let Ok(index) = usize::try_from(index) else {
        return f32::NAN;
    };
    with_runtime(|runtime| {
        runtime
            .last_attribute_probabilities
            .get(index)
            .copied()
            .unwrap_or(f32::NAN)
    })
    .unwrap_or(f32::NAN)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_triple_score(index: u32) -> f32 {
    let Ok(index) = usize::try_from(index) else {
        return f32::NAN;
    };
    with_runtime(|runtime| {
        runtime
            .last_triple_scores
            .get(index)
            .copied()
            .unwrap_or(f32::NAN)
    })
    .unwrap_or(f32::NAN)
}

#[cfg(test)]
mod tests {
    use super::{
        ATTRIBUTE_LOGITS, CARD_PIXELS, Runtime, attribute_probabilities, summarize_scores,
    };

    #[test]
    fn attribute_softmax_is_stable_and_grouped_per_attribute() {
        let probabilities = attribute_probabilities(&[1_000.0, 999.0, 998.0], 1);
        assert!(probabilities.is_none());

        let logits = [1_000.0, 999.0, 998.0].repeat(ATTRIBUTE_LOGITS / 3);
        let probabilities = attribute_probabilities(&logits, 1)
            .unwrap_or_else(|| unreachable!("finite complete logits must normalize"));
        assert_eq!(probabilities.len(), ATTRIBUTE_LOGITS);
        let (groups, []) = probabilities.as_chunks::<3>() else {
            unreachable!("probability fixture must contain complete groups");
        };
        for group in groups {
            assert!((group.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
            assert!(group[0] > group[1] && group[1] > group[2]);
        }
    }

    #[test]
    fn score_summary_is_stable_and_ignores_non_finite_values() {
        let summary = summarize_scores(&[0.0, 1.0, f64::NAN])
            .unwrap_or_else(|| unreachable!("finite scores must summarize"));
        assert!((summary.mean - 0.5).abs() < 1.0e-12);
        assert!((summary.stddev - 0.5).abs() < 1.0e-12);
        assert!((summary.logsumexp - (1.0_f64.exp() + 1.0).ln()).abs() < 1.0e-12);
        assert!((summary.positive_fraction - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn embedded_frankentorch_policy_loads_and_scores() -> Result<(), String> {
        let mut runtime = Runtime::new()?;
        assert!(runtime.reset_cards(3));
        for (index, pixel) in runtime.cards.iter_mut().enumerate() {
            *pixel =
                f64::from(u32::try_from(index % 251).map_err(|error| error.to_string())?) / 250.0;
        }
        assert_eq!(runtime.cards.len(), 3 * CARD_PIXELS);
        assert_eq!(runtime.best_triple()?, 0x00_01_02);
        assert!(runtime.last_score.is_finite());
        assert!(runtime.last_score_margin.is_infinite());
        assert!(runtime.last_attribute_entropy.is_finite());
        assert!(runtime.last_attribute_margin.is_finite());
        assert_eq!(runtime.last_action_is_set, runtime.last_score >= 0.0);
        assert!(runtime.last_score_mean.is_finite());
        assert!(runtime.last_score_stddev.is_finite());
        assert!(runtime.last_score_logsumexp.is_finite());
        assert!(runtime.last_positive_score_fraction.is_finite());
        assert_eq!(
            runtime.last_attribute_probabilities.len(),
            3 * ATTRIBUTE_LOGITS
        );
        assert_eq!(runtime.last_triple_scores.len(), 1);
        Ok(())
    }
}
