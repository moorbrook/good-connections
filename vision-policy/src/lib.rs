#![forbid(unsafe_code)]

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};
use ft_nn::{AdaptiveAvgPool2d, Conv2d, Linear, Module};
pub use set_vision_contract::{
    ATTRIBUTE_COUNT, ATTRIBUTE_LOGITS, ATTRIBUTE_VALUE_COUNT, CARD_HEIGHT, CARD_PIXELS, CARD_WIDTH,
    CHANNELS,
};
use std::fmt;

const SPECIALIST_EMBED_DIM: usize = 24;
pub const ATTRIBUTE_RELATION_DIM: usize = 32;
pub const RELATION_DIM: usize = 16;
const CONV_CHANNELS: usize = 12;
const SPECIALIST_CHANNELS: usize = 8;
const CONV_OUTPUT_HEIGHT: usize = 6;
const CONV_OUTPUT_WIDTH: usize = 8;
const SPECIALIST_FEATURES: usize = SPECIALIST_CHANNELS * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH;
const FRONTEND_POOL_OUTPUT: (usize, usize) = (CONV_OUTPUT_HEIGHT, CONV_OUTPUT_WIDTH);

#[derive(Debug)]
pub struct BoardScores {
    pub triples: Vec<[usize; 3]>,
    pub scores: Vec<f64>,
    pub attribute_logits: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoardDecision {
    pub triple: [usize; 3],
    pub score: f64,
    pub runner_up_score: f64,
    pub score_margin: f64,
    pub mean_attribute_entropy: f64,
    pub min_attribute_margin: f64,
}

impl BoardScores {
    /// Returns the best action and diagnostics used by the abstention layer.
    /// Entropy is normalized to `[0, 1]`; attribute margin is the weakest
    /// top-one minus top-two softmax gap among the selected cards.
    pub fn best_decision(&self) -> Option<BoardDecision> {
        let mut ranked = self
            .scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        let &(best_index, score) = ranked.first()?;
        let runner_up_score = ranked.get(1).map_or(f64::NEG_INFINITY, |(_, score)| *score);
        let triple = *self.triples.get(best_index)?;
        let mut entropy_total = 0.0;
        let mut min_attribute_margin = 1.0_f64;
        let mut attribute_count = 0_u32;
        for card in triple {
            let card_start = card.checked_mul(ATTRIBUTE_LOGITS)?;
            let card_end = card_start.checked_add(ATTRIBUTE_LOGITS)?;
            let logits = self.attribute_logits.get(card_start..card_end)?;
            let (attributes, remainder) = logits.as_chunks::<ATTRIBUTE_VALUE_COUNT>();
            if !remainder.is_empty() {
                return None;
            }
            for attribute in attributes {
                let probabilities = softmax_three(attribute);
                let entropy = -probabilities
                    .iter()
                    .filter(|&&probability| probability > 0.0)
                    .map(|&probability| probability * probability.ln())
                    .sum::<f64>()
                    / 3.0_f64.ln();
                let mut ranked_probabilities = probabilities;
                ranked_probabilities.sort_by(|left, right| right.total_cmp(left));
                min_attribute_margin =
                    min_attribute_margin.min(ranked_probabilities[0] - ranked_probabilities[1]);
                entropy_total += entropy;
                attribute_count += 1;
            }
        }
        if attribute_count == 0 {
            return None;
        }
        Some(BoardDecision {
            triple,
            score,
            runner_up_score,
            score_margin: score - runner_up_score,
            mean_attribute_entropy: entropy_total / f64::from(attribute_count),
            min_attribute_margin,
        })
    }
}

fn softmax_three(logits: &[f64; ATTRIBUTE_VALUE_COUNT]) -> [f64; ATTRIBUTE_VALUE_COUNT] {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = logits.map(|logit| (logit - maximum).exp());
    let total = probabilities.iter().sum::<f64>();
    if total.is_finite() && total > 0.0 {
        for probability in &mut probabilities {
            *probability /= total;
        }
    }
    probabilities
}

#[derive(Debug)]
pub enum PolicyError {
    InvalidCardCount(usize),
    InvalidPixelCount { expected: usize, actual: usize },
    Autograd(AutogradError),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCardCount(count) => {
                write!(
                    formatter,
                    "a board must contain at least three cards, got {count}"
                )
            }
            Self::InvalidPixelCount { expected, actual } => write!(
                formatter,
                "board pixel count must be {expected}, got {actual}"
            ),
            Self::Autograd(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Autograd(error) => Some(error),
            Self::InvalidCardCount(_) | Self::InvalidPixelCount { .. } => None,
        }
    }
}

impl From<AutogradError> for PolicyError {
    fn from(error: AutogradError) -> Self {
        Self::Autograd(error)
    }
}

struct AttributeHead {
    convolution: Conv2d,
    projection: Linear,
    classifier: Linear,
}

impl AttributeHead {
    fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            convolution: Conv2d::new(
                session,
                CONV_CHANNELS,
                SPECIALIST_CHANNELS,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
            )?,
            projection: Linear::new(session, SPECIALIST_FEATURES, SPECIALIST_EMBED_DIM, true)?,
            classifier: Linear::new(session, SPECIALIST_EMBED_DIM, ATTRIBUTE_VALUE_COUNT, true)?,
        })
    }
}

impl Module for AttributeHead {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let card_count = session.tensor_shape(input)?[0];
        let encoded = self.convolution.forward(session, input)?;
        let encoded = session.tensor_gelu(encoded)?;
        let encoded = session.tensor_reshape(encoded, vec![card_count, SPECIALIST_FEATURES])?;
        let encoded = self.projection.forward(session, encoded)?;
        let encoded = session.tensor_gelu(encoded)?;
        self.classifier.forward(session, encoded)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.named_children()
            .into_iter()
            .flat_map(|(_, module)| module.parameters())
            .collect()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("convolution".to_string(), &self.convolution),
            ("projection".to_string(), &self.projection),
            ("classifier".to_string(), &self.classifier),
        ]
    }
}

/// The one learned policy definition shared by training and browser inference.
///
/// It has a shared low-level convolution, an adaptive spatial pooling frontend,
/// four attribute-specialist branches, and a learned permutation-invariant
/// relation scorer. It is not a transformer, VLM, or VLA.
pub struct VisualSetPolicy {
    shared_convolution: Conv2d,
    frontend_pool: AdaptiveAvgPool2d,
    attribute_heads: [AttributeHead; ATTRIBUTE_COUNT],
    attribute_relation_in: Linear,
    attribute_relation_out: Linear,
    set_relation_in: Linear,
    set_relation_out: Linear,
}

impl fmt::Debug for VisualSetPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VisualSetPolicy")
            .field("input", &(CARD_HEIGHT, CARD_WIDTH, CHANNELS))
            .field("shared_channels", &CONV_CHANNELS)
            .field("specialist_channels", &SPECIALIST_CHANNELS)
            .field("specialist_embedding_width", &SPECIALIST_EMBED_DIM)
            .finish_non_exhaustive()
    }
}

impl VisualSetPolicy {
    pub fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            shared_convolution: Conv2d::new(
                session,
                CHANNELS,
                CONV_CHANNELS,
                (5, 5),
                (1, 1),
                (2, 2),
                true,
            )?,
            frontend_pool: AdaptiveAvgPool2d::new(FRONTEND_POOL_OUTPUT),
            attribute_heads: [
                AttributeHead::new(session)?,
                AttributeHead::new(session)?,
                AttributeHead::new(session)?,
                AttributeHead::new(session)?,
            ],
            attribute_relation_in: Linear::new(
                session,
                ATTRIBUTE_VALUE_COUNT,
                ATTRIBUTE_RELATION_DIM,
                true,
            )?,
            attribute_relation_out: Linear::new(session, ATTRIBUTE_RELATION_DIM, 1, true)?,
            set_relation_in: Linear::new(session, ATTRIBUTE_COUNT, RELATION_DIM, true)?,
            set_relation_out: Linear::new(session, RELATION_DIM, 1, true)?,
        })
    }

    pub fn forward_with_attributes(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        let batch_size = session.tensor_shape(input)?[0];
        let attribute_logits = self.attribute_logits(session, input)?;
        let attribute_probabilities = session.tensor_reshape(
            attribute_logits,
            vec![batch_size, 3, ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        )?;
        let attribute_probabilities = session.tensor_softmax(attribute_probabilities, 3)?;
        let set_logits = self.forward_relation(session, attribute_probabilities, batch_size)?;
        Ok((set_logits, attribute_logits))
    }

    /// Encodes any number of cards once. Browser inference uses this before
    /// enumerating triples, avoiding 220 repeated CNN passes on a 12-card board.
    fn attribute_logits_for_cards(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        card_count: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let encoded =
            session.tensor_reshape(input, vec![card_count, CARD_HEIGHT, CARD_WIDTH, CHANNELS])?;
        let encoded = session.tensor_permute(encoded, vec![0, 3, 1, 2])?;
        let encoded = self.shared_convolution.forward(session, encoded)?;
        let encoded = session.tensor_gelu(encoded)?;
        let encoded = self.frontend_pool.forward(session, encoded)?;
        let attribute_logits = self
            .attribute_heads
            .iter()
            .map(|head| head.forward(session, encoded))
            .collect::<Result<Vec<_>, _>>()?;
        session.tensor_cat(&attribute_logits, 1)
    }

    fn attribute_logits(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let batch_size = session.tensor_shape(input)?[0];
        self.attribute_logits_for_cards(session, input, batch_size * 3)
    }

    fn forward_relation(
        &self,
        session: &mut FrankenTorchSession,
        attribute_probabilities: TensorNodeId,
        batch_size: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let relation = session.tensor_sum_dim(attribute_probabilities, 1)?;
        let relation = session.tensor_reshape(
            relation,
            vec![batch_size * ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        )?;
        let relation = self.attribute_relation_in.forward(session, relation)?;
        let relation = session.tensor_gelu(relation)?;
        let relation = self.attribute_relation_out.forward(session, relation)?;
        let relation = session.tensor_reshape(relation, vec![batch_size, ATTRIBUTE_COUNT])?;
        let relation = self.set_relation_in.forward(session, relation)?;
        let relation = session.tensor_gelu(relation)?;
        self.set_relation_out.forward(session, relation)
    }

    /// Scores every three-card action after encoding each board card exactly once.
    /// Training evaluation and browser inference both call this method.
    pub fn score_board(
        &self,
        session: &mut FrankenTorchSession,
        pixels: Vec<f64>,
        card_count: usize,
    ) -> Result<BoardScores, PolicyError> {
        if card_count < 3 {
            return Err(PolicyError::InvalidCardCount(card_count));
        }
        let Some(expected_pixels) = card_count.checked_mul(CARD_PIXELS) else {
            return Err(PolicyError::InvalidCardCount(card_count));
        };
        if pixels.len() != expected_pixels {
            return Err(PolicyError::InvalidPixelCount {
                expected: expected_pixels,
                actual: pixels.len(),
            });
        }
        let input = session.tensor_variable(pixels, vec![card_count, CARD_PIXELS], false)?;
        let attribute_logits = self.attribute_logits_for_cards(session, input, card_count)?;
        let attribute_logit_values = session.tensor_values(attribute_logits)?;
        let attribute_logits = session.tensor_reshape(
            attribute_logits,
            vec![card_count, ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        )?;
        let attribute_probabilities = session.tensor_softmax(attribute_logits, 2)?;
        let card_probabilities = session.tensor_values(attribute_probabilities)?;

        let triple_count = card_count * (card_count - 1) * (card_count - 2) / 6;
        let mut triples = Vec::with_capacity(triple_count);
        let mut triple_probabilities = Vec::with_capacity(triple_count * 3 * ATTRIBUTE_LOGITS);
        for first in 0..card_count - 2 {
            for second in first + 1..card_count - 1 {
                for third in second + 1..card_count {
                    triples.push([first, second, third]);
                    for card in [first, second, third] {
                        triple_probabilities.extend_from_slice(
                            &card_probabilities
                                [card * ATTRIBUTE_LOGITS..(card + 1) * ATTRIBUTE_LOGITS],
                        );
                    }
                }
            }
        }

        let relation_input = session.tensor_variable(
            triple_probabilities,
            vec![triple_count, 3, ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
            false,
        )?;
        let scores = self.forward_relation(session, relation_input, triple_count)?;
        Ok(BoardScores {
            triples,
            scores: session.tensor_values(scores)?,
            attribute_logits: attribute_logit_values,
        })
    }
}

impl Module for VisualSetPolicy {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.forward_with_attributes(session, input)
            .map(|(set_logits, _)| set_logits)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.named_children()
            .into_iter()
            .flat_map(|(_, module)| module.parameters())
            .collect()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        let mut children: Vec<(String, &dyn Module)> = vec![
            ("shared_convolution".to_string(), &self.shared_convolution),
            ("frontend_pool".to_string(), &self.frontend_pool),
            (
                "attribute_relation_in".to_string(),
                &self.attribute_relation_in,
            ),
            (
                "attribute_relation_out".to_string(),
                &self.attribute_relation_out,
            ),
            ("set_relation_in".to_string(), &self.set_relation_in),
            ("set_relation_out".to_string(), &self.set_relation_out),
        ];
        children.extend(
            self.attribute_heads
                .iter()
                .enumerate()
                .map(|(index, head)| (format!("attribute_head_{index}"), head as &dyn Module)),
        );
        children
    }
}

#[cfg(test)]
mod tests {
    use ft_api::FrankenTorchSession;
    use ft_core::ExecutionMode;
    use ft_nn::Module;

    use super::{
        ATTRIBUTE_LOGITS, BoardScores, CARD_HEIGHT, CARD_PIXELS, CARD_WIDTH, CHANNELS, PolicyError,
        VisualSetPolicy,
    };

    #[test]
    fn decision_diagnostics_rank_scores_and_measure_attribute_certainty() {
        let scores = BoardScores {
            triples: vec![[0, 1, 2], [0, 1, 3]],
            scores: vec![2.5, 1.75],
            attribute_logits: (0..4).flat_map(|_| [8.0, -4.0, -4.0].repeat(4)).collect(),
        };
        let decision = scores.best_decision().unwrap_or_else(|| unreachable!());
        assert_eq!(decision.triple, [0, 1, 2]);
        assert!((decision.score_margin - 0.75).abs() < 1.0e-12);
        assert!(decision.mean_attribute_entropy < 0.001);
        assert!(decision.min_attribute_margin > 0.999);
    }

    #[test]
    fn board_scoring_rejects_invalid_input_at_its_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let policy = VisualSetPolicy::new(&mut session)?;
        assert!(matches!(
            policy.score_board(&mut session, Vec::new(), 2),
            Err(PolicyError::InvalidCardCount(2))
        ));
        assert!(matches!(
            policy.score_board(&mut session, vec![0.0], 3),
            Err(PolicyError::InvalidPixelCount { expected, actual: 1 })
                if expected == 3 * CARD_PIXELS
        ));
        Ok(())
    }

    #[test]
    fn adaptive_frontend_emits_the_fixed_specialist_map() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let policy = VisualSetPolicy::new(&mut session)?;
        let input = session.tensor_variable(
            vec![0.0; 2 * CHANNELS * CARD_HEIGHT * CARD_WIDTH],
            vec![2, CHANNELS, CARD_HEIGHT, CARD_WIDTH],
            false,
        )?;
        let shared = policy.shared_convolution.forward(&mut session, input)?;
        assert_eq!(
            session.tensor_shape(shared)?,
            vec![2, 12, CARD_HEIGHT, CARD_WIDTH]
        );
        let pooled = policy.frontend_pool.forward(&mut session, shared)?;
        assert_eq!(session.tensor_shape(pooled)?, vec![2, 12, 6, 8]);

        let cards =
            session.tensor_variable(vec![0.0; 2 * CARD_PIXELS], vec![2, CARD_PIXELS], false)?;
        let logits = policy.attribute_logits_for_cards(&mut session, cards, 2)?;
        assert_eq!(session.tensor_shape(logits)?, vec![2, ATTRIBUTE_LOGITS]);
        let parameter_count = policy
            .parameters()
            .into_iter()
            .map(|parameter| session.tensor_numel(parameter))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<usize>();
        assert_eq!(parameter_count, 41_918);
        Ok(())
    }

    #[test]
    fn board_path_matches_direct_triple_scoring() -> Result<(), Box<dyn std::error::Error>> {
        const CARD_COUNT: usize = 5;
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let policy = VisualSetPolicy::new(&mut session)?;
        session.no_grad_enter();
        let pixels = (0..CARD_COUNT * CARD_PIXELS)
            .map(|index| u32::try_from(index % 251).map(|value| f64::from(value) / 250.0))
            .collect::<Result<Vec<_>, _>>()?;

        let board = policy.score_board(&mut session, pixels.clone(), CARD_COUNT)?;
        let repeated = board
            .triples
            .iter()
            .flat_map(|triple| {
                triple.iter().flat_map(|&card| {
                    pixels[card * CARD_PIXELS..(card + 1) * CARD_PIXELS]
                        .iter()
                        .copied()
                })
            })
            .collect::<Vec<_>>();
        let input =
            session.tensor_variable(repeated, vec![board.triples.len(), 3, CARD_PIXELS], false)?;
        let direct = policy.forward_with_attributes(&mut session, input)?.0;
        let direct = session.tensor_values(direct)?;

        assert_eq!(board.scores.len(), direct.len());
        for (cached, repeated) in board.scores.iter().zip(direct) {
            assert!((cached - repeated).abs() < 1.0e-10);
        }
        Ok(())
    }
}
