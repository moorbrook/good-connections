#![forbid(unsafe_code)]

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};
use ft_nn::{Conv2d, Linear, Module};
pub use set_vision_contract::{
    ATTRIBUTE_COUNT, ATTRIBUTE_LOGITS, ATTRIBUTE_VALUE_COUNT, CARD_HEIGHT, CARD_PIXELS, CARD_WIDTH,
    CHANNELS, RELATION_CLASS_COUNT,
};
use std::fmt;

const SPECIALIST_EMBED_DIM: usize = 24;
pub const ATTRIBUTE_RELATION_DIM: usize = 32;
pub const RELATION_DIM: usize = 16;
const CONV_CHANNELS: usize = 12;
const SPECIALIST_CHANNELS: usize = 8;
const WIDE_CONV_CHANNELS: usize = 24;
const WIDE_SPECIALIST_CHANNELS: usize = 16;
const WIDE_SPECIALIST_EMBED_DIM: usize = 48;
const CONV_OUTPUT_HEIGHT: usize = 6;
const CONV_OUTPUT_WIDTH: usize = 8;
const SPECIALIST_FEATURES: usize = SPECIALIST_CHANNELS * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH;
const FILL_TEXTURE_CHANNELS: usize = 8;
const FILL_TEXTURE_EMBED_DIM: usize = 24;
const FILL_TEXTURE_FEATURES: usize = FILL_TEXTURE_CHANNELS * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH;

#[derive(Debug)]
pub struct BoardScores {
    pub triples: Vec<[usize; 3]>,
    pub scores: Vec<f64>,
    pub attribute_logits: Vec<f64>,
    /// Per-triple, per-attribute relation logits in triple-major order.
    /// Typed policies emit three classes (`same`, `different`, `invalid`);
    /// legacy policies emit one untyped relation logit.
    pub relation_logits: Vec<f64>,
    pub relation_class_count: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawBoardAction {
    Set([usize; 3]),
    NoSet,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawBoardDecision {
    pub action: RawBoardAction,
    pub strongest_triple: BoardDecision,
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

    /// Chooses from the complete raw action space. `NoSet` is a supervised
    /// game action, not an abstention: non-negative logits denote a predicted
    /// SET and negative logits denote that every enumerated triple is invalid.
    pub fn best_raw_decision(&self) -> Option<RawBoardDecision> {
        let strongest_triple = self.best_decision()?;
        let action = if strongest_triple.score >= 0.0 {
            RawBoardAction::Set(strongest_triple.triple)
        } else {
            RawBoardAction::NoSet
        };
        Some(RawBoardDecision {
            action,
            strongest_triple,
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

struct DeployedAttributeHead {
    convolution: Conv2d,
    projection: Linear,
    classifier: Linear,
}

impl DeployedAttributeHead {
    fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            convolution: Conv2d::new(
                session,
                CONV_CHANNELS,
                SPECIALIST_CHANNELS,
                (3, 3),
                (2, 2),
                (1, 1),
                true,
            )?,
            projection: Linear::new(session, SPECIALIST_FEATURES, SPECIALIST_EMBED_DIM, true)?,
            classifier: Linear::new(session, SPECIALIST_EMBED_DIM, ATTRIBUTE_VALUE_COUNT, true)?,
        })
    }
}

impl Module for DeployedAttributeHead {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let card_count = session.tensor_shape(input)?[0];
        let encoded = self.convolution.forward(session, input)?;
        finish_attribute_head(
            session,
            encoded,
            card_count,
            SPECIALIST_FEATURES,
            &self.projection,
            &self.classifier,
        )
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

struct ResearchAttributeHead<
    const SHARED_CHANNELS: usize,
    const HEAD_CHANNELS: usize,
    const EMBED_DIM: usize,
> {
    local_convolution: Conv2d,
    dilated_residual_convolution: Conv2d,
    projection: Linear,
    classifier: Linear,
}

impl<const SHARED_CHANNELS: usize, const HEAD_CHANNELS: usize, const EMBED_DIM: usize>
    ResearchAttributeHead<SHARED_CHANNELS, HEAD_CHANNELS, EMBED_DIM>
{
    fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            local_convolution: Conv2d::new(
                session,
                SHARED_CHANNELS,
                HEAD_CHANNELS,
                (3, 3),
                (2, 2),
                (1, 1),
                true,
            )?,
            dilated_residual_convolution: Conv2d::new(
                session,
                HEAD_CHANNELS,
                HEAD_CHANNELS,
                (3, 3),
                (1, 1),
                (2, 2),
                true,
            )?
            .dilation((2, 2)),
            projection: Linear::new(
                session,
                HEAD_CHANNELS * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH,
                EMBED_DIM,
                true,
            )?,
            classifier: Linear::new(session, EMBED_DIM, ATTRIBUTE_VALUE_COUNT, true)?,
        })
    }
}

fn finish_attribute_head(
    session: &mut FrankenTorchSession,
    encoded: TensorNodeId,
    card_count: usize,
    specialist_features: usize,
    projection: &Linear,
    classifier: &Linear,
) -> Result<TensorNodeId, AutogradError> {
    let encoded = session.tensor_gelu(encoded)?;
    let encoded = session.tensor_reshape(encoded, vec![card_count, specialist_features])?;
    let encoded = projection.forward(session, encoded)?;
    let encoded = session.tensor_gelu(encoded)?;
    classifier.forward(session, encoded)
}

impl<const SHARED_CHANNELS: usize, const HEAD_CHANNELS: usize, const EMBED_DIM: usize> Module
    for ResearchAttributeHead<SHARED_CHANNELS, HEAD_CHANNELS, EMBED_DIM>
{
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let card_count = session.tensor_shape(input)?[0];
        let local = self.local_convolution.forward(session, input)?;
        let residual = self.dilated_residual_convolution.forward(session, local)?;
        let encoded = session.tensor_add(local, residual)?;
        finish_attribute_head(
            session,
            encoded,
            card_count,
            HEAD_CHANNELS * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH,
            &self.projection,
            &self.classifier,
        )
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.named_children()
            .into_iter()
            .flat_map(|(_, module)| module.parameters())
            .collect()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("local_convolution".to_string(), &self.local_convolution),
            (
                "dilated_residual_convolution".to_string(),
                &self.dilated_residual_convolution,
            ),
            ("projection".to_string(), &self.projection),
            ("classifier".to_string(), &self.classifier),
        ]
    }
}

/// A small fill-only residual branch that detects texture before the shared
/// stride-2 encoder. It refines fill logits without changing count, colour,
/// shape, or the standard fill specialist.
struct FillTextureHead {
    fine_convolution: Conv2d,
    downsample_convolution_1: Conv2d,
    downsample_convolution_2: Conv2d,
    projection: Linear,
    classifier: Linear,
}

impl FillTextureHead {
    fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            fine_convolution: Conv2d::new(
                session,
                CHANNELS,
                FILL_TEXTURE_CHANNELS,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
            )?,
            downsample_convolution_1: Conv2d::new(
                session,
                FILL_TEXTURE_CHANNELS,
                FILL_TEXTURE_CHANNELS,
                (3, 3),
                (2, 2),
                (1, 1),
                true,
            )?,
            downsample_convolution_2: Conv2d::new(
                session,
                FILL_TEXTURE_CHANNELS,
                FILL_TEXTURE_CHANNELS,
                (3, 3),
                (2, 2),
                (1, 1),
                true,
            )?,
            projection: Linear::new(session, FILL_TEXTURE_FEATURES, FILL_TEXTURE_EMBED_DIM, true)?,
            classifier: Linear::new(session, FILL_TEXTURE_EMBED_DIM, ATTRIBUTE_VALUE_COUNT, true)?,
        })
    }
}

impl Module for FillTextureHead {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let card_count = session.tensor_shape(input)?[0];
        let encoded = self.fine_convolution.forward(session, input)?;
        let encoded = session.tensor_gelu(encoded)?;
        let encoded = self.downsample_convolution_1.forward(session, encoded)?;
        let encoded = session.tensor_gelu(encoded)?;
        let encoded = self.downsample_convolution_2.forward(session, encoded)?;
        finish_attribute_head(
            session,
            encoded,
            card_count,
            FILL_TEXTURE_FEATURES,
            &self.projection,
            &self.classifier,
        )
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.named_children()
            .into_iter()
            .flat_map(|(_, module)| module.parameters())
            .collect()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("fine_convolution".to_string(), &self.fine_convolution),
            (
                "downsample_convolution_1".to_string(),
                &self.downsample_convolution_1,
            ),
            (
                "downsample_convolution_2".to_string(),
                &self.downsample_convolution_2,
            ),
            ("projection".to_string(), &self.projection),
            ("classifier".to_string(), &self.classifier),
        ]
    }
}

/// Shared topology-independent implementation for the deployed and research
/// policy wrappers. The wrapper's attribute-head type determines the exported
/// state-dict key family, so strict loading cannot silently mix topologies.
struct PolicyCore<H, const RELATION_CLASSES: usize, const SHARED_CHANNELS: usize> {
    shared_convolution: Conv2d,
    attribute_heads: [H; ATTRIBUTE_COUNT],
    fill_texture_head: Option<FillTextureHead>,
    attribute_relation_in: Linear,
    attribute_relation_out: Linear,
    set_relation_in: Linear,
    set_relation_out: Linear,
}

impl<H: Module, const RELATION_CLASSES: usize, const SHARED_CHANNELS: usize>
    PolicyCore<H, RELATION_CLASSES, SHARED_CHANNELS>
{
    fn new<F>(session: &mut FrankenTorchSession, mut head_new: F) -> Result<Self, AutogradError>
    where
        F: FnMut(&mut FrankenTorchSession) -> Result<H, AutogradError>,
    {
        Self::new_with_fill_texture(session, &mut head_new, false)
    }

    fn new_with_fill_texture<F>(
        session: &mut FrankenTorchSession,
        mut head_new: F,
        fill_texture: bool,
    ) -> Result<Self, AutogradError>
    where
        F: FnMut(&mut FrankenTorchSession) -> Result<H, AutogradError>,
    {
        Ok(Self {
            shared_convolution: Conv2d::new(
                session,
                CHANNELS,
                SHARED_CHANNELS,
                (5, 5),
                (2, 2),
                (2, 2),
                true,
            )?,
            attribute_heads: [
                head_new(session)?,
                head_new(session)?,
                head_new(session)?,
                head_new(session)?,
            ],
            fill_texture_head: if fill_texture {
                Some(FillTextureHead::new(session)?)
            } else {
                None
            },
            attribute_relation_in: Linear::new(
                session,
                ATTRIBUTE_VALUE_COUNT,
                ATTRIBUTE_RELATION_DIM,
                true,
            )?,
            attribute_relation_out: Linear::new(
                session,
                ATTRIBUTE_RELATION_DIM,
                RELATION_CLASSES,
                true,
            )?,
            set_relation_in: Linear::new(
                session,
                ATTRIBUTE_COUNT * RELATION_CLASSES,
                RELATION_DIM,
                true,
            )?,
            set_relation_out: Linear::new(session, RELATION_DIM, 1, true)?,
        })
    }

    fn forward_with_attributes(
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
        let (set_logits, _) =
            self.forward_relation_with_types(session, attribute_probabilities, batch_size)?;
        Ok((set_logits, attribute_logits))
    }

    fn forward_with_relations(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId, TensorNodeId), AutogradError> {
        let batch_size = session.tensor_shape(input)?[0];
        let attribute_logits = self.attribute_logits(session, input)?;
        let attribute_probabilities = session.tensor_reshape(
            attribute_logits,
            vec![batch_size, 3, ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        )?;
        let attribute_probabilities = session.tensor_softmax(attribute_probabilities, 3)?;
        let (set_logits, relation_logits) =
            self.forward_relation_with_types(session, attribute_probabilities, batch_size)?;
        Ok((set_logits, attribute_logits, relation_logits))
    }

    /// Encodes any number of cards once. Browser inference uses this before
    /// enumerating triples, avoiding 220 repeated CNN passes on a 12-card board.
    fn attribute_logits_for_cards(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        card_count: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let pixels =
            session.tensor_reshape(input, vec![card_count, CARD_HEIGHT, CARD_WIDTH, CHANNELS])?;
        let pixels = session.tensor_permute(pixels, vec![0, 3, 1, 2])?;
        let encoded = pixels;
        let encoded = self.shared_convolution.forward(session, encoded)?;
        let encoded = session.tensor_gelu(encoded)?;
        let mut attribute_logits = self
            .attribute_heads
            .iter()
            .map(|head| head.forward(session, encoded))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(fill_texture_head) = &self.fill_texture_head {
            let texture_logits = fill_texture_head.forward(session, pixels)?;
            attribute_logits[3] = session.tensor_add(attribute_logits[3], texture_logits)?;
        }
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

    fn forward_relation_with_types(
        &self,
        session: &mut FrankenTorchSession,
        attribute_probabilities: TensorNodeId,
        batch_size: usize,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        let relation = session.tensor_sum_dim(attribute_probabilities, 1)?;
        let relation = session.tensor_reshape(
            relation,
            vec![batch_size * ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        )?;
        let relation = self.attribute_relation_in.forward(session, relation)?;
        let relation = session.tensor_gelu(relation)?;
        let relation_logits = self.attribute_relation_out.forward(session, relation)?;
        let relation_features = if RELATION_CLASSES == 1 {
            session.tensor_reshape(
                relation_logits,
                vec![batch_size, ATTRIBUTE_COUNT * RELATION_CLASSES],
            )?
        } else {
            let probabilities = session.tensor_softmax(relation_logits, 1)?;
            session.tensor_reshape(
                probabilities,
                vec![batch_size, ATTRIBUTE_COUNT * RELATION_CLASSES],
            )?
        };
        let relation = self.set_relation_in.forward(session, relation_features)?;
        let relation = session.tensor_gelu(relation)?;
        let set_logits = self.set_relation_out.forward(session, relation)?;
        Ok((set_logits, relation_logits))
    }

    /// Scores every three-card action after encoding each board card exactly once.
    /// Training evaluation and browser inference both call this method.
    fn score_board(
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
        let (scores, relation_logits) =
            self.forward_relation_with_types(session, relation_input, triple_count)?;
        Ok(BoardScores {
            triples,
            scores: session.tensor_values(scores)?,
            attribute_logits: attribute_logit_values,
            relation_logits: session.tensor_values(relation_logits)?,
            relation_class_count: RELATION_CLASSES,
        })
    }
}

impl<H: Module, const RELATION_CLASSES: usize, const SHARED_CHANNELS: usize> Module
    for PolicyCore<H, RELATION_CLASSES, SHARED_CHANNELS>
{
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
        if let Some(fill_texture_head) = &self.fill_texture_head {
            children.push((
                "fill_texture_head".to_string(),
                fill_texture_head as &dyn Module,
            ));
        }
        children
    }
}

/// The shipped browser policy. Its state dict uses the legacy
/// `attribute_head_N.convolution.*` keys and the original single-convolution
/// specialist topology.
pub struct DeployedVisualSetPolicy {
    core: PolicyCore<DeployedAttributeHead, 1, CONV_CHANNELS>,
}

impl fmt::Debug for DeployedVisualSetPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeployedVisualSetPolicy")
            .field("input", &(CARD_HEIGHT, CARD_WIDTH, CHANNELS))
            .field("shared_channels", &CONV_CHANNELS)
            .field("specialist_channels", &SPECIALIST_CHANNELS)
            .field("specialist_embedding_width", &SPECIALIST_EMBED_DIM)
            .finish_non_exhaustive()
    }
}

impl DeployedVisualSetPolicy {
    pub fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            core: PolicyCore::new(session, DeployedAttributeHead::new)?,
        })
    }

    pub fn forward_with_attributes(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        self.core.forward_with_attributes(session, input)
    }

    pub fn score_board(
        &self,
        session: &mut FrankenTorchSession,
        pixels: Vec<f64>,
        card_count: usize,
    ) -> Result<BoardScores, PolicyError> {
        self.core.score_board(session, pixels, card_count)
    }
}

impl Module for DeployedVisualSetPolicy {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.core.forward(session, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.core.parameters()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.core.named_children()
    }
}

/// The native model-search policy. Its state dict uses the current
/// `attribute_head_N.local_convolution.*` and
/// `attribute_head_N.dilated_residual_convolution.*` keys.
pub struct ResearchVisualSetPolicy {
    core: PolicyCore<
        ResearchAttributeHead<CONV_CHANNELS, SPECIALIST_CHANNELS, SPECIALIST_EMBED_DIM>,
        1,
        CONV_CHANNELS,
    >,
}

/// V9 research policy with explicit `same`, `different`, and `invalid`
/// supervision for each of the four card attributes. Its typed relation
/// probabilities feed the final SET scorer without a discrete operation in
/// the training graph.
pub struct TypedRelationResearchPolicy {
    core: PolicyCore<
        ResearchAttributeHead<CONV_CHANNELS, SPECIALIST_CHANNELS, SPECIALIST_EMBED_DIM>,
        RELATION_CLASS_COUNT,
        CONV_CHANNELS,
    >,
}

/// V9 fill-frequency candidate. The standard typed policy is retained and a
/// small raw-pixel texture branch adds a residual only to the three fill
/// logits.
pub struct FillTextureTypedRelationResearchPolicy {
    core: PolicyCore<
        ResearchAttributeHead<CONV_CHANNELS, SPECIALIST_CHANNELS, SPECIALIST_EMBED_DIM>,
        RELATION_CLASS_COUNT,
        CONV_CHANNELS,
    >,
}

/// V9 width-search candidate. It keeps the typed relation and SET heads but
/// doubles the shared channels, specialist channels, and specialist embedding
/// width. The distinct Rust type and strict tensor shapes prevent accidental
/// loading as the standard research policy.
pub struct WideTypedRelationResearchPolicy {
    core: PolicyCore<
        ResearchAttributeHead<
            WIDE_CONV_CHANNELS,
            WIDE_SPECIALIST_CHANNELS,
            WIDE_SPECIALIST_EMBED_DIM,
        >,
        RELATION_CLASS_COUNT,
        WIDE_CONV_CHANNELS,
    >,
}

impl fmt::Debug for TypedRelationResearchPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TypedRelationResearchPolicy")
            .field("input", &(CARD_HEIGHT, CARD_WIDTH, CHANNELS))
            .field("relation_classes", &RELATION_CLASS_COUNT)
            .finish_non_exhaustive()
    }
}

impl TypedRelationResearchPolicy {
    pub fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            core: PolicyCore::new(session, ResearchAttributeHead::new)?,
        })
    }

    pub fn forward_with_relations(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId, TensorNodeId), AutogradError> {
        self.core.forward_with_relations(session, input)
    }

    pub fn score_board(
        &self,
        session: &mut FrankenTorchSession,
        pixels: Vec<f64>,
        card_count: usize,
    ) -> Result<BoardScores, PolicyError> {
        self.core.score_board(session, pixels, card_count)
    }
}

impl fmt::Debug for FillTextureTypedRelationResearchPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FillTextureTypedRelationResearchPolicy")
            .field("input", &(CARD_HEIGHT, CARD_WIDTH, CHANNELS))
            .field("fill_texture_channels", &FILL_TEXTURE_CHANNELS)
            .field("fill_texture_embedding_width", &FILL_TEXTURE_EMBED_DIM)
            .field("relation_classes", &RELATION_CLASS_COUNT)
            .finish_non_exhaustive()
    }
}

impl FillTextureTypedRelationResearchPolicy {
    pub fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            core: PolicyCore::new_with_fill_texture(session, ResearchAttributeHead::new, true)?,
        })
    }

    pub fn forward_with_relations(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId, TensorNodeId), AutogradError> {
        self.core.forward_with_relations(session, input)
    }

    pub fn score_board(
        &self,
        session: &mut FrankenTorchSession,
        pixels: Vec<f64>,
        card_count: usize,
    ) -> Result<BoardScores, PolicyError> {
        self.core.score_board(session, pixels, card_count)
    }
}

impl Module for FillTextureTypedRelationResearchPolicy {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.core.forward(session, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.core.parameters()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.core.named_children()
    }
}

impl fmt::Debug for WideTypedRelationResearchPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WideTypedRelationResearchPolicy")
            .field("input", &(CARD_HEIGHT, CARD_WIDTH, CHANNELS))
            .field("shared_channels", &WIDE_CONV_CHANNELS)
            .field("specialist_channels", &WIDE_SPECIALIST_CHANNELS)
            .field("specialist_embedding_width", &WIDE_SPECIALIST_EMBED_DIM)
            .field("relation_classes", &RELATION_CLASS_COUNT)
            .finish_non_exhaustive()
    }
}

impl WideTypedRelationResearchPolicy {
    pub fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            core: PolicyCore::new(
                session,
                ResearchAttributeHead::<
                    WIDE_CONV_CHANNELS,
                    WIDE_SPECIALIST_CHANNELS,
                    WIDE_SPECIALIST_EMBED_DIM,
                >::new,
            )?,
        })
    }

    pub fn forward_with_relations(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId, TensorNodeId), AutogradError> {
        self.core.forward_with_relations(session, input)
    }

    pub fn score_board(
        &self,
        session: &mut FrankenTorchSession,
        pixels: Vec<f64>,
        card_count: usize,
    ) -> Result<BoardScores, PolicyError> {
        self.core.score_board(session, pixels, card_count)
    }
}

impl Module for WideTypedRelationResearchPolicy {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.core.forward(session, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.core.parameters()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.core.named_children()
    }
}

impl Module for TypedRelationResearchPolicy {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.core.forward(session, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.core.parameters()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.core.named_children()
    }
}

impl fmt::Debug for ResearchVisualSetPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResearchVisualSetPolicy")
            .field("input", &(CARD_HEIGHT, CARD_WIDTH, CHANNELS))
            .field("shared_channels", &CONV_CHANNELS)
            .field("specialist_channels", &SPECIALIST_CHANNELS)
            .field("specialist_embedding_width", &SPECIALIST_EMBED_DIM)
            .finish_non_exhaustive()
    }
}

impl ResearchVisualSetPolicy {
    pub fn new(session: &mut FrankenTorchSession) -> Result<Self, AutogradError> {
        Ok(Self {
            core: PolicyCore::new(session, ResearchAttributeHead::new)?,
        })
    }

    pub fn forward_with_attributes(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        self.core.forward_with_attributes(session, input)
    }

    pub fn score_board(
        &self,
        session: &mut FrankenTorchSession,
        pixels: Vec<f64>,
        card_count: usize,
    ) -> Result<BoardScores, PolicyError> {
        self.core.score_board(session, pixels, card_count)
    }
}

impl Module for ResearchVisualSetPolicy {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.core.forward(session, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.core.parameters()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.core.named_children()
    }
}

#[cfg(test)]
mod tests {
    use ft_api::FrankenTorchSession;
    use ft_core::ExecutionMode;
    use ft_nn::Module;

    use super::{
        ATTRIBUTE_COUNT, BoardScores, CARD_PIXELS, DeployedVisualSetPolicy,
        FillTextureTypedRelationResearchPolicy, PolicyError, RawBoardAction,
        ResearchVisualSetPolicy, TypedRelationResearchPolicy, WideTypedRelationResearchPolicy,
    };

    #[test]
    fn decision_diagnostics_rank_scores_and_measure_attribute_certainty() {
        let scores = BoardScores {
            triples: vec![[0, 1, 2], [0, 1, 3]],
            scores: vec![2.5, 1.75],
            attribute_logits: (0..4).flat_map(|_| [8.0, -4.0, -4.0].repeat(4)).collect(),
            relation_logits: Vec::new(),
            relation_class_count: 0,
        };
        let decision = scores.best_decision().unwrap_or_else(|| unreachable!());
        assert_eq!(decision.triple, [0, 1, 2]);
        assert!((decision.score_margin - 0.75).abs() < 1.0e-12);
        assert!(decision.mean_attribute_entropy < 0.001);
        assert!(decision.min_attribute_margin > 0.999);
    }

    #[test]
    fn raw_decision_uses_no_set_as_a_real_action_below_zero() {
        let attributes = (0..4).flat_map(|_| [8.0, -4.0, -4.0].repeat(4));
        let no_set_scores = BoardScores {
            triples: vec![[0, 1, 2], [0, 1, 3]],
            scores: vec![-0.25, -0.75],
            attribute_logits: attributes.collect(),
            relation_logits: Vec::new(),
            relation_class_count: 0,
        };
        assert_eq!(
            no_set_scores
                .best_raw_decision()
                .map(|decision| decision.action),
            Some(RawBoardAction::NoSet)
        );

        let set_scores = BoardScores {
            triples: no_set_scores.triples,
            scores: vec![0.25, -0.75],
            attribute_logits: no_set_scores.attribute_logits,
            relation_logits: no_set_scores.relation_logits,
            relation_class_count: no_set_scores.relation_class_count,
        };
        assert_eq!(
            set_scores
                .best_raw_decision()
                .map(|decision| decision.action),
            Some(RawBoardAction::Set([0, 1, 2]))
        );
    }

    #[test]
    fn board_scoring_rejects_invalid_input_at_its_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let policy = DeployedVisualSetPolicy::new(&mut session)?;
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
    fn board_path_matches_direct_triple_scoring() -> Result<(), Box<dyn std::error::Error>> {
        const CARD_COUNT: usize = 5;
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let policy = ResearchVisualSetPolicy::new(&mut session)?;
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

    #[test]
    fn policy_state_dict_key_families_are_topology_specific_and_strict()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut deployed_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deployed = DeployedVisualSetPolicy::new(&mut deployed_session)?;
        let deployed_state = deployed.state_dict(&deployed_session)?;
        for index in 0..ATTRIBUTE_COUNT {
            let legacy_weight = format!("attribute_head_{index}.convolution.weight");
            let legacy_bias = format!("attribute_head_{index}.convolution.bias");
            let research_weight = format!("attribute_head_{index}.local_convolution.weight");
            assert!(deployed_state.contains_key(&legacy_weight));
            assert!(deployed_state.contains_key(&legacy_bias));
            assert!(!deployed_state.contains_key(&research_weight));
        }

        let mut deployed_reload_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deployed_reload = DeployedVisualSetPolicy::new(&mut deployed_reload_session)?;
        let deployed_report =
            deployed_reload.load_state_dict(&mut deployed_reload_session, &deployed_state, true)?;
        assert_eq!(deployed_report.missing_keys, Vec::<String>::new());
        assert_eq!(deployed_report.unexpected_keys, Vec::<String>::new());

        let mut research_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let research = ResearchVisualSetPolicy::new(&mut research_session)?;
        let research_state = research.state_dict(&research_session)?;
        for index in 0..ATTRIBUTE_COUNT {
            let legacy_weight = format!("attribute_head_{index}.convolution.weight");
            let local_weight = format!("attribute_head_{index}.local_convolution.weight");
            let residual_weight =
                format!("attribute_head_{index}.dilated_residual_convolution.weight");
            assert!(!research_state.contains_key(&legacy_weight));
            assert!(research_state.contains_key(&local_weight));
            assert!(research_state.contains_key(&residual_weight));
        }

        let mut research_reload_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let research_reload = ResearchVisualSetPolicy::new(&mut research_reload_session)?;
        let research_report =
            research_reload.load_state_dict(&mut research_reload_session, &research_state, true)?;
        assert_eq!(research_report.missing_keys, Vec::<String>::new());
        assert_eq!(research_report.unexpected_keys, Vec::<String>::new());

        let mut mismatch_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mismatch = DeployedVisualSetPolicy::new(&mut mismatch_session)?;
        assert!(
            mismatch
                .load_state_dict(&mut mismatch_session, &research_state, true)
                .is_err()
        );

        let mut typed_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let typed = TypedRelationResearchPolicy::new(&mut typed_session)?;
        let typed_state = typed.state_dict(&typed_session)?;
        assert_eq!(typed.parameters().len(), research.parameters().len());
        assert!(
            research
                .load_state_dict(&mut research_session, &typed_state, true)
                .is_err()
        );

        let mut wide_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let wide = WideTypedRelationResearchPolicy::new(&mut wide_session)?;
        let wide_state = wide.state_dict(&wide_session)?;
        assert_eq!(wide.parameters().len(), typed.parameters().len());
        assert!(
            typed
                .load_state_dict(&mut typed_session, &wide_state, true)
                .is_err()
        );

        let mut texture_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let texture = FillTextureTypedRelationResearchPolicy::new(&mut texture_session)?;
        let texture_state = texture.state_dict(&texture_session)?;
        assert!(texture_state.contains_key("fill_texture_head.fine_convolution.weight"));
        assert!(texture_state.contains_key("fill_texture_head.classifier.weight"));
        assert!(
            typed
                .load_state_dict(&mut typed_session, &texture_state, true)
                .is_err()
        );

        Ok(())
    }
}
