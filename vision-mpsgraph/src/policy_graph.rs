use objc2::{Message, rc::Retained};
use objc2_foundation::NSArray;
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphLossReductionType, MPSGraphPaddingStyle,
    MPSGraphTensor, MPSGraphTensorNamedDataLayout,
};
use set_vision_contract::{
    ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT, CARD_HEIGHT, CARD_WIDTH, CHANNELS, RELATION_CLASS_COUNT,
};

use crate::shape;

pub const BATCH_SIZE: usize = 64;
const CARD_COUNT: usize = BATCH_SIZE * 3;
const SHARED_CHANNELS: usize = 12;
const HEAD_CHANNELS: usize = 8;
const HEAD_EMBED: usize = 24;
const WIDE_SHARED_CHANNELS: usize = 24;
const WIDE_HEAD_CHANNELS: usize = 16;
const WIDE_HEAD_EMBED: usize = 48;
const FILL_TEXTURE_CHANNELS: usize = 8;
const FILL_TEXTURE_EMBED: usize = 24;
const CONV_OUTPUT_HEIGHT: usize = 6;
const CONV_OUTPUT_WIDTH: usize = 8;

pub struct Parameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub tensor: Retained<MPSGraphTensor>,
    pub values: Vec<f32>,
}

#[derive(Clone, Copy)]
pub struct NegativeAggregateConfig {
    pub weight: f32,
    pub board_groups: usize,
    pub count_correction: bool,
}

pub struct PolicyGraph {
    pub graph: Retained<MPSGraph>,
    pub pixels: Retained<MPSGraphTensor>,
    pub set_labels: Retained<MPSGraphTensor>,
    pub attribute_labels: Retained<MPSGraphTensor>,
    pub relation_labels: Retained<MPSGraphTensor>,
    pub loss: Retained<MPSGraphTensor>,
    pub parameters: Vec<Parameter>,
    pub gradients: Vec<Retained<MPSGraphTensor>>,
}

struct Builder {
    graph: Retained<MPSGraph>,
    parameters: Vec<Parameter>,
    random_state: u64,
}

#[derive(Clone, Copy)]
struct ConvolutionSpec {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Builder {
    fn new(random_state: u64) -> Self {
        Self {
            graph: unsafe { MPSGraph::new() },
            parameters: Vec::new(),
            random_state: random_state.max(1),
        }
    }

    fn placeholder(&self, dimensions: &[usize]) -> Retained<MPSGraphTensor> {
        unsafe {
            self.graph.placeholderWithShape_dataType_name(
                Some(&shape(dimensions)),
                MPSDataType::Float32,
                None,
            )
        }
    }

    fn parameter(
        &mut self,
        name: impl Into<String>,
        dimensions: &[usize],
    ) -> Retained<MPSGraphTensor> {
        let fan_in = dimensions.iter().skip(1).product::<usize>().max(1);
        let scale = (2.0_f32 / fan_in as f32).sqrt();
        let count = dimensions.iter().product();
        let values = (0..count)
            .map(|_| {
                self.random_state ^= self.random_state << 13;
                self.random_state ^= self.random_state >> 7;
                self.random_state ^= self.random_state << 17;
                let unit = (self.random_state >> 40) as f32 / (1_u32 << 24) as f32;
                (unit * 2.0 - 1.0) * scale
            })
            .collect::<Vec<_>>();
        let tensor = self.placeholder(dimensions);
        self.parameters.push(Parameter {
            name: name.into(),
            shape: dimensions.to_vec(),
            tensor: tensor.clone(),
            values,
        });
        tensor
    }

    fn zeros(&mut self, name: impl Into<String>, dimensions: &[usize]) -> Retained<MPSGraphTensor> {
        let tensor = self.placeholder(dimensions);
        self.parameters.push(Parameter {
            name: name.into(),
            shape: dimensions.to_vec(),
            tensor: tensor.clone(),
            values: vec![0.0; dimensions.iter().product()],
        });
        tensor
    }

    fn add(&self, left: &MPSGraphTensor, right: &MPSGraphTensor) -> Retained<MPSGraphTensor> {
        unsafe {
            self.graph
                .additionWithPrimaryTensor_secondaryTensor_name(left, right, None)
        }
    }

    fn mul(&self, left: &MPSGraphTensor, right: &MPSGraphTensor) -> Retained<MPSGraphTensor> {
        unsafe {
            self.graph
                .multiplicationWithPrimaryTensor_secondaryTensor_name(left, right, None)
        }
    }

    fn activation(&self, input: &MPSGraphTensor) -> Retained<MPSGraphTensor> {
        unsafe {
            let half = self
                .graph
                .constantWithScalar_dataType(0.5, MPSDataType::Float32);
            let one = self
                .graph
                .constantWithScalar_dataType(1.0, MPSDataType::Float32);
            let inverse_sqrt_two = self
                .graph
                .constantWithScalar_dataType(std::f64::consts::FRAC_1_SQRT_2, MPSDataType::Float32);
            let scaled = self.mul(input, &inverse_sqrt_two);
            let erf = self.graph.erfWithTensor_name(&scaled, None);
            let shifted = self.add(&one, &erf);
            let gated = self.mul(input, &shifted);
            self.mul(&half, &gated)
        }
    }

    fn linear(
        &mut self,
        prefix: &str,
        input: &MPSGraphTensor,
        input_dim: usize,
        output_dim: usize,
    ) -> Retained<MPSGraphTensor> {
        let weight = self.parameter(format!("{prefix}.weight"), &[input_dim, output_dim]);
        let bias = self.zeros(format!("{prefix}.bias"), &[output_dim]);
        unsafe {
            let output = self
                .graph
                .matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(input, &weight, None);
            self.add(&output, &bias)
        }
    }

    fn convolution(
        &mut self,
        prefix: &str,
        input: &MPSGraphTensor,
        spec: ConvolutionSpec,
    ) -> Retained<MPSGraphTensor> {
        let weight = self.parameter(
            format!("{prefix}.weight"),
            &[
                spec.output_channels,
                spec.input_channels,
                spec.kernel,
                spec.kernel,
            ],
        );
        let bias = self.zeros(format!("{prefix}.bias"), &[1, spec.output_channels, 1, 1]);
        let descriptor = unsafe {
            MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                spec.stride,
                spec.stride,
                spec.dilation,
                spec.dilation,
                1,
                spec.padding,
                spec.padding,
                spec.padding,
                spec.padding,
                MPSGraphPaddingStyle::Explicit,
                MPSGraphTensorNamedDataLayout::NCHW,
                MPSGraphTensorNamedDataLayout::OIHW,
            )
        }.expect("valid convolution descriptor");
        unsafe {
            let output = self
                .graph
                .convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                    input,
                    &weight,
                    &descriptor,
                    None,
                );
            self.add(&output, &bias)
        }
    }

    fn reshape(&self, input: &MPSGraphTensor, dimensions: &[usize]) -> Retained<MPSGraphTensor> {
        unsafe {
            self.graph
                .reshapeTensor_withShape_name(input, &shape(dimensions), None)
        }
    }

    fn concat(&self, tensors: &[&MPSGraphTensor], dimension: isize) -> Retained<MPSGraphTensor> {
        unsafe {
            self.graph
                .concatTensors_dimension_name(&NSArray::from_slice(tensors), dimension, None)
        }
    }
}

impl PolicyGraph {
    pub fn new(
        init_seed: u64,
        typed_relations: bool,
        negative_aggregate: NegativeAggregateConfig,
        wide_vision: bool,
        shape_loss_multiplier: f32,
        fill_loss_multiplier: f32,
    ) -> Self {
        let mut builder = Builder::new(init_seed);
        let shared_channels = if wide_vision {
            WIDE_SHARED_CHANNELS
        } else {
            SHARED_CHANNELS
        };
        let head_channels = if wide_vision {
            WIDE_HEAD_CHANNELS
        } else {
            HEAD_CHANNELS
        };
        let head_embed = if wide_vision {
            WIDE_HEAD_EMBED
        } else {
            HEAD_EMBED
        };
        let head_features = head_channels * 6 * 8;
        let pixels = builder.placeholder(&[CARD_COUNT, CHANNELS, CARD_HEIGHT, CARD_WIDTH]);
        let set_labels = builder.placeholder(&[BATCH_SIZE, 2]);
        let attribute_labels =
            builder.placeholder(&[BATCH_SIZE * 3 * ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT]);
        let relation_labels =
            builder.placeholder(&[BATCH_SIZE * ATTRIBUTE_COUNT, RELATION_CLASS_COUNT]);

        let shared = builder.convolution(
            "shared_convolution",
            &pixels,
            ConvolutionSpec {
                input_channels: CHANNELS,
                output_channels: shared_channels,
                kernel: 5,
                stride: 2,
                padding: 2,
                dilation: 1,
            },
        );
        let shared = builder.activation(&shared);
        let mut heads = Vec::with_capacity(ATTRIBUTE_COUNT);
        for index in 0..ATTRIBUTE_COUNT {
            let prefix = format!("attribute_head_{index}");
            let encoded = builder.convolution(
                &format!("{prefix}.local_convolution"),
                &shared,
                ConvolutionSpec {
                    input_channels: shared_channels,
                    output_channels: head_channels,
                    kernel: 3,
                    stride: 2,
                    padding: 1,
                    dilation: 1,
                },
            );
            let residual = builder.convolution(
                &format!("{prefix}.dilated_residual_convolution"),
                &encoded,
                ConvolutionSpec {
                    input_channels: head_channels,
                    output_channels: head_channels,
                    kernel: 3,
                    stride: 1,
                    padding: 2,
                    dilation: 2,
                },
            );
            let encoded = builder.add(&encoded, &residual);
            #[cfg(feature = "high-resolution")]
            let encoded = {
                let encoded = builder.activation(&encoded);
                builder.convolution(
                    &format!("{prefix}.resolution_downsample_convolution"),
                    &encoded,
                    ConvolutionSpec {
                        input_channels: head_channels,
                        output_channels: head_channels,
                        kernel: 3,
                        stride: 2,
                        padding: 1,
                        dilation: 1,
                    },
                )
            };
            let encoded = builder.activation(&encoded);
            let encoded = builder.reshape(&encoded, &[CARD_COUNT, head_features]);
            let encoded = builder.linear(
                &format!("{prefix}.projection"),
                &encoded,
                head_features,
                head_embed,
            );
            let encoded = builder.activation(&encoded);
            heads.push(builder.linear(
                &format!("{prefix}.classifier"),
                &encoded,
                head_embed,
                ATTRIBUTE_VALUE_COUNT,
            ));
        }
        if typed_relations && !wide_vision {
            let texture = builder.convolution(
                "fill_texture_head.fine_convolution",
                &pixels,
                ConvolutionSpec {
                    input_channels: CHANNELS,
                    output_channels: FILL_TEXTURE_CHANNELS,
                    kernel: 3,
                    stride: 1,
                    padding: 1,
                    dilation: 1,
                },
            );
            let texture = builder.activation(&texture);
            let texture = builder.convolution(
                "fill_texture_head.downsample_convolution_1",
                &texture,
                ConvolutionSpec {
                    input_channels: FILL_TEXTURE_CHANNELS,
                    output_channels: FILL_TEXTURE_CHANNELS,
                    kernel: 3,
                    stride: 2,
                    padding: 1,
                    dilation: 1,
                },
            );
            let texture = builder.activation(&texture);
            let texture = builder.convolution(
                "fill_texture_head.downsample_convolution_2",
                &texture,
                ConvolutionSpec {
                    input_channels: FILL_TEXTURE_CHANNELS,
                    output_channels: FILL_TEXTURE_CHANNELS,
                    kernel: 3,
                    stride: 2,
                    padding: 1,
                    dilation: 1,
                },
            );
            #[cfg(feature = "high-resolution")]
            let texture = {
                let texture = builder.activation(&texture);
                builder.convolution(
                    "fill_texture_head.resolution_downsample_convolution",
                    &texture,
                    ConvolutionSpec {
                        input_channels: FILL_TEXTURE_CHANNELS,
                        output_channels: FILL_TEXTURE_CHANNELS,
                        kernel: 3,
                        stride: 2,
                        padding: 1,
                        dilation: 1,
                    },
                )
            };
            let texture = builder.activation(&texture);
            let texture = builder.reshape(
                &texture,
                &[
                    CARD_COUNT,
                    FILL_TEXTURE_CHANNELS * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH,
                ],
            );
            let texture = builder.linear(
                "fill_texture_head.projection",
                &texture,
                FILL_TEXTURE_CHANNELS * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH,
                FILL_TEXTURE_EMBED,
            );
            let texture = builder.activation(&texture);
            let texture = builder.linear(
                "fill_texture_head.classifier",
                &texture,
                FILL_TEXTURE_EMBED,
                ATTRIBUTE_VALUE_COUNT,
            );
            heads[3] = builder.add(&heads[3], &texture);
        }
        let head_refs = heads.iter().map(|tensor| &**tensor).collect::<Vec<_>>();
        let attributes = builder.concat(&head_refs, 1);
        let attribute_classes = builder.reshape(
            &attributes,
            &[BATCH_SIZE * 3 * ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        );
        let mut attribute_loss = unsafe {
            builder
                .graph
                .softMaxCrossEntropyWithSourceTensor_labelsTensor_axis_reductionType_name(
                    &attribute_classes,
                    &attribute_labels,
                    1,
                    MPSGraphLossReductionType::Mean,
                    None,
                )
        };
        if (shape_loss_multiplier - 1.0).abs() > f32::EPSILON
            || (fill_loss_multiplier - 1.0).abs() > f32::EPSILON
        {
            let attribute_classes_by_kind = builder.reshape(
                &attribute_classes,
                &[CARD_COUNT, ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
            );
            let attribute_labels_by_kind = builder.reshape(
                &attribute_labels,
                &[CARD_COUNT, ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
            );
            let shape_classes = unsafe {
                builder.graph.sliceTensor_dimension_start_length_name(
                    &attribute_classes_by_kind,
                    1,
                    2,
                    1,
                    None,
                )
            };
            let shape_classes =
                builder.reshape(&shape_classes, &[CARD_COUNT, ATTRIBUTE_VALUE_COUNT]);
            let fill_classes = unsafe {
                builder.graph.sliceTensor_dimension_start_length_name(
                    &attribute_classes_by_kind,
                    1,
                    3,
                    1,
                    None,
                )
            };
            let fill_classes = builder.reshape(&fill_classes, &[CARD_COUNT, ATTRIBUTE_VALUE_COUNT]);
            let shape_labels = unsafe {
                builder.graph.sliceTensor_dimension_start_length_name(
                    &attribute_labels_by_kind,
                    1,
                    2,
                    1,
                    None,
                )
            };
            let shape_labels = builder.reshape(&shape_labels, &[CARD_COUNT, ATTRIBUTE_VALUE_COUNT]);
            let fill_labels = unsafe {
                builder.graph.sliceTensor_dimension_start_length_name(
                    &attribute_labels_by_kind,
                    1,
                    3,
                    1,
                    None,
                )
            };
            let fill_labels = builder.reshape(&fill_labels, &[CARD_COUNT, ATTRIBUTE_VALUE_COUNT]);
            let shape_loss = unsafe {
                builder
                    .graph
                    .softMaxCrossEntropyWithSourceTensor_labelsTensor_axis_reductionType_name(
                        &shape_classes,
                        &shape_labels,
                        1,
                        MPSGraphLossReductionType::Mean,
                        None,
                    )
            };
            let fill_loss = unsafe {
                builder
                    .graph
                    .softMaxCrossEntropyWithSourceTensor_labelsTensor_axis_reductionType_name(
                        &fill_classes,
                        &fill_labels,
                        1,
                        MPSGraphLossReductionType::Mean,
                        None,
                    )
            };
            let denominator = 2.0 + shape_loss_multiplier + fill_loss_multiplier;
            let base_scale = unsafe {
                builder
                    .graph
                    .constantWithScalar_dataType(f64::from(4.0 / denominator), MPSDataType::Float32)
            };
            let shape_scale = unsafe {
                builder.graph.constantWithScalar_dataType(
                    f64::from((shape_loss_multiplier - 1.0) / denominator),
                    MPSDataType::Float32,
                )
            };
            let fill_scale = unsafe {
                builder.graph.constantWithScalar_dataType(
                    f64::from((fill_loss_multiplier - 1.0) / denominator),
                    MPSDataType::Float32,
                )
            };
            let scaled_base = builder.mul(&attribute_loss, &base_scale);
            let scaled_shape = builder.mul(&shape_loss, &shape_scale);
            let scaled_fill = builder.mul(&fill_loss, &fill_scale);
            let scaled = builder.add(&scaled_base, &scaled_shape);
            attribute_loss = builder.add(&scaled, &scaled_fill);
        }

        let probabilities = builder.reshape(
            &attributes,
            &[BATCH_SIZE, 3, ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        );
        let probabilities = unsafe {
            builder
                .graph
                .softMaxWithTensor_axis_name(&probabilities, 3, None)
        };
        let relation = unsafe {
            builder
                .graph
                .reductionSumWithTensor_axis_name(&probabilities, 1, None)
        };
        let relation = builder.reshape(
            &relation,
            &[BATCH_SIZE * ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        );
        let relation = builder.linear(
            "attribute_relation_in",
            &relation,
            ATTRIBUTE_VALUE_COUNT,
            32,
        );
        let relation = builder.activation(&relation);
        let relation_classes = if typed_relations {
            RELATION_CLASS_COUNT
        } else {
            1
        };
        let relation_logits =
            builder.linear("attribute_relation_out", &relation, 32, relation_classes);
        let relation_features = if typed_relations {
            let probabilities = unsafe {
                builder
                    .graph
                    .softMaxWithTensor_axis_name(&relation_logits, 1, None)
            };
            builder.reshape(
                &probabilities,
                &[BATCH_SIZE, ATTRIBUTE_COUNT * RELATION_CLASS_COUNT],
            )
        } else {
            builder.reshape(&relation_logits, &[BATCH_SIZE, ATTRIBUTE_COUNT])
        };
        let relation = builder.linear(
            "set_relation_in",
            &relation_features,
            ATTRIBUTE_COUNT * relation_classes,
            16,
        );
        let relation = builder.activation(&relation);
        let set_logits = builder.linear("set_relation_out", &relation, 16, 1);
        let zero_logits = unsafe {
            builder.graph.constantWithScalar_shape_dataType(
                0.0,
                &shape(&[BATCH_SIZE, 1]),
                MPSDataType::Float32,
            )
        };
        let set_classes = builder.concat(&[&zero_logits, &set_logits], 1);
        let set_loss = unsafe {
            builder
                .graph
                .softMaxCrossEntropyWithSourceTensor_labelsTensor_axis_reductionType_name(
                    &set_classes,
                    &set_labels,
                    1,
                    MPSGraphLossReductionType::Mean,
                    None,
                )
        };
        let weight = unsafe {
            builder
                .graph
                .constantWithScalar_dataType(4.0, MPSDataType::Float32)
        };
        let weighted_attributes = builder.mul(&attribute_loss, &weight);
        let mut loss = builder.add(&set_loss, &weighted_attributes);
        if typed_relations {
            let relation_loss = unsafe {
                builder
                    .graph
                    .softMaxCrossEntropyWithSourceTensor_labelsTensor_axis_reductionType_name(
                        &relation_logits,
                        &relation_labels,
                        1,
                        MPSGraphLossReductionType::Mean,
                        None,
                    )
            };
            loss = builder.add(&loss, &relation_loss);
        }
        if negative_aggregate.weight > 0.0 {
            let negatives_per_board = (BATCH_SIZE / 2) / negative_aggregate.board_groups;
            // Slice each board explicitly and use the already parity-proven
            // batch-axis reduction. This also makes the graph's grouping
            // boundary independent of reshape/reduction layout semantics.
            let negative_logsumexp_groups = (0..negative_aggregate.board_groups)
                .map(|group| {
                    let group_logits = unsafe {
                        builder.graph.sliceTensor_dimension_start_length_name(
                            &set_logits,
                            0,
                            (BATCH_SIZE / 2 + group * negatives_per_board).cast_signed(),
                            negatives_per_board.cast_signed(),
                            None,
                        )
                    };
                    let exponentials =
                        unsafe { builder.graph.exponentWithTensor_name(&group_logits, None) };
                    let exponential_sum = unsafe {
                        builder
                            .graph
                            .reductionSumWithTensor_axis_name(&exponentials, 0, None)
                    };
                    let logsumexp = unsafe {
                        builder
                            .graph
                            .logarithmWithTensor_name(&exponential_sum, None)
                    };
                    builder.reshape(&logsumexp, &[1, 1])
                })
                .collect::<Vec<_>>();
            let negative_logsumexp_refs = negative_logsumexp_groups
                .iter()
                .map(|tensor| &**tensor)
                .collect::<Vec<_>>();
            let mut negative_logsumexp = builder.concat(&negative_logsumexp_refs, 0);
            if negative_aggregate.count_correction {
                let Ok(group_count) = u32::try_from(negative_aggregate.board_groups) else {
                    unreachable!("negative board groups cannot exceed the batch size")
                };
                let correction = unsafe {
                    builder.graph.constantWithScalar_shape_dataType(
                        f64::from(group_count).ln(),
                        &shape(&[negative_aggregate.board_groups, 1]),
                        MPSDataType::Float32,
                    )
                };
                negative_logsumexp = builder.add(&negative_logsumexp, &correction);
            }
            let zero = unsafe {
                builder.graph.constantWithScalar_shape_dataType(
                    0.0,
                    &shape(&[negative_aggregate.board_groups, 1]),
                    MPSDataType::Float32,
                )
            };
            let aggregate_classes = builder.concat(&[&zero, &negative_logsumexp], 1);
            let negative_label = unsafe {
                builder.graph.sliceTensor_dimension_start_length_name(
                    &set_labels,
                    0,
                    (BATCH_SIZE / 2).cast_signed(),
                    negative_aggregate.board_groups.cast_signed(),
                    None,
                )
            };
            let aggregate_loss = unsafe {
                builder
                    .graph
                    .softMaxCrossEntropyWithSourceTensor_labelsTensor_axis_reductionType_name(
                        &aggregate_classes,
                        &negative_label,
                        1,
                        MPSGraphLossReductionType::Mean,
                        None,
                    )
            };
            let weight = unsafe {
                builder.graph.constantWithScalar_dataType(
                    f64::from(negative_aggregate.weight),
                    MPSDataType::Float32,
                )
            };
            let weighted_aggregate = builder.mul(&aggregate_loss, &weight);
            loss = builder.add(&loss, &weighted_aggregate);
        }

        let parameter_tensors = builder
            .parameters
            .iter()
            .map(|parameter| &*parameter.tensor)
            .collect::<Vec<_>>();
        let gradient_map = unsafe {
            builder.graph.gradientForPrimaryTensor_withTensors_name(
                &loss,
                &NSArray::from_slice(&parameter_tensors),
                None,
            )
        };
        let gradients = builder
            .parameters
            .iter()
            .map(|parameter| unsafe {
                gradient_map
                    .objectForKey_unchecked(&parameter.tensor)
                    .expect("gradient for every parameter")
                    .retain()
            })
            .collect();

        Self {
            graph: builder.graph,
            pixels,
            set_labels,
            attribute_labels,
            relation_labels,
            loss,
            parameters: builder.parameters,
            gradients,
        }
    }
}
