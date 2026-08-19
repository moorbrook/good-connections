use objc2::{Message, rc::Retained};
use objc2_foundation::NSArray;
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphLossReductionType, MPSGraphPaddingStyle,
    MPSGraphTensor, MPSGraphTensorNamedDataLayout,
};
use set_vision_contract::{
    ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT, CARD_HEIGHT, CARD_WIDTH, CHANNELS,
};

use crate::shape;

pub const BATCH_SIZE: usize = 64;
const CARD_COUNT: usize = BATCH_SIZE * 3;
const SHARED_CHANNELS: usize = 12;
const HEAD_CHANNELS: usize = 8;
const HEAD_FEATURES: usize = HEAD_CHANNELS * 6 * 8;
const HEAD_EMBED: usize = 24;

pub struct Parameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub tensor: Retained<MPSGraphTensor>,
    pub values: Vec<f32>,
}

pub struct PolicyGraph {
    pub graph: Retained<MPSGraph>,
    pub pixels: Retained<MPSGraphTensor>,
    pub set_labels: Retained<MPSGraphTensor>,
    pub attribute_labels: Retained<MPSGraphTensor>,
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
    pub fn new(init_seed: u64) -> Self {
        let mut builder = Builder::new(init_seed);
        let pixels = builder.placeholder(&[CARD_COUNT, CHANNELS, CARD_HEIGHT, CARD_WIDTH]);
        let set_labels = builder.placeholder(&[BATCH_SIZE, 2]);
        let attribute_labels =
            builder.placeholder(&[BATCH_SIZE * 3 * ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT]);

        let shared = builder.convolution(
            "shared_convolution",
            &pixels,
            ConvolutionSpec {
                input_channels: CHANNELS,
                output_channels: SHARED_CHANNELS,
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
                    input_channels: SHARED_CHANNELS,
                    output_channels: HEAD_CHANNELS,
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
                    input_channels: HEAD_CHANNELS,
                    output_channels: HEAD_CHANNELS,
                    kernel: 3,
                    stride: 1,
                    padding: 2,
                    dilation: 2,
                },
            );
            let encoded = builder.add(&encoded, &residual);
            let encoded = builder.activation(&encoded);
            let encoded = builder.reshape(&encoded, &[CARD_COUNT, HEAD_FEATURES]);
            let encoded = builder.linear(
                &format!("{prefix}.projection"),
                &encoded,
                HEAD_FEATURES,
                HEAD_EMBED,
            );
            let encoded = builder.activation(&encoded);
            heads.push(builder.linear(
                &format!("{prefix}.classifier"),
                &encoded,
                HEAD_EMBED,
                ATTRIBUTE_VALUE_COUNT,
            ));
        }
        let head_refs = heads.iter().map(|tensor| &**tensor).collect::<Vec<_>>();
        let attributes = builder.concat(&head_refs, 1);
        let attribute_classes = builder.reshape(
            &attributes,
            &[BATCH_SIZE * 3 * ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT],
        );
        let attribute_loss = unsafe {
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
        let relation = builder.linear("attribute_relation_out", &relation, 32, 1);
        let relation = builder.reshape(&relation, &[BATCH_SIZE, ATTRIBUTE_COUNT]);
        let relation = builder.linear("set_relation_in", &relation, ATTRIBUTE_COUNT, 16);
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
        let loss = builder.add(&set_loss, &weighted_attributes);

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
            loss,
            parameters: builder.parameters,
            gradients,
        }
    }
}
