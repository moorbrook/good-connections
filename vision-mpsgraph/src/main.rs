#[cfg(not(target_os = "macos"))]
compile_error!("set-vision-mpsgraph requires macOS");

mod policy_graph;

use std::{collections::BTreeMap, error::Error, path::PathBuf, ptr::NonNull, time::Instant};

use ft_core::{DenseTensor, Device};
use objc2::{
    AnyThread, Message,
    rc::{Retained, autoreleasepool},
};
use objc2_foundation::{NSArray, NSData, NSDictionary, NSNumber, ns_string};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
use objc2_metal_performance_shaders::{MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{MPSGraph, MPSGraphDevice, MPSGraphTensorData};
use set_vision_contract::{
    ATTRIBUTE_COUNT, ATTRIBUTE_VALUE_COUNT, CARD_HEIGHT, CARD_WIDTH, CHANNELS, RELATION_CLASS_COUNT,
};
use set_vision_data::{AtlasCollection, Batch, Rng};

const DEFAULT_STEPS: usize = 100;
const DEFAULT_DATA_SEED: u64 = 99_005;
const DEFAULT_INIT_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const LEARNING_RATE: f32 = 8.0e-4;
const WEIGHT_DECAY: f32 = 1.0e-4;

#[allow(clippy::struct_excessive_bools)]
struct TrainConfig {
    steps: usize,
    output: Option<PathBuf>,
    atlas_dir: Option<PathBuf>,
    augment: bool,
    data_seed: u64,
    init_seed: u64,
    typed_relations: bool,
    fill_loss_multiplier: f32,
    negative_logsumexp_loss_weight: f32,
    board_correlated_negatives: bool,
    wide_vision: bool,
    fill_texture_branch: bool,
}

struct PreparedBatch {
    pixels: Vec<f32>,
    set_labels: Vec<f32>,
    attribute_labels: Vec<f32>,
    relation_labels: Vec<f32>,
}

struct AdamW {
    first_moments: Vec<Vec<f32>>,
    second_moments: Vec<Vec<f32>>,
    step: i32,
}

impl AdamW {
    fn new(parameters: &[policy_graph::Parameter]) -> Self {
        Self {
            first_moments: parameters
                .iter()
                .map(|parameter| vec![0.0; parameter.values.len()])
                .collect(),
            second_moments: parameters
                .iter()
                .map(|parameter| vec![0.0; parameter.values.len()])
                .collect(),
            step: 0,
        }
    }

    fn step(
        &mut self,
        parameters: &mut [policy_graph::Parameter],
        gradients: &[Vec<f32>],
        learning_rate: f32,
    ) {
        const BETA_1: f32 = 0.9;
        const BETA_2: f32 = 0.999;
        const EPSILON: f32 = 1.0e-8;
        self.step += 1;
        let first_correction = 1.0 - BETA_1.powi(self.step);
        let second_correction = 1.0 - BETA_2.powi(self.step);
        for (((parameter, gradient), first), second) in parameters
            .iter_mut()
            .zip(gradients)
            .zip(&mut self.first_moments)
            .zip(&mut self.second_moments)
        {
            for (((value, &gradient), first), second) in parameter
                .values
                .iter_mut()
                .zip(gradient)
                .zip(first)
                .zip(second)
            {
                *first = BETA_1 * *first + (1.0 - BETA_1) * gradient;
                *second = BETA_2 * *second + (1.0 - BETA_2) * gradient * gradient;
                let first_hat = *first / first_correction;
                let second_hat = *second / second_correction;
                let adam_delta = learning_rate * first_hat / (second_hat.sqrt() + EPSILON);
                let decay_delta = *value * learning_rate * WEIGHT_DECAY;
                *value -= decay_delta + adam_delta;
            }
        }
    }
}

fn shape(dimensions: &[usize]) -> Retained<MPSShape> {
    let numbers = dimensions
        .iter()
        .copied()
        .map(NSNumber::new_usize)
        .collect::<Vec<_>>();
    let references = numbers.iter().map(|number| &**number).collect::<Vec<_>>();
    NSArray::from_slice(&references)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn tensor_data(
    device: &MPSGraphDevice,
    values: &[f32],
    dimensions: &[usize],
) -> Retained<MPSGraphTensorData> {
    let data = NSData::from_vec(f32_bytes(values));
    let tensor_shape = shape(dimensions);
    // SAFETY: `data` contains exactly one native-endian f32 per statically
    // shaped element, and MPSGraph copies it into device-owned tensor data.
    unsafe {
        MPSGraphTensorData::initWithDevice_data_shape_dataType(
            MPSGraphTensorData::alloc(),
            device,
            &data,
            &tensor_shape,
            MPSDataType::Float32,
        )
    }
}

fn read_f32(data: &MPSGraphTensorData, count: usize) -> Vec<f32> {
    let array = unsafe { data.mpsndarray() };
    let mut values = vec![0.0_f32; count];
    let destination = NonNull::new(values.as_mut_ptr().cast()).expect("vector pointer is non-null");
    // SAFETY: `values` has room for `count` contiguous f32 values, matching
    // the result tensor shape, and null strides request packed output.
    unsafe { array.readBytes_strideBytes(destination, std::ptr::null_mut()) };
    values
}

fn cards_nchw(pixels: &[f64]) -> Vec<f32> {
    let card_pixels = CARD_HEIGHT * CARD_WIDTH * CHANNELS;
    let card_count = pixels.len() / card_pixels;
    let mut reordered = vec![0.0_f32; pixels.len()];
    for card in 0..card_count {
        for channel in 0..CHANNELS {
            for y in 0..CARD_HEIGHT {
                for x in 0..CARD_WIDTH {
                    let source = card * card_pixels + (y * CARD_WIDTH + x) * CHANNELS + channel;
                    let destination = card * card_pixels
                        + channel * CARD_HEIGHT * CARD_WIDTH
                        + y * CARD_WIDTH
                        + x;
                    reordered[destination] = pixels[source] as f32;
                }
            }
        }
    }
    reordered
}

fn binary_one_hot(targets: &[f64]) -> Vec<f32> {
    targets
        .iter()
        .flat_map(|&target| [1.0 - target as f32, target as f32])
        .collect()
}

fn attribute_one_hot(targets: &[f64]) -> Vec<f32> {
    targets
        .iter()
        .flat_map(|&target| {
            let mut encoded = [0.0_f32; ATTRIBUTE_VALUE_COUNT];
            encoded[target as usize] = 1.0;
            encoded
        })
        .collect()
}

fn prepare_batch(batch: &Batch) -> PreparedBatch {
    PreparedBatch {
        pixels: cards_nchw(&batch.pixels),
        set_labels: binary_one_hot(&batch.set_targets),
        attribute_labels: attribute_one_hot(&batch.attribute_targets),
        relation_labels: attribute_one_hot(&batch.relation_targets),
    }
}

fn run_policy(
    policy: &policy_graph::PolicyGraph,
    device: &MPSGraphDevice,
    batch: &PreparedBatch,
) -> Result<(f32, Vec<Vec<f32>>), Box<dyn Error>> {
    let mut feed_tensors = vec![
        policy.pixels.clone(),
        policy.set_labels.clone(),
        policy.attribute_labels.clone(),
        policy.relation_labels.clone(),
    ];
    feed_tensors.extend(
        policy
            .parameters
            .iter()
            .map(|parameter| parameter.tensor.clone()),
    );
    let mut feed_data = vec![
        tensor_data(
            device,
            &batch.pixels,
            &[
                policy_graph::BATCH_SIZE * 3,
                CHANNELS,
                CARD_HEIGHT,
                CARD_WIDTH,
            ],
        ),
        tensor_data(device, &batch.set_labels, &[policy_graph::BATCH_SIZE, 2]),
        tensor_data(
            device,
            &batch.attribute_labels,
            &[
                policy_graph::BATCH_SIZE * 3 * ATTRIBUTE_COUNT,
                ATTRIBUTE_VALUE_COUNT,
            ],
        ),
        tensor_data(
            device,
            &batch.relation_labels,
            &[
                policy_graph::BATCH_SIZE * ATTRIBUTE_COUNT,
                RELATION_CLASS_COUNT,
            ],
        ),
    ];
    feed_data.extend(
        policy
            .parameters
            .iter()
            .map(|parameter| tensor_data(device, &parameter.values, &parameter.shape)),
    );
    let feed_keys = feed_tensors
        .iter()
        .map(|tensor| &**tensor)
        .collect::<Vec<_>>();
    let feed_values = feed_data.iter().map(|data| &**data).collect::<Vec<_>>();
    let feeds = NSDictionary::from_slices(&feed_keys, &feed_values);
    let mut targets = vec![policy.loss.clone()];
    targets.extend(policy.gradients.iter().cloned());
    let target_refs = targets.iter().map(|tensor| &**tensor).collect::<Vec<_>>();
    let target_array = NSArray::from_slice(&target_refs);
    let results = unsafe {
        policy
            .graph
            .runWithFeeds_targetTensors_targetOperations(&feeds, &target_array, None)
    };
    let policy_loss = unsafe {
        results
            .objectForKey_unchecked(&policy.loss)
            .ok_or("MPSGraph returned no policy loss")?
    };
    let loss = read_f32(policy_loss, 1)[0];
    let mut gradients = Vec::with_capacity(policy.parameters.len());
    for (parameter, gradient) in policy.parameters.iter().zip(&policy.gradients) {
        let data = unsafe {
            results
                .objectForKey_unchecked(gradient)
                .ok_or("MPSGraph returned no parameter gradient")?
        };
        let values = read_f32(data, parameter.values.len());
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("non-finite gradient for {}", parameter.name).into());
        }
        gradients.push(values);
    }
    Ok((loss, gradients))
}

fn parse_config() -> Result<TrainConfig, Box<dyn Error>> {
    let mut config = TrainConfig {
        steps: DEFAULT_STEPS,
        output: None,
        atlas_dir: None,
        augment: true,
        data_seed: DEFAULT_DATA_SEED,
        init_seed: DEFAULT_INIT_SEED,
        typed_relations: false,
        fill_loss_multiplier: 1.0,
        negative_logsumexp_loss_weight: 0.0,
        board_correlated_negatives: false,
        wide_vision: false,
        fill_texture_branch: false,
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--steps" => {
                config.steps = arguments
                    .next()
                    .ok_or("--steps requires a value")?
                    .parse::<usize>()?;
            }
            "--output" => {
                config.output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires a path")?,
                ));
            }
            "--atlas-dir" => {
                config.atlas_dir = Some(PathBuf::from(
                    arguments.next().ok_or("--atlas-dir requires a path")?,
                ));
            }
            "--no-augment" => config.augment = false,
            "--typed-relations" => config.typed_relations = true,
            "--fill-loss-multiplier" => {
                config.fill_loss_multiplier = arguments
                    .next()
                    .ok_or("--fill-loss-multiplier requires a value")?
                    .parse::<f32>()?;
            }
            "--board-correlated-negatives" => config.board_correlated_negatives = true,
            "--wide-vision" => config.wide_vision = true,
            "--fill-texture-branch" => config.fill_texture_branch = true,
            "--negative-logsumexp-loss-weight" => {
                config.negative_logsumexp_loss_weight = arguments
                    .next()
                    .ok_or("--negative-logsumexp-loss-weight requires a value")?
                    .parse::<f32>()?;
            }
            "--data-seed" => {
                config.data_seed = arguments
                    .next()
                    .ok_or("--data-seed requires a value")?
                    .parse::<u64>()?;
            }
            "--init-seed" => {
                config.init_seed = arguments
                    .next()
                    .ok_or("--init-seed requires a value")?
                    .parse::<u64>()?;
            }
            unknown => {
                return Err(format!(
                    "unknown argument {unknown}; usage: [--steps N] [--output PATH] [--atlas-dir PATH] [--no-augment] [--typed-relations] [--fill-loss-multiplier M] [--negative-logsumexp-loss-weight W] [--board-correlated-negatives] [--wide-vision] [--fill-texture-branch] [--data-seed N] [--init-seed N]"
                )
                .into());
            }
        }
    }
    if config.steps == 0 {
        return Err("steps must be positive".into());
    }
    if !config.fill_loss_multiplier.is_finite() || config.fill_loss_multiplier <= 0.0 {
        return Err("fill-loss multiplier must be finite and positive".into());
    }
    if !config.negative_logsumexp_loss_weight.is_finite()
        || config.negative_logsumexp_loss_weight < 0.0
    {
        return Err("negative-logsumexp loss weight must be finite and non-negative".into());
    }
    if config.negative_logsumexp_loss_weight > 0.0 && !config.typed_relations {
        return Err("--negative-logsumexp-loss-weight requires --typed-relations".into());
    }
    if config.board_correlated_negatives && config.negative_logsumexp_loss_weight == 0.0 {
        return Err(
            "--board-correlated-negatives requires --negative-logsumexp-loss-weight".into(),
        );
    }
    if config.wide_vision && !config.typed_relations {
        return Err("--wide-vision requires --typed-relations".into());
    }
    if config.fill_texture_branch && !config.typed_relations {
        return Err("--fill-texture-branch requires --typed-relations".into());
    }
    if config.fill_texture_branch && config.wide_vision {
        return Err("--fill-texture-branch and --wide-vision are mutually exclusive".into());
    }
    Ok(config)
}

fn percentile_milliseconds(samples: &[f64], percentile: usize) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let index = (ordered.len() - 1) * percentile / 100;
    ordered[index]
}

fn transpose_linear(values: &[f32], input: usize, output: usize) -> Vec<f64> {
    let mut transposed = vec![0.0; values.len()];
    for input_index in 0..input {
        for output_index in 0..output {
            transposed[output_index * input + input_index] =
                f64::from(values[input_index * output + output_index]);
        }
    }
    transposed
}

fn save_checkpoint(
    parameters: &[policy_graph::Parameter],
    path: &std::path::Path,
) -> Result<(), Box<dyn Error>> {
    let mut state = BTreeMap::new();
    for parameter in parameters {
        let (values, shape) = if parameter.shape.len() == 2 && parameter.name.ends_with(".weight") {
            (
                transpose_linear(&parameter.values, parameter.shape[0], parameter.shape[1]),
                vec![parameter.shape[1], parameter.shape[0]],
            )
        } else if parameter.shape.len() == 4 && parameter.name.ends_with(".bias") {
            (
                parameter.values.iter().copied().map(f64::from).collect(),
                vec![parameter.shape[1]],
            )
        } else {
            (
                parameter.values.iter().copied().map(f64::from).collect(),
                parameter.shape.clone(),
            )
        };
        let tensor = DenseTensor::from_contiguous_values(values, shape, Device::Cpu)?;
        if state.insert(parameter.name.clone(), tensor).is_some() {
            return Err(format!("duplicate checkpoint key {}", parameter.name).into());
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    ft_serialize::save_state_dict(&state, path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_config()?;
    let steps = config.steps;
    let metal = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
    let device_name = metal.name();
    let device = unsafe { MPSGraphDevice::deviceWithMTLDevice(&metal) };
    let graph = unsafe { MPSGraph::new() };
    let input_shape = shape(&[4]);
    let axes = shape(&[0]);

    // SAFETY: All Objective-C objects are retained for the graph's lifetime;
    // shapes and dtypes agree with the supplied feed and requested gradient.
    let (input, loss, gradient) = unsafe {
        let input = graph.placeholderWithShape_dataType_name(
            Some(&input_shape),
            MPSDataType::Float32,
            Some(ns_string!("input")),
        );
        let squared = graph.squareWithTensor_name(&input, Some(ns_string!("squared")));
        let loss = graph.meanOfTensor_axes_name(&squared, &axes, Some(ns_string!("loss")));
        let differentiated = NSArray::from_slice(&[&*input]);
        let gradients = graph.gradientForPrimaryTensor_withTensors_name(
            &loss,
            &differentiated,
            Some(ns_string!("gradients")),
        );
        let gradient = gradients
            .objectForKey_unchecked(&input)
            .ok_or("MPSGraph did not create the input gradient")?
            .retain();
        (input, loss, gradient)
    };

    let input_data = tensor_data(&device, &[1.0, 2.0, 3.0, 4.0], &[4]);
    let feeds = NSDictionary::from_slices(&[&*input], &[&*input_data]);
    let targets = NSArray::from_slice(&[&*loss, &*gradient]);
    let results =
        unsafe { graph.runWithFeeds_targetTensors_targetOperations(&feeds, &targets, None) };
    let (loss_data, gradient_data) = unsafe {
        (
            results
                .objectForKey_unchecked(&loss)
                .ok_or("MPSGraph returned no loss")?,
            results
                .objectForKey_unchecked(&gradient)
                .ok_or("MPSGraph returned no gradient")?,
        )
    };
    let loss_value = read_f32(loss_data, 1)[0];
    let gradient_values = read_f32(gradient_data, 4);

    if (loss_value - 7.5).abs() > 1.0e-5
        || gradient_values
            .iter()
            .zip([0.5, 1.0, 1.5, 2.0])
            .any(|(actual, expected)| (actual - expected).abs() > 1.0e-5)
    {
        return Err(format!(
            "unexpected MPSGraph result: loss={loss_value}, gradient={gradient_values:?}"
        )
        .into());
    }

    println!("device: {device_name}");
    println!("loss: {loss_value:.6}");
    println!("gradient: {gradient_values:?}");

    println!("data_seed: {}", config.data_seed);
    println!("init_seed: {}", config.init_seed);
    let mut policy = policy_graph::PolicyGraph::new(
        config.init_seed,
        config.typed_relations,
        config.negative_logsumexp_loss_weight,
        config.wide_vision,
        config.fill_texture_branch,
        config.fill_loss_multiplier,
    );
    println!("typed_relations: {}", config.typed_relations);
    println!("fill_loss_multiplier: {}", config.fill_loss_multiplier);
    println!(
        "negative_logsumexp_loss_weight: {}",
        config.negative_logsumexp_loss_weight
    );
    println!(
        "board_correlated_negatives: {}",
        config.board_correlated_negatives
    );
    println!("wide_vision: {}", config.wide_vision);
    println!("fill_texture_branch: {}", config.fill_texture_branch);
    let parameter_count = policy
        .parameters
        .iter()
        .map(|parameter| parameter.values.len())
        .sum::<usize>();
    println!("policy_parameters: {parameter_count}");

    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("MPSGraph crate has no repository parent")?;
    let atlas_dir = config
        .atlas_dir
        .unwrap_or_else(|| repository.join("vision-artifacts/atlases"));
    let atlases = AtlasCollection::load(&atlas_dir)?;
    let mut rng = Rng::new(config.data_seed);
    let probe_batch = if config.board_correlated_negatives {
        atlases.make_board_correlated_training_batch(&mut rng, policy_graph::BATCH_SIZE, false)
    } else if config.negative_logsumexp_loss_weight > 0.0 {
        atlases.make_grouped_training_batch(&mut rng, policy_graph::BATCH_SIZE, false)
    } else {
        atlases.make_batch(&mut rng, policy_graph::BATCH_SIZE, false)
    };
    let probe = prepare_batch(&probe_batch);
    let compile_started = Instant::now();
    let (initial_probe_loss, _) = autoreleasepool(|_| run_policy(&policy, &device, &probe))?;
    let compile_elapsed = compile_started.elapsed();
    println!(
        "policy_compile_and_first_ms: {:.3}",
        compile_elapsed.as_secs_f64() * 1_000.0
    );
    println!("initial_probe_loss: {initial_probe_loss:.6}");

    let mut optimizer = AdamW::new(&policy.parameters);
    let mut step_milliseconds = Vec::with_capacity(steps);
    let mut first_training_loss = None;
    let mut last_training_loss = 0.0;
    println!(
        "training {steps} steps x {} triples with MPSGraph",
        policy_graph::BATCH_SIZE
    );
    for step in 1..=steps {
        let started = Instant::now();
        let training_batch = if config.board_correlated_negatives {
            atlases.make_board_correlated_training_batch(
                &mut rng,
                policy_graph::BATCH_SIZE,
                config.augment,
            )
        } else if config.negative_logsumexp_loss_weight > 0.0 {
            atlases.make_grouped_training_batch(&mut rng, policy_graph::BATCH_SIZE, config.augment)
        } else {
            atlases.make_training_batch(&mut rng, policy_graph::BATCH_SIZE, config.augment)
        };
        let batch = prepare_batch(&training_batch);
        let (loss, gradients) = autoreleasepool(|_| run_policy(&policy, &device, &batch))?;
        let progress = (step - 1) as f32 / steps as f32;
        let cosine = 1.0_f32.midpoint((std::f32::consts::PI * progress).cos());
        let learning_rate = LEARNING_RATE * (0.05 + 0.95 * cosine);
        optimizer.step(&mut policy.parameters, &gradients, learning_rate);
        step_milliseconds.push(started.elapsed().as_secs_f64() * 1_000.0);
        first_training_loss.get_or_insert(loss);
        last_training_loss = loss;
        if step == 1 || step == steps || step % 100 == 0 {
            println!("step {step:>5}/{steps} loss {loss:.6}");
        }
    }
    let (final_probe_loss, final_probe_gradients) =
        autoreleasepool(|_| run_policy(&policy, &device, &probe))?;
    let total_seconds = step_milliseconds.iter().sum::<f64>() / 1_000.0;
    println!(
        "first_training_loss: {:.6}",
        first_training_loss.unwrap_or(0.0)
    );
    println!("last_training_loss: {last_training_loss:.6}");
    println!("final_probe_loss: {final_probe_loss:.6}");
    println!(
        "probe_shared_weight_gradient_0: {:?}",
        &final_probe_gradients[0][..8]
    );
    println!("training_seconds: {total_seconds:.3}");
    println!("steps_per_second: {:.2}", steps as f64 / total_seconds);
    println!(
        "step_latency_ms: p50={:.3} p95={:.3}",
        percentile_milliseconds(&step_milliseconds, 50),
        percentile_milliseconds(&step_milliseconds, 95)
    );
    if let Some(output) = config.output {
        save_checkpoint(&policy.parameters, &output)?;
        println!("checkpoint: {}", output.display());
    }
    Ok(())
}
