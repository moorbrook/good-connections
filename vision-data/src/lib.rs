use std::{collections::BTreeMap, fs::File, io::BufReader, path::Path};

use set_vision_contract::{CARD_HEIGHT, CARD_PIXELS, CARD_WIDTH, CHANNELS};

const ATLAS_COLUMNS: usize = 9;
const ATLAS_ROWS: usize = 9;
const ATLAS_GAP: usize = 4;
const ATLAS_PADDING: usize = 10;
const DECK_SIZE: usize = 81;
const ATTRIBUTE_VALUES: [f64; 3] = [0.0, 1.0, 2.0];

type SamplingGroups = Vec<Vec<Vec<usize>>>;
type SeedSamplingGroups = Vec<Vec<Vec<Vec<usize>>>>;

#[derive(Debug)]
pub struct Batch {
    pub pixels: Vec<f64>,
    pub set_targets: Vec<f64>,
    pub attribute_targets: Vec<f64>,
    pub atlas_indices: Vec<usize>,
}

#[derive(Debug)]
pub struct BoardBatch {
    pub pixels: Vec<f64>,
    pub set_targets: Vec<f64>,
    pub ids: Vec<usize>,
    pub atlas_index: usize,
}

#[derive(Debug)]
pub struct AtlasCollection {
    cards: Vec<Vec<Vec<f64>>>,
    names: Vec<String>,
    // theme -> source raster -> procedural atlas. Sampling each level
    // independently prevents a populous fixed-width bucket from dominating.
    sampling_groups: SamplingGroups,
    // theme -> source raster -> procedural seed -> capture/raster variants.
    // Training samples a seed family before choosing one of its variants so
    // extra capture variants cannot crowd out procedural seed diversity.
    seed_sampling_groups: SeedSamplingGroups,
}

impl AtlasCollection {
    pub fn load(directory: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut paths = std::fs::read_dir(directory)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "png"))
            .collect::<Vec<_>>();
        paths.sort();
        if paths.is_empty() {
            return Err(format!("no PNG atlases found in {}", directory.display()).into());
        }
        let cards = paths
            .iter()
            .map(|path| load_atlas(path))
            .collect::<Result<Vec<_>, _>>()?;
        let names = paths
            .iter()
            .map(|path| {
                path.file_stem().map_or_else(
                    || "unknown".to_string(),
                    |name| name.to_string_lossy().into(),
                )
            })
            .collect::<Vec<_>>();
        let mut groups = BTreeMap::<String, BTreeMap<String, Vec<usize>>>::new();
        let mut seed_groups =
            BTreeMap::<String, BTreeMap<String, BTreeMap<String, Vec<usize>>>>::new();
        for (index, name) in names.iter().enumerate() {
            let (theme, raster) = atlas_sampling_keys(name);
            groups
                .entry(theme.to_string())
                .or_default()
                .entry(raster.to_string())
                .or_default()
                .push(index);
            seed_groups
                .entry(theme.to_string())
                .or_default()
                .entry(raster.to_string())
                .or_default()
                .entry(atlas_seed_key(name).to_string())
                .or_default()
                .push(index);
        }
        let sampling_groups = groups
            .into_values()
            .map(|rasters| rasters.into_values().collect())
            .collect();
        let seed_sampling_groups = seed_groups
            .into_values()
            .map(|rasters| {
                rasters
                    .into_values()
                    .map(|seeds| seeds.into_values().collect())
                    .collect()
            })
            .collect();
        Ok(Self {
            cards,
            names,
            sampling_groups,
            seed_sampling_groups,
        })
    }

    pub fn make_batch(&self, rng: &mut Rng, size: usize, augment: bool) -> Batch {
        self.make_batch_with_sampling(rng, size, augment, false)
    }

    /// Make a training batch with seed-stratified atlas sampling. This is a
    /// training-only path; evaluation and probe callers should use
    /// [`Self::make_batch`] to retain the locked evaluator stream.
    pub fn make_training_batch(&self, rng: &mut Rng, size: usize, augment: bool) -> Batch {
        self.make_batch_with_sampling(rng, size, augment, true)
    }

    fn make_batch_with_sampling(
        &self,
        rng: &mut Rng,
        size: usize,
        augment: bool,
        seed_stratified: bool,
    ) -> Batch {
        let mut pixels = Vec::with_capacity(size * 3 * CARD_PIXELS);
        let mut set_targets = Vec::with_capacity(size);
        let mut attribute_targets = Vec::with_capacity(size * 3 * 4);
        let mut atlas_indices = Vec::with_capacity(size);
        let seed_offset = if seed_stratified { rng.next_u64() } else { 0 };

        for sample in 0..size {
            let is_set = sample % 2 == 0;
            let mut ids = if is_set {
                positive_triple(rng)
            } else if sample % 8 != 7 {
                hard_negative_triple(rng)
            } else {
                negative_triple(rng)
            };
            shuffle_three(rng, &mut ids);
            let theme_group_index = rng.index(self.sampling_groups.len());
            let raster_group_index = rng.index(self.sampling_groups[theme_group_index].len());
            let theme = if seed_stratified {
                sample_seed_stratified_atlas(
                    rng,
                    &self.seed_sampling_groups[theme_group_index][raster_group_index],
                    sample,
                    seed_offset,
                )
            } else {
                let raster_group = &self.sampling_groups[theme_group_index][raster_group_index];
                raster_group[rng.index(raster_group.len())]
            };
            atlas_indices.push(theme);
            for id in ids {
                append_card(&mut pixels, &self.cards[theme][id], rng, augment);
                attribute_targets.extend(decode_card(id).map(|value| ATTRIBUTE_VALUES[value]));
            }
            set_targets.push(f64::from(is_set));
        }

        Batch {
            pixels,
            set_targets,
            attribute_targets,
            atlas_indices,
        }
    }

    pub fn make_board(&self, rng: &mut Rng, card_count: usize) -> BoardBatch {
        let ids = loop {
            let mut deck = (0..DECK_SIZE).collect::<Vec<_>>();
            for index in 0..card_count {
                let replacement = index + rng.index(DECK_SIZE - index);
                deck.swap(index, replacement);
            }
            deck.truncate(card_count);
            if board_has_set(&deck) {
                break deck;
            }
        };
        let theme = rng.index(self.cards.len());
        let triple_count = card_count * (card_count - 1) * (card_count - 2) / 6;
        let mut pixels = Vec::with_capacity(card_count * CARD_PIXELS);
        for &id in &ids {
            append_card(&mut pixels, &self.cards[theme][id], rng, false);
        }
        let mut set_targets = Vec::with_capacity(triple_count);
        for first in 0..card_count - 2 {
            for second in first + 1..card_count - 1 {
                for third in second + 1..card_count {
                    set_targets.push(f64::from(
                        completing_card(ids[first], ids[second]) == ids[third],
                    ));
                }
            }
        }
        BoardBatch {
            pixels,
            set_targets,
            ids,
            atlas_index: theme,
        }
    }

    pub fn len(&self) -> usize {
        self.cards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    pub fn name(&self, index: usize) -> &str {
        &self.names[index]
    }
}

fn atlas_sampling_keys(name: &str) -> (&str, &str) {
    let theme = name.split_once('-').map_or(name, |(theme, _)| theme);
    let raster = name
        .rsplit_once("-w")
        .filter(|(_, width)| width.chars().all(|character| character.is_ascii_digit()))
        .map_or("reference", |(_, width)| width);
    (theme, raster)
}

fn atlas_seed_key(name: &str) -> &str {
    name.split('-')
        .find(|part| !part.is_empty() && part.chars().all(|character| character.is_ascii_digit()))
        .unwrap_or("unknown")
}

fn sample_seed_stratified_atlas(
    rng: &mut Rng,
    seed_groups: &[Vec<usize>],
    sample: usize,
    seed_offset: u64,
) -> usize {
    let group_count = u64::try_from(seed_groups.len()).unwrap_or(1);
    let random_start = usize::try_from(seed_offset % group_count).unwrap_or(0);
    let seed_group = &seed_groups[(random_start + sample) % seed_groups.len()];
    seed_group[rng.index(seed_group.len())]
}

fn atlas_device_scale(name: &str) -> usize {
    name.split('-')
        .find_map(|part| part.strip_prefix("dpr")?.parse::<usize>().ok())
        .filter(|&scale| scale > 0)
        .unwrap_or(1)
}

fn board_has_set(ids: &[usize]) -> bool {
    for first in 0..ids.len() - 2 {
        for second in first + 1..ids.len() - 1 {
            if ids[second + 1..].contains(&completing_card(ids[first], ids[second])) {
                return true;
            }
        }
    }
    false
}

fn load_atlas(path: &Path) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let decoder = png::Decoder::new(BufReader::new(File::open(path)?));
    let mut reader = decoder.read_info()?;
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer)?;
    let bytes = &buffer[..info.buffer_size()];
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => {
            return Err(
                format!("unsupported atlas color type {other:?}: {}", path.display()).into(),
            );
        }
    };
    let width = usize::try_from(info.width)?;
    let height = usize::try_from(info.height)?;
    let name = path
        .file_stem()
        .map_or_else(|| "unknown".into(), |name| name.to_string_lossy());
    let device_scale = atlas_device_scale(&name);
    let padding = ATLAS_PADDING * device_scale;
    let gap = ATLAS_GAP * device_scale;
    let fixed_width = padding * 2 + (ATLAS_COLUMNS - 1) * gap;
    let fixed_height = padding * 2 + (ATLAS_ROWS - 1) * gap;
    let content_width = width.checked_sub(fixed_width);
    let content_height = height.checked_sub(fixed_height);
    let Some((source_card_width, source_card_height)) = content_width
        .zip(content_height)
        .filter(|(content_width, content_height)| {
            content_width % ATLAS_COLUMNS == 0 && content_height % ATLAS_ROWS == 0
        })
        .map(|(content_width, content_height)| {
            (content_width / ATLAS_COLUMNS, content_height / ATLAS_ROWS)
        })
    else {
        return Err(format!(
            "atlas {} is {width}x{height}; dimensions do not encode a 9x9 card grid",
            path.display()
        )
        .into());
    };

    let mut cards = Vec::with_capacity(DECK_SIZE);
    for id in 0..DECK_SIZE {
        let column = id % ATLAS_COLUMNS;
        let row = id / ATLAS_COLUMNS;
        let source_x = padding + column * (source_card_width + gap);
        let source_y = padding + row * (source_card_height + gap);
        cards.push(area_resample_rgb(
            bytes,
            width,
            channels,
            source_x,
            source_y,
            source_card_width,
            source_card_height,
        )?);
    }
    Ok(cards)
}

fn area_resample_rgb(
    bytes: &[u8],
    image_width: usize,
    channels: usize,
    source_x: usize,
    source_y: usize,
    source_width: usize,
    source_height: usize,
) -> Result<Vec<f64>, std::num::TryFromIntError> {
    let mut card = Vec::with_capacity(CARD_PIXELS);
    for y in 0..CARD_HEIGHT {
        let row_start = source_y + y * source_height / CARD_HEIGHT;
        let row_end = source_y + (y + 1) * source_height / CARD_HEIGHT;
        for x in 0..CARD_WIDTH {
            let column_start = source_x + x * source_width / CARD_WIDTH;
            let column_end = source_x + (x + 1) * source_width / CARD_WIDTH;
            let sample_count = (column_end - column_start) * (row_end - row_start);
            for channel in 0..CHANNELS {
                let mut total = 0_u32;
                for sample_y in row_start..row_end {
                    for sample_x in column_start..column_end {
                        let offset = (sample_y * image_width + sample_x) * channels;
                        total += u32::from(bytes[offset + channel]);
                    }
                }
                let denominator = u32::try_from(sample_count)? * 255;
                card.push(f64::from(total) / f64::from(denominator));
            }
        }
    }
    Ok(card)
}

fn append_card(output: &mut Vec<f64>, card: &[f64], rng: &mut Rng, augment: bool) {
    if !augment {
        output.extend_from_slice(card);
        return;
    }

    let plan = AugmentationPlan::sample(rng);
    let geometric = plan.needs_geometry().then(|| resample_card(card, &plan));
    let blur_input = geometric.as_deref().unwrap_or(card);
    let blurred = plan
        .blur_strength
        .map(|strength| blur_card(blur_input, strength));
    let source = blurred.as_deref().unwrap_or(blur_input);
    let contrast = rng.range(0.86, 1.14);
    let brightness = rng.range(-0.055, 0.055);
    for (offset, &pixel) in source.iter().enumerate() {
        let noise = rng.range(-0.018, 0.018);
        let channel = offset % CHANNELS;
        let mut pixel = ((pixel - 0.5) * contrast + 0.5 + brightness + noise)
            * plan.gamut_gains[channel]
            + plan.gamut_offsets[channel];
        let x = (offset / CHANNELS) % CARD_WIDTH;
        let y = (offset / CHANNELS) / CARD_WIDTH;
        pixel *= plan.shadow_factor(x, y);
        if let Some(band) = plan.occlusion {
            let coverage = band.coverage(y);
            let alpha = plan.occlusion_alpha * coverage;
            pixel = pixel * (1.0 - alpha) + plan.occlusion_tint * alpha;
        }
        output.push(pixel.clamp(0.0, 1.0));
    }
}

/// Per-card training augmentation sampled from the same project RNG by both
/// native backends. The geometric ranges are deliberately small: they model
/// capture variation while retaining the card's symbol and colour labels.
#[derive(Debug, Clone, Copy)]
struct AugmentationPlan {
    scale: f64,
    rotation_radians: f64,
    translate_x: f64,
    translate_y: f64,
    blur_strength: Option<f64>,
    shadow_strength: f64,
    shadow_x: f64,
    shadow_y: f64,
    gamut_gains: [f64; CHANNELS],
    gamut_offsets: [f64; CHANNELS],
    occlusion: Option<HorizontalBand>,
    occlusion_alpha: f64,
    occlusion_tint: f64,
}

impl AugmentationPlan {
    fn sample(rng: &mut Rng) -> Self {
        let scale = if rng.chance(0.38) {
            rng.range(0.93, 1.07)
        } else {
            1.0
        };
        let rotation_radians = if rng.chance(0.28) {
            rng.range(-4.0, 4.0).to_radians()
        } else {
            0.0
        };
        let translate_x = if rng.chance(0.32) {
            rng.range(-1.2, 1.2)
        } else {
            0.0
        };
        let translate_y = if rng.chance(0.32) {
            rng.range(-0.8, 0.8)
        } else {
            0.0
        };
        let blur_strength = rng.chance(0.20).then(|| rng.range(0.35, 0.75));
        let (shadow_strength, shadow_x, shadow_y) = if rng.chance(0.18) {
            let angle = rng.range(0.0, std::f64::consts::TAU);
            (rng.range(0.08, 0.24), angle.cos(), angle.sin())
        } else {
            (0.0, 0.0, 0.0)
        };
        let (gamut_gains, gamut_offsets) = if rng.chance(0.35) {
            (
                std::array::from_fn(|_| rng.range(0.97, 1.03)),
                std::array::from_fn(|_| rng.range(-0.008, 0.008)),
            )
        } else {
            ([1.0; CHANNELS], [0.0; CHANNELS])
        };
        let (occlusion, occlusion_alpha, occlusion_tint) = if rng.chance(0.16) {
            let height = rng.range(2.0, 8.0);
            let top = rng.range(0.35, 0.56) * f64::from(u32::try_from(CARD_HEIGHT).unwrap_or(0));
            (
                Some(HorizontalBand { top, height }),
                rng.range(0.30, 0.50),
                rng.range(0.04, 0.16),
            )
        } else {
            (None, 0.0, 0.0)
        };
        Self {
            scale,
            rotation_radians,
            translate_x,
            translate_y,
            blur_strength,
            shadow_strength,
            shadow_x,
            shadow_y,
            gamut_gains,
            gamut_offsets,
            occlusion,
            occlusion_alpha,
            occlusion_tint,
        }
    }

    fn needs_geometry(self) -> bool {
        (self.scale - 1.0).abs() > f64::EPSILON
            || self.rotation_radians.abs() > f64::EPSILON
            || self.translate_x.abs() > f64::EPSILON
            || self.translate_y.abs() > f64::EPSILON
    }

    #[allow(clippy::cast_precision_loss)]
    fn shadow_factor(self, x: usize, y: usize) -> f64 {
        if self.shadow_strength == 0.0 {
            return 1.0;
        }
        let normalized_x = x as f64 / (CARD_WIDTH - 1) as f64 * 2.0 - 1.0;
        let normalized_y = y as f64 / (CARD_HEIGHT - 1) as f64 * 2.0 - 1.0;
        let edge = f64::midpoint(
            1.0,
            normalized_x * self.shadow_x + normalized_y * self.shadow_y,
        )
        .clamp(0.0, 1.0);
        1.0 - self.shadow_strength * edge
    }
}

#[derive(Debug, Clone, Copy)]
struct HorizontalBand {
    top: f64,
    height: f64,
}

impl HorizontalBand {
    #[allow(clippy::cast_precision_loss)]
    fn coverage(self, y: usize) -> f64 {
        let row_start = y as f64;
        let row_end = row_start + 1.0;
        let bottom = self.top + self.height;
        (row_end.min(bottom) - row_start.max(self.top)).clamp(0.0, 1.0)
    }
}

#[allow(clippy::cast_precision_loss)]
fn resample_card(card: &[f64], plan: &AugmentationPlan) -> Vec<f64> {
    let mut output = vec![0.0; card.len()];
    let center_x = (CARD_WIDTH - 1) as f64 * 0.5;
    let center_y = (CARD_HEIGHT - 1) as f64 * 0.5;
    let sine = plan.rotation_radians.sin();
    let cosine = plan.rotation_radians.cos();
    let max_x = (CARD_WIDTH - 1) as f64;
    let max_y = (CARD_HEIGHT - 1) as f64;
    for y in 0..CARD_HEIGHT {
        for x in 0..CARD_WIDTH {
            let centered_x = (x as f64 - center_x) / plan.scale;
            let centered_y = (y as f64 - center_y) / plan.scale;
            let source_x = (cosine * centered_x + sine * centered_y + center_x - plan.translate_x)
                .clamp(0.0, max_x);
            let source_y = (-sine * centered_x + cosine * centered_y + center_y - plan.translate_y)
                .clamp(0.0, max_y);
            for channel in 0..CHANNELS {
                let offset = (y * CARD_WIDTH + x) * CHANNELS + channel;
                output[offset] = bilinear_sample(card, source_x, source_y, channel);
            }
        }
    }
    output
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn bilinear_sample(card: &[f64], x: f64, y: f64, channel: usize) -> f64 {
    let left = x.floor() as usize;
    let top = y.floor() as usize;
    let right = (left + 1).min(CARD_WIDTH - 1);
    let bottom = (top + 1).min(CARD_HEIGHT - 1);
    let x_weight = x - left as f64;
    let y_weight = y - top as f64;
    let top_left = card[(top * CARD_WIDTH + left) * CHANNELS + channel];
    let top_right = card[(top * CARD_WIDTH + right) * CHANNELS + channel];
    let bottom_left = card[(bottom * CARD_WIDTH + left) * CHANNELS + channel];
    let bottom_right = card[(bottom * CARD_WIDTH + right) * CHANNELS + channel];
    let top_value = top_left + (top_right - top_left) * x_weight;
    let bottom_value = bottom_left + (bottom_right - bottom_left) * x_weight;
    top_value + (bottom_value - top_value) * y_weight
}

fn blur_card(card: &[f64], strength: f64) -> Vec<f64> {
    let mut output = vec![0.0; card.len()];
    for y in 0..CARD_HEIGHT {
        for x in 0..CARD_WIDTH {
            for channel in 0..CHANNELS {
                let mut weighted = 0.0;
                for (dy, y_weight) in [(-1_i32, 1.0), (0, 2.0), (1, 1.0)] {
                    let sample_y = offset_coordinate(y, dy, CARD_HEIGHT);
                    for (dx, x_weight) in [(-1_i32, 1.0), (0, 2.0), (1, 1.0)] {
                        let sample_x = offset_coordinate(x, dx, CARD_WIDTH);
                        let offset = (sample_y * CARD_WIDTH + sample_x) * CHANNELS + channel;
                        weighted += card[offset] * y_weight * x_weight;
                    }
                }
                let offset = (y * CARD_WIDTH + x) * CHANNELS + channel;
                let kernel_value = weighted / 16.0;
                output[offset] = card[offset] + (kernel_value - card[offset]) * strength;
            }
        }
    }
    output
}

fn offset_coordinate(value: usize, delta: i32, limit: usize) -> usize {
    if delta < 0 {
        value.saturating_sub(usize::try_from(delta.unsigned_abs()).unwrap_or(0))
    } else {
        value
            .saturating_add(usize::try_from(delta).unwrap_or(0))
            .min(limit - 1)
    }
}

fn positive_triple(rng: &mut Rng) -> [usize; 3] {
    let first = rng.index(DECK_SIZE);
    let mut second = rng.index(DECK_SIZE - 1);
    if second >= first {
        second += 1;
    }
    [first, second, completing_card(first, second)]
}

fn negative_triple(rng: &mut Rng) -> [usize; 3] {
    loop {
        let ids = [
            rng.index(DECK_SIZE),
            rng.index(DECK_SIZE),
            rng.index(DECK_SIZE),
        ];
        if ids[0] != ids[1]
            && ids[0] != ids[2]
            && ids[1] != ids[2]
            && completing_card(ids[0], ids[1]) != ids[2]
        {
            return ids;
        }
    }
}

fn hard_negative_triple(rng: &mut Rng) -> [usize; 3] {
    loop {
        let first = rng.index(DECK_SIZE);
        let mut second = rng.index(DECK_SIZE - 1);
        if second >= first {
            second += 1;
        }
        let completion = decode_card(completing_card(first, second));
        let attribute = rng.index(4);
        let mut near_miss = completion;
        near_miss[attribute] = (near_miss[attribute] + 1 + rng.index(2)) % 3;
        let third = encode_card(near_miss);
        if third != first && third != second {
            return [first, second, third];
        }
    }
}

fn completing_card(first: usize, second: usize) -> usize {
    let first = decode_card(first);
    let second = decode_card(second);
    let third = std::array::from_fn(|index| (3 - (first[index] + second[index]) % 3) % 3);
    encode_card(third)
}

// Cross-language data contract: this order must match ATTRIBUTES and cardId in
// src/game/cards.ts because atlas grid position is the offline training label.
pub fn decode_card(id: usize) -> [usize; 4] {
    [id / 27, (id / 9) % 3, (id / 3) % 3, id % 3]
}

fn encode_card(attributes: [usize; 4]) -> usize {
    ((attributes[0] * 3 + attributes[1]) * 3 + attributes[2]) * 3 + attributes[3]
}

fn shuffle_three(rng: &mut Rng, values: &mut [usize; 3]) {
    let last = rng.index(3);
    values.swap(2, last);
    let middle = rng.index(2);
    values.swap(1, middle);
}

#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn index(&mut self, upper: usize) -> usize {
        let upper = u64::try_from(upper).unwrap_or(u64::MAX);
        usize::try_from(self.next_u64() % upper).unwrap_or(0)
    }

    fn chance(&mut self, probability: f64) -> bool {
        self.range(0.0, 1.0) < probability
    }

    #[allow(clippy::cast_precision_loss)]
    fn range(&mut self, lower: f64, upper: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64);
        lower + (upper - lower) * unit
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AtlasCollection, AugmentationPlan, CARD_HEIGHT, CARD_PIXELS, CARD_WIDTH, DECK_SIZE, Rng,
        append_card, area_resample_rgb, atlas_device_scale, atlas_sampling_keys, atlas_seed_key,
        board_has_set, completing_card, decode_card, hard_negative_triple,
        sample_seed_stratified_atlas,
    };

    #[test]
    fn atlas_sampling_keys_separate_theme_and_source_raster() {
        assert_eq!(atlas_sampling_keys("classic-0007-w128"), ("classic", "128"));
        assert_eq!(atlas_sampling_keys("signal-0192"), ("signal", "reference"));
        assert_eq!(atlas_sampling_keys("nocturne"), ("nocturne", "reference"));
    }

    #[test]
    fn atlas_seed_key_separates_procedural_seed_from_capture_suffixes() {
        assert_eq!(
            atlas_seed_key("classic-15001-pocclusion-dpr2-w107"),
            "15001"
        );
        assert_eq!(atlas_seed_key("signal-0007-w128"), "0007");
        assert_eq!(atlas_seed_key("nocturne"), "unknown");
    }

    #[test]
    fn seed_stratified_sampling_walks_each_seed_before_repeating() {
        let seed_groups = vec![vec![11], vec![22], vec![33]];
        let mut rng = Rng::new(0x5eed);
        let sampled = (0..6)
            .map(|sample| sample_seed_stratified_atlas(&mut rng, &seed_groups, sample, 1))
            .collect::<Vec<_>>();
        assert_eq!(sampled, [22, 33, 11, 22, 33, 11]);
    }

    #[test]
    fn legacy_batch_sampling_path_remains_byte_identical() {
        let collection = fixture_collection();
        let mut legacy_rng = Rng::new(0x5eed);
        let mut explicit_legacy_rng = Rng::new(0x5eed);
        let legacy = collection.make_batch(&mut legacy_rng, 12, false);
        let explicit_legacy =
            collection.make_batch_with_sampling(&mut explicit_legacy_rng, 12, false, false);
        assert_eq!(legacy.pixels, explicit_legacy.pixels);
        assert_eq!(legacy.set_targets, explicit_legacy.set_targets);
        assert_eq!(legacy.attribute_targets, explicit_legacy.attribute_targets);
        assert_eq!(legacy.atlas_indices, explicit_legacy.atlas_indices);
    }

    #[test]
    fn training_batch_cycles_seed_families_before_repeating() {
        let collection = fixture_collection();
        let mut rng = Rng::new(0x5eed);
        let batch = collection.make_training_batch(&mut rng, 6, false);
        let first_cycle = &batch.atlas_indices[..3];
        let second_cycle = &batch.atlas_indices[3..];
        assert_eq!(first_cycle.len(), 3);
        assert_eq!(
            first_cycle
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        assert_eq!(second_cycle, first_cycle);
    }

    fn fixture_collection() -> AtlasCollection {
        let card = vec![0.0; CARD_PIXELS];
        let cards = vec![vec![card; DECK_SIZE]; 3];
        AtlasCollection {
            cards,
            names: vec![
                "classic-0001-w128".to_string(),
                "classic-0002-w128".to_string(),
                "classic-0003-w128".to_string(),
            ],
            sampling_groups: vec![vec![vec![0, 1, 2]]],
            seed_sampling_groups: vec![vec![vec![vec![0], vec![1], vec![2]]]],
        }
    }

    #[test]
    fn atlas_device_scale_reads_capture_suffix() {
        assert_eq!(atlas_device_scale("classic-11001-pblur-dpr2-w144"), 2);
        assert_eq!(atlas_device_scale("classic-11001-pblur-w144"), 1);
    }

    #[test]
    fn card_ids_follow_the_typescript_attribute_order() {
        assert_eq!(decode_card(0), [0, 0, 0, 0]);
        assert_eq!(decode_card(1), [0, 0, 0, 1]);
        assert_eq!(decode_card(3), [0, 0, 1, 0]);
        assert_eq!(decode_card(9), [0, 1, 0, 0]);
        assert_eq!(decode_card(27), [1, 0, 0, 0]);
        assert_eq!(decode_card(80), [2, 2, 2, 2]);
    }

    #[test]
    fn area_resampling_matches_the_browser_contract_fixture() {
        let width = CARD_WIDTH * 2;
        let height = CARD_HEIGHT * 2;
        let mut rgba = vec![0_u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 4;
                rgba[offset] = u8::try_from(x).unwrap_or(0);
                rgba[offset + 1] = u8::try_from(y).unwrap_or(0);
                rgba[offset + 2] = u8::try_from(x + y).unwrap_or(0);
                rgba[offset + 3] = 255;
            }
        }
        let output = area_resample_rgb(&rgba, width, 4, 0, 0, width, height)
            .unwrap_or_else(|error| unreachable!("small fixture must convert: {error}"));
        for (x, y) in [(0, 0), (17, 9), (29, 20)] {
            let offset = (y * CARD_WIDTH + x) * 3;
            let x = u32::try_from(x).unwrap_or(0);
            let y = u32::try_from(y).unwrap_or(0);
            assert!((output[offset] - (f64::from(2 * x) + 0.5) / 255.0).abs() < 1.0e-12);
            assert!((output[offset + 1] - (f64::from(2 * y) + 0.5) / 255.0).abs() < 1.0e-12);
            assert!(
                (output[offset + 2] - (f64::from(2 * x + 2 * y) + 1.0) / 255.0).abs() < 1.0e-12
            );
        }
    }

    #[test]
    fn completion_makes_each_attribute_sum_divisible_by_three() {
        for first in 0..81 {
            for second in 0..81 {
                let third = completing_card(first, second);
                let cards = [decode_card(first), decode_card(second), decode_card(third)];
                for ((first, second), third) in cards[0].iter().zip(cards[1]).zip(cards[2]) {
                    assert_eq!((first + second + third) % 3, 0);
                }
            }
        }
    }

    #[test]
    fn board_set_detection_distinguishes_known_examples() {
        assert!(board_has_set(&[0, 1, 2]));
        assert!(!board_has_set(&[0, 1, 3]));
    }

    #[test]
    fn hard_negatives_are_distinct_non_sets() {
        let mut rng = Rng::new(42);
        for _ in 0..1_000 {
            let ids = hard_negative_triple(&mut rng);
            assert_ne!(ids[0], ids[1]);
            assert_ne!(ids[0], ids[2]);
            assert_ne!(ids[1], ids[2]);
            assert_ne!(completing_card(ids[0], ids[1]), ids[2]);
        }
    }

    #[test]
    fn augmentation_is_deterministic_for_a_seed() {
        let card = vec![0.37; CARD_PIXELS];
        let mut first_rng = Rng::new(0x5eed);
        let mut second_rng = Rng::new(0x5eed);
        let mut first = Vec::new();
        let mut second = Vec::new();
        append_card(&mut first, &card, &mut first_rng, true);
        append_card(&mut second, &card, &mut second_rng, true);
        assert_eq!(first, second);
    }

    #[test]
    fn augmentation_preserves_card_shape_and_unit_interval() {
        let card = (0..CARD_PIXELS)
            .map(|index| f64::from(u32::try_from(index % 17).unwrap_or(0)) / 16.0)
            .collect::<Vec<_>>();
        let mut rng = Rng::new(0xb0a7_0007);
        for _ in 0..512 {
            let mut output = Vec::new();
            append_card(&mut output, &card, &mut rng, true);
            assert_eq!(output.len(), CARD_PIXELS);
            assert!(output.iter().all(|pixel| pixel.is_finite()));
            assert!(output.iter().all(|pixel| (0.0..=1.0).contains(pixel)));
        }
    }

    #[test]
    fn sampled_geometry_and_occlusion_stay_within_label_preserving_bounds() {
        let mut rng = Rng::new(0xface_0007);
        for _ in 0..10_000 {
            let plan = AugmentationPlan::sample(&mut rng);
            assert!((0.93..=1.07).contains(&plan.scale));
            assert!(plan.rotation_radians.abs() <= 4.0_f64.to_radians());
            assert!(plan.translate_x.abs() <= 1.2);
            assert!(plan.translate_y.abs() <= 0.8);
            assert!(plan.shadow_strength <= 0.24);
            if let Some(band) = plan.occlusion {
                let card_height = f64::from(u32::try_from(CARD_HEIGHT).unwrap_or(0));
                assert!((0.35 * card_height..=0.56 * card_height).contains(&band.top));
                assert!((2.0..=8.0).contains(&band.height));
                assert!(band.top + band.height <= card_height);
                let covered_height = (0..CARD_HEIGHT).map(|row| band.coverage(row)).sum::<f64>();
                assert!((covered_height - band.height).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn horizontal_band_has_fractional_edges_and_spans_the_card() {
        let band = super::HorizontalBand {
            top: 8.25,
            height: 3.5,
        };
        assert_eq!(band.coverage(7), 0.0);
        assert!((band.coverage(8) - 0.75).abs() < 1.0e-12);
        assert_eq!(band.coverage(9), 1.0);
        assert_eq!(band.coverage(10), 1.0);
        assert!((band.coverage(11) - 0.75).abs() < 1.0e-12);
        assert_eq!(band.coverage(12), 0.0);
        assert!((0..CARD_WIDTH).all(|_| (band.coverage(9) - 1.0).abs() < 1.0e-12));
    }

    #[test]
    fn augmentation_false_is_an_exact_copy() {
        let card = vec![0.37; CARD_PIXELS];
        let mut rng = Rng::new(0x5eed);
        let mut output = Vec::new();
        append_card(&mut output, &card, &mut rng, false);
        assert_eq!(output, card);
    }

    #[test]
    #[ignore = "microbenchmark; run explicitly with --release --ignored"]
    fn augmentation_microbenchmark() {
        use std::hint::black_box;
        use std::time::Instant;

        let card = (0..CARD_PIXELS)
            .map(|index| f64::from(u32::try_from(index % 17).unwrap_or(0)) / 16.0)
            .collect::<Vec<_>>();
        let iterations = 4_096;
        let mut augmented_rng = Rng::new(0xface_0007);
        let augmented_start = Instant::now();
        let mut augmented_checksum = 0.0;
        for _ in 0..iterations {
            let mut output = Vec::with_capacity(CARD_PIXELS);
            append_card(&mut output, &card, &mut augmented_rng, true);
            augmented_checksum += output.iter().sum::<f64>();
            black_box(output);
        }
        let augmented_elapsed = augmented_start.elapsed();
        let mut baseline_rng = Rng::new(0xface_0007);
        let baseline_start = Instant::now();
        let mut baseline_checksum = 0.0;
        for _ in 0..iterations {
            let mut output = Vec::with_capacity(CARD_PIXELS);
            append_card(&mut output, &card, &mut baseline_rng, false);
            baseline_checksum += output.iter().sum::<f64>();
            black_box(output);
        }
        let baseline_elapsed = baseline_start.elapsed();
        let nanos_per_augmented_card =
            augmented_elapsed.as_secs_f64() * 1.0e9 / f64::from(iterations);
        let nanos_per_baseline_card =
            baseline_elapsed.as_secs_f64() * 1.0e9 / f64::from(iterations);
        println!(
            "augmentation: {iterations} cards in {augmented_elapsed:?}; {nanos_per_augmented_card:.1} ns/card; checksum={augmented_checksum:.3}"
        );
        println!(
            "copy baseline: {iterations} cards in {baseline_elapsed:?}; {nanos_per_baseline_card:.1} ns/card; checksum={baseline_checksum:.3}"
        );
        assert!(augmented_checksum.is_finite());
        assert!(baseline_checksum.is_finite());

        let mut legacy_rng = Rng::new(0xface_0007);
        let legacy_start = Instant::now();
        let mut legacy_checksum = 0.0;
        for _ in 0..iterations {
            let mut output = Vec::with_capacity(CARD_PIXELS);
            append_legacy_photometric(&mut output, &card, &mut legacy_rng);
            legacy_checksum += output.iter().sum::<f64>();
            black_box(output);
        }
        let legacy_elapsed = legacy_start.elapsed();
        let nanos_per_legacy_card = legacy_elapsed.as_secs_f64() * 1.0e9 / f64::from(iterations);
        println!(
            "legacy photometric: {iterations} cards in {legacy_elapsed:?}; {nanos_per_legacy_card:.1} ns/card; checksum={legacy_checksum:.3}"
        );
        assert!(legacy_checksum.is_finite());
    }

    fn append_legacy_photometric(output: &mut Vec<f64>, card: &[f64], rng: &mut Rng) {
        let contrast = rng.range(0.86, 1.14);
        let brightness = rng.range(-0.055, 0.055);
        for &pixel in card {
            let noise = rng.range(-0.018, 0.018);
            output.push(((pixel - 0.5) * contrast + 0.5 + brightness + noise).clamp(0.0, 1.0));
        }
    }
}
