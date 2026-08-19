use std::{collections::BTreeMap, fs::File, io::BufReader, path::Path};

use set_vision_contract::{CARD_HEIGHT, CARD_PIXELS, CARD_WIDTH, CHANNELS};

const ATLAS_COLUMNS: usize = 9;
const ATLAS_ROWS: usize = 9;
const ATLAS_GAP: usize = 4;
const ATLAS_PADDING: usize = 10;
const DECK_SIZE: usize = 81;
const ATTRIBUTE_VALUES: [f64; 3] = [0.0, 1.0, 2.0];

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
    sampling_groups: Vec<Vec<Vec<usize>>>,
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
        for (index, name) in names.iter().enumerate() {
            let (theme, raster) = atlas_sampling_keys(name);
            groups
                .entry(theme.to_string())
                .or_default()
                .entry(raster.to_string())
                .or_default()
                .push(index);
        }
        let sampling_groups = groups
            .into_values()
            .map(|rasters| rasters.into_values().collect())
            .collect();
        Ok(Self {
            cards,
            names,
            sampling_groups,
        })
    }

    pub fn make_batch(&self, rng: &mut Rng, size: usize, augment: bool) -> Batch {
        let mut pixels = Vec::with_capacity(size * 3 * CARD_PIXELS);
        let mut set_targets = Vec::with_capacity(size);
        let mut attribute_targets = Vec::with_capacity(size * 3 * 4);
        let mut atlas_indices = Vec::with_capacity(size);

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
            let theme_group = &self.sampling_groups[rng.index(self.sampling_groups.len())];
            let raster_group = &theme_group[rng.index(theme_group.len())];
            let theme = raster_group[rng.index(raster_group.len())];
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
    let contrast = rng.range(0.86, 1.14);
    let brightness = rng.range(-0.055, 0.055);
    for &pixel in card {
        let noise = rng.range(-0.018, 0.018);
        output.push(((pixel - 0.5) * contrast + 0.5 + brightness + noise).clamp(0.0, 1.0));
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

    #[allow(clippy::cast_precision_loss)]
    fn range(&mut self, lower: f64, upper: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64);
        lower + (upper - lower) * unit
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CARD_HEIGHT, CARD_WIDTH, Rng, area_resample_rgb, atlas_device_scale, atlas_sampling_keys,
        board_has_set, completing_card, decode_card, hard_negative_triple,
    };

    #[test]
    fn atlas_sampling_keys_separate_theme_and_source_raster() {
        assert_eq!(atlas_sampling_keys("classic-0007-w128"), ("classic", "128"));
        assert_eq!(atlas_sampling_keys("signal-0192"), ("signal", "reference"));
        assert_eq!(atlas_sampling_keys("nocturne"), ("nocturne", "reference"));
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
}
