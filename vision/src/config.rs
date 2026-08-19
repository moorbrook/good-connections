use std::path::PathBuf;

pub(crate) const DEFAULT_TRIPLE_SEED: u64 = 0xa11c_e5ed_5eed;
pub(crate) const DEFAULT_BOARD_SEED: u64 = 0xb0a4_d5e7_5eed;

#[derive(Debug, Clone)]
pub(crate) struct TrainConfig {
    pub(crate) atlas_dir: PathBuf,
    pub(crate) validation_atlas_dir: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) ensemble_output_dir: Option<PathBuf>,
    pub(crate) steps: usize,
    pub(crate) batch_size: usize,
    pub(crate) learning_rate: f64,
    pub(crate) attribute_loss_weight: f64,
    pub(crate) seed: u64,
    pub(crate) eval_only: bool,
    pub(crate) probe_only: bool,
    pub(crate) eval_batches: usize,
    pub(crate) eval_boards: usize,
    pub(crate) triple_seed: u64,
    pub(crate) board_seed: u64,
    pub(crate) augment: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map_or_else(|| PathBuf::from("."), std::path::Path::to_path_buf);
        Self {
            atlas_dir: root.join("vision-artifacts/atlases"),
            validation_atlas_dir: root.join("vision-artifacts/validation-atlases"),
            output_dir: root.join("vision-artifacts/model"),
            ensemble_output_dir: None,
            steps: 10_000,
            batch_size: 64,
            learning_rate: 8.0e-4,
            attribute_loss_weight: 0.35,
            seed: 0x5e7_2026,
            eval_only: false,
            probe_only: false,
            eval_batches: 32,
            eval_boards: 32,
            triple_seed: DEFAULT_TRIPLE_SEED,
            board_seed: DEFAULT_BOARD_SEED,
            augment: true,
        }
    }
}

impl TrainConfig {
    pub(crate) fn from_args() -> Result<Option<Self>, String> {
        Self::from_iter(std::env::args().skip(1))
    }

    fn from_iter(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, String> {
        let mut config = Self::default();
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--help" | "-h" => return Ok(None),
                "--steps" => config.steps = parse_value(&mut args, &flag)?,
                "--batch-size" => config.batch_size = parse_value(&mut args, &flag)?,
                "--learning-rate" => config.learning_rate = parse_value(&mut args, &flag)?,
                "--attribute-loss-weight" => {
                    config.attribute_loss_weight = parse_value(&mut args, &flag)?;
                }
                "--seed" => config.seed = parse_value(&mut args, &flag)?,
                "--eval-only" => config.eval_only = true,
                "--probe-only" => config.probe_only = true,
                "--eval-batches" => config.eval_batches = parse_value(&mut args, &flag)?,
                "--eval-boards" => config.eval_boards = parse_value(&mut args, &flag)?,
                "--triple-seed" => config.triple_seed = parse_value(&mut args, &flag)?,
                "--board-seed" => config.board_seed = parse_value(&mut args, &flag)?,
                "--no-augment" => config.augment = false,
                "--atlas-dir" => config.atlas_dir = PathBuf::from(next_value(&mut args, &flag)?),
                "--validation-atlas-dir" => {
                    config.validation_atlas_dir = PathBuf::from(next_value(&mut args, &flag)?);
                }
                "--output-dir" => config.output_dir = PathBuf::from(next_value(&mut args, &flag)?),
                "--ensemble-output-dir" => {
                    config.ensemble_output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?));
                }
                unknown => return Err(format!("unknown argument: {unknown}")),
            }
        }
        if config.steps == 0
            || config.batch_size == 0
            || config.eval_batches == 0
            || config.eval_boards == 0
        {
            return Err("step, batch, and evaluation counts must be positive".to_string());
        }
        if config.learning_rate <= 0.0 || config.attribute_loss_weight < 0.0 {
            return Err("learning rate must be positive and loss weights non-negative".to_string());
        }
        if config.ensemble_output_dir.is_some() && (!config.eval_only || config.probe_only) {
            return Err(
                "--ensemble-output-dir requires --eval-only and is unavailable with --probe-only"
                    .to_string(),
            );
        }
        Ok(Some(config))
    }

    pub(crate) fn usage() -> &'static str {
        "Usage: cargo run --release -- [--steps N] [--batch-size N] \
         [--learning-rate RATE] [--attribute-loss-weight WEIGHT] [--seed N] [--atlas-dir PATH] \
         [--validation-atlas-dir PATH] [--output-dir PATH] [--eval-only] \
         [--ensemble-output-dir PATH] \
         [--probe-only] \
         [--eval-batches N] [--eval-boards N] [--triple-seed N] [--board-seed N] \
         [--no-augment]"
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    let value = next_value(args, flag)?;
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{DEFAULT_BOARD_SEED, DEFAULT_TRIPLE_SEED, TrainConfig};

    #[test]
    fn default_evaluation_seeds_match_locked_protocol() {
        let config = TrainConfig::default();
        assert_eq!(config.triple_seed, DEFAULT_TRIPLE_SEED);
        assert_eq!(config.board_seed, DEFAULT_BOARD_SEED);
    }

    #[test]
    fn explicit_evaluation_seeds_parse_without_changing_other_defaults() {
        let args = [
            "--eval-only",
            "--triple-seed",
            "123456789",
            "--board-seed",
            "987654321",
        ]
        .into_iter()
        .map(str::to_string);
        let config = TrainConfig::from_iter(args)
            .ok()
            .flatten()
            .unwrap_or_default();

        assert!(config.eval_only);
        assert_eq!(config.triple_seed, 123_456_789);
        assert_eq!(config.board_seed, 987_654_321);
        assert_eq!(config.eval_batches, 32);
        assert_eq!(config.eval_boards, 32);
    }

    #[test]
    fn ensemble_output_dir_parses_for_eval_only() {
        let args = [
            "--eval-only",
            "--ensemble-output-dir",
            "/tmp/second-checkpoint",
        ]
        .into_iter()
        .map(str::to_string);
        let config = TrainConfig::from_iter(args)
            .ok()
            .flatten()
            .unwrap_or_default();

        assert_eq!(
            config.ensemble_output_dir,
            Some(PathBuf::from("/tmp/second-checkpoint"))
        );
    }

    #[test]
    fn ensemble_output_dir_requires_eval_only() {
        let args = ["--ensemble-output-dir", "/tmp/second-checkpoint"]
            .into_iter()
            .map(str::to_string);
        assert!(TrainConfig::from_iter(args).is_err());

        let args = [
            "--probe-only",
            "--ensemble-output-dir",
            "/tmp/second-checkpoint",
        ]
        .into_iter()
        .map(str::to_string);
        assert!(TrainConfig::from_iter(args).is_err());
    }
}
