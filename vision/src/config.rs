use std::path::PathBuf;

pub(crate) const DEFAULT_TRIPLE_SEED: u64 = 0xa11c_e5ed_5eed;
pub(crate) const DEFAULT_BOARD_SEED: u64 = 0xb0a4_d5e7_5eed;

#[derive(Debug, Clone)]
pub(crate) struct TrainConfig {
    pub(crate) atlas_dir: PathBuf,
    pub(crate) validation_atlas_dir: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) ensemble_output_dir: Option<PathBuf>,
    pub(crate) ensemble_third_output_dir: Option<PathBuf>,
    pub(crate) ensemble_primary_weight: f64,
    ensemble_primary_weight_explicit: bool,
    pub(crate) ensemble_weights: Option<[f64; 3]>,
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
            ensemble_third_output_dir: None,
            ensemble_primary_weight: 0.5,
            ensemble_primary_weight_explicit: false,
            ensemble_weights: None,
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
                "--ensemble-third-output-dir" => {
                    config.ensemble_third_output_dir =
                        Some(PathBuf::from(next_value(&mut args, &flag)?));
                }
                "--ensemble-primary-weight" => {
                    config.ensemble_primary_weight = parse_value(&mut args, &flag)?;
                    config.ensemble_primary_weight_explicit = true;
                }
                "--ensemble-weights" => {
                    config.ensemble_weights = Some([
                        parse_value(&mut args, &flag)?,
                        parse_value(&mut args, &flag)?,
                        parse_value(&mut args, &flag)?,
                    ]);
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
        config.validate_ensemble_options()?;
        Ok(Some(config))
    }

    fn validate_ensemble_options(&mut self) -> Result<(), String> {
        if !(self.ensemble_primary_weight.is_finite()
            && 0.0 < self.ensemble_primary_weight
            && self.ensemble_primary_weight < 1.0)
        {
            return Err(
                "ensemble primary weight must be finite and strictly between 0 and 1".to_string(),
            );
        }
        if self.ensemble_output_dir.is_some() && (!self.eval_only || self.probe_only) {
            return Err(
                "--ensemble-output-dir requires --eval-only and is unavailable with --probe-only"
                    .to_string(),
            );
        }
        if self.ensemble_third_output_dir.is_some()
            && (self.ensemble_output_dir.is_none() || !self.eval_only || self.probe_only)
        {
            return Err(
                "--ensemble-third-output-dir requires --ensemble-output-dir with --eval-only"
                    .to_string(),
            );
        }
        if self.ensemble_third_output_dir.is_some() && self.ensemble_weights.is_none() {
            return Err(
                "a third ensemble checkpoint requires --ensemble-weights W1 W2 W3".to_string(),
            );
        }
        if self.ensemble_weights.is_some() && self.ensemble_third_output_dir.is_none() {
            return Err("--ensemble-weights requires --ensemble-third-output-dir".to_string());
        }
        if self.ensemble_weights.is_some() && self.ensemble_primary_weight_explicit {
            return Err(
                "--ensemble-primary-weight cannot be combined with --ensemble-weights".to_string(),
            );
        }
        if self.ensemble_primary_weight_explicit
            && (self.ensemble_output_dir.is_none() || !self.eval_only || self.probe_only)
        {
            return Err(
                "--ensemble-primary-weight requires --ensemble-output-dir with --eval-only"
                    .to_string(),
            );
        }
        if let Some(weights) = self.ensemble_weights {
            self.ensemble_weights = Some(normalize_ensemble_weights(weights)?);
        }
        Ok(())
    }

    pub(crate) fn usage() -> &'static str {
        "Usage: cargo run --release -- [--steps N] [--batch-size N] \
         [--learning-rate RATE] [--attribute-loss-weight WEIGHT] [--seed N] [--atlas-dir PATH] \
         [--validation-atlas-dir PATH] [--output-dir PATH] [--eval-only] \
         [--ensemble-output-dir PATH] \
         [--ensemble-third-output-dir PATH] \
         [--ensemble-primary-weight W] \
         [--ensemble-weights W1 W2 W3] \
         [--probe-only] \
         [--eval-batches N] [--eval-boards N] [--triple-seed N] [--board-seed N] \
         [--no-augment]"
    }
}

fn normalize_ensemble_weights(weights: [f64; 3]) -> Result<[f64; 3], String> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err("three ensemble weights must be finite and positive".to_string());
    }
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err("three ensemble weights must have a finite positive sum".to_string());
    }
    Ok(weights.map(|weight| weight / total))
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
        assert_eq!(config.ensemble_primary_weight, 0.5);
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
        assert_eq!(config.ensemble_primary_weight, 0.5);
    }

    #[test]
    fn explicit_ensemble_primary_weight_parses_with_ensemble() {
        let args = [
            "--eval-only",
            "--ensemble-output-dir",
            "/tmp/second-checkpoint",
            "--ensemble-primary-weight",
            "0.7",
        ]
        .into_iter()
        .map(str::to_string);
        let config = TrainConfig::from_iter(args)
            .ok()
            .flatten()
            .unwrap_or_default();

        assert!((config.ensemble_primary_weight - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn explicit_three_checkpoint_weights_are_normalized() {
        let args = [
            "--eval-only",
            "--ensemble-output-dir",
            "/tmp/second-checkpoint",
            "--ensemble-third-output-dir",
            "/tmp/third-checkpoint",
            "--ensemble-weights",
            "2",
            "3",
            "5",
        ]
        .into_iter()
        .map(str::to_string);
        let config = TrainConfig::from_iter(args)
            .ok()
            .flatten()
            .unwrap_or_default();

        let weights = config.ensemble_weights.unwrap_or([0.0; 3]);
        assert!((weights[0] - 0.2).abs() < f64::EPSILON);
        assert!((weights[1] - 0.3).abs() < f64::EPSILON);
        assert!((weights[2] - 0.5).abs() < f64::EPSILON);
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

    #[test]
    fn ensemble_primary_weight_is_strictly_bounded_and_scoped() {
        for value in ["0", "1", "-0.1", "NaN", "inf"] {
            let args = [
                "--eval-only",
                "--ensemble-output-dir",
                "/tmp/second-checkpoint",
                "--ensemble-primary-weight",
                value,
            ]
            .into_iter()
            .map(str::to_string);
            assert!(
                TrainConfig::from_iter(args).is_err(),
                "accepted weight {value}"
            );
        }

        let args = ["--eval-only", "--ensemble-primary-weight", "0.7"]
            .into_iter()
            .map(str::to_string);
        assert!(TrainConfig::from_iter(args).is_err());
    }

    #[test]
    fn three_checkpoint_weights_require_third_checkpoint_and_positive_values() {
        let args = [
            "--eval-only",
            "--ensemble-output-dir",
            "/tmp/second-checkpoint",
            "--ensemble-weights",
            "1",
            "1",
            "1",
        ]
        .into_iter()
        .map(str::to_string);
        assert!(TrainConfig::from_iter(args).is_err());

        let args = [
            "--eval-only",
            "--ensemble-third-output-dir",
            "/tmp/third-checkpoint",
            "--ensemble-weights",
            "1",
            "1",
            "1",
        ]
        .into_iter()
        .map(str::to_string);
        assert!(TrainConfig::from_iter(args).is_err());

        for value in ["0", "-1", "NaN", "inf"] {
            let args = [
                "--eval-only",
                "--ensemble-output-dir",
                "/tmp/second-checkpoint",
                "--ensemble-third-output-dir",
                "/tmp/third-checkpoint",
                "--ensemble-weights",
                value,
                "1",
                "1",
            ]
            .into_iter()
            .map(str::to_string);
            assert!(
                TrainConfig::from_iter(args).is_err(),
                "accepted weight {value}"
            );
        }
    }

    #[test]
    fn three_checkpoint_weights_cannot_be_combined_with_two_checkpoint_weight() {
        let args = [
            "--eval-only",
            "--ensemble-output-dir",
            "/tmp/second-checkpoint",
            "--ensemble-third-output-dir",
            "/tmp/third-checkpoint",
            "--ensemble-primary-weight",
            "0.6",
            "--ensemble-weights",
            "1",
            "1",
            "1",
        ]
        .into_iter()
        .map(str::to_string);
        assert!(TrainConfig::from_iter(args).is_err());
    }
}
