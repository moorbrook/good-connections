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
    pub(crate) ensemble_fourth_output_dir: Option<PathBuf>,
    pub(crate) ensemble_primary_weight: f64,
    ensemble_primary_weight_explicit: bool,
    pub(crate) ensemble_weights: Option<[f64; 3]>,
    pub(crate) ensemble_four_weights: Option<[f64; 4]>,
    pub(crate) steps: usize,
    pub(crate) batch_size: usize,
    pub(crate) learning_rate: f64,
    pub(crate) attribute_loss_weight: f64,
    pub(crate) fill_loss_multiplier: f64,
    pub(crate) relation_loss_weight: f64,
    pub(crate) negative_logsumexp_loss_weight: f64,
    pub(crate) seed: u64,
    pub(crate) eval_only: bool,
    pub(crate) probe_only: bool,
    pub(crate) eval_batches: usize,
    pub(crate) eval_boards: usize,
    pub(crate) triple_seed: u64,
    pub(crate) board_seed: u64,
    pub(crate) augment: bool,
    pub(crate) typed_relations: bool,
    pub(crate) export_board_evidence: bool,
    pub(crate) mixed_boards: bool,
    pub(crate) board_correlated_negatives: bool,
    pub(crate) wide_vision: bool,
    pub(crate) fill_texture_branch: bool,
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
            ensemble_fourth_output_dir: None,
            ensemble_primary_weight: 0.5,
            ensemble_primary_weight_explicit: false,
            ensemble_weights: None,
            ensemble_four_weights: None,
            steps: 10_000,
            batch_size: 64,
            learning_rate: 8.0e-4,
            attribute_loss_weight: 0.35,
            fill_loss_multiplier: 1.0,
            relation_loss_weight: 1.0,
            negative_logsumexp_loss_weight: 0.0,
            seed: 0x5e7_2026,
            eval_only: false,
            probe_only: false,
            eval_batches: 32,
            eval_boards: 32,
            triple_seed: DEFAULT_TRIPLE_SEED,
            board_seed: DEFAULT_BOARD_SEED,
            augment: true,
            typed_relations: false,
            export_board_evidence: false,
            mixed_boards: false,
            board_correlated_negatives: false,
            wide_vision: false,
            fill_texture_branch: false,
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
                "--fill-loss-multiplier" => {
                    config.fill_loss_multiplier = parse_value(&mut args, &flag)?;
                }
                "--relation-loss-weight" => {
                    config.relation_loss_weight = parse_value(&mut args, &flag)?;
                }
                "--negative-logsumexp-loss-weight" => {
                    config.negative_logsumexp_loss_weight = parse_value(&mut args, &flag)?;
                }
                "--seed" => config.seed = parse_value(&mut args, &flag)?,
                "--eval-only" => config.eval_only = true,
                "--probe-only" => config.probe_only = true,
                "--eval-batches" => config.eval_batches = parse_value(&mut args, &flag)?,
                "--eval-boards" => config.eval_boards = parse_value(&mut args, &flag)?,
                "--triple-seed" => config.triple_seed = parse_value(&mut args, &flag)?,
                "--board-seed" => config.board_seed = parse_value(&mut args, &flag)?,
                "--no-augment" => config.augment = false,
                "--typed-relations" => config.typed_relations = true,
                "--export-board-evidence" => config.export_board_evidence = true,
                "--mixed-boards" => config.mixed_boards = true,
                "--board-correlated-negatives" => config.board_correlated_negatives = true,
                "--wide-vision" => config.wide_vision = true,
                "--fill-texture-branch" => config.fill_texture_branch = true,
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
                "--ensemble-fourth-output-dir" => {
                    config.ensemble_fourth_output_dir =
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
                "--ensemble-four-weights" => {
                    config.ensemble_four_weights = Some([
                        parse_value(&mut args, &flag)?,
                        parse_value(&mut args, &flag)?,
                        parse_value(&mut args, &flag)?,
                        parse_value(&mut args, &flag)?,
                    ]);
                }
                unknown => return Err(format!("unknown argument: {unknown}")),
            }
        }
        config.validate_training_options()?;
        config.validate_ensemble_options()?;
        Ok(Some(config))
    }

    fn validate_training_options(&self) -> Result<(), String> {
        if self.steps == 0
            || self.batch_size == 0
            || self.eval_batches == 0
            || self.eval_boards == 0
        {
            return Err("step, batch, and evaluation counts must be positive".to_string());
        }
        if self.learning_rate <= 0.0
            || self.attribute_loss_weight < 0.0
            || !self.fill_loss_multiplier.is_finite()
            || self.fill_loss_multiplier <= 0.0
            || self.relation_loss_weight < 0.0
            || !self.negative_logsumexp_loss_weight.is_finite()
            || self.negative_logsumexp_loss_weight < 0.0
        {
            return Err(
                "learning rate and fill multiplier must be positive; loss weights must be non-negative"
                    .to_string(),
            );
        }
        if self.negative_logsumexp_loss_weight > 0.0 && !self.typed_relations {
            return Err("--negative-logsumexp-loss-weight requires --typed-relations".to_string());
        }
        if self.negative_logsumexp_loss_weight > 0.0 && !self.batch_size.is_multiple_of(2) {
            return Err("--negative-logsumexp-loss-weight requires an even batch size".to_string());
        }
        if self.board_correlated_negatives && self.negative_logsumexp_loss_weight == 0.0 {
            return Err(
                "--board-correlated-negatives requires --negative-logsumexp-loss-weight"
                    .to_string(),
            );
        }
        if self.wide_vision && !self.typed_relations {
            return Err("--wide-vision requires --typed-relations".to_string());
        }
        if self.fill_texture_branch && !self.typed_relations {
            return Err("--fill-texture-branch requires --typed-relations".to_string());
        }
        if self.fill_texture_branch && self.wide_vision {
            return Err(
                "--fill-texture-branch and --wide-vision are mutually exclusive".to_string(),
            );
        }
        if self.export_board_evidence && !self.typed_relations {
            return Err("--export-board-evidence requires --typed-relations".to_string());
        }
        Ok(())
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
        if self.ensemble_fourth_output_dir.is_some()
            && (self.ensemble_output_dir.is_none()
                || self.ensemble_third_output_dir.is_none()
                || !self.eval_only
                || self.probe_only)
        {
            return Err(
                "--ensemble-fourth-output-dir requires second and third checkpoints with --eval-only"
                    .to_string(),
            );
        }
        if self.ensemble_fourth_output_dir.is_some() && self.ensemble_four_weights.is_none() {
            return Err(
                "a fourth ensemble checkpoint requires --ensemble-four-weights W1 W2 W3 W4"
                    .to_string(),
            );
        }
        if self.ensemble_four_weights.is_some() && self.ensemble_fourth_output_dir.is_none() {
            return Err(
                "--ensemble-four-weights requires --ensemble-fourth-output-dir".to_string(),
            );
        }
        if self.ensemble_four_weights.is_some() && self.ensemble_weights.is_some() {
            return Err(
                "--ensemble-four-weights cannot be combined with --ensemble-weights".to_string(),
            );
        }
        if self.ensemble_four_weights.is_some() && self.ensemble_primary_weight_explicit {
            return Err(
                "--ensemble-primary-weight cannot be combined with --ensemble-four-weights"
                    .to_string(),
            );
        }
        if self.ensemble_third_output_dir.is_some()
            && self.ensemble_weights.is_none()
            && self.ensemble_four_weights.is_none()
        {
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
        if let Some(weights) = self.ensemble_four_weights {
            self.ensemble_four_weights = Some(normalize_ensemble_four_weights(weights)?);
        }
        Ok(())
    }

    pub(crate) fn usage() -> &'static str {
        "Usage: cargo run --release -- [--steps N] [--batch-size N] \
         [--learning-rate RATE] [--attribute-loss-weight WEIGHT] [--seed N] [--atlas-dir PATH] \
         [--fill-loss-multiplier MULTIPLIER] \
         [--relation-loss-weight WEIGHT] [--negative-logsumexp-loss-weight WEIGHT] \
         [--typed-relations] [--mixed-boards] [--board-correlated-negatives] [--wide-vision] [--fill-texture-branch] \
         [--export-board-evidence] \
         [--validation-atlas-dir PATH] [--output-dir PATH] [--eval-only] \
         [--ensemble-output-dir PATH] \
         [--ensemble-third-output-dir PATH] \
         [--ensemble-fourth-output-dir PATH] \
         [--ensemble-primary-weight W] \
         [--ensemble-weights W1 W2 W3] \
         [--ensemble-four-weights W1 W2 W3 W4] \
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

fn normalize_ensemble_four_weights(weights: [f64; 4]) -> Result<[f64; 4], String> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err("four ensemble weights must be finite and positive".to_string());
    }
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err("four ensemble weights must have a finite positive sum".to_string());
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
    fn typed_relation_and_mixed_board_modes_are_explicit() {
        let args = [
            "--typed-relations",
            "--mixed-boards",
            "--export-board-evidence",
            "--relation-loss-weight",
            "1.5",
            "--fill-loss-multiplier",
            "3.0",
            "--negative-logsumexp-loss-weight",
            "2.0",
            "--board-correlated-negatives",
            "--wide-vision",
        ]
        .into_iter()
        .map(str::to_string);
        let config = TrainConfig::from_iter(args)
            .ok()
            .flatten()
            .unwrap_or_default();
        assert!(config.typed_relations);
        assert!(config.mixed_boards);
        assert!(config.export_board_evidence);
        assert!((config.relation_loss_weight - 1.5).abs() < f64::EPSILON);
        assert!((config.fill_loss_multiplier - 3.0).abs() < f64::EPSILON);
        assert!((config.negative_logsumexp_loss_weight - 2.0).abs() < f64::EPSILON);
        assert!(config.board_correlated_negatives);
        assert!(config.wide_vision);
    }

    #[test]
    fn negative_logsumexp_loss_requires_typed_relations_and_even_batch() {
        let untyped = ["--negative-logsumexp-loss-weight", "1.0"]
            .into_iter()
            .map(str::to_string);
        assert!(TrainConfig::from_iter(untyped).is_err());

        let odd = [
            "--typed-relations",
            "--negative-logsumexp-loss-weight",
            "1.0",
            "--batch-size",
            "63",
        ]
        .into_iter()
        .map(str::to_string);
        assert!(TrainConfig::from_iter(odd).is_err());

        let missing_loss = ["--typed-relations", "--board-correlated-negatives"]
            .into_iter()
            .map(str::to_string);
        assert!(TrainConfig::from_iter(missing_loss).is_err());

        let untyped_wide = ["--wide-vision"].into_iter().map(str::to_string);
        assert!(TrainConfig::from_iter(untyped_wide).is_err());

        let untyped_texture = ["--fill-texture-branch"].into_iter().map(str::to_string);
        assert!(TrainConfig::from_iter(untyped_texture).is_err());

        let conflicting_vision = [
            "--typed-relations",
            "--wide-vision",
            "--fill-texture-branch",
        ]
        .into_iter()
        .map(str::to_string);
        assert!(TrainConfig::from_iter(conflicting_vision).is_err());

        let untyped_export = ["--export-board-evidence"].into_iter().map(str::to_string);
        assert!(TrainConfig::from_iter(untyped_export).is_err());

        let invalid_fill = ["--fill-loss-multiplier", "0"]
            .into_iter()
            .map(str::to_string);
        assert!(TrainConfig::from_iter(invalid_fill).is_err());
    }

    #[test]
    fn fill_texture_branch_is_an_explicit_typed_candidate() {
        let args = ["--typed-relations", "--fill-texture-branch"]
            .into_iter()
            .map(str::to_string);
        let config = TrainConfig::from_iter(args)
            .ok()
            .flatten()
            .unwrap_or_default();
        assert!(config.typed_relations);
        assert!(config.fill_texture_branch);
        assert!(!config.wide_vision);
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

    #[test]
    fn explicit_four_checkpoint_weights_are_normalized() {
        let args = [
            "--eval-only",
            "--ensemble-output-dir",
            "/tmp/second-checkpoint",
            "--ensemble-third-output-dir",
            "/tmp/third-checkpoint",
            "--ensemble-fourth-output-dir",
            "/tmp/fourth-checkpoint",
            "--ensemble-four-weights",
            "1",
            "2",
            "3",
            "4",
        ]
        .into_iter()
        .map(str::to_string);
        let config = TrainConfig::from_iter(args)
            .ok()
            .flatten()
            .unwrap_or_default();

        let weights = config.ensemble_four_weights.unwrap_or([0.0; 4]);
        assert!((weights[0] - 0.1).abs() < 1.0e-12);
        assert!((weights[1] - 0.2).abs() < 1.0e-12);
        assert!((weights[2] - 0.3).abs() < 1.0e-12);
        assert!((weights[3] - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn four_checkpoint_weights_require_the_fourth_checkpoint_and_positive_values() {
        let args = [
            "--eval-only",
            "--ensemble-output-dir",
            "/tmp/second-checkpoint",
            "--ensemble-third-output-dir",
            "/tmp/third-checkpoint",
            "--ensemble-four-weights",
            "1",
            "1",
            "1",
            "1",
        ]
        .into_iter()
        .map(str::to_string);
        assert!(TrainConfig::from_iter(args).is_err());

        let args = [
            "--eval-only",
            "--ensemble-fourth-output-dir",
            "/tmp/fourth-checkpoint",
            "--ensemble-four-weights",
            "1",
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
                "--ensemble-fourth-output-dir",
                "/tmp/fourth-checkpoint",
                "--ensemble-four-weights",
                value,
                "1",
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
    fn four_checkpoint_weights_cannot_be_combined_with_three_checkpoint_weights() {
        let args = [
            "--eval-only",
            "--ensemble-output-dir",
            "/tmp/second-checkpoint",
            "--ensemble-third-output-dir",
            "/tmp/third-checkpoint",
            "--ensemble-fourth-output-dir",
            "/tmp/fourth-checkpoint",
            "--ensemble-weights",
            "1",
            "1",
            "1",
            "--ensemble-four-weights",
            "1",
            "1",
            "1",
            "1",
        ]
        .into_iter()
        .map(str::to_string);
        assert!(TrainConfig::from_iter(args).is_err());
    }
}
