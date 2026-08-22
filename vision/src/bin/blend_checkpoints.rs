use std::{collections::BTreeMap, env, error::Error, path::PathBuf};

use ft_core::{DType, DenseTensor};
use ft_nn::StateDict;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let primary_path = PathBuf::from(arguments.next().ok_or(usage())?);
    let secondary_path = PathBuf::from(arguments.next().ok_or(usage())?);
    let primary_weight = arguments.next().ok_or(usage())?.parse::<f64>()?;
    let output_path = PathBuf::from(arguments.next().ok_or(usage())?);
    if arguments.next().is_some() {
        return Err(usage().into());
    }
    if !primary_weight.is_finite() || !(0.0..=1.0).contains(&primary_weight) {
        return Err("primary weight must be finite and between zero and one".into());
    }

    let primary = ft_serialize::load_state_dict(&primary_path)?;
    let secondary = ft_serialize::load_state_dict(&secondary_path)?;
    let blended = blend_state_dicts(&primary, &secondary, primary_weight)?;
    ft_serialize::save_state_dict(&blended, &output_path)?;
    println!("primary: {}", primary_path.display());
    println!("secondary: {}", secondary_path.display());
    println!("primary_weight: {primary_weight}");
    println!("tensors: {}", blended.len());
    println!("output: {}", output_path.display());
    Ok(())
}

fn usage() -> &'static str {
    "usage: blend_checkpoints PRIMARY SECONDARY PRIMARY_WEIGHT OUTPUT"
}

fn blend_state_dicts(
    primary: &StateDict,
    secondary: &StateDict,
    primary_weight: f64,
) -> Result<StateDict, Box<dyn Error>> {
    if primary.keys().ne(secondary.keys()) {
        return Err("checkpoint keys differ".into());
    }
    let mut blended = BTreeMap::new();
    for (key, primary_tensor) in primary {
        let secondary_tensor = secondary.get(key).ok_or("checkpoint keys differ")?;
        if primary_tensor.meta() != secondary_tensor.meta() {
            return Err(format!("tensor metadata differs for {key}").into());
        }
        if primary_tensor.meta().dtype() != DType::F64 {
            return Err(format!("tensor {key} is not f64").into());
        }
        let primary_values = primary_tensor.contiguous_values_as_f64()?;
        let secondary_values = secondary_tensor.contiguous_values_as_f64()?;
        let values = blend_values(&primary_values, &secondary_values, primary_weight)?;
        let tensor = DenseTensor::from_storage(primary_tensor.meta().clone(), values)?;
        blended.insert(key.clone(), tensor);
    }
    Ok(blended)
}

fn blend_values(
    primary: &[f64],
    secondary: &[f64],
    primary_weight: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    if primary.len() != secondary.len() {
        return Err("tensor lengths differ".into());
    }
    let secondary_weight = 1.0 - primary_weight;
    let values = primary
        .iter()
        .zip(secondary)
        .map(|(&left, &right)| left * primary_weight + right * secondary_weight)
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("blended tensor contains a non-finite value".into());
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::blend_values;

    #[test]
    fn blends_values_with_primary_weight() {
        let blended = blend_values(&[2.0, 6.0], &[10.0, 14.0], 0.25)
            .unwrap_or_else(|error| unreachable!("valid values must blend: {error}"));
        assert_eq!(blended, [8.0, 12.0]);
    }

    #[test]
    fn rejects_different_lengths() {
        assert!(blend_values(&[1.0], &[1.0, 2.0], 0.5).is_err());
    }
}
