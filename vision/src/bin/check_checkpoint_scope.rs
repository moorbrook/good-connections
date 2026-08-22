#![forbid(unsafe_code)]

use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let base_path = PathBuf::from(arguments.next().ok_or(usage())?);
    let candidate_path = PathBuf::from(arguments.next().ok_or(usage())?);
    let mut allowed_prefixes = Vec::new();
    while let Some(argument) = arguments.next() {
        if argument != "--allow-prefix" {
            return Err(usage().into());
        }
        allowed_prefixes.push(arguments.next().ok_or("--allow-prefix requires a value")?);
    }
    if allowed_prefixes.is_empty() {
        allowed_prefixes = vec![
            "attribute_relation_".to_string(),
            "set_relation_".to_string(),
        ];
    }
    let base = ft_serialize::load_state_dict(&base_path)?;
    let candidate = ft_serialize::load_state_dict(&candidate_path)?;
    if base.keys().ne(candidate.keys()) {
        return Err("checkpoint keys differ".into());
    }
    let mut changed = Vec::new();
    let mut unchanged = 0_usize;
    for (name, base_tensor) in &base {
        let candidate_tensor = candidate.get(name).ok_or("checkpoint keys differ")?;
        if base_tensor.meta() != candidate_tensor.meta() {
            return Err(format!("tensor metadata differs for {name}").into());
        }
        let base_values = base_tensor.contiguous_values_as_f64()?;
        let candidate_values = candidate_tensor.contiguous_values_as_f64()?;
        let differs = base_values
            .iter()
            .zip(&candidate_values)
            .any(|(left, right)| left.to_bits() != right.to_bits());
        if differs {
            if !allowed_prefixes
                .iter()
                .any(|prefix| name.starts_with(prefix))
            {
                return Err(format!("frozen tensor changed: {name}").into());
            }
            changed.push(name.clone());
        } else {
            unchanged += 1;
        }
    }
    if changed.is_empty() {
        return Err("no decision-head tensor changed".into());
    }
    println!("base: {}", base_path.display());
    println!("candidate: {}", candidate_path.display());
    println!("allowed_prefixes: {}", allowed_prefixes.join(","));
    println!("changed_tensors: {}", changed.len());
    println!("unchanged_tensors: {unchanged}");
    for name in changed {
        println!("changed: {name}");
    }
    Ok(())
}

fn usage() -> &'static str {
    "usage: check_checkpoint_scope BASE CANDIDATE [--allow-prefix PREFIX]..."
}
