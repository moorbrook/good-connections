#![deny(unsafe_op_in_unsafe_fn)]

// `no_mangle` is an unsafe attribute in Rust 2024 because symbol collisions can
// be unsound. These eight names are the crate's deliberately tiny browser ABI;
// there are no unsafe blocks or raw-memory interfaces behind them.

use std::cell::RefCell;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_nn::Module;
use set_vision_policy::{CARD_HEIGHT, CARD_PIXELS, CARD_WIDTH, CHANNELS, VisualSetPolicy};

const MIN_CARDS: usize = 3;
const MAX_CARDS: usize = 32;
const EMBEDDED_CHECKPOINT: &[u8] = include_bytes!("../../vision-artifacts/model/set-vision.ftsv");

struct Runtime {
    session: FrankenTorchSession,
    policy: VisualSetPolicy,
    graph_boundary: usize,
    cards: Vec<f64>,
    card_count: usize,
    last_score: f32,
}

impl Runtime {
    fn new() -> Result<Self, String> {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let policy = VisualSetPolicy::new(&mut session).map_err(|error| error.to_string())?;
        let state = ft_serialize::load_state_dict_from_bytes(EMBEDDED_CHECKPOINT)
            .map_err(|error| error.to_string())?;
        policy
            .load_state_dict(&mut session, &state, true)
            .map_err(|error| error.to_string())?;
        session.no_grad_enter();
        let graph_boundary = session.autograd_graph_node_count();
        Ok(Self {
            session,
            policy,
            graph_boundary,
            cards: Vec::new(),
            card_count: 0,
            last_score: f32::NEG_INFINITY,
        })
    }

    fn reset_cards(&mut self, card_count: usize) -> bool {
        if !(MIN_CARDS..=MAX_CARDS).contains(&card_count) {
            return false;
        }
        self.cards = vec![0.0; card_count * CARD_PIXELS];
        self.card_count = card_count;
        self.last_score = f32::NEG_INFINITY;
        true
    }

    fn best_triple(&mut self) -> Result<u32, String> {
        if self.card_count < MIN_CARDS {
            return Err("too few cards".to_string());
        }
        let output = self
            .policy
            .score_board(&mut self.session, self.cards.clone(), self.card_count)
            .map_err(|error| error.to_string())?;
        self.session.truncate_autograd_graph(self.graph_boundary);

        let Some((best_index, best_score)) = output
            .scores
            .into_iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
        else {
            return Err("policy produced no finite score".to_string());
        };
        let [first, second, third] = output.triples[best_index];
        #[allow(clippy::cast_possible_truncation)]
        {
            self.last_score = best_score as f32;
        }
        let first = u32::try_from(first).map_err(|error| error.to_string())?;
        let second = u32::try_from(second).map_err(|error| error.to_string())?;
        let third = u32::try_from(third).map_err(|error| error.to_string())?;
        Ok((first << 16) | (second << 8) | third)
    }
}

thread_local! {
    // Browser WASM is single-threaded. Thread-local ownership avoids pretending
    // FrankenTorch sessions are cross-thread state and requires no unsafe code.
    static RUNTIME: RefCell<Option<Result<Runtime, String>>> = const { RefCell::new(None) };
}

fn with_runtime<T>(action: impl FnOnce(&mut Runtime) -> T) -> Option<T> {
    RUNTIME.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Runtime::new());
        }
        slot.as_mut()?.as_mut().ok().map(action)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn model_ready() -> u32 {
    u32::from(with_runtime(|_| ()).is_some())
}

#[unsafe(no_mangle)]
pub extern "C" fn card_width() -> u32 {
    u32::try_from(CARD_WIDTH).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn card_height() -> u32 {
    u32::try_from(CARD_HEIGHT).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn card_channels() -> u32 {
    u32::try_from(CHANNELS).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn reset_cards(card_count: u32) -> u32 {
    let Ok(card_count) = usize::try_from(card_count) else {
        return 0;
    };
    u32::from(with_runtime(|runtime| runtime.reset_cards(card_count)).unwrap_or(false))
}

#[unsafe(no_mangle)]
pub extern "C" fn set_pixel(index: u32, value: f32) -> u32 {
    let Ok(index) = usize::try_from(index) else {
        return 0;
    };
    with_runtime(|runtime| {
        let Some(pixel) = runtime.cards.get_mut(index) else {
            return 0;
        };
        *pixel = f64::from(value.clamp(0.0, 1.0));
        1
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn best_triple() -> u32 {
    with_runtime(|runtime| runtime.best_triple().unwrap_or(u32::MAX)).unwrap_or(u32::MAX)
}

#[unsafe(no_mangle)]
pub extern "C" fn last_score() -> f32 {
    with_runtime(|runtime| runtime.last_score).unwrap_or(f32::NEG_INFINITY)
}

#[cfg(test)]
mod tests {
    use super::{CARD_PIXELS, Runtime};

    #[test]
    fn embedded_frankentorch_policy_loads_and_scores() -> Result<(), String> {
        let mut runtime = Runtime::new()?;
        assert!(runtime.reset_cards(3));
        for (index, pixel) in runtime.cards.iter_mut().enumerate() {
            *pixel =
                f64::from(u32::try_from(index % 251).map_err(|error| error.to_string())?) / 250.0;
        }
        assert_eq!(runtime.cards.len(), 3 * CARD_PIXELS);
        assert_eq!(runtime.best_triple()?, 0x00_01_02);
        assert!(runtime.last_score.is_finite());
        Ok(())
    }
}
