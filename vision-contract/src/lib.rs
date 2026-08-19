#![forbid(unsafe_code)]

pub const CARD_WIDTH: usize = 30;
pub const CARD_HEIGHT: usize = 21;
pub const CHANNELS: usize = 3;
pub const CARD_PIXELS: usize = CARD_WIDTH * CARD_HEIGHT * CHANNELS;
pub const ATTRIBUTE_COUNT: usize = 4;
pub const ATTRIBUTE_VALUE_COUNT: usize = 3;
pub const ATTRIBUTE_LOGITS: usize = ATTRIBUTE_COUNT * ATTRIBUTE_VALUE_COUNT;
pub const RELATION_CLASS_COUNT: usize = 3;
pub const RELATION_SAME: u8 = 0;
pub const RELATION_DIFFERENT: u8 = 1;
pub const RELATION_INVALID: u8 = 2;
