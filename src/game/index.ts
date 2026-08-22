/**
 * The pure Set game core: card representation, deck generation, set detection
 * and enumeration, and the two board generators. No React, no I/O, no
 * dependencies — this is the slice-1 module that moves server-side in slice 3
 * unchanged.
 */

export {
  type Attr,
  type Attribute,
  type Card,
  ATTRIBUTES,
  COUNTS,
  COLOURS,
  SHAPES,
  FILLS,
  DECK_SIZE,
  cardId,
  cardFromId,
  fullDeck,
  cardsEqual,
} from './cards'

export {
  type Triple,
  isSet,
  completingCard,
  enumerateSets,
  enumerateSetsBrute,
  countSets,
} from './set'

export {
  type RNG,
  type Mode,
  type Board,
  BOARD_SIZE,
  MODE_A_SET_COUNT,
  mulberry32,
  shuffle,
  generateModeA,
  generateModeB,
  generateNoSetBoard,
  generateBoard,
} from './board'

export {
  type GameContext,
  type CardIndex,
  type TelemetryEvent,
  type TelemetryEventType,
  type GameRecord,
  type GameStats,
  type SolveTimeAggregate,
  type StandingRow,
  GameRecorder,
  deriveStats,
  aggregateSolveTimes,
  completionRate,
  standings,
} from './telemetry'
