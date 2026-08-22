import { describe, expect, it } from 'vitest'

import {
  BOARD_HEAD_FEATURES,
  boardActionIsSet,
  boardExistenceLogit,
  boardHeadRequiresTripleScores,
  validateBoardHead,
} from './board-head.js'

describe('board existence head', () => {
  it('preserves the raw zero-threshold action by default', () => {
    expect(boardActionIsSet({ score: 0.1 }, { type: 'score-threshold', threshold: 0 })).toBe(true)
    expect(boardActionIsSet({ score: -0.1 }, { type: 'score-threshold', threshold: 0 })).toBe(false)
  })

  it('standardizes every logistic feature', () => {
    const observation = Object.fromEntries(BOARD_HEAD_FEATURES.map((feature, index) => [feature, index + 2]))
    const config = {
      type: 'logistic',
      means: BOARD_HEAD_FEATURES.map(() => 1),
      stddevs: BOARD_HEAD_FEATURES.map(() => 2),
      weights: BOARD_HEAD_FEATURES.map(() => 0.5),
      bias: -1,
    }
    const expected = BOARD_HEAD_FEATURES.reduce((total, _, index) => total + 0.25 * (index + 1), -1)
    expect(boardExistenceLogit(observation, config)).toBeCloseTo(expected)
  })

  it('reproduces top-score and nearest-rank quantile features', () => {
    const observation = {
      ...Object.fromEntries(BOARD_HEAD_FEATURES.map((feature, index) => [feature, index + 1])),
      tripleScores: [4, 1, 3, 2],
    }
    const featureValues = [
      ...BOARD_HEAD_FEATURES.map((_, index) => index + 1),
      4, 3,
      1, 3, 4,
    ]
    const config = {
      type: 'logistic-score-distribution',
      baseFeatures: BOARD_HEAD_FEATURES,
      tripleScoreCount: 4,
      topScoreCount: 2,
      quantiles: [0, 0.5, 1],
      means: featureValues.map(() => 0),
      stddevs: featureValues.map(() => 1),
      weights: featureValues.map((_, index) => index + 1),
      bias: -2,
    }
    const expected = featureValues.reduce((total, value, index) => total + value * (index + 1), -2)
    expect(boardHeadRequiresTripleScores(config)).toBe(true)
    expect(boardExistenceLogit(observation, config)).toBe(expected)
  })

  it('rejects a score-distribution feature contract mismatch', () => {
    expect(() => validateBoardHead({
      type: 'logistic-score-distribution',
      baseFeatures: [...BOARD_HEAD_FEATURES].reverse(),
    })).toThrow(/base features/)
  })
})
