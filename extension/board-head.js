export const BOARD_HEAD_FEATURES = [
  'score',
  'scoreMargin',
  'attributeEntropy',
  'attributeMargin',
  'scoreMean',
  'scoreStddev',
  'scoreLogsumexp',
  'positiveScoreFraction',
]

function finiteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value)
}

function validateLogisticParameters(config, expected) {
  for (const key of ['means', 'stddevs', 'weights']) {
    if (!Array.isArray(config[key]) || config[key].length !== expected || config[key].some((value) => !finiteNumber(value))) {
      throw new Error(`board head ${key} must contain ${expected} finite values`)
    }
  }
  if (config.stddevs.some((value) => value <= 0)) throw new Error('board head stddevs must be positive')
  if (!finiteNumber(config.bias)) throw new Error('board head bias must be finite')
}

export function validateBoardHead(config) {
  if (!config || typeof config !== 'object') throw new Error('board head config is missing')
  if (config.type === 'score-threshold') {
    if (!finiteNumber(config.threshold)) throw new Error('board head threshold must be finite')
    return config
  }
  if (config.type === 'logistic') {
    validateLogisticParameters(config, BOARD_HEAD_FEATURES.length)
    return config
  }
  if (config.type === 'logistic-score-distribution') {
    if (JSON.stringify(config.baseFeatures) !== JSON.stringify(BOARD_HEAD_FEATURES)) {
      throw new Error('score-distribution head base features do not match the runtime contract')
    }
    if (!Number.isInteger(config.tripleScoreCount) || config.tripleScoreCount <= 0) {
      throw new Error('score-distribution head tripleScoreCount must be a positive integer')
    }
    if (!Number.isInteger(config.topScoreCount) || config.topScoreCount <= 0 || config.topScoreCount > config.tripleScoreCount) {
      throw new Error('score-distribution head topScoreCount is invalid')
    }
    if (!Array.isArray(config.quantiles) || config.quantiles.length === 0
      || config.quantiles.some((value) => !finiteNumber(value) || value < 0 || value > 1)) {
      throw new Error('score-distribution head quantiles must be finite values in [0, 1]')
    }
    validateLogisticParameters(config, BOARD_HEAD_FEATURES.length + config.topScoreCount + config.quantiles.length)
    return config
  }
  throw new Error(`unsupported board head type: ${config.type}`)
}

export function boardHeadRequiresTripleScores(config) {
  return validateBoardHead(config).type === 'logistic-score-distribution'
}

function logisticFeatures(observation, config) {
  const base = BOARD_HEAD_FEATURES.map((feature) => {
    const value = observation[feature]
    if (!finiteNumber(value)) throw new Error(`board observation ${feature} must be finite`)
    return value
  })
  if (config.type !== 'logistic-score-distribution') return base
  if (!Array.isArray(observation.tripleScores)
    || observation.tripleScores.length !== config.tripleScoreCount
    || observation.tripleScores.some((value) => !finiteNumber(value))) {
    throw new Error(`board observation tripleScores must contain ${config.tripleScoreCount} finite values`)
  }
  const descending = [...observation.tripleScores].sort((left, right) => right - left)
  const ascending = [...descending].reverse()
  return [
    ...base,
    ...descending.slice(0, config.topScoreCount),
    ...config.quantiles.map((quantile) => ascending[Math.round(quantile * (ascending.length - 1))]),
  ]
}

export function boardExistenceLogit(observation, config) {
  validateBoardHead(config)
  if (config.type === 'score-threshold') return observation.score - config.threshold
  return logisticFeatures(observation, config).reduce((total, value, index) => {
    return total + config.weights[index] * (value - config.means[index]) / config.stddevs[index]
  }, config.bias)
}

export function boardActionIsSet(observation, config) {
  return boardExistenceLogit(observation, config) >= 0
}
