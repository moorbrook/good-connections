const ATTRIBUTE_COUNT = 4
const ATTRIBUTE_VALUES = 3

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value))
}

function boundedBox(left, top, right, bottom, imageWidth, imageHeight) {
  const boundedLeft = clamp(left, 0, imageWidth - 1)
  const boundedTop = clamp(top, 0, imageHeight - 1)
  const boundedRight = clamp(right, boundedLeft + 1, imageWidth)
  const boundedBottom = clamp(bottom, boundedTop + 1, imageHeight)
  return {
    left: boundedLeft,
    top: boundedTop,
    width: boundedRight - boundedLeft,
    height: boundedBottom - boundedTop,
  }
}

/** Four integer interpretations of a fractional CSS rectangle in screenshot pixels. */
export function pixelGridBoxes(rect, scaleX, scaleY, imageWidth, imageHeight) {
  const left = rect.left * scaleX
  const top = rect.top * scaleY
  const right = (rect.left + rect.width) * scaleX
  const bottom = (rect.top + rect.height) * scaleY
  const baseLeft = Math.floor(left)
  const baseTop = Math.floor(top)
  return [
    boundedBox(
      baseLeft,
      baseTop,
      baseLeft + Math.max(1, Math.round(rect.width * scaleX)),
      baseTop + Math.max(1, Math.round(rect.height * scaleY)),
      imageWidth,
      imageHeight,
    ),
    boundedBox(Math.floor(left), Math.floor(top), Math.ceil(right), Math.ceil(bottom), imageWidth, imageHeight),
    boundedBox(Math.ceil(left), Math.ceil(top), Math.floor(right), Math.floor(bottom), imageWidth, imageHeight),
    boundedBox(Math.round(left), Math.round(top), Math.round(right), Math.round(bottom), imageWidth, imageHeight),
  ]
}

function triples(cardCount) {
  const values = []
  for (let first = 0; first < cardCount - 2; first++) {
    for (let second = first + 1; second < cardCount - 1; second++) {
      for (let third = second + 1; third < cardCount; third++) values.push([first, second, third])
    }
  }
  return values
}

function averageArrays(views, field) {
  const length = views[0][field].length
  if (views.some((view) => view[field].length !== length)) throw new Error(`multiview ${field} lengths disagree`)
  return Array.from({ length }, (_, index) =>
    views.reduce((total, view) => total + view[field][index], 0) / views.length)
}

function selectedAttributeDiagnostics(probabilities, selected) {
  let entropy = 0
  let minimumMargin = 1
  let count = 0
  for (const card of selected) {
    for (let attribute = 0; attribute < ATTRIBUTE_COUNT; attribute++) {
      const start = (card * ATTRIBUTE_COUNT + attribute) * ATTRIBUTE_VALUES
      const values = probabilities.slice(start, start + ATTRIBUTE_VALUES)
      const ranked = [...values].sort((left, right) => right - left)
      minimumMargin = Math.min(minimumMargin, ranked[0] - ranked[1])
      entropy += -values.reduce(
        (total, probability) => total + (probability > 0 ? probability * Math.log(probability) : 0),
        0,
      ) / Math.log(ATTRIBUTE_VALUES)
      count++
    }
  }
  return { attributeEntropy: entropy / count, attributeMargin: minimumMargin }
}

/** Average neural outputs across views and recompute the complete board-head observation. */
export function aggregateViewEvidence(views, cardCount) {
  if (!Array.isArray(views) || views.length === 0) throw new Error('multiview evidence is empty')
  const triplePositions = triples(cardCount)
  const tripleScores = averageArrays(views, 'tripleScores')
  const attributeProbabilities = averageArrays(views, 'attributeProbabilities')
  if (tripleScores.length !== triplePositions.length) throw new Error('multiview triple count is invalid')
  if (attributeProbabilities.length !== cardCount * ATTRIBUTE_COUNT * ATTRIBUTE_VALUES) {
    throw new Error('multiview attribute count is invalid')
  }
  const ranked = tripleScores
    .map((score, index) => ({ score, index }))
    .sort((left, right) => right.score - left.score)
  const score = ranked[0].score
  const runnerUp = ranked[1]?.score ?? Number.NEGATIVE_INFINITY
  const scoreMean = tripleScores.reduce((total, value) => total + value, 0) / tripleScores.length
  const variance = tripleScores.reduce((total, value) => total + (value - scoreMean) ** 2, 0) / tripleScores.length
  const maximum = Math.max(...tripleScores)
  const scoreLogsumexp = maximum + Math.log(
    tripleScores.reduce((total, value) => total + Math.exp(value - maximum), 0),
  )
  const choice = triplePositions[ranked[0].index]
  return {
    choice,
    score,
    scoreMargin: score - runnerUp,
    ...selectedAttributeDiagnostics(attributeProbabilities, choice),
    scoreMean,
    scoreStddev: Math.sqrt(variance),
    scoreLogsumexp,
    positiveScoreFraction: tripleScores.filter((value) => value >= 0).length / tripleScores.length,
    attributeProbabilities,
    tripleScores,
  }
}
