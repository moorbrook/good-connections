import initFrankenTorch from './set_vision_runtime.js'
import { areaResampleRgb } from './preprocess.js'
import {
  boardTransitionPass,
  repeatedObservationPass,
} from './guard.js'

const status = document.querySelector('#status')
const choice = document.querySelector('#choice')
const confidence = document.querySelector('#confidence')
const scanButton = document.querySelector('#scan')
const playButton = document.querySelector('#play')
const autoButton = document.querySelector('#auto')
const selectorInput = document.querySelector('#selector')

let wasm
let modelShape
let currentTab
let currentObservation
let autoPlaying = false

async function loadModel() {
  wasm = await initFrankenTorch({
    module_or_path: chrome.runtime.getURL('set_vision_runtime_bg.wasm'),
  })
  if (wasm.model_ready() !== 1) throw new Error('embedded model failed validation')
  modelShape = {
    width: wasm.card_width(),
    height: wasm.card_height(),
    channels: wasm.card_channels(),
  }
  if (!modelShape.width || !modelShape.height || modelShape.channels !== 3) {
    throw new Error('embedded model requires an unsupported image shape')
  }
  status.textContent = 'Local FrankenTorch/WASM policy ready'
}

async function activeTab() {
  if (globalThis.__setVisionHeadlessIo) return globalThis.__setVisionHeadlessIo.activeTab()
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
  if (!tab?.id) throw new Error('No active web page')
  return tab
}

async function installReader(tabId) {
  await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] })
}

async function screenshotPixels(tab, rects, viewport) {
  if (!modelShape) throw new Error('Model dimensions are unavailable')
  const { width: cardWidth, height: cardHeight } = modelShape
  // Chrome's headless mode cannot exercise these two UI-bound APIs. The local
  // integration harness supplies only this I/O boundary; normal extension
  // execution never installs the hook.
  const captureVisibleTab = globalThis.__setVisionHeadlessIo?.captureVisibleTab
    ?? chrome.tabs.captureVisibleTab.bind(chrome.tabs)
  const dataUrl = await captureVisibleTab(tab.windowId, { format: 'png' })
  const image = new Image()
  image.src = dataUrl
  await image.decode()
  const scaleX = image.naturalWidth / viewport.width
  const scaleY = image.naturalHeight / viewport.height
  const canvas = document.createElement('canvas')
  canvas.width = image.naturalWidth
  canvas.height = image.naturalHeight
  const context = canvas.getContext('2d', { willReadFrequently: true })
  context.drawImage(image, 0, 0)
  return rects.flatMap((rect) => {
    const left = Math.max(0, Math.floor(rect.left * scaleX))
    const top = Math.max(0, Math.floor(rect.top * scaleY))
    const width = Math.max(1, Math.min(image.naturalWidth - left, Math.round(rect.width * scaleX)))
    const height = Math.max(1, Math.min(image.naturalHeight - top, Math.round(rect.height * scaleY)))
    const rgba = context.getImageData(left, top, width, height).data
    return areaResampleRgb(rgba, width, height, cardWidth, cardHeight)
  })
}

async function observeBoard(tab = undefined) {
  if (!wasm) throw new Error('Model is still loading')
  currentTab = tab ?? await activeTab()
  await installReader(currentTab.id)
  const selector = selectorInput.value.trim() || '.card-btn'
  const geometry = await chrome.tabs.sendMessage(currentTab.id, {
    type: 'set-vision:rects', selector,
  })
  if (!geometry?.rects || geometry.rects.length < 3) {
    throw new Error(`Found ${geometry?.rects?.length ?? 0} visible cards with ${selector}`)
  }
  status.textContent = `Capturing ${geometry.rects.length} cards…`
  const pixels = await screenshotPixels(currentTab, geometry.rects, geometry.viewport)
  status.textContent = `Running local policy on ${geometry.rects.length} cards…`
  const inferenceStarted = performance.now()
  if (wasm.reset_cards(geometry.rects.length) !== 1) throw new Error('Board size is unsupported')
  for (let index = 0; index < pixels.length; index++) wasm.set_pixel(index, pixels[index])
  const packed = wasm.best_triple()
  if (packed === 0xffffffff) throw new Error('Model could not score this board')
  const observation = {
    tab: currentTab,
    selector,
    cardCount: geometry.rects.length,
    pixels,
    choice: [(packed >>> 16) & 255, (packed >>> 8) & 255, packed & 255],
    actionIsSet: wasm.last_action_is_set() === 1,
    score: wasm.last_score(),
    scoreMargin: wasm.last_score_margin(),
    attributeEntropy: wasm.last_attribute_entropy(),
    attributeMargin: wasm.last_attribute_margin(),
  }
  const tripleCount = geometry.rects.length * (geometry.rects.length - 1) * (geometry.rects.length - 2) / 6
  observation.summary = `Scored ${tripleCount} visual triples in ${(performance.now() - inferenceStarted).toFixed(1)} ms`
  return observation
}

function renderObservation(observation, accepted) {
  choice.textContent = observation.actionIsSet
    ? observation.choice.map((index) => index + 1).join(' · ')
    : 'No SET'
  confidence.textContent = `score ${observation.score.toFixed(3)} · relation margin ${observation.scoreMargin.toFixed(3)} · attribute entropy ${observation.attributeEntropy.toFixed(3)} · weakest attribute margin ${observation.attributeMargin.toFixed(3)}`
  status.textContent = accepted
    ? observation.summary
    : observation.actionIsSet
      ? 'Abstained: repeated frames or decision diagnostics did not pass'
      : 'No SET predicted; no cards will be clicked'
  playButton.disabled = !accepted
}

async function scanBoard() {
  const first = await observeBoard()
  await new Promise((resolve) => setTimeout(resolve, 90))
  const second = await observeBoard(first.tab)
  const accepted = first.actionIsSet && second.actionIsSet && repeatedObservationPass(first, second)
  currentObservation = accepted ? second : undefined
  renderObservation(second, accepted)
  await chrome.tabs.sendMessage(currentTab.id, {
    type: 'set-vision:highlight', selector: second.selector, indices: accepted ? second.choice : [],
  })
  return accepted
}

async function playChoice() {
  if (!currentTab?.id || !currentObservation) return false
  const before = currentObservation
  await chrome.tabs.sendMessage(currentTab.id, {
    type: 'set-vision:highlight', selector: before.selector, indices: [],
  })
  await chrome.tabs.sendMessage(currentTab.id, {
    type: 'set-vision:click',
    selector: before.selector,
    indices: before.choice,
  })
  currentObservation = undefined
  playButton.disabled = true
  status.textContent = 'Waiting for a stable board change…'
  await new Promise((resolve) => setTimeout(resolve, 650))
  for (let attempt = 0; attempt < 10; attempt++) {
    const firstAfter = await observeBoard(before.tab)
    await new Promise((resolve) => setTimeout(resolve, 120))
    const secondAfter = await observeBoard(before.tab)
    if (boardTransitionPass(before, firstAfter, secondAfter)) {
      status.textContent = 'Board change verified from pixels'
      return true
    }
    await new Promise((resolve) => setTimeout(resolve, 220))
  }
  throw new Error('Auto-play stopped: board did not change after the click action')
}

async function guardedAutoLoop() {
  while (autoPlaying) {
    const accepted = await scanBoard()
    if (!accepted) {
      autoPlaying = false
      status.textContent = 'Auto-play stopped: abstention guard rejected the board'
      break
    }
    await playChoice()
  }
  autoButton.textContent = 'Start guarded auto-play'
  autoButton.classList.remove('running')
}

async function reportErrors(action) {
  try { await action() } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error)
    playButton.disabled = true
    autoPlaying = false
  }
}

scanButton.addEventListener('click', () => reportErrors(scanBoard))
playButton.addEventListener('click', () => reportErrors(playChoice))
autoButton.addEventListener('click', () => {
  autoPlaying = !autoPlaying
  autoButton.textContent = autoPlaying ? 'Stop auto-play' : 'Start guarded auto-play'
  autoButton.classList.toggle('running', autoPlaying)
  if (autoPlaying) void reportErrors(guardedAutoLoop)
})

void reportErrors(loadModel)
