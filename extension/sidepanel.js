import initFrankenTorch from './set_vision_runtime.js'
import { areaResampleRgb } from './preprocess.js'

const status = document.querySelector('#status')
const choice = document.querySelector('#choice')
const confidence = document.querySelector('#confidence')
const scanButton = document.querySelector('#scan')
const playButton = document.querySelector('#play')
const autoButton = document.querySelector('#auto')
const selectorInput = document.querySelector('#selector')
const thresholdInput = document.querySelector('#threshold')

let wasm
let modelShape
let currentTab
let currentChoice
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

async function scanBoard() {
  if (!wasm) throw new Error('Model is still loading')
  currentTab = await activeTab()
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
  currentChoice = [(packed >>> 16) & 255, (packed >>> 8) & 255, packed & 255]
  const probability = 1 / (1 + Math.exp(-wasm.last_score()))
  choice.textContent = currentChoice.map((index) => index + 1).join(' · ')
  confidence.textContent = `${(probability * 100).toFixed(1)}% learned-set confidence`
  const tripleCount = geometry.rects.length * (geometry.rects.length - 1) * (geometry.rects.length - 2) / 6
  status.textContent = `Scored ${tripleCount} visual triples in ${(performance.now() - inferenceStarted).toFixed(1)} ms`
  playButton.disabled = probability < Number(thresholdInput.value)
  await chrome.tabs.sendMessage(currentTab.id, {
    type: 'set-vision:highlight', selector, indices: currentChoice,
  })
  return probability
}

async function playChoice() {
  if (!currentTab?.id || !currentChoice) return
  await chrome.tabs.sendMessage(currentTab.id, {
    type: 'set-vision:click',
    selector: selectorInput.value.trim() || '.card-btn',
    indices: currentChoice,
  })
}

async function guardedAutoLoop() {
  while (autoPlaying) {
    const probability = await scanBoard()
    if (probability < Number(thresholdInput.value)) {
      autoPlaying = false
      status.textContent = 'Auto-play stopped: confidence below threshold'
      break
    }
    await playChoice()
    await new Promise((resolve) => setTimeout(resolve, 850))
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
