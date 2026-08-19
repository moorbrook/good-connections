#!/usr/bin/env node

import { readFile } from 'node:fs/promises'
import { performance } from 'node:perf_hooks'
import { fileURLToPath } from 'node:url'
import initFrankenTorch from '../extension/set_vision_runtime.js'
const iterations = Number.parseInt(process.argv[2] ?? '100', 10)
if (!Number.isSafeInteger(iterations) || iterations < 1) {
  throw new Error('iteration count must be a positive integer')
}

const wasmPath = fileURLToPath(new URL('../extension/set_vision_runtime_bg.wasm', import.meta.url))
const model = await initFrankenTorch({ module_or_path: await readFile(wasmPath) })
if (model.model_ready() !== 1) throw new Error('embedded model failed validation')

const cardCount = 12
const cardPixels = model.card_width() * model.card_height() * model.card_channels()
const pixels = Float32Array.from(
  { length: cardCount * cardPixels },
  (_, index) => (index * 73 % 251) / 250,
)

function transferBoard() {
  if (model.reset_cards(cardCount) !== 1) throw new Error('reset_cards failed')
  for (let index = 0; index < pixels.length; index++) {
    if (model.set_pixel(index, pixels[index]) !== 1) throw new Error('set_pixel failed')
  }
}

transferBoard()
for (let iteration = 0; iteration < 10; iteration++) model.best_triple()

let started = performance.now()
for (let iteration = 0; iteration < iterations; iteration++) model.best_triple()
const inferenceMs = (performance.now() - started) / iterations

started = performance.now()
for (let iteration = 0; iteration < iterations; iteration++) {
  transferBoard()
  model.best_triple()
}
const endToEndMs = (performance.now() - started) / iterations

console.log(JSON.stringify({ iterations, inferenceMs, endToEndMs }, null, 2))
