#!/usr/bin/env node

import assert from 'node:assert/strict'
import { areaResampleRgb } from '../extension/preprocess.js'

const width = 60
const height = 42
const rgba = new Uint8ClampedArray(width * height * 4)
for (let y = 0; y < height; y++) {
  for (let x = 0; x < width; x++) {
    const offset = (y * width + x) * 4
    rgba[offset] = x
    rgba[offset + 1] = y
    rgba[offset + 2] = x + y
    rgba[offset + 3] = 255
  }
}

const output = areaResampleRgb(rgba, width, height, 30, 21)
assert.equal(output.length, 30 * 21 * 3)
for (const [x, y] of [[0, 0], [17, 9], [29, 20]]) {
  const offset = (y * 30 + x) * 3
  assert.ok(Math.abs(output[offset] - (2 * x + 0.5) / 255) < 1e-12)
  assert.ok(Math.abs(output[offset + 1] - (2 * y + 0.5) / 255) < 1e-12)
  assert.ok(Math.abs(output[offset + 2] - (2 * x + 2 * y + 1) / 255) < 1e-12)
}
console.log('browser area-resampling contract: ok')
