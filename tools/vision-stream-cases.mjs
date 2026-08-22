export const VISION_THEMES = ['classic', 'nocturne', 'atelier', 'signal']
export const VISION_CAPTURE_PROFILES = [
  'clean',
  'blur',
  'crop',
  'shadow',
  'lowcontrast',
  'defocus',
  'tilt',
  'occlusion',
  'gamut',
  'compound',
]

export function mix32(value) {
  let mixed = value >>> 0
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x21f0aaad)
  mixed = Math.imul(mixed ^ (mixed >>> 15), 0x735a2d97)
  return (mixed ^ (mixed >>> 15)) >>> 0
}

export function unitRandom(value) {
  return mix32(value) / 0x1_0000_0000
}

export function visionCaseEntropy(rootSeed, iteration) {
  return mix32(rootSeed ^ Math.imul(iteration + 1, 0x9e3779b1))
}

export function freshVisionCase(rootSeed, iteration) {
  const entropy = visionCaseEntropy(rootSeed, iteration)
  return {
    iteration,
    source: 'fresh',
    kind: ((iteration + (rootSeed & 1)) & 1) === 0 ? 'has-set' : 'no-set',
    theme: VISION_THEMES[mix32(entropy ^ 0x6c8e9cf5) % VISION_THEMES.length],
    profile: VISION_CAPTURE_PROFILES[
      mix32(entropy ^ 0xd1310ba6) % VISION_CAPTURE_PROFILES.length
    ],
    viewport: [
      240 + (mix32(entropy ^ 0x243f6a88) % 861),
      600 + (mix32(entropy ^ 0x85a308d3) % 521),
    ],
    dpr: 1 + (mix32(entropy ^ 0x13198a2e) & 1),
    boardSeed: mix32(entropy ^ 0x3c6ef372),
    visualSeed: mix32(entropy ^ 0xbb67ae85),
  }
}
