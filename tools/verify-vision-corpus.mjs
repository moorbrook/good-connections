import { readdir, readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const [directoryArgument, expectedCountArgument] = process.argv.slice(2)
const expectedCount = Number(expectedCountArgument)
if (!directoryArgument || !Number.isSafeInteger(expectedCount) || expectedCount < 1) {
  console.error('Usage: node tools/verify-vision-corpus.mjs DIRECTORY EXPECTED_COUNT')
  process.exit(2)
}

const directory = resolve(directoryArgument)
const files = (await readdir(directory)).filter((file) => file.endsWith('.png')).sort()
if (files.length !== expectedCount) throw new Error(`Expected ${expectedCount} PNGs, found ${files.length}`)
const pattern = /^(classic|nocturne|atelier|signal)-(\d+)(?:-p(clean|blur|crop|shadow|lowcontrast|jpeg|defocus|tilt|occlusion|gamut|compound))?(?:-dpr([12]))?(?:-w(\d+))?\.png$/
const counts = { themes: {}, profiles: {}, dprs: {}, widths: {}, seeds: {} }
const increment = (group, value) => { group[value] = (group[value] ?? 0) + 1 }

for (const file of files) {
  const match = file.match(pattern)
  if (!match) throw new Error(`Unexpected corpus filename: ${file}`)
  const [, theme, seed, profile = 'clean', dprText = '1', widthText = '190'] = match
  const dpr = Number(dprText)
  const cardWidth = Number(widthText)
  const cardHeight = Math.round(cardWidth * 140 / 190)
  const expectedWidth = (20 + 9 * cardWidth + 8 * 4) * dpr
  const expectedHeight = (20 + 9 * cardHeight + 8 * 4) * dpr
  const bytes = await readFile(resolve(directory, file))
  if (bytes.length < 24 || bytes.toString('hex', 0, 8) !== '89504e470d0a1a0a') {
    throw new Error(`${file} is not a PNG`)
  }
  const width = bytes.readUInt32BE(16)
  const height = bytes.readUInt32BE(20)
  if (width !== expectedWidth || height !== expectedHeight) {
    throw new Error(`${file} is ${width}x${height}; expected ${expectedWidth}x${expectedHeight}`)
  }
  increment(counts.themes, theme)
  increment(counts.profiles, profile)
  increment(counts.dprs, dprText)
  increment(counts.widths, widthText)
  increment(counts.seeds, seed)
}

console.log(JSON.stringify({ directory, files: files.length, ...counts }, null, 2))
