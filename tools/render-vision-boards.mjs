#!/usr/bin/env node

import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { access, mkdtemp, readFile, rm } from 'node:fs/promises'
import { createServer as createNetServer } from 'node:net'
import { homedir, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { freshVisionCase, mix32 } from './vision-stream-cases.mjs'

const MAGIC = 'SET_VISION_CURRICULUM_V1'
const projectRoot = resolve(fileURLToPath(new URL('..', import.meta.url)))
const chromeCandidates = [
  process.env.CHROME_BIN,
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean)

function parsePositiveInteger(value, name) {
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`${name} must be a positive integer`)
  return parsed
}

function parseUint32(value, name) {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0 || parsed > 0xffff_ffff) {
    throw new Error(`${name} must be an integer between 0 and 4294967295`)
  }
  return parsed
}

function parseOptions(args) {
  const options = {
    curriculum: undefined,
    generateBoards: undefined,
    seed: undefined,
    start: 0,
    limit: Number.POSITIVE_INFINITY,
    variantsPerFailure: 1,
    modelWidth: 30,
    modelHeight: 21,
  }
  for (let index = 0; index < args.length; index++) {
    const argument = args[index]
    if (argument === '--curriculum') options.curriculum = resolve(args[++index] ?? '')
    else if (argument === '--limit') {
      options.limit = parsePositiveInteger(args[++index], '--limit')
    } else if (argument === '--generate-boards') {
      options.generateBoards = parsePositiveInteger(args[++index], '--generate-boards')
    } else if (argument === '--seed') {
      options.seed = parseUint32(args[++index], '--seed')
    } else if (argument === '--start') {
      options.start = parseUint32(args[++index], '--start')
    } else if (argument === '--variants-per-failure') {
      options.variantsPerFailure = parsePositiveInteger(args[++index], '--variants-per-failure')
    } else if (argument === '--model-width') {
      options.modelWidth = parsePositiveInteger(args[++index], '--model-width')
    } else if (argument === '--model-height') {
      options.modelHeight = parsePositiveInteger(args[++index], '--model-height')
    } else if (argument === '--help') {
      console.error(`Usage: node tools/render-vision-boards.mjs --curriculum PATH [--limit N]
    [--variants-per-failure N] [--seed UINT32]

  node tools/render-vision-boards.mjs --generate-boards N --seed UINT32 [--start UINT32]
    [--model-width 30] [--model-height 21]

  node tools/render-vision-boards.mjs --curriculum PATH --variants-per-failure N
    --generate-boards N --seed UINT32

Writes a framed binary tensor stream to stdout. Generated-board mode uses the
same deterministic fresh-case distribution as the browser evaluator.
Diagnostics go to stderr.`)
      process.exit(0)
    } else throw new Error(`unknown argument: ${argument}`)
  }
  if (!options.curriculum && !options.generateBoards) {
    throw new Error('provide --curriculum, --generate-boards, or both')
  }
  if (options.generateBoards && options.seed === undefined) {
    throw new Error('--generate-boards requires --seed')
  }
  if (options.curriculum && options.start !== 0) {
    throw new Error('--start is only available with --generate-boards')
  }
  if (!options.curriculum && options.variantsPerFailure !== 1) {
    throw new Error('--variants-per-failure requires --curriculum')
  }
  if (options.variantsPerFailure > 1 && options.seed === undefined) {
    throw new Error('multiple curriculum variants require --seed')
  }
  if (options.modelWidth > 256 || options.modelHeight > 256) {
    throw new Error('model dimensions must not exceed 256 pixels')
  }
  return options
}

function failureKey(failure) {
  return JSON.stringify([
    failure.kind,
    failure.theme,
    failure.profile,
    failure.viewport,
    failure.dpr,
    failure.boardSeed,
    failure.visualSeed,
  ])
}

async function loadFailures(path, limit) {
  const failures = (await readFile(path, 'utf8'))
    .split('\n')
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line)
      } catch (error) {
        throw new Error(`invalid curriculum JSONL at line ${index + 1}: ${error}`)
      }
    })
    .filter((record) => record.event === undefined || record.event === 'failure')
  const uniqueByCondition = new Map()
  for (const failure of failures) {
    const key = failureKey(failure)
    // Keep the newest observation so schema additions (for example boardIds)
    // replace an older replay of the same rendering condition.
    uniqueByCondition.set(key, failure)
  }
  const unique = [...uniqueByCondition.values()].slice(0, limit)
  if (unique.length === 0) throw new Error('curriculum contains no failures')
  return unique
}

export function curriculumVariants(failure, failureIndex, count, seed) {
  if (!Number.isSafeInteger(count) || count <= 0) throw new Error('curriculum variant count must be positive')
  const entropy = mix32(seed ^ Math.imul(failureIndex + 1, 0x9e3779b1))
  return Array.from({ length: count }, (_, variant) => {
    const testCase = {
      ...failure,
      source: variant === 0 ? 'curriculum-exact' : 'curriculum-variant',
      curriculumIndex: failureIndex,
      curriculumVariant: variant,
    }
    if (variant === 1) {
      testCase.visualSeed = mix32(entropy ^ 0xbb67ae85)
    } else if (variant === 2) {
      testCase.viewport = [
        Math.max(230, Math.min(1100, failure.viewport[0] + (mix32(entropy ^ 0x17a1465) % 81) - 40)),
        Math.max(560, Math.min(1200, failure.viewport[1] + (mix32(entropy ^ 0x2c1b3c6d) % 101) - 50)),
      ]
    } else if (variant >= 3) {
      testCase.boardSeed = mix32(entropy ^ Math.imul(variant, 0x3c6ef372))
      testCase.visualSeed = mix32(entropy ^ Math.imul(variant, 0xbb67ae85))
      delete testCase.boardIds
      delete testCase.chosenIds
    }
    return testCase
  })
}

async function freePort() {
  return new Promise((resolvePort, reject) => {
    const server = createNetServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      server.close(() => resolvePort(address.port))
    })
  })
}

async function waitFor(check, label, timeoutMs = 15_000) {
  const deadline = Date.now() + timeoutMs
  let lastError
  while (Date.now() < deadline) {
    try {
      const value = await check()
      if (value) return value
    } catch (error) {
      lastError = error
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 5))
  }
  throw new Error(`timed out waiting for ${label}${lastError ? `: ${lastError}` : ''}`)
}

class Cdp {
  constructor(url) {
    this.socket = new WebSocket(url)
    this.nextId = 1
    this.pending = new Map()
  }

  async open() {
    await new Promise((resolveOpen, reject) => {
      this.socket.addEventListener('open', resolveOpen, { once: true })
      this.socket.addEventListener('error', reject, { once: true })
    })
    this.socket.addEventListener('message', ({ data }) => {
      const message = JSON.parse(data)
      if (!message.id) return
      const pending = this.pending.get(message.id)
      if (!pending) return
      this.pending.delete(message.id)
      if (message.error) pending.reject(new Error(message.error.message))
      else pending.resolve(message.result)
    })
    return this
  }

  send(method, params = {}) {
    const id = this.nextId++
    return new Promise((resolveSend, reject) => {
      this.pending.set(id, { resolve: resolveSend, reject })
      this.socket.send(JSON.stringify({ id, method, params }))
    })
  }

  async evaluate(expression) {
    const result = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    })
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text)
    }
    return result.result.value
  }

  close() {
    this.socket.close()
  }
}

function validateBoardCase(testCase, index) {
  if (!['has-set', 'no-set'].includes(testCase.kind)) throw new Error(`board ${index}: invalid kind`)
  if (!['classic', 'nocturne', 'atelier', 'signal'].includes(testCase.theme)) {
    throw new Error(`board ${index}: invalid theme`)
  }
  if (!['clean', 'blur', 'crop', 'shadow', 'lowcontrast', 'defocus', 'tilt', 'occlusion', 'gamut', 'compound'].includes(testCase.profile)) {
    throw new Error(`board ${index}: invalid capture profile`)
  }
  if (!Array.isArray(testCase.viewport) || testCase.viewport.length !== 2
    || testCase.viewport.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error(`board ${index}: invalid viewport`)
  }
  if (![1, 2].includes(testCase.dpr)) throw new Error(`board ${index}: invalid DPR`)
  for (const field of ['boardSeed', 'visualSeed']) {
    if (!Number.isInteger(testCase[field]) || testCase[field] < 0 || testCase[field] > 0xffff_ffff) {
      throw new Error(`board ${index}: invalid ${field}`)
    }
  }
}

async function writeOutput(value) {
  if (!process.stdout.write(value)) await once(process.stdout, 'drain')
}

function float32LeBuffer(values) {
  const buffer = Buffer.allocUnsafe(values.length * 4)
  for (let index = 0; index < values.length; index++) {
    buffer.writeFloatLE(values[index], index * 4)
  }
  return buffer
}

async function renderBoardCase(page, baseUrl, testCase, index, modelWidth, modelHeight) {
  validateBoardCase(testCase, index)
  console.error(`rendering board ${index}: opening live board`)
  const [width, height] = testCase.viewport
  await page.send('Emulation.setDeviceMetricsOverride', {
    width,
    height,
    deviceScaleFactor: testCase.dpr,
    mobile: false,
  })
  const parameters = new URLSearchParams({
    'vision-board': testCase.theme,
    'board-kind': testCase.kind,
    'board-seed': String(testCase.boardSeed),
    'vision-seed': String(testCase.visualSeed),
    'vision-capture': testCase.profile ?? 'clean',
  })
  if (index === 1) {
    await page.send('Page.navigate', { url: `${baseUrl}/?${parameters}` })
  } else {
    await page.evaluate(`(() => {
      document.dispatchEvent(new CustomEvent('set-vision-render-case', { detail: ${JSON.stringify({
        theme: testCase.theme,
        boardKind: testCase.kind,
        boardSeed: testCase.boardSeed,
        visualSeed: testCase.visualSeed,
        captureProfile: testCase.profile ?? 'clean',
      })} }))
      return true
    })()`)
  }
  await waitFor(
    () => page.evaluate(`document.documentElement.dataset.visionBoardReady === 'true'
      && document.querySelectorAll('.card-btn').length === 12`),
    `board ${index}`,
  )
  await page.evaluate('document.fonts.ready.then(() => true)')
  const geometry = await page.evaluate(`(() => ({
    viewport: { width: innerWidth, height: innerHeight },
    rects: Array.from(document.querySelectorAll('.card-btn'), (card) => {
      const rect = card.getBoundingClientRect()
      return { left: rect.left, top: rect.top, width: rect.width, height: rect.height }
    }),
    boardIds: Array.from(document.querySelectorAll('.card-btn'), (card) => Number(card.dataset.cardId)),
  }))()`)
  if (testCase.boardIds && JSON.stringify(testCase.boardIds) !== JSON.stringify(geometry.boardIds)) {
    throw new Error(`board ${index}: recorded board IDs do not reproduce from its seeds`)
  }
  const screenshot = await page.send('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: false,
    fromSurface: true,
  })
  console.error(`rendering board ${index}: preprocessing live capture`)
  const pixels = await page.evaluate(`(async () => {
    const image = new Image()
    image.src = ${JSON.stringify(`data:image/png;base64,${screenshot.data}`)}
    await image.decode()
    const geometry = ${JSON.stringify(geometry)}
    const scaleX = image.naturalWidth / geometry.viewport.width
    const scaleY = image.naturalHeight / geometry.viewport.height
    const canvas = document.createElement('canvas')
    canvas.width = image.naturalWidth
    canvas.height = image.naturalHeight
    const context = canvas.getContext('2d', { willReadFrequently: true })
    context.drawImage(image, 0, 0)
    const output = []
    for (const rect of geometry.rects) {
      const left = Math.max(0, Math.floor(rect.left * scaleX))
      const top = Math.max(0, Math.floor(rect.top * scaleY))
      const width = Math.max(1, Math.min(image.naturalWidth - left, Math.round(rect.width * scaleX)))
      const height = Math.max(1, Math.min(image.naturalHeight - top, Math.round(rect.height * scaleY)))
      const rgba = context.getImageData(left, top, width, height).data
      for (let y = 0; y < ${modelHeight}; y++) {
        const sourceYStart = Math.floor(y * height / ${modelHeight})
        const sourceYEnd = Math.max(sourceYStart + 1, Math.floor((y + 1) * height / ${modelHeight}))
        for (let x = 0; x < ${modelWidth}; x++) {
          const sourceXStart = Math.floor(x * width / ${modelWidth})
          const sourceXEnd = Math.max(sourceXStart + 1, Math.floor((x + 1) * width / ${modelWidth}))
          const sampleCount = (sourceXEnd - sourceXStart) * (sourceYEnd - sourceYStart)
          const totals = [0, 0, 0]
          for (let sourceY = sourceYStart; sourceY < sourceYEnd; sourceY++) {
            for (let sourceX = sourceXStart; sourceX < sourceXEnd; sourceX++) {
              const source = (sourceY * width + sourceX) * 4
              totals[0] += rgba[source]
              totals[1] += rgba[source + 1]
              totals[2] += rgba[source + 2]
            }
          }
          output.push(totals[0] / (sampleCount * 255))
          output.push(totals[1] / (sampleCount * 255))
          output.push(totals[2] / (sampleCount * 255))
        }
      }
    }
    return output
  })()`)
  const expectedPixels = 12 * modelWidth * modelHeight * 3
  if (pixels.length !== expectedPixels) {
    throw new Error(`board ${index}: rendered ${pixels.length} floats, expected ${expectedPixels}`)
  }
  return { boardIds: geometry.boardIds, pixels }
}

export async function main() {
  const options = parseOptions(process.argv.slice(2))
  await access(join(projectRoot, 'dist', 'index.html'))
  const failures = options.curriculum ? await loadFailures(options.curriculum, options.limit) : []
  const curriculumCases = failures.flatMap((failure, index) =>
    curriculumVariants(failure, index, options.variantsPerFailure, options.seed ?? 0))
  const freshCases = options.generateBoards
    ? Array.from(
      { length: options.generateBoards },
      (_, offset) => freshVisionCase(options.seed, options.start + offset),
    )
    : []
  const cases = [...curriculumCases, ...freshCases]
  console.error(`prepared ${cases.length} rendered training case(s) at ${options.modelWidth}x${options.modelHeight}: ${failures.length} unique failures × ${options.variantsPerFailure} variants + ${freshCases.length} fresh`)
  const chrome = await waitFor(async () => {
    for (const candidate of chromeCandidates) {
      try {
        await access(candidate)
        return candidate
      } catch {}
    }
    return undefined
  }, 'a Chrome binary', 500)
  const previewPort = await freePort()
  const debugPort = await freePort()
  const temporaryRoot = await mkdtemp(join(tmpdir(), 'set-vision-rendered-boards.'))
  const profileDir = join(temporaryRoot, 'profile')
  const processes = []
  let page
  try {
    const preview = spawn('npm', [
      'run', 'preview', '--', '--host', '127.0.0.1', '--port', String(previewPort), '--strictPort',
    ], { cwd: projectRoot, stdio: ['ignore', 'ignore', 'inherit'] })
    processes.push(preview)
    await waitFor(async () => (await fetch(`http://127.0.0.1:${previewPort}/`)).ok, 'Vite preview')
    console.error('rendered-board preview ready')
    const browser = spawn(chrome, [
      '--headless=new',
      '--no-first-run',
      '--no-default-browser-check',
      '--hide-scrollbars',
      `--remote-debugging-port=${debugPort}`,
      `--user-data-dir=${profileDir}`,
      '--window-size=1200,1000',
      'about:blank',
    ], { stdio: ['ignore', 'ignore', 'ignore'] })
    processes.push(browser)
    const target = await waitFor(async () => {
      const response = await fetch(`http://127.0.0.1:${debugPort}/json/list`)
      const targets = await response.json()
      return targets.find((candidate) => candidate.type === 'page')
    }, 'headless browser page')
    console.error('rendered-board browser ready')
    page = await new Cdp(target.webSocketDebuggerUrl).open()
    await page.send('Page.enable')
    await page.send('Runtime.enable')

    await writeOutput(`${MAGIC}\n${cases.length}\n`)
    for (const [index, testCase] of cases.entries()) {
      const rendered = await renderBoardCase(
        page,
        `http://127.0.0.1:${previewPort}`,
        testCase,
        index + 1,
        options.modelWidth,
        options.modelHeight,
      )
      const name = options.curriculum
        && index < curriculumCases.length
        ? `failure-${testCase.rootSeed ?? 0}-${testCase.iteration ?? index}-v${testCase.curriculumVariant}`
        : `browser-train-${options.seed}-${testCase.iteration}`
      const chosen = Array.isArray(testCase.chosenIds) && testCase.chosenIds.length === 3
        ? testCase.chosenIds.join(',')
        : '-'
      await writeOutput(`${name}\t${testCase.kind}\t${rendered.boardIds.join(',')}\t${chosen}\t${rendered.pixels.length}\n`)
      await writeOutput(float32LeBuffer(rendered.pixels))
      console.error(`rendered board ${index + 1}/${cases.length}: ${name}`)
    }
  } finally {
    page?.close()
    for (const process of processes.reverse()) process.kill('SIGTERM')
    await rm(temporaryRoot, { recursive: true, force: true, maxRetries: 8, retryDelay: 100 })
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack ?? String(error))
    process.exitCode = 1
  })
}
