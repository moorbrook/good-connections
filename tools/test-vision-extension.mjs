#!/usr/bin/env node

import { spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { access, appendFile, cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer as createNetServer } from 'node:net'
import { homedir, tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  freshVisionCase,
  mix32,
  unitRandom,
  visionCaseEntropy,
} from './vision-stream-cases.mjs'

const projectRoot = resolve(fileURLToPath(new URL('..', import.meta.url)))
const chromeCandidates = [
  process.env.CHROME_BIN,
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean)

function optionValue(args, index, name) {
  const value = args[index + 1]
  if (!value || value.startsWith('--')) throw new Error(`${name} requires a value`)
  return value
}

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

function parseUnitInterval(value, name) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) throw new Error(`${name} must be between 0 and 1`)
  return parsed
}

function parseOptions(args) {
  const options = {
    stream: false,
    audit: false,
    boards: Number.POSITIVE_INFINITY,
    start: 0,
    seed: randomBytes(4).readUInt32LE(0),
    reportEvery: 100,
    curriculumPath: join(projectRoot, 'vision-artifacts', 'stream-curriculum.jsonl'),
    focusRate: 0.5,
    minRawAccuracy: undefined,
    evidencePath: undefined,
    multiView: false,
  }
  for (let index = 0; index < args.length; index++) {
    const argument = args[index]
    if (argument === '--stream') options.stream = true
    else if (argument === '--audit') options.audit = true
    else if (argument === '--boards') options.boards = parsePositiveInteger(optionValue(args, index++, argument), argument)
    else if (argument === '--start') options.start = parseUint32(optionValue(args, index++, argument), argument)
    else if (argument === '--seed') options.seed = parseUint32(optionValue(args, index++, argument), argument)
    else if (argument === '--report-every') options.reportEvery = parsePositiveInteger(optionValue(args, index++, argument), argument)
    else if (argument === '--curriculum') options.curriculumPath = resolve(optionValue(args, index++, argument))
    else if (argument === '--focus-rate') options.focusRate = parseUnitInterval(optionValue(args, index++, argument), argument)
    else if (argument === '--min-raw-accuracy') options.minRawAccuracy = parseUnitInterval(optionValue(args, index++, argument), argument)
    else if (argument === '--evidence') options.evidencePath = resolve(optionValue(args, index++, argument))
    else if (argument === '--multi-view') options.multiView = true
    else if (argument === '--help') {
      console.log(`Usage:
  node tools/test-vision-extension.mjs
  node tools/test-vision-extension.mjs --stream [--boards N] [--start N] [--seed N]
    [--audit] [--curriculum PATH] [--focus-rate 0..1]
    [--report-every N] [--min-raw-accuracy 0..1] [--evidence PATH] [--multi-view]

Streaming mine mode journals every failure and samples nearby conditions from
the growing curriculum. --audit disables both feedback and curriculum input.`)
      process.exit(0)
    } else throw new Error(`Unknown option: ${argument}`)
  }
  if (options.audit && !options.stream) throw new Error('--audit requires --stream')
  if (options.evidencePath && !options.stream) throw new Error('--evidence requires --stream')
  if (options.evidencePath && options.audit) throw new Error('--evidence is forbidden in audit mode')
  return options
}

const options = parseOptions(process.argv.slice(2))

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
    await new Promise((resolveWait) => setTimeout(resolveWait, 100))
  }
  throw new Error(`Timed out waiting for ${label}${lastError ? `: ${lastError}` : ''}`)
}

class Cdp {
  constructor(url) {
    this.socket = new WebSocket(url)
    this.nextId = 1
    this.pending = new Map()
    this.listeners = new Map()
  }

  async open() {
    await new Promise((resolveOpen, reject) => {
      this.socket.addEventListener('open', resolveOpen, { once: true })
      this.socket.addEventListener('error', reject, { once: true })
    })
    this.socket.addEventListener('message', ({ data }) => {
      const message = JSON.parse(data)
      if (!message.id) {
        for (const listener of this.listeners.get(message.method) ?? []) listener(message.params)
        return
      }
      const pending = this.pending.get(message.id)
      if (!pending) return
      this.pending.delete(message.id)
      if (message.error) pending.reject(new Error(message.error.message))
      else pending.resolve(message.result)
    })
    return this
  }

  on(method, listener) {
    const listeners = this.listeners.get(method) ?? []
    listeners.push(listener)
    this.listeners.set(method, listeners)
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
      throw new Error(
        result.exceptionDetails.exception?.description ?? result.exceptionDetails.text,
      )
    }
    return result.result.value
  }

  close() {
    this.socket.close()
  }
}

function isSet(ids) {
  const cards = ids.map((id) => [
    Math.floor(id / 27) % 3,
    Math.floor(id / 9) % 3,
    Math.floor(id / 3) % 3,
    id % 3,
  ])
  return [0, 1, 2, 3].every((attribute) =>
    (cards[0][attribute] + cards[1][attribute] + cards[2][attribute]) % 3 === 0)
}

function enumerateSets(ids) {
  const triples = []
  for (let first = 0; first < ids.length - 2; first++) {
    for (let second = first + 1; second < ids.length - 1; second++) {
      for (let third = second + 1; third < ids.length; third++) {
        if (isSet([ids[first], ids[second], ids[third]])) triples.push([first, second, third])
      }
    }
  }
  return triples
}

async function readCurriculum(path) {
  try {
    return (await readFile(path, 'utf8'))
      .split('\n')
      .filter(Boolean)
      .map((line, index) => {
        try {
          return JSON.parse(line)
        } catch (error) {
          throw new Error(`Invalid curriculum JSONL at line ${index + 1}: ${error}`)
        }
      })
  } catch (error) {
    if (error?.code === 'ENOENT') return []
    throw error
  }
}

async function scanPanel(panel, label) {
  await panel.evaluate("document.querySelector('#status').textContent = 'test pending'; document.querySelector('#scan').click()")
  const status = await waitFor(async () => {
    const value = await panel.evaluate("document.querySelector('#status').textContent")
    if (value.startsWith('Scored') || value.startsWith('Abstained:') || value.startsWith('No SET predicted;')) return value
    if (
      value !== 'test pending'
      && !value.startsWith('Capturing ')
      && !value.startsWith('Running local policy ')
    ) throw new Error(value)
    return undefined
  }, label, 20_000)
  const choiceText = await panel.evaluate("document.querySelector('#choice').textContent")
  const positions = choiceText === 'No SET'
    ? []
    : choiceText.split('·').map((value) => Number(value.trim()) - 1)
  const playEnabled = await panel.evaluate("!document.querySelector('#play').disabled")
  const confidence = await panel.evaluate("document.querySelector('#confidence').textContent")
  const diagnostics = await panel.evaluate('globalThis.__setVisionLastDiagnostics')
  if (!diagnostics || !Number.isFinite(diagnostics.score)) {
    throw new Error('extension did not expose finite headless diagnostics')
  }
  return { status, choiceText, positions, playEnabled, confidence, diagnostics }
}

const processes = []
const browserDiagnostics = []
let temporaryRoot
try {
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
  temporaryRoot = await mkdtemp(join(tmpdir(), 'set-vision-extension-test.'))
  const extensionDir = join(temporaryRoot, 'extension')
  const profileDir = join(temporaryRoot, 'profile')
  await cp(join(projectRoot, 'extension'), extensionDir, { recursive: true })
  const manifestPath = join(extensionDir, 'manifest.json')
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
  // Headless CDP clicks do not grant activeTab's browser-gesture capability.
  // Broaden only this throwaway manifest so captureVisibleTab can exercise the
  // same screenshot path; the production extension remains activeTab-only.
  manifest.host_permissions = ['<all_urls>']
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)

  const preview = spawn('npm', [
    'run', 'preview', '--', '--host', '127.0.0.1', '--port', String(previewPort), '--strictPort',
  ], { cwd: projectRoot, stdio: ['ignore', 'ignore', 'pipe'] })
  processes.push(preview)
  await waitFor(async () => (await fetch(`http://127.0.0.1:${previewPort}/`)).ok, 'Vite preview')

  const browser = spawn(chrome, [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profileDir}`,
    `--disable-extensions-except=${extensionDir}`,
    `--load-extension=${extensionDir}`,
    '--window-size=1600,1000',
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] })
  browser.stderr.on('data', (chunk) => browserDiagnostics.push(String(chunk)))
  processes.push(browser)

  const extensionTarget = await waitFor(async () => {
    const response = await fetch(`http://127.0.0.1:${debugPort}/json/list`)
    const list = await response.json()
    for (const target of list.filter((candidate) => candidate.url.endsWith('/background.js'))) {
      const candidate = await new Cdp(target.webSocketDebuggerUrl).open()
      await candidate.send('Runtime.enable')
      const name = await candidate.evaluate('chrome.runtime.getManifest().name')
      candidate.close()
      if (name === manifest.name) return target
    }
    return undefined
  }, 'vision extension service worker')
  const extensionId = new URL(extensionTarget.url).host
  const createTarget = async (url) => {
    const response = await fetch(
      `http://127.0.0.1:${debugPort}/json/new?${encodeURIComponent(url)}`,
      { method: 'PUT' },
    )
    return response.json()
  }
  const gameTarget = await createTarget(`http://127.0.0.1:${previewPort}/`)
  const worker = await new Cdp(extensionTarget.webSocketDebuggerUrl).open()
  await worker.send('Runtime.enable')
  const panelResource = await worker.evaluate(`fetch(chrome.runtime.getURL('sidepanel.html'))
    .then(async (response) => ({ ok: response.ok, status: response.status, text: await response.text() }))
    .catch((error) => ({ error: String(error) }))`)
  if (!panelResource?.ok) {
    const loadedManifest = await worker.evaluate('chrome.runtime.getManifest()')
    throw new Error(
      `Extension page resource is unavailable for ${extensionTarget.url} ${JSON.stringify(loadedManifest)}: ${JSON.stringify(panelResource)}`,
    )
  }
  const createdPanelTab = await worker.evaluate(`chrome.tabs.create({
    url: chrome.runtime.getURL('sidepanel.html'),
    active: false,
  })`)
  if (!createdPanelTab?.id) {
    throw new Error(`Extension page tab was not created: ${JSON.stringify(createdPanelTab)}`)
  }
  const panelTarget = await waitFor(async () => {
    const response = await fetch(`http://127.0.0.1:${debugPort}/json/list`)
    const list = await response.json()
    return list.find((target) => target.url === `chrome-extension://${extensionId}/sidepanel.html`)
  }, 'extension page')
  worker.close()
  const game = await new Cdp(gameTarget.webSocketDebuggerUrl).open()
  const panel = await new Cdp(panelTarget.webSocketDebuggerUrl).open()
  await game.send('Page.enable')
  await panel.send('Page.enable')
  await panel.send('Runtime.enable')
  try {
    await waitFor(async () => {
      const value = await panel.evaluate("document.querySelector('#status')?.textContent")
      if (value?.includes('ready')) return value
      if (value && value !== 'Loading local Rust model…') {
        throw new Error(`extension status: ${value}`)
      }
      return undefined
    }, 'FrankenTorch extension initialization')
  } catch (error) {
    const href = await panel.evaluate('location.href')
    const probe = await panel.evaluate(`(async () => {
      try {
        const module = await import('./set_vision_runtime.js')
        const runtime = await module.default({
          module_or_path: chrome.runtime.getURL('set_vision_runtime_bg.wasm'),
        })
        return { ready: runtime.model_ready() }
      } catch (failure) {
        return { error: String(failure), stack: failure?.stack }
      }
    })()`)
    const stderr = browserDiagnostics.join('').split('\n').slice(-20).join('\n')
    throw new Error(
      `${error}; created tab: ${JSON.stringify(createdPanelTab)}; panel URL: ${href}; direct runtime probe: ${JSON.stringify(probe)}; browser stderr: ${stderr}`,
    )
  }

  const gameTab = await panel.evaluate(
    `chrome.tabs.query({}).then((tabs) => tabs.find((tab) => tab.url?.startsWith(${JSON.stringify(`http://127.0.0.1:${previewPort}/`)})))`,
  )
  if (!gameTab?.id) throw new Error('Could not resolve the responsive-board tab')

  // Chrome's headless captureVisibleTab call can remain pending. Keep the
  // extension's real screenshot/crop/WASM path, but bridge that one UI-bound
  // call through a DevTools binding to a live capture of the current game tab.
  // Every post-click capture therefore observes the actual page state.
  const captureBinding = '__setVisionRequestCapture'
  await panel.send('Runtime.addBinding', { name: captureBinding })
  panel.on('Runtime.bindingCalled', ({ name, payload }) => {
    if (name !== captureBinding) return
    if (process.env.VISION_CAPTURE_DEBUG === '1') console.error(`capture request ${payload}`)
    void (async () => {
      try {
        const screenshot = await game.send('Page.captureScreenshot', {
          format: 'png',
          captureBeyondViewport: false,
          fromSurface: true,
        })
        await panel.evaluate(
          `globalThis.__setVisionDeliverCapture(${JSON.stringify(payload)}, ${JSON.stringify(`data:image/png;base64,${screenshot.data}`)})`,
        )
        if (process.env.VISION_CAPTURE_DEBUG === '1') console.error(`capture delivered ${payload}`)
      } catch (error) {
        if (process.env.VISION_CAPTURE_DEBUG === '1') console.error(`capture failed ${payload}: ${error}`)
        await panel.evaluate(
          `globalThis.__setVisionDeliverCapture(${JSON.stringify(payload)}, undefined, ${JSON.stringify(String(error))})`,
        )
      }
    })()
  })
  await panel.evaluate(`(() => {
    let nextCapture = 1
    const pendingCaptures = new Map()
    globalThis.__setVisionDeliverCapture = (id, dataUrl, error) => {
      const pending = pendingCaptures.get(id)
      if (!pending) return
      pendingCaptures.delete(id)
      clearTimeout(pending.timeout)
      if (error) pending.reject(new Error(error))
      else pending.resolve(dataUrl)
    }
    globalThis.__setVisionHeadlessIo = {
      collectEvidence: ${Boolean(options.evidencePath)},
      multiView: ${options.multiView},
      captureVisibleTab: async function () {
        const id = String(nextCapture++)
        const result = new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            pendingCaptures.delete(id)
            reject(new Error('headless screenshot capture timed out after 15 seconds'))
          }, 15_000)
          pendingCaptures.set(id, { resolve, reject, timeout })
        })
        globalThis.${captureBinding}(id)
        return result
      },
      activeTab: async () => (${JSON.stringify(gameTab)}),
    }
  })()`)

  async function renderCase(testCase) {
    const [width, height] = testCase.viewport
    await game.send('Emulation.setDeviceMetricsOverride', {
      width, height, deviceScaleFactor: testCase.dpr, mobile: false,
    })
    const query = new URLSearchParams({
      'vision-board': testCase.theme,
      'board-kind': testCase.kind,
      'board-seed': String(testCase.boardSeed),
      'vision-seed': String(testCase.visualSeed),
      'vision-capture': testCase.profile,
    })
    await game.send('Page.navigate', {
      url: `http://127.0.0.1:${previewPort}/?${query}`,
    })
    await waitFor(
      () => game.evaluate("document.documentElement.dataset.visionBoardReady === 'true'"),
      `${testCase.theme} ${testCase.kind} responsive board`,
    )
    const cardRect = await game.evaluate(`(() => {
      const rect = document.querySelector('.card-btn').getBoundingClientRect()
      return { width: rect.width, height: rect.height }
    })()`)
    await fetch(`http://127.0.0.1:${debugPort}/json/activate/${panelTarget.id}`)
    const prediction = await scanPanel(panel, `${testCase.theme} ${width}x${height} scan`)
    // Oracle access begins only after the frozen pixel-only prediction.
    const boardIds = await game.evaluate(
      `Array.from(document.querySelectorAll('.card-btn'), (card) => Number(card.dataset.cardId))`,
    )
    const sets = enumerateSets(boardIds)
    const chosenIds = prediction.positions.map((position) => boardIds[position])
    const chosenIsSet = prediction.positions.length === 3
      && prediction.positions.every((position) => Number.isInteger(position) && position >= 0 && position < boardIds.length)
      && isSet(chosenIds)
    const predictedNoSet = prediction.choiceText === 'No SET'
    const actionCorrect = predictedNoSet ? sets.length === 0 : chosenIsSet
    return { ...testCase, ...prediction, cardRect, boardIds, setCount: sets.length, chosenIds, chosenIsSet, predictedNoSet, actionCorrect }
  }

  const screenshotProbe = await panel.evaluate(`Promise.race([
    (async () => {
      const dataUrl = await globalThis.__setVisionHeadlessIo.captureVisibleTab()
      return { ok: typeof dataUrl === 'string' && dataUrl.startsWith('data:image/png;base64,'), bytes: dataUrl.length }
    })(),
    new Promise((resolve) => setTimeout(() => resolve({ ok: false, timeout: true }), 5_000)),
  ])`)
  if (!screenshotProbe?.ok) {
    throw new Error(`Headless screenshot adapter failed: ${JSON.stringify(screenshotProbe)}`)
  }

  if (options.stream) {
    const curriculum = options.audit ? [] : await readCurriculum(options.curriculumPath)
    const curriculumByKind = {
      'has-set': curriculum.filter((failure) => failure.kind === 'has-set'),
      'no-set': curriculum.filter((failure) => failure.kind === 'no-set'),
    }
    if (!options.audit) await mkdir(dirname(options.curriculumPath), { recursive: true })
    if (options.evidencePath) {
      await mkdir(dirname(options.evidencePath), { recursive: true })
      await writeFile(options.evidencePath, '', { flag: 'wx' })
    }
    console.log(JSON.stringify({
      event: 'stream_start',
      mode: options.audit ? 'audit' : 'mine',
      seed: options.seed,
      start: options.start,
      boards: Number.isFinite(options.boards) ? options.boards : 'infinite',
      priorCurriculumFailures: curriculum.length,
      curriculum: options.audit ? null : options.curriculumPath,
      focusRate: options.audit ? 0 : options.focusRate,
    }))

    let stop = false
    process.once('SIGINT', () => { stop = true })
    const stats = {
      boards: 0,
      correct: 0,
      failures: 0,
      accepted: 0,
      acceptedCorrect: 0,
      freshCases: 0,
      curriculumCases: 0,
      exactReplayCases: 0,
      hasSet: { boards: 0, correct: 0 },
      noSet: { boards: 0, correct: 0 },
    }
    const summary = () => ({
      event: 'stream_summary',
      mode: options.audit ? 'audit' : 'mine',
      seed: options.seed,
      start: options.start,
      boards: stats.boards,
      rawAccuracy: stats.boards === 0 ? null : stats.correct / stats.boards,
      failures: stats.failures,
      hasSetAccuracy: stats.hasSet.boards === 0 ? null : stats.hasSet.correct / stats.hasSet.boards,
      noSetAccuracy: stats.noSet.boards === 0 ? null : stats.noSet.correct / stats.noSet.boards,
      guardedCoverage: stats.boards === 0 ? null : stats.accepted / stats.boards,
      guardedAccuracy: stats.accepted === 0 ? null : stats.acceptedCorrect / stats.accepted,
      freshCases: stats.freshCases,
      curriculumCases: stats.curriculumCases,
      exactReplayCases: stats.exactReplayCases,
      curriculumFailures: curriculum.length,
    })

    for (let offset = 0; offset < options.boards && !stop; offset++) {
      const iteration = options.start + offset
      const entropy = visionCaseEntropy(options.seed, iteration)
      const freshCase = freshVisionCase(options.seed, iteration)
      const kind = freshCase.kind
      const sameKind = curriculumByKind[kind]
      const focused = !options.audit
        && sameKind.length > 0
        && unitRandom(entropy ^ 0xa5a5a5a5) < options.focusRate
      const parent = focused ? sameKind[mix32(entropy ^ 0x51ed270b) % sameKind.length] : undefined
      const exactReplay = Boolean(parent && mix32(entropy ^ 0x94d049bb) % 4 === 0)
      const widthJitter = (mix32(entropy ^ 0x17a1465) % 81) - 40
      const heightJitter = (mix32(entropy ^ 0x2c1b3c6d) % 101) - 50
      const testCase = {
        ...freshCase,
        source: exactReplay ? 'replay' : parent ? 'curriculum' : 'fresh',
        parentIteration: parent?.iteration,
        parentRootSeed: parent?.rootSeed,
        kind,
        theme: parent?.theme ?? freshCase.theme,
        profile: parent?.profile ?? freshCase.profile,
        viewport: exactReplay
          ? parent.viewport
          : parent
          ? [Math.max(230, Math.min(1100, parent.viewport[0] + widthJitter)), Math.max(560, Math.min(1200, parent.viewport[1] + heightJitter))]
          : freshCase.viewport,
        dpr: parent?.dpr ?? freshCase.dpr,
        boardSeed: exactReplay || (parent && (mix32(entropy ^ 0x37_79b9) & 1) === 0)
          ? parent.boardSeed
          : freshCase.boardSeed,
        visualSeed: exactReplay ? parent.visualSeed : freshCase.visualSeed,
      }
      const result = await renderCase(testCase)
      stats.boards++
      stats.correct += Number(result.actionCorrect)
      stats.failures += Number(!result.actionCorrect)
      stats.accepted += Number(result.playEnabled)
      stats.acceptedCorrect += Number(result.playEnabled && result.actionCorrect)
      stats.freshCases += Number(result.source === 'fresh')
      stats.curriculumCases += Number(result.source !== 'fresh')
      stats.exactReplayCases += Number(result.source === 'replay')
      const kindStats = kind === 'has-set' ? stats.hasSet : stats.noSet
      kindStats.boards++
      kindStats.correct += Number(result.actionCorrect)

      if (options.evidencePath) {
        const expectedAttributeProbabilities = result.boardIds.length * 4 * 3
        const expectedTripleScores = result.boardIds.length * (result.boardIds.length - 1) * (result.boardIds.length - 2) / 6
        if (result.diagnostics.attributeProbabilities?.length !== expectedAttributeProbabilities
          || result.diagnostics.tripleScores?.length !== expectedTripleScores
          || result.diagnostics.attributeProbabilities.some((value) => !Number.isFinite(value))
          || result.diagnostics.tripleScores.some((value) => !Number.isFinite(value))) {
          throw new Error(`iteration ${iteration}: incomplete neural evidence arrays`)
        }
        const strongestIds = result.diagnostics.choice.map((position) => result.boardIds[position])
        await appendFile(options.evidencePath, `${JSON.stringify({
          version: 1,
          rootSeed: options.seed,
          iteration,
          source: result.source,
          kind,
          theme: result.theme,
          profile: result.profile,
          viewport: result.viewport,
          dpr: result.dpr,
          boardSeed: result.boardSeed,
          visualSeed: result.visualSeed,
          boardIds: result.boardIds,
          setCount: result.setCount,
          strongestPositions: result.diagnostics.choice,
          strongestIds,
          strongestIsSet: isSet(strongestIds),
          actionIsSet: result.diagnostics.actionIsSet,
          score: result.diagnostics.score,
          scoreMargin: result.diagnostics.scoreMargin,
          attributeEntropy: result.diagnostics.attributeEntropy,
          attributeMargin: result.diagnostics.attributeMargin,
          scoreMean: result.diagnostics.scoreMean,
          scoreStddev: result.diagnostics.scoreStddev,
          scoreLogsumexp: result.diagnostics.scoreLogsumexp,
          positiveScoreFraction: result.diagnostics.positiveScoreFraction,
          attributeProbabilities: result.diagnostics.attributeProbabilities,
          tripleScores: result.diagnostics.tripleScores,
        })}\n`)
      }

      if (!result.actionCorrect) {
        const failure = {
          version: 1,
          rootSeed: options.seed,
          iteration,
          source: result.source,
          parentIteration: result.parentIteration,
          parentRootSeed: result.parentRootSeed,
          kind,
          theme: result.theme,
          profile: result.profile,
          viewport: result.viewport,
          dpr: result.dpr,
          cardWidth: result.cardRect.width,
          cardHeight: result.cardRect.height,
          boardSeed: result.boardSeed,
          visualSeed: result.visualSeed,
          boardIds: result.boardIds,
          setCount: result.setCount,
          action: result.choiceText,
          chosenIds: result.chosenIds,
          confidence: result.confidence,
        }
        console.log(JSON.stringify({ event: 'failure', ...failure }))
        if (!options.audit) {
          curriculum.push(failure)
          curriculumByKind[kind].push(failure)
          await appendFile(options.curriculumPath, `${JSON.stringify(failure)}\n`)
        }
      }
      if (stats.boards % options.reportEvery === 0) console.log(JSON.stringify(summary()))
    }
    console.log(JSON.stringify(summary()))
    if (options.minRawAccuracy !== undefined && stats.correct / stats.boards < options.minRawAccuracy) {
      throw new Error(`stream raw accuracy ${stats.correct / stats.boards} is below ${options.minRawAccuracy}`)
    }
  } else {
    const cases = []
    for (const viewport of [[390, 800], [600, 900], [900, 1000]]) {
      for (const theme of ['classic', 'nocturne', 'atelier', 'signal']) {
        cases.push({ viewport, theme, kind: 'has-set', profile: 'clean', dpr: 1 })
      }
    }
    let modelFailures = 0
    let actionFailures = 0
    let acceptedActions = 0
    for (const [index, testCase] of cases.entries()) {
      const result = await renderCase({
        ...testCase,
        boardSeed: 100 + index,
        visualSeed: 9000 + index,
      })
      modelFailures += Number(!result.actionCorrect)
      acceptedActions += Number(result.playEnabled)
      if (result.playEnabled) await panel.evaluate("document.querySelector('#play').click(); true")
      const clicked = result.playEnabled ? await waitFor(async () => {
        const selected = await game.evaluate(`Array.from(document.querySelectorAll('.card-btn'))
          .map((card, cardIndex) => card.getAttribute('aria-pressed') === 'true' ? cardIndex : -1)
          .filter((cardIndex) => cardIndex >= 0)`)
        return selected.length === 3 ? selected : undefined
      }, `${testCase.theme} click action`) : []
      const actionValid = !result.playEnabled || JSON.stringify(clicked) === JSON.stringify(result.positions)
      const transitionVerified = result.playEnabled ? await waitFor(async () => {
        const value = await panel.evaluate("document.querySelector('#status').textContent")
        if (value === 'Board change verified from pixels') return true
        if (value.startsWith('Auto-play stopped:') || value.includes('did not change')) throw new Error(value)
        return undefined
      }, `${testCase.theme} board transition`, 20_000) : false
      actionFailures += Number(result.playEnabled && (!actionValid || !transitionVerified))
      const casePassed = result.actionCorrect && actionValid && (!result.playEnabled || transitionVerified)
      console.log(`${casePassed ? 'PASS' : 'FAIL'} ${testCase.theme.padEnd(9)} viewport=${testCase.viewport.join('x')} card=${result.cardRect.width.toFixed(1)}x${result.cardRect.height.toFixed(1)} ids=${result.chosenIds.join(',')} ${result.status}`)
    }
    if (modelFailures > 0 || actionFailures > 0 || acceptedActions === 0) {
      throw new Error(
        `${modelFailures}/${cases.length} invalid selections; ${actionFailures}/${acceptedActions} accepted action mismatches; ${acceptedActions}/${cases.length} actions accepted`,
      )
    }
    console.log(`extension screenshot path: ${cases.length}/${cases.length} valid proposals; ${acceptedActions}/${cases.length} guarded actions clicked and live transition-verified`)
  }
  game.close()
  panel.close()
} finally {
  for (const process of processes.reverse()) process.kill('SIGTERM')
  if (temporaryRoot) {
    await rm(temporaryRoot, { recursive: true, force: true, maxRetries: 8, retryDelay: 100 })
  }
}
