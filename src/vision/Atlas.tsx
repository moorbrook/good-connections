import { useEffect, type CSSProperties } from 'react'
import { fullDeck } from '../game/cards'
import { Card } from '../ui/Card'
import { CardThemeProvider } from '../ui/CardThemeContext'
import { isCardTheme, type CardTheme } from '../ui/cardThemes'
import { variationFor } from './proceduralVariation'

/**
 * Seeded, development-only render surface used by the Rust vision trainer.
 *
 * It deliberately uses the production Card component. That keeps the pixels
 * used for training identical to the pixels the extension sees while playing.
 * The route is selected with `?vision-atlas=<theme>` in src/main.tsx.
 */

const REFERENCE_CARD_WIDTH = 190
const REFERENCE_CARD_HEIGHT = 140
const COLUMNS = 9
const GAP = 4
const PADDING = 10

type CaptureProfile = 'clean' | 'blur' | 'crop' | 'shadow' | 'lowcontrast' | 'jpeg'

export const VISION_ATLAS_GEOMETRY = {
  cardWidth: REFERENCE_CARD_WIDTH,
  cardHeight: REFERENCE_CARD_HEIGHT,
  columns: COLUMNS,
  gap: GAP,
  padding: PADDING,
} as const

function requestedTheme(): CardTheme {
  const value = new URLSearchParams(window.location.search).get('vision-atlas')
  return isCardTheme(value) ? value : 'classic'
}

function requestedSeed(): number {
  const raw = Number(new URLSearchParams(window.location.search).get('vision-seed') ?? 0)
  return Number.isSafeInteger(raw) ? raw >>> 0 : 0
}

function requestedCardWidth(): number {
  const raw = Number(new URLSearchParams(window.location.search).get('vision-card-width') ?? REFERENCE_CARD_WIDTH)
  return Number.isSafeInteger(raw) && raw >= 96 && raw <= 380 ? raw : REFERENCE_CARD_WIDTH
}

function requestedCaptureProfile(): CaptureProfile {
  const value = new URLSearchParams(window.location.search).get('vision-capture')
  return value === 'blur' || value === 'crop' || value === 'shadow'
    || value === 'lowcontrast' || value === 'jpeg'
    ? value
    : 'clean'
}

function captureStyles(profile: CaptureProfile): {
  outer: CSSProperties
  inner: CSSProperties
} {
  switch (profile) {
    case 'blur':
      return { outer: { overflow: 'hidden' }, inner: { filter: 'blur(0.55px) contrast(0.92)' } }
    case 'crop':
      return {
        outer: { overflow: 'hidden' },
        inner: { transform: 'translate(3px, -2px) scale(1.055)', transformOrigin: 'center' },
      }
    case 'shadow':
      return {
        outer: { overflow: 'hidden', background: '#c7c8cf' },
        inner: {
          boxShadow: '4px 5px 9px rgb(0 0 0 / 48%)',
          transform: 'translate(-2px, -2px) scale(0.94)',
          transformOrigin: 'center',
        },
      }
    case 'lowcontrast':
      return { outer: { overflow: 'hidden' }, inner: { filter: 'contrast(0.72) brightness(0.94) saturate(0.78)' } }
    case 'jpeg':
    case 'clean':
      return { outer: {}, inner: {} }
  }
}

export function VisionAtlas() {
  const theme = requestedTheme()
  const seed = requestedSeed()
  const cardWidth = requestedCardWidth()
  const cardHeight = Math.round(cardWidth * REFERENCE_CARD_HEIGHT / REFERENCE_CARD_WIDTH)
  const captureProfile = requestedCaptureProfile()
  const styles = captureStyles(captureProfile)

  useEffect(() => {
    document.body.classList.add('vision-atlas-page')
    document.documentElement.dataset.visionAtlasReady = 'true'
    return () => {
      document.body.classList.remove('vision-atlas-page')
      delete document.documentElement.dataset.visionAtlasReady
    }
  }, [])

  return (
    <CardThemeProvider theme={theme}>
      <main
        aria-label={`${theme} vision training atlas`}
        data-vision-atlas={theme}
        data-vision-capture={captureProfile}
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${COLUMNS}, ${cardWidth}px)`,
          gap: GAP,
          padding: PADDING,
          width: PADDING * 2 + COLUMNS * cardWidth + (COLUMNS - 1) * GAP,
          background: '#0f1220',
        }}
      >
        {fullDeck().map((card, id) => (
          <div
            data-card-id={id}
            key={id}
            style={{ width: cardWidth, height: cardHeight, ...styles.outer }}
          >
            <div style={{ width: '100%', height: '100%', ...styles.inner }}>
              <Card card={card} theme={theme} visualVariation={variationFor(seed, id)} />
            </div>
          </div>
        ))}
      </main>
    </CardThemeProvider>
  )
}
