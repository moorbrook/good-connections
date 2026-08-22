import { useEffect, useMemo, useState } from 'react'
import { generateModeB, generateNoSetBoard, mulberry32 } from '../game/board'
import { cardId } from '../game/cards'
import { Card } from '../ui/Card'
import { CardThemeProvider } from '../ui/CardThemeContext'
import { isCardTheme, type CardTheme } from '../ui/cardThemes'
import { variationFor } from './proceduralVariation'
import { captureProfileFrom, captureStyles } from './captureProfile'

function integerParameter(name: string, fallback: number): number {
  const value = Number(new URLSearchParams(window.location.search).get(name) ?? fallback)
  return Number.isSafeInteger(value) ? value >>> 0 : fallback
}

function requestedTheme(): CardTheme {
  const value = new URLSearchParams(window.location.search).get('vision-board')
  return isCardTheme(value) ? value : 'classic'
}

function requestedBoardKind(): 'has-set' | 'no-set' {
  return new URLSearchParams(window.location.search).get('board-kind') === 'no-set'
    ? 'no-set'
    : 'has-set'
}

interface VisionBoardRequest {
  theme: CardTheme
  boardSeed: number
  visualSeed: number
  boardKind: 'has-set' | 'no-set'
  captureProfile: ReturnType<typeof captureProfileFrom>
}

function requestedBoard(): VisionBoardRequest {
  const parameters = new URLSearchParams(window.location.search)
  return {
    theme: requestedTheme(),
    boardSeed: integerParameter('board-seed', 1),
    visualSeed: integerParameter('vision-seed', 0),
    boardKind: requestedBoardKind(),
    captureProfile: captureProfileFrom(parameters.get('vision-capture')),
  }
}

/**
 * Production-card, production-layout surface for end-to-end extension tests.
 * Unlike the dense atlas, this deliberately exercises responsive raster sizes.
 */
export function ResponsiveVisionBoard() {
  const [request, setRequest] = useState(requestedBoard)
  const { theme, boardSeed, visualSeed, boardKind, captureProfile } = request
  const styles = captureStyles(captureProfile)
  const [round, setRound] = useState(0)
  const board = useMemo(
    () => boardKind === 'no-set'
      ? generateNoSetBoard(mulberry32(boardSeed + round))
      : generateModeB(mulberry32(boardSeed + round)),
    [boardKind, boardSeed, round],
  )
  const [selected, setSelected] = useState<number[]>([])

  const selectCard = (index: number) => {
    setSelected((current) => {
      if (current.includes(index)) return current
      const next = [...current, index]
      if (next.length === 3) {
        const sorted = [...next].sort((left, right) => left - right)
        const valid = board.sets.some((triple) => triple.every((value, position) => value === sorted[position]))
        if (valid) {
          window.setTimeout(() => {
            setRound((value) => value + 1)
            setSelected([])
          }, 420)
        } else {
          window.setTimeout(() => setSelected([]), 420)
        }
      }
      return next
    })
  }

  useEffect(() => {
    const update = (event: Event) => {
      const detail = (event as CustomEvent<Partial<VisionBoardRequest>>).detail
      const theme = detail?.theme
      const boardKind = detail?.boardKind
      if (!isCardTheme(theme) || (boardKind !== 'has-set' && boardKind !== 'no-set')) return
      document.documentElement.dataset.visionBoardReady = 'false'
      setSelected([])
      setRound(0)
      setRequest({
        theme,
        boardKind,
        boardSeed: Number(detail.boardSeed) >>> 0,
        visualSeed: Number(detail.visualSeed) >>> 0,
        captureProfile: captureProfileFrom(detail.captureProfile ?? null),
      })
    }
    document.addEventListener('set-vision-render-case', update)
    return () => document.removeEventListener('set-vision-render-case', update)
  }, [])

  useEffect(() => {
    document.documentElement.dataset.visionBoardReady = 'true'
    return () => {
      delete document.documentElement.dataset.visionBoardReady
    }
  }, [boardKind, boardSeed, captureProfile, theme, visualSeed])

  return (
    <CardThemeProvider theme={theme}>
      <main className="game-shell" data-vision-board={theme} data-board-kind={boardKind}>
        <div className={`board${selected.length > 0 ? ' has-selection' : ''}`}>
          {board.cards.map((card, index) => (
            <button
              type="button"
              className={`card-btn${selected.includes(index) ? ' is-selected' : ''}`}
              data-card-id={cardId(card)}
              key={cardId(card)}
              aria-pressed={selected.includes(index)}
              onClick={() => selectCard(index)}
              style={styles.outer}
            >
              <div style={{ width: '100%', height: '100%', ...styles.inner }}>
                <Card
                  card={card}
                  theme={theme}
                  visualVariation={variationFor(visualSeed, cardId(card))}
                />
              </div>
              {captureProfile === 'occlusion' && (
                <div aria-hidden="true" data-vision-occlusion="true" style={styles.occlusion} />
              )}
            </button>
          ))}
        </div>
      </main>
    </CardThemeProvider>
  )
}
