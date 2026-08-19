import { useEffect, useMemo, useState } from 'react'
import { generateModeB, mulberry32 } from '../game/board'
import { cardId } from '../game/cards'
import { Card } from '../ui/Card'
import { CardThemeProvider } from '../ui/CardThemeContext'
import { isCardTheme, type CardTheme } from '../ui/cardThemes'
import { variationFor } from './proceduralVariation'

function integerParameter(name: string, fallback: number): number {
  const value = Number(new URLSearchParams(window.location.search).get(name) ?? fallback)
  return Number.isSafeInteger(value) ? value >>> 0 : fallback
}

function requestedTheme(): CardTheme {
  const value = new URLSearchParams(window.location.search).get('vision-board')
  return isCardTheme(value) ? value : 'classic'
}

/**
 * Production-card, production-layout surface for end-to-end extension tests.
 * Unlike the dense atlas, this deliberately exercises responsive raster sizes.
 */
export function ResponsiveVisionBoard() {
  const theme = requestedTheme()
  const boardSeed = integerParameter('board-seed', 1)
  const visualSeed = integerParameter('vision-seed', 0)
  const [round, setRound] = useState(0)
  const board = useMemo(() => generateModeB(mulberry32(boardSeed + round)), [boardSeed, round])
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
    document.documentElement.dataset.visionBoardReady = 'true'
    return () => {
      delete document.documentElement.dataset.visionBoardReady
    }
  }, [])

  return (
    <CardThemeProvider theme={theme}>
      <main className="game-shell" data-vision-board={theme}>
        <div className={`board${selected.length > 0 ? ' has-selection' : ''}`}>
          {board.cards.map((card, index) => (
            <button
              type="button"
              className={`card-btn${selected.includes(index) ? ' is-selected' : ''}`}
              data-card-id={cardId(card)}
              key={cardId(card)}
              aria-pressed={selected.includes(index)}
              onClick={() => selectCard(index)}
            >
              <Card
                card={card}
                theme={theme}
                visualVariation={variationFor(visualSeed, cardId(card))}
              />
            </button>
          ))}
        </div>
      </main>
    </CardThemeProvider>
  )
}
