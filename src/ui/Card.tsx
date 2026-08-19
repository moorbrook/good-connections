import { useId } from 'react'
import { COLOURS, SHAPES, FILLS, type Card as SetCard } from '../game/cards'
import { useCardTheme } from './CardThemeContext'
import { CARD_THEME_BY_ID, type CardTheme } from './cardThemes'

/**
 * The one component that draws all 81 logical cards in every visual theme.
 * Count, colour, shape, and fill remain the only gameplay inputs; frames and
 * material treatments are vector decoration and never alter card identity.
 */

const VB_W = 100
const VB_H = 250
const SYMBOL_W = 40
const SYMBOL_H = 100
const CARD_W = 190
const CARD_H = 140
const STRIPE_STROKE = 6

const CLASSIC_DIAMOND_PATH = `M50 4 L96 ${VB_H / 2} L50 ${VB_H - 4} L4 ${VB_H / 2} Z`
const PREMIUM_DIAMOND_PATH = `M50 5 L96 ${VB_H / 2} L50 ${VB_H - 5} L4 ${VB_H / 2} Z`
const CLASSIC_SQUIGGLE_PATH =
  'M50 8 C82 14, 78 52, 60 86 C44 120, 88 140, 88 172 ' +
  'C88 210, 72 238, 50 242 C18 236, 22 198, 40 164 ' +
  'C56 130, 12 110, 12 78 C12 40, 28 12, 50 8 Z'
const PREMIUM_SQUIGGLE_PATH =
  'M49 7 C81 12, 80 50, 61 85 C43 118, 88 140, 88 173 ' +
  'C88 211, 72 238, 51 243 C19 238, 20 200, 39 165 ' +
  'C57 132, 12 110, 12 77 C12 39, 28 12, 49 7 Z'

interface ShapeProps {
  theme: CardTheme
  shape: number
  fill: string
  stroke: string
  strokeWidth: number
  opacity?: number
  transform?: string
}

function Shape({
  theme,
  shape,
  fill,
  stroke,
  strokeWidth,
  opacity = 1,
  transform = '',
}: ShapeProps) {
  const common = {
    fill,
    stroke,
    strokeWidth,
    opacity,
    transform,
    strokeLinejoin: 'round' as const,
  }
  const diamondPath = theme === 'classic' ? CLASSIC_DIAMOND_PATH : PREMIUM_DIAMOND_PATH
  const squigglePath = theme === 'classic' ? CLASSIC_SQUIGGLE_PATH : PREMIUM_SQUIGGLE_PATH
  return shape === 2 ? (
    <rect x={8} y={4} width={VB_W - 16} height={VB_H - 8} rx={(VB_W - 16) / 2} {...common} />
  ) : (
    <path d={shape === 0 ? diamondPath : squigglePath} {...common} />
  )
}

function StripePattern({
  id,
  colour,
  theme,
  periodScale = 1,
}: {
  id: string
  colour: string
  theme: CardTheme
  periodScale?: number | undefined
}) {
  const period = (theme === 'classic' ? 24 : 21) * periodScale
  return (
    <pattern id={id} patternUnits="userSpaceOnUse" width={VB_W} height={period}>
      <line
        x1={0}
        y1={period / 2}
        x2={VB_W}
        y2={period / 2}
        stroke={colour}
        strokeWidth={STRIPE_STROKE}
      />
    </pattern>
  )
}

function ThemeDefinitions({ theme, ids }: { theme: CardTheme; ids: FrameIds }) {
  if (theme === 'nocturne') {
    return (
      <>
        <linearGradient id={ids.face} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#13223d" />
          <stop offset="0.55" stopColor="#091426" />
          <stop offset="1" stopColor="#050b17" />
        </linearGradient>
        <linearGradient id={ids.edge} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#75d8ff" />
          <stop offset="0.35" stopColor="#d6ad69" />
          <stop offset="0.7" stopColor="#a66cff" />
          <stop offset="1" stopColor="#75d8ff" />
        </linearGradient>
      </>
    )
  }
  if (theme === 'atelier') {
    return (
      <>
        <linearGradient id={ids.face} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#fffaf0" />
          <stop offset="1" stopColor="#f1eadb" />
        </linearGradient>
        <pattern id={ids.paper} width="7" height="7" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="2" r="0.35" fill="#a59984" opacity="0.18" />
          <circle cx="5" cy="6" r="0.25" fill="#fff" opacity="0.75" />
        </pattern>
      </>
    )
  }
  if (theme === 'signal') {
    return (
      <linearGradient id={ids.face} x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#fff" />
        <stop offset="0.62" stopColor="#f8fbfb" />
        <stop offset="1" stopColor="#e8f0f1" />
      </linearGradient>
    )
  }
  return null
}

interface FrameIds {
  face: string
  edge: string
  paper: string
}

function NocturneFrame({ colour, ids }: { colour: string; ids: FrameIds }) {
  return (
    <>
      <rect x="1.5" y="1.5" width="187" height="137" rx="13" fill={`url(#${ids.face})`} stroke={`url(#${ids.edge})`} strokeWidth="2.4" />
      <rect x="6" y="6" width="178" height="128" rx="9" fill="none" stroke="#d9b979" strokeOpacity="0.55" strokeWidth="0.8" />
      <ellipse cx="95" cy="70" rx="76" ry="52" fill="none" stroke={colour} strokeOpacity="0.23" strokeWidth="0.8" />
      <ellipse cx="95" cy="70" rx="62" ry="43" fill="none" stroke="#d9b979" strokeOpacity="0.25" strokeWidth="0.6" strokeDasharray="2 3" />
      <path d="M16 70 H30 M160 70 H174" stroke="#d9b979" strokeOpacity="0.7" strokeWidth="0.8" />
      <circle cx="21" cy="70" r="2.3" fill="#e8ca8b" />
      <circle cx="169" cy="70" r="2.3" fill="#e8ca8b" />
    </>
  )
}

const ATELIER_CORNERS = [
  'M14 29 C14 19 19 14 29 14 M14 24 C21 24 24 21 24 14 M18 29 C18 22 22 18 29 18',
  'M176 29 C176 19 171 14 161 14 M176 24 C169 24 166 21 166 14 M172 29 C172 22 168 18 161 18',
  'M14 111 C14 121 19 126 29 126 M14 116 C21 116 24 119 24 126 M18 111 C18 118 22 122 29 122',
  'M176 111 C176 121 171 126 161 126 M176 116 C169 116 166 119 166 126 M172 111 C172 118 168 122 161 122',
] as const

function AtelierFrame({ colour, ids }: { colour: string; ids: FrameIds }) {
  return (
    <>
      <rect x="1.5" y="1.5" width="187" height="137" rx="12" fill={`url(#${ids.face})`} stroke="#c4bba9" strokeWidth="1.3" />
      <rect x="1.5" y="1.5" width="187" height="137" rx="12" fill={`url(#${ids.paper})`} />
      <rect x="6" y="6" width="178" height="128" rx="8" fill="none" stroke={colour} strokeOpacity="0.9" strokeWidth="1.1" />
      <rect x="9" y="9" width="172" height="122" rx="6" fill="none" stroke="#9d927e" strokeOpacity="0.38" strokeWidth="0.65" />
      <g fill="none" stroke="#b8ad98" strokeOpacity="0.38" strokeWidth="1">
        {ATELIER_CORNERS.map((path) => <path d={path} key={path} />)}
      </g>
    </>
  )
}

function SignalFrame({ colour, ids }: { colour: string; ids: FrameIds }) {
  return (
    <>
      <rect x="1.5" y="1.5" width="187" height="137" rx="13" fill={`url(#${ids.face})`} stroke="#26343b" strokeWidth="2.2" />
      <path d="M164 2 A24 24 0 0 0 188 26 M170 2 A18 18 0 0 0 188 20 M176 2 A12 12 0 0 0 188 14" fill="none" stroke={colour} strokeWidth="1.5" opacity="0.8" />
      <path d="M2 114 A24 24 0 0 1 26 138 M2 120 A18 18 0 0 1 20 138 M2 126 A12 12 0 0 1 14 138" fill="none" stroke={colour} strokeWidth="1.5" opacity="0.8" />
      <circle cx="14" cy="14" r="3" fill={colour} />
      <circle cx="176" cy="126" r="3" fill={colour} />
      <path d="M24 13 H28 M31 13 H35 M155 126 H159 M162 126 H166" stroke="#26343b" strokeWidth="2.2" />
    </>
  )
}

function ThemeFrame({ theme, colour, ids }: { theme: CardTheme; colour: string; ids: FrameIds }) {
  if (theme === 'nocturne') return <NocturneFrame colour={colour} ids={ids} />
  if (theme === 'atelier') return <AtelierFrame colour={colour} ids={ids} />
  if (theme === 'signal') return <SignalFrame colour={colour} ids={ids} />
  return null
}

interface SymbolProps {
  theme: CardTheme
  shape: number
  fill: number
  colour: string
  x: number
  patternId: string
  variation?: CardVisualVariation | undefined
}

function Symbol({ theme, shape, fill, colour, x, patternId, variation }: SymbolProps) {
  const fillValue = fill === 0 ? colour : fill === 1 ? `url(#${patternId})` : 'none'
  const strokeWidth = (theme === 'classic' ? 7 : fill === 2 ? 8 : 7) * (variation?.strokeScale ?? 1)
  const symbolScale = variation?.symbolScale ?? 1
  const width = SYMBOL_W * symbolScale
  const height = SYMBOL_H * symbolScale
  const symbolX = x + (SYMBOL_W - width) / 2
  const symbolY = (CARD_H - height) / 2 + (variation?.symbolY ?? 0)
  const rotation = variation?.symbolRotation ?? 0
  const echo = theme === 'classic' ? null : (
    <Shape
      theme={theme}
      shape={shape}
      fill={fill === 2 || theme === 'nocturne' ? 'none' : colour}
      stroke={theme === 'nocturne' || fill === 2 ? colour : 'none'}
      strokeWidth={theme === 'nocturne' ? 18 : fill === 2 ? 11 : 0}
      opacity={theme === 'nocturne' ? 0.12 : 0.13}
      transform={
        theme === 'signal' ? 'translate(8 9)' : theme === 'atelier' ? 'translate(3 4)' : ''
      }
    />
  )

  return (
    <svg x={symbolX} y={symbolY} width={width} height={height} viewBox={`0 0 ${VB_W} ${VB_H}`} overflow="visible">
      <g transform={`rotate(${rotation} ${VB_W / 2} ${VB_H / 2})`}>
        {echo}
        <Shape theme={theme} shape={shape} fill={fillValue} stroke={colour} strokeWidth={strokeWidth} />
      </g>
    </svg>
  )
}

/** Training-only visual degrees of freedom. Omit this in normal gameplay. */
export interface CardVisualVariation {
  readonly brightness: number
  readonly contrast: number
  readonly hueRotation: number
  readonly saturation: number
  readonly spacingScale: number
  readonly stripePeriodScale: number
  readonly strokeScale: number
  readonly symbolRotation: number
  readonly symbolScale: number
  readonly symbolY: number
}

export interface CardProps {
  card: SetCard
  /** Render a particular theme instead of the current app theme (used by previews). */
  theme?: CardTheme
  /** Accessible label override; defaults to a description of the four attributes. */
  label?: string
  /** Seeded renderer variation used only by the hidden vision-training surface. */
  visualVariation?: CardVisualVariation | undefined
}

export function Card({ card, theme: themeOverride, label, visualVariation }: CardProps) {
  const currentTheme = useCardTheme()
  const theme = themeOverride ?? currentTheme
  const uid = useId().replaceAll(':', '')
  const patternId = `stripe-${uid}`
  const ids: FrameIds = {
    face: `face-${uid}`,
    edge: `edge-${uid}`,
    paper: `paper-${uid}`,
  }
  const colour = CARD_THEME_BY_ID[theme].palette[card.colour]
  const count = card.count + 1
  const gap = (theme === 'classic' ? 16 : count === 3 ? 12 : 16) * (visualVariation?.spacingScale ?? 1)
  const groupW = count * SYMBOL_W + (count - 1) * gap
  const startX = (CARD_W - groupW) / 2
  const xs = Array.from({ length: count }, (_, i) => startX + i * (SYMBOL_W + gap))
  const describe = label ?? `${count} ${COLOURS[card.colour]} ${FILLS[card.fill]} ${SHAPES[card.shape]}`

  return (
    <svg
      className="set-card"
      data-card-theme={theme}
      viewBox={`0 0 ${CARD_W} ${CARD_H}`}
      role="img"
      aria-label={describe}
      xmlns="http://www.w3.org/2000/svg"
      style={visualVariation ? {
        filter: `hue-rotate(${visualVariation.hueRotation}deg) saturate(${visualVariation.saturation}) contrast(${visualVariation.contrast}) brightness(${visualVariation.brightness})`,
      } : undefined}
    >
      <defs>
        {card.fill === 1 && (
          <StripePattern
            id={patternId}
            colour={colour}
            theme={theme}
            periodScale={visualVariation?.stripePeriodScale}
          />
        )}
        <ThemeDefinitions theme={theme} ids={ids} />
      </defs>
      <ThemeFrame theme={theme} colour={colour} ids={ids} />
      {xs.map((x, i) => (
        <Symbol
          key={i}
          theme={theme}
          shape={card.shape}
          fill={card.fill}
          colour={colour}
          x={x}
          patternId={patternId}
          variation={visualVariation}
        />
      ))}
    </svg>
  )
}
