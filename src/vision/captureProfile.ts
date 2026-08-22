import type { CSSProperties } from 'react'

export const CAPTURE_PROFILES = [
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
] as const

export type CaptureProfile = (typeof CAPTURE_PROFILES)[number] | 'jpeg'

export function captureProfileFrom(value: string | null): CaptureProfile {
  return value === 'jpeg' || CAPTURE_PROFILES.some((profile) => profile === value)
    ? value as CaptureProfile
    : 'clean'
}

export function captureStyles(profile: CaptureProfile): {
  outer: CSSProperties
  inner: CSSProperties
  occlusion: CSSProperties
} {
  switch (profile) {
    case 'blur':
      return { outer: { overflow: 'hidden' }, inner: { filter: 'blur(0.55px) contrast(0.92)' }, occlusion: {} }
    case 'crop':
      return {
        outer: { overflow: 'hidden' },
        inner: { transform: 'translate(3px, -2px) scale(1.055)', transformOrigin: 'center' },
        occlusion: {},
      }
    case 'shadow':
      return {
        outer: { overflow: 'hidden', background: '#c7c8cf' },
        inner: {
          boxShadow: '4px 5px 9px rgb(0 0 0 / 48%)',
          transform: 'translate(-2px, -2px) scale(0.94)',
          transformOrigin: 'center',
        },
        occlusion: {},
      }
    case 'lowcontrast':
      return {
        outer: { overflow: 'hidden' },
        inner: { filter: 'contrast(0.72) brightness(0.94) saturate(0.78)' },
        occlusion: {},
      }
    case 'defocus':
      return {
        outer: { overflow: 'hidden' },
        inner: { filter: 'blur(1.15px) contrast(0.88)' },
        occlusion: {},
      }
    case 'tilt':
      return {
        outer: { overflow: 'hidden' },
        inner: { transform: 'rotate(2.2deg) scale(1.04)', transformOrigin: 'center' },
        occlusion: {},
      }
    case 'occlusion':
      return {
        outer: { overflow: 'hidden', position: 'relative' },
        inner: {},
        occlusion: {
          position: 'absolute',
          left: 0,
          top: '42%',
          width: '100%',
          height: '22px',
          background: 'rgb(15 18 32 / 46%)',
          pointerEvents: 'none',
        },
      }
    case 'gamut':
      return {
        outer: { overflow: 'hidden' },
        inner: { filter: 'saturate(0.45) hue-rotate(24deg) contrast(1.08) brightness(1.05)' },
        occlusion: {},
      }
    case 'compound':
      return {
        outer: { overflow: 'hidden', background: '#b8bbc5' },
        inner: {
          filter: 'blur(0.75px) contrast(0.82) brightness(0.96) saturate(0.72)',
          boxShadow: '5px 6px 12px rgb(0 0 0 / 42%)',
          transform: 'translate(-2px, -1px) scale(1.065)',
          transformOrigin: 'center',
        },
        occlusion: {},
      }
    case 'jpeg':
    case 'clean':
      return { outer: {}, inner: {}, occlusion: {} }
  }
}
