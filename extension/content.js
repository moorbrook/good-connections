if (!globalThis.__goodConnectionsVisionInstalled) {
  globalThis.__goodConnectionsVisionInstalled = true

  const visibleCards = (selector) => [...document.querySelectorAll(selector)].filter((element) => {
    const rect = element.getBoundingClientRect()
    const style = getComputedStyle(element)
    return rect.width > 8 && rect.height > 8 && style.visibility !== 'hidden' && style.display !== 'none'
  })

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    const selector = typeof message.selector === 'string' ? message.selector : '.card-btn'
    if (message.type === 'set-vision:rects') {
      const rects = visibleCards(selector).map((element) => {
        const rect = element.getBoundingClientRect()
        return { left: rect.left, top: rect.top, width: rect.width, height: rect.height }
      })
      sendResponse({ rects, viewport: { width: innerWidth, height: innerHeight } })
      return false
    }
    if (message.type === 'set-vision:highlight') {
      const cards = visibleCards(selector)
      for (const card of cards) card.style.removeProperty('box-shadow')
      for (const index of message.indices ?? []) {
        cards[index]?.style.setProperty('box-shadow', '0 0 0 6px #23d5ab, 0 0 24px #23d5ab')
      }
      sendResponse({ ok: true })
      return false
    }
    if (message.type === 'set-vision:click') {
      const cards = visibleCards(selector)
      ;(async () => {
        for (const index of message.indices ?? []) {
          cards[index]?.click()
          await new Promise((resolve) => setTimeout(resolve, 130))
        }
        sendResponse({ ok: true })
      })()
      return true
    }
    return false
  })
}
