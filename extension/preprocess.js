/** Area-average an RGBA screenshot crop into the RGB input contract exported by WASM. */
export function areaResampleRgb(rgba, width, height, targetWidth, targetHeight) {
  const rgb = new Array(targetWidth * targetHeight * 3)
  let target = 0
  for (let y = 0; y < targetHeight; y++) {
    const sourceYStart = Math.floor(y * height / targetHeight)
    const sourceYEnd = Math.max(sourceYStart + 1, Math.floor((y + 1) * height / targetHeight))
    for (let x = 0; x < targetWidth; x++) {
      const sourceXStart = Math.floor(x * width / targetWidth)
      const sourceXEnd = Math.max(sourceXStart + 1, Math.floor((x + 1) * width / targetWidth))
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
      rgb[target++] = totals[0] / (sampleCount * 255)
      rgb[target++] = totals[1] / (sampleCount * 255)
      rgb[target++] = totals[2] / (sampleCount * 255)
    }
  }
  return rgb
}
