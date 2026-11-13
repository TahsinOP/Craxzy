// Tiny line-plot helper for the interactive figures. Classic script (no modules) so it
// also runs from file:// in the offline build. Colours come from the Starlight theme.
(function () {
  var NS = 'http://www.w3.org/2000/svg'
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag)
    for (var k in attrs) e.setAttribute(k, attrs[k])
    if (parent) parent.appendChild(e)
    return e
  }
  function nice(v) {
    if (!(v > 0)) return 1
    var p = Math.pow(10, Math.floor(Math.log10(v)))
    var f = v / p
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p
  }
  function fmt(v) {
    var a = Math.abs(v)
    if (a === 0) return '0'
    if (a >= 100) return v.toFixed(0)
    if (a >= 10) return v.toFixed(0)
    if (a >= 1) return v.toFixed(1).replace(/\.0$/, '')
    return v.toPrecision(1)
  }
  // CCPlot(svg, {xlabel, ylabel}) -> draw(series, opts)
  //   series: [{x: [...], y: [...], color: 'css colour', dash: '4 3' | undefined, label: '...'}]
  //   opts:   {xmax, ymax (optional), ymin (optional, default 0)}
  window.CCPlot = function (svg, cfg) {
    var W = 560, H = 280, L = 62, R = 14, Tp = 14, B = 44
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H)
    var g = el('g', {}, svg)
    return function draw(series, opts) {
      while (g.firstChild) g.removeChild(g.firstChild)
      var xmax = opts.xmax, ymin = opts.ymin || 0
      var ymax = opts.ymax
      if (ymax === undefined) {
        ymax = 0
        series.forEach(function (s) { s.y.forEach(function (v) { if (isFinite(v) && v > ymax) ymax = v }) })
        ymax = nice(ymax * 1.05)
      }
      var X = function (x) { return L + (x / xmax) * (W - L - R) }
      var Y = function (y) { return H - B - ((y - ymin) / (ymax - ymin)) * (H - B - Tp) }
      for (var i = 0; i <= 4; i++) {
        var yv = ymin + (ymax - ymin) * i / 4
        el('line', { x1: L, x2: W - R, y1: Y(yv), y2: Y(yv), stroke: 'var(--sl-color-hairline)', 'stroke-width': 1 }, g)
        el('text', { x: L - 8, y: Y(yv) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--sl-color-gray-3)' }, g).textContent = fmt(yv)
        var xv = xmax * i / 4
        el('text', { x: X(xv), y: H - B + 16, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--sl-color-gray-3)' }, g).textContent = fmt(xv)
      }
      el('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: 'var(--sl-color-gray-4)', 'stroke-width': 1.5 }, g)
      el('line', { x1: L, x2: L, y1: Tp, y2: H - B, stroke: 'var(--sl-color-gray-4)', 'stroke-width': 1.5 }, g)
      el('text', { x: (L + W - R) / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--sl-color-gray-3)' }, g).textContent = cfg.xlabel
      el('text', { x: 16, y: (Tp + H - B) / 2, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--sl-color-gray-3)',
        transform: 'rotate(-90 16 ' + (Tp + H - B) / 2 + ')' }, g).textContent = cfg.ylabel
      series.forEach(function (s, j) {
        var pts = []
        for (var k = 0; k < s.x.length; k++) {
          var yy = Math.max(ymin, Math.min(ymax, s.y[k]))
          pts.push(X(s.x[k]).toFixed(1) + ',' + Y(yy).toFixed(1))
        }
        var a = { points: pts.join(' '), fill: 'none', stroke: s.color, 'stroke-width': 2.5, 'stroke-linejoin': 'round' }
        if (s.dash) a['stroke-dasharray'] = s.dash
        el('polyline', a, g)
        // legend, top-left, one row per series
        var ly = Tp + 10 + j * 17
        var la = { x1: L + 12, x2: L + 36, y1: ly, y2: ly, stroke: s.color, 'stroke-width': 2.5 }
        if (s.dash) la['stroke-dasharray'] = s.dash
        el('line', la, g)
        el('text', { x: L + 42, y: ly + 4, 'font-size': 11.5, fill: 'var(--sl-color-text)' }, g).textContent = s.label
      })
    }
  }
})()
