/* Global helpers for all dashboard pages. */

/* ------------------------------------------------------------------ *
 *  Adaptive-color sentinels (trace line color)                        *
 *  We mark lines that should follow the theme color with one of       *
 *  these two hex codes. renderChart() + MutationObserver swap          *
 *  bidirectionally between dark/light.                                *
 * ------------------------------------------------------------------ */
const _ADAPTIVE_LIGHT = '#111';
const _ADAPTIVE_DARK  = '#e2e8f0';
const _ADAPTIVE_SET   = new Set([_ADAPTIVE_LIGHT.toLowerCase(), _ADAPTIVE_DARK.toLowerCase()]);
function _adaptiveColor(isDark) { return isDark ? _ADAPTIVE_DARK : _ADAPTIVE_LIGHT; }

/* ------------------------------------------------------------------ *
 *  Theme overrides: layout (for Plotly.relayout) and updatemenus      *
 * ------------------------------------------------------------------ */
function _darkLayoutOverrides() {
  return {
    paper_bgcolor: 'rgba(27,27,44,0)',
    plot_bgcolor:  'rgba(27,27,44,0)',
    'font.color':  '#E2E2ED',
    'font.family': 'Inter, ui-sans-serif, system-ui, sans-serif',
    'xaxis.gridcolor': '#2A2A40',
    'xaxis.zerolinecolor': '#2A2A40',
    'yaxis.gridcolor': '#2A2A40',
    'yaxis.zerolinecolor': '#2A2A40',
    'legend.bgcolor': 'rgba(27,27,44,0.0)',
    'legend.font.color': '#E2E2ED',
    'hoverlabel.bgcolor':   '#12121F',
    'hoverlabel.bordercolor': '#3B3B55',
    'hoverlabel.font.color':  '#EEEEF5',
  };
}
function _lightLayoutOverrides() {
  return {
    paper_bgcolor: 'rgba(255,255,255,0)',
    plot_bgcolor:  'rgba(255,255,255,0)',
    'font.color':  '#121220',
    'font.family': 'Inter, ui-sans-serif, system-ui, sans-serif',
    'xaxis.gridcolor': '#E9E9F2',
    'xaxis.zerolinecolor': '#E9E9F2',
    'yaxis.gridcolor': '#E9E9F2',
    'yaxis.zerolinecolor': '#E9E9F2',
    'legend.bgcolor': 'rgba(255,255,255,0.0)',
    'legend.font.color': '#121220',
    'hoverlabel.bgcolor':   '#ffffff',
    'hoverlabel.bordercolor': '#CBCBDC',
    'hoverlabel.font.color':  '#121220',
  };
}

/** Map adaptive trace lines (#111 ↔ #e2e8f0) to the theme. */
function _swapAdaptiveTraceColors(fig, isDark) {
  const target = _adaptiveColor(isDark);
  fig.data = (fig.data || []).map((t) => {
    if (t.line && typeof t.line.color === 'string' &&
        _ADAPTIVE_SET.has(t.line.color.toLowerCase())) {
      return Object.assign({}, t, {
        line: Object.assign({}, t.line, { color: target }),
      });
    }
    return t;
  });
}

/** Render Plotly chart from /api/... */
async function renderChart(elId, url) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.innerHTML = '<div class="text-sm text-slate-500 p-4">loading…</div>';
  try {
    const r = await fetch(url);
    if (!r.ok) {
      const detail = await r.text();
      el.innerHTML = `<div class="p-4 text-sm text-rose-600 bg-rose-50 dark:bg-rose-900/10 rounded border border-rose-200 dark:border-rose-800">
        ${r.status} — ${detail.slice(0, 400)}</div>`;
      return;
    }
    const fig = await r.json();
    const isDark = document.documentElement.classList.contains('dark');
    fig.layout = fig.layout || {};

    const fontFamily = 'Inter, ui-sans-serif, system-ui, sans-serif';
    if (isDark) {
      fig.layout.template = undefined;
      fig.layout.paper_bgcolor = 'rgba(27,27,44,0)';
      fig.layout.plot_bgcolor  = 'rgba(27,27,44,0)';
      fig.layout.font = Object.assign({}, fig.layout.font, { color: '#E2E2ED', family: fontFamily });
      fig.layout.xaxis = Object.assign({}, fig.layout.xaxis, { gridcolor: '#2A2A40', zerolinecolor: '#2A2A40' });
      fig.layout.yaxis = Object.assign({}, fig.layout.yaxis, { gridcolor: '#2A2A40', zerolinecolor: '#2A2A40' });
      if (fig.layout.yaxis2) {
        fig.layout.yaxis2 = Object.assign({}, fig.layout.yaxis2, { gridcolor: '#2A2A40', zerolinecolor: '#2A2A40' });
      }
      fig.layout.legend = Object.assign({}, fig.layout.legend, {
        bgcolor: 'rgba(27,27,44,0.0)',
        font: Object.assign({}, (fig.layout.legend || {}).font, { color: '#E2E2ED' }),
      });
      fig.layout.hoverlabel = Object.assign({}, fig.layout.hoverlabel, {
        bgcolor:   '#12121F',
        bordercolor: '#3B3B55',
        font: Object.assign({}, (fig.layout.hoverlabel || {}).font, { color: '#EEEEF5' }),
      });
    } else {
      fig.layout.paper_bgcolor = 'rgba(255,255,255,0)';
      fig.layout.plot_bgcolor  = 'rgba(255,255,255,0)';
      fig.layout.font = Object.assign({}, fig.layout.font, { family: fontFamily });
      fig.layout.hoverlabel = Object.assign({}, fig.layout.hoverlabel, {
        bgcolor:   '#ffffff',
        bordercolor: '#CBCBDC',
        font: Object.assign({}, (fig.layout.hoverlabel || {}).font, { color: '#121220' }),
      });
    }

    // Always apply (light+dark): adaptive trace lines
    _swapAdaptiveTraceColors(fig, isDark);

    el.innerHTML = '';
    Plotly.newPlot(el, fig.data, fig.layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });
  } catch (e) {
    el.innerHTML = `<div class="p-4 text-sm text-rose-600">Error: ${e}</div>`;
  }
}

/** Load markdown from /api/markdown/{name} and render as HTML. */
async function fetchMd(name) {
  try {
    const r = await fetch(`/api/markdown/${encodeURIComponent(name)}`);
    if (!r.ok) return `<p class="text-slate-500 italic">Not yet available: ${name}</p>`;
    const { content } = await r.json();
    if (window.marked) {
      marked.setOptions({ gfm: true, breaks: false });
      return marked.parse(content);
    }
    return `<pre>${content.replace(/</g, '&lt;')}</pre>`;
  } catch (e) {
    return `<p class="text-rose-500">Load error: ${e}</p>`;
  }
}

/** Toast-Notification. */
function toast(msg, kind = 'info', ttl = 4000) {
  const c = document.getElementById('toast-container');
  if (!c) return;
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity 0.3s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 300);
  }, ttl);
}

/* ------------------------------------------------------------------ *
 *  Title "running…" indicator                                         *
 *  Use via: setRunning(true) ... setRunning(false)                    *
 * ------------------------------------------------------------------ */
const _ORIGINAL_TITLE = document.title;
let _runningCount = 0;
function setRunning(isRunning) {
  _runningCount += isRunning ? 1 : -1;
  if (_runningCount < 0) _runningCount = 0;
  const prefix = _runningCount > 0 ? 'running… ' : '';
  document.title = prefix + _ORIGINAL_TITLE;
}
// Reset cleanly on navigation/reload
window.addEventListener('beforeunload', () => { document.title = _ORIGINAL_TITLE; });

/* ------------------------------------------------------------------ *
 *  Image lightbox: every <img data-lightbox> is clickable → overlay   *
 * ------------------------------------------------------------------ */
function _initLightbox() {
  const modal = document.getElementById('img-lightbox');
  const target = document.getElementById('img-lightbox-target');
  if (!modal || !target) return;
  document.body.addEventListener('click', (e) => {
    const img = e.target.closest('img[data-lightbox]');
    if (!img) return;
    target.src = img.src;
    target.alt = img.alt || '';
    modal.classList.remove('hidden');
    modal.classList.add('flex');
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { modal.classList.add('hidden'); modal.classList.remove('flex'); }
  });
}

/** All <figure><img> under the body get data-lightbox + cursor-zoom-in. */
function _enableLightboxOnAssets() {
  document.querySelectorAll('img[src^="/api/asset/"]').forEach((img) => {
    if (!img.hasAttribute('data-lightbox')) {
      img.setAttribute('data-lightbox', '');
      img.classList.add('cursor-zoom-in', 'hover:opacity-90', 'transition');
    }
  });
}

/** Theme switch triggers a re-render of all Plotly charts. */
document.addEventListener('DOMContentLoaded', () => {
  _initLightbox();
  _enableLightboxOnAssets();
  // MutationObserver: dynamically inserted <img> elements also get the lightbox.
  new MutationObserver(() => _enableLightboxOnAssets())
    .observe(document.body, { childList: true, subtree: true });

  const obs = new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.attributeName !== 'class') continue;
      const isDark = document.documentElement.classList.contains('dark');
      const baseOverrides = isDark ? _darkLayoutOverrides() : _lightLayoutOverrides();
      const adaptive = _adaptiveColor(isDark);

      document.querySelectorAll('.js-plotly-plot').forEach((el) => {
        // 1) layout overrides
        Plotly.relayout(el, baseOverrides);

        // 2) adaptive trace line colors (Plotly.restyle — relayout is not enough)
        if (el.data && el.data.length) {
          const indices = [];
          el.data.forEach((t, i) => {
            if (t.line && typeof t.line.color === 'string' &&
                _ADAPTIVE_SET.has(t.line.color.toLowerCase())) {
              indices.push(i);
            }
          });
          if (indices.length > 0) {
            Plotly.restyle(el, { 'line.color': adaptive }, indices);
          }
        }
      });
    }
  });
  obs.observe(document.documentElement, { attributes: true });
});
