/* Globale Helper für alle Dashboard-Seiten. */

/* ------------------------------------------------------------------ *
 *  Dark-Mode-Layout-Overrides für Plotly                              *
 * ------------------------------------------------------------------ */
function _darkLayoutOverrides() {
  return {
    paper_bgcolor: 'rgba(30,41,59,0)',
    plot_bgcolor:  'rgba(30,41,59,0)',
    'font.color':  '#e2e8f0',
    'xaxis.gridcolor': '#334155',
    'xaxis.zerolinecolor': '#334155',
    'yaxis.gridcolor': '#334155',
    'yaxis.zerolinecolor': '#334155',
    'legend.bgcolor': 'rgba(30,41,59,0.0)',
    'legend.font.color': '#e2e8f0',
    'hoverlabel.bgcolor':   '#0f172a',
    'hoverlabel.bordercolor': '#475569',
    'hoverlabel.font.color':  '#f1f5f9',
  };
}
function _lightLayoutOverrides() {
  return {
    paper_bgcolor: 'white',
    plot_bgcolor:  'white',
    'font.color':  '#0f172a',
    'xaxis.gridcolor': '#e2e8f0',
    'xaxis.zerolinecolor': '#e2e8f0',
    'yaxis.gridcolor': '#e2e8f0',
    'yaxis.zerolinecolor': '#e2e8f0',
    'legend.bgcolor': 'rgba(255,255,255,0.0)',
    'legend.font.color': '#0f172a',
    'hoverlabel.bgcolor':   '#ffffff',
    'hoverlabel.bordercolor': '#cbd5e1',
    'hoverlabel.font.color':  '#0f172a',
  };
}

/** Plotly-Chart aus /api/... rendern. */
async function renderChart(elId, url) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.innerHTML = '<div class="text-sm text-slate-500 p-4">lädt…</div>';
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
    if (isDark) {
      fig.layout.template = undefined;
      fig.layout.paper_bgcolor = 'rgba(30,41,59,0)';
      fig.layout.plot_bgcolor  = 'rgba(30,41,59,0)';
      fig.layout.font = Object.assign({}, fig.layout.font, { color: '#e2e8f0' });
      fig.layout.xaxis = Object.assign({}, fig.layout.xaxis, { gridcolor: '#334155', zerolinecolor: '#334155' });
      fig.layout.yaxis = Object.assign({}, fig.layout.yaxis, { gridcolor: '#334155', zerolinecolor: '#334155' });
      if (fig.layout.yaxis2) {
        fig.layout.yaxis2 = Object.assign({}, fig.layout.yaxis2, { gridcolor: '#334155', zerolinecolor: '#334155' });
      }
      fig.layout.legend = Object.assign({}, fig.layout.legend, {
        bgcolor: 'rgba(30,41,59,0.0)',
        font: Object.assign({}, (fig.layout.legend || {}).font, { color: '#e2e8f0' }),
      });
      fig.layout.hoverlabel = Object.assign({}, fig.layout.hoverlabel, {
        bgcolor:   '#0f172a',
        bordercolor: '#475569',
        font: Object.assign({}, (fig.layout.hoverlabel || {}).font, { color: '#f1f5f9' }),
      });
    } else {
      fig.layout.hoverlabel = Object.assign({}, fig.layout.hoverlabel, {
        bgcolor:   '#ffffff',
        bordercolor: '#cbd5e1',
        font: Object.assign({}, (fig.layout.hoverlabel || {}).font, { color: '#0f172a' }),
      });
    }
    el.innerHTML = '';
    Plotly.newPlot(el, fig.data, fig.layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });
  } catch (e) {
    el.innerHTML = `<div class="p-4 text-sm text-rose-600">Fehler: ${e}</div>`;
  }
}

/** Markdown aus /api/markdown/{name} laden und als HTML rendern. */
async function fetchMd(name) {
  try {
    const r = await fetch(`/api/markdown/${encodeURIComponent(name)}`);
    if (!r.ok) return `<p class="text-slate-500 italic">Noch nicht verfügbar: ${name}</p>`;
    const { content } = await r.json();
    if (window.marked) {
      marked.setOptions({ gfm: true, breaks: false });
      return marked.parse(content);
    }
    return `<pre>${content.replace(/</g, '&lt;')}</pre>`;
  } catch (e) {
    return `<p class="text-rose-500">Load-Fehler: ${e}</p>`;
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
 *  Title „läuft…“-Indikator                                           *
 *  Verwenden via: setRunning(true) ... setRunning(false)              *
 * ------------------------------------------------------------------ */
const _ORIGINAL_TITLE = document.title;
let _runningCount = 0;
function setRunning(isRunning) {
  _runningCount += isRunning ? 1 : -1;
  if (_runningCount < 0) _runningCount = 0;
  const prefix = _runningCount > 0 ? 'läuft… ' : '';
  document.title = prefix + _ORIGINAL_TITLE;
}
// Beim Navigieren/Reload sauber zurücksetzen
window.addEventListener('beforeunload', () => { document.title = _ORIGINAL_TITLE; });

/* ------------------------------------------------------------------ *
 *  Image-Lightbox: jede <img data-lightbox> klickbar → Overlay        *
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

/** Alle <figure><img> unter dem Body bekommen data-lightbox + cursor-zoom-in. */
function _enableLightboxOnAssets() {
  document.querySelectorAll('img[src^="/api/asset/"]').forEach((img) => {
    if (!img.hasAttribute('data-lightbox')) {
      img.setAttribute('data-lightbox', '');
      img.classList.add('cursor-zoom-in', 'hover:opacity-90', 'transition');
    }
  });
}

/** Theme-Switch triggert Re-Render aller Plotly-Charts. */
document.addEventListener('DOMContentLoaded', () => {
  _initLightbox();
  _enableLightboxOnAssets();
  // Mutation-Observer: auch später dynamisch eingefügte <img> bekommen Lightbox.
  new MutationObserver(() => _enableLightboxOnAssets())
    .observe(document.body, { childList: true, subtree: true });

  const obs = new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.attributeName === 'class') {
        const isDark = document.documentElement.classList.contains('dark');
        const overrides = isDark ? _darkLayoutOverrides() : _lightLayoutOverrides();
        document.querySelectorAll('.js-plotly-plot').forEach((el) => {
          Plotly.relayout(el, overrides);
        });
      }
    }
  });
  obs.observe(document.documentElement, { attributes: true });
});
