/* Globale Helper für alle Dashboard-Seiten. */

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
    if (isDark) {
      fig.layout = fig.layout || {};
      fig.layout.template = undefined;
      fig.layout.paper_bgcolor = 'rgba(30,41,59,0)';
      fig.layout.plot_bgcolor = 'rgba(30,41,59,0)';
      fig.layout.font = Object.assign({}, fig.layout.font, { color: '#e2e8f0' });
      fig.layout.xaxis = Object.assign({}, fig.layout.xaxis, { gridcolor: '#334155', zerolinecolor: '#334155' });
      fig.layout.yaxis = Object.assign({}, fig.layout.yaxis, { gridcolor: '#334155', zerolinecolor: '#334155' });
    }
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

/** Theme-Switch triggert Re-Render aller Plotly-Charts. */
document.addEventListener('DOMContentLoaded', () => {
  const obs = new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.attributeName === 'class') {
        document.querySelectorAll('.js-plotly-plot').forEach((el) => {
          // Plotly.relayout mit neuen Theme-Farben
          const isDark = document.documentElement.classList.contains('dark');
          Plotly.relayout(el, {
            'paper_bgcolor': isDark ? 'rgba(30,41,59,0)' : 'white',
            'plot_bgcolor':  isDark ? 'rgba(30,41,59,0)' : 'white',
            'font.color':    isDark ? '#e2e8f0' : '#0f172a',
            'xaxis.gridcolor': isDark ? '#334155' : '#e2e8f0',
            'yaxis.gridcolor': isDark ? '#334155' : '#e2e8f0',
          });
        });
      }
    }
  });
  obs.observe(document.documentElement, { attributes: true });
});
