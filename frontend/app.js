/* ─── State ─── */
let state = { modelTrained: false, featureSchema: [], notebookCache: {} };

/* ─── Navigation ─── */
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tab) {
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.nav-btn[data-tab="${tab}"]`).classList.add('active');
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
}

/* ─── Status ─── */
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    state.modelTrained = data.model_trained;
    state.featureSchema = data.feature_schema || [];

    document.getElementById('ds-model-trained').textContent = data.model_trained ? '✓ Ready' : '✗ Not trained';
    document.getElementById('ds-data-ready').textContent = data.data_exists ? '✓ Yes' : '✗ No';
    document.getElementById('ds-features').textContent = data.feature_count ?? '—';

    const dot = document.querySelector('#status-badge .status-dot');
    const label = document.querySelector('#status-badge span');
    if (data.model_trained) {
      dot.className = 'status-dot ready';
      label.textContent = 'Model Ready';
    } else {
      dot.className = 'status-dot error';
      label.textContent = 'Model Missing';
    }

    updatePipelineSteps(data);
    buildPredictForm(data);
    if (data.model_trained) {
      document.getElementById('btn-run-setup').textContent = 'Re-run Full Pipeline';
    }
  } catch (e) {
    console.error('Status fetch failed:', e);
  }
}

function updatePipelineSteps(data) {
  const s1 = document.getElementById('step-data');
  const s2 = document.getElementById('step-features');
  const s3 = document.getElementById('step-train');
  s1.classList.toggle('done', data.data_exists);
  s2.classList.toggle('done', data.data_exists);
  s3.classList.toggle('done', data.model_trained);
}

/* ─── Setup ─── */
async function runSetup() {
  const btn = document.getElementById('btn-run-setup');
  const out = document.getElementById('setup-output');
  btn.disabled = true;
  btn.textContent = 'Running...';
  out.className = 'output-box';
  out.textContent = 'Starting pipeline...\n';

  try {
    const res = await fetch('/api/setup', { method: 'POST' });
    const data = await res.json();

    if (data.status === 'success') {
      out.textContent += `✓ Data ingestion complete\n`;
      out.textContent += `✓ Feature engineering complete\n`;
      out.textContent += `✓ Training complete\n`;
      out.textContent += `\nR² (train): ${data.metrics.r2_train}\n`;
      out.textContent += `R² (test):  ${data.metrics.r2_test}\n`;
      out.textContent += `Features:   ${data.metrics.feature_count}\n`;
    } else {
      out.textContent += `✗ Error: ${data.message}\n`;
      out.className = 'output-box error';
    }
  } catch (e) {
    out.textContent += `✗ Network error: ${e.message}\n`;
    out.className = 'output-box error';
  }

  btn.disabled = false;
  btn.textContent = 'Run Full Pipeline';
  fetchStatus();
}

/* ─── Predict ─── */
function buildPredictForm(data) {
  const schema = data.feature_schema || [];
  if (!schema.length) return;

  for (const group of schema) {
    const sel = document.getElementById(`p-${group.name}`);
    if (!sel || !group.values) continue;
    sel.innerHTML = group.values.map(v =>
      `<option value="${v}">${v.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</option>`
    ).join('');
  }
}

async function handlePredict(e) {
  e.preventDefault();
  const resultEl = document.getElementById('predict-result');
  const errorEl = document.getElementById('predict-error');
  resultEl.classList.add('hidden');
  errorEl.classList.add('hidden');

  const body = {
    make: document.getElementById('p-make').value,
    body_style: document.getElementById('p-body-style').value,
    engine_type: document.getElementById('p-engine-type').value,
    fuel_system: document.getElementById('p-fuel-system').value,
    num_of_cylinders: document.getElementById('p-num-of-cylinders').value,
    drive_wheels: document.getElementById('p-drive-wheels').value,
    aspiration_turbo: parseInt(document.getElementById('p-aspiration').value),
    engine_location_rear: parseInt(document.getElementById('p-engine-location').value),
    curb_weight: parseFloat(document.getElementById('p-curb-weight').value),
    engine_size: parseFloat(document.getElementById('p-engine-size').value),
    horsepower: parseFloat(document.getElementById('p-horsepower').value),
    bore: parseFloat(document.getElementById('p-bore').value),
    stroke: parseFloat(document.getElementById('p-stroke').value),
    height: parseFloat(document.getElementById('p-height').value),
    symboling: parseInt(document.getElementById('p-symboling').value),
    peak_rpm: parseFloat(document.getElementById('p-peak-rpm').value),
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (res.ok) {
      document.getElementById('predict-price').textContent =
        '$' + data.predicted_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      resultEl.classList.remove('hidden');
    } else {
      errorEl.textContent = 'Error: ' + (data.detail || 'Prediction failed');
      errorEl.classList.remove('hidden');
    }
  } catch (e) {
    errorEl.textContent = 'Network error: ' + e.message;
    errorEl.classList.remove('hidden');
  }
}

/* ─── Evaluate ─── */
async function runEvaluate() {
  const metricsEl = document.getElementById('eval-metrics');
  const plotsEl = document.getElementById('eval-plots');
  const errorEl = document.getElementById('eval-error');
  metricsEl.classList.add('hidden');
  plotsEl.classList.add('hidden');
  errorEl.classList.add('hidden');

  try {
    const res = await fetch('/api/evaluate', { method: 'POST' });
    const data = await res.json();

    if (data.status === 'success') {
      const m = data.metrics;
      metricsEl.innerHTML = `
        <div class="metric-card">
          <div class="metric-value">${m.r2}</div>
          <div class="metric-label">R² Score</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">$${(m.mae).toLocaleString()}</div>
          <div class="metric-label">MAE</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">$${(m.rmse).toLocaleString()}</div>
          <div class="metric-label">RMSE</div>
        </div>
      `;
      metricsEl.classList.remove('hidden');

      plotsEl.innerHTML = '';
      for (const plot of m.plots) {
        const img = document.createElement('img');
        img.src = `/reports/${plot}?t=${Date.now()}`;
        img.alt = plot;
        plotsEl.appendChild(img);
      }
      plotsEl.classList.remove('hidden');
    } else {
      errorEl.textContent = 'Error: ' + (data.message || 'Evaluation failed');
      errorEl.classList.remove('hidden');
    }
  } catch (e) {
    errorEl.textContent = 'Network error: ' + e.message;
    errorEl.classList.remove('hidden');
  }
}

/* ─── Notebooks ─── */
document.getElementById('btn-notebooks-toggle').addEventListener('click', function(e) {
  const group = this.closest('.nav-group');
  const isOpening = !group.classList.contains('open');

  document.querySelectorAll('.nav-group.open').forEach(g => {
    if (g !== group) g.classList.remove('open');
  });

  group.classList.toggle('open');

  if (isOpening) {
    switchTab('notebooks');
  } else {
    document.querySelectorAll('.nav-subitem').forEach(s => s.classList.remove('active'));
    document.getElementById('notebook-viewer').innerHTML = '';
  }
});

async function preloadNotebooks() {
  try {
    const res = await fetch('/api/notebooks');
    const data = await res.json();
    const menu = document.getElementById('notebook-submenu');
    menu.innerHTML = data.notebooks.map(nb =>
      `<button class="nav-subitem" onclick="openNotebook('${nb}')">${nb.replace('.ipynb', '')}</button>`
    ).join('');
    data.notebooks.forEach(async (name) => {
      const r = await fetch(`/api/notebooks/${name}`);
      state.notebookCache[name] = await r.json();
    });
  } catch (e) {
    document.getElementById('notebook-submenu').innerHTML =
      '<span style="color:var(--error);font-size:11px">Failed to load</span>';
  }
}

function renderNotebookCells(nb, container) {
  for (const cell of nb.cells) {
    if (cell.cell_type === 'markdown') {
      const src = cell.source.join('');
      const html = marked.parse(src);
      const tmp = document.createElement('div');
      tmp.innerHTML = html;
      while (tmp.firstChild) container.appendChild(tmp.firstChild);
    } else if (cell.cell_type === 'code') {
      const src = cell.source.join('');

      const cellGroup = document.createElement('div');
      cellGroup.className = 'code-cell-group';

      const codeEl = document.createElement('pre');
      codeEl.className = 'code-block';
      const codeTag = document.createElement('code');
      codeTag.className = 'language-python';
      codeTag.textContent = src;
      codeEl.appendChild(codeTag);
      cellGroup.appendChild(codeEl);

      for (const output of cell.outputs || []) {
        let text = null;
        if (output.text) {
          text = (output.text.join ? output.text.join('') : output.text).trim();
        } else if (output.data && output.data['text/plain']) {
          text = output.data['text/plain'].join ? output.data['text/plain'].join('') : output.data['text/plain'];
        }
        if (text) {
          const pre = document.createElement('pre');
          pre.className = 'cell-output';
          pre.textContent = text;
          cellGroup.appendChild(pre);
        }
      }

      container.appendChild(cellGroup);
    }
  }

  container.querySelectorAll('pre.code-block code').forEach(hljs.highlightElement);
}

function openNotebook(name) {
  document.querySelectorAll('.nav-subitem').forEach(s => s.classList.remove('active'));
  const btn = document.querySelector(`.nav-subitem[onclick*="'${name}'"]`);
  if (btn) btn.classList.add('active');

  const viewerEl = document.getElementById('notebook-viewer');
  viewerEl.innerHTML = '';

  const nb = state.notebookCache[name];
  if (!nb) {
    viewerEl.innerHTML = '<p style="color:var(--error)">Notebook not loaded yet.</p>';
    return;
  }

  const title = document.createElement('h1');
  title.textContent = name.replace('.ipynb', '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  viewerEl.appendChild(title);

  renderNotebookCells(nb, viewerEl);
}

/* ─── Init ─── */
fetchStatus();
preloadNotebooks();
