/* ─── State ─── */
let state = { modelTrained: false, featureSchema: [] };

/* ─── Navigation ─── */
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tab) {
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.nav-btn[data-tab="${tab}"]`).classList.add('active');
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  if (tab === 'notebooks') loadNotebookList();
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
    num_of_cylinders: document.getElementById('p-cylinders').value,
    drive_wheels: document.getElementById('p-drive').value,
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
async function loadNotebookList() {
  const listEl = document.getElementById('notebook-list');
  const viewerEl = document.getElementById('notebook-viewer');
  viewerEl.classList.add('hidden');

  try {
    const res = await fetch('/api/notebooks');
    const data = await res.json();
    listEl.innerHTML = data.notebooks.map(nb => {
      const desc = { 'features.ipynb': 'Feature engineering & selection', 'initial_eda.ipynb': 'Exploratory data analysis', 'training.ipynb': 'Model training & evaluation' }[nb] || '';
      return `<div class="nb-card" onclick="openNotebook('${nb}')">
        <div class="nb-name">${nb}</div>
        <div class="nb-desc">${desc}</div>
      </div>`;
    }).join('');
  } catch (e) {
    listEl.innerHTML = '<p style="color:var(--text-secondary)">Failed to load notebooks.</p>';
  }
}

async function openNotebook(name) {
  const viewerEl = document.getElementById('notebook-viewer');

  try {
    const res = await fetch(`/api/notebooks/${name}`);
    const nb = await res.json();
    viewerEl.innerHTML = '<div class="cell-marker">' + name + '</div>';
    viewerEl.classList.remove('hidden');

    for (const cell of nb.cells) {
      const div = document.createElement('div');
      div.className = 'cell';

      if (cell.cell_type === 'markdown') {
        const src = cell.source.join('');
        div.innerHTML = `<div class="cell-marker">md</div><div class="cell-md">${marked.parse(src)}</div>`;
      } else if (cell.cell_type === 'code') {
        const src = cell.source.join('');
        div.innerHTML = `<div class="cell-marker">code</div><pre class="cell-code">${escHtml(src)}</pre>`;

        for (const output of cell.outputs || []) {
          if (output.text) {
            const outDiv = document.createElement('div');
            outDiv.className = 'cell-output';
            outDiv.textContent = (output.text.join ? output.text.join('') : output.text).trim();
            div.appendChild(outDiv);
          }
          if (output.data && output.data['text/plain']) {
            const outDiv = document.createElement('div');
            outDiv.className = 'cell-output';
            outDiv.textContent = output.data['text/plain'].join ? output.data['text/plain'].join('') : output.data['text/plain'];
            div.appendChild(outDiv);
          }
        }
      }

      viewerEl.appendChild(div);
    }

    viewerEl.scrollTop = 0;
  } catch (e) {
    viewerEl.innerHTML = '<p style="color:var(--error)">Failed to load notebook.</p>';
    viewerEl.classList.remove('hidden');
  }
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/* ─── Init ─── */
fetchStatus();
