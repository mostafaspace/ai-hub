/**
 * AI-Hub Dashboard — Frontend Engine
 * 
 * Polls the orchestrator at /v1/hub/services and /v1/hub/tasks
 * to render live service status, VRAM bars, task feeds, and logs.
 */

// --- Configuration ---
const API_BASE = window.location.origin; // Same-origin as orchestrator
const POLL_SERVICES_MS = 5000;
const POLL_TASKS_MS = 10000;

// Service registry — defines the cards to render.
// This is the single source of truth for service display metadata.
const SERVICE_REGISTRY = [
  { key: 'tts', name: 'Qwen3 TTS', emoji: '🎙️', port: 8000 },
  { key: 'music', name: 'ACE-Step Music', emoji: '🎵', port: 8001 },
  { key: 'asr', name: 'Qwen3 ASR', emoji: '🎧', port: 8002 },
  { key: 'vision', name: 'Z-Image', emoji: '🖼️', port: 8003 },
  { key: 'video', name: 'LTX-2 Video', emoji: '🎬', port: 8004 },
];

// --- State ---
let servicesData = {};
let systemLogs = [];

// ==============================================
//  INITIALIZATION
// ==============================================
document.addEventListener('DOMContentLoaded', () => {
  renderServiceCards();
  startClock();
  pollServices();
  pollTasks();

  // Start polling loops
  setInterval(pollServices, POLL_SERVICES_MS);
  setInterval(pollTasks, POLL_TASKS_MS);

  addLog('Dashboard connected to orchestrator', 'ok');
});

// ==============================================
//  CLOCK
// ==============================================
function startClock() {
  const el = document.getElementById('clock');
  function tick() {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('en-US', { hour12: false }) +
      ' · ' + now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }
  tick();
  setInterval(tick, 1000);
}

// ==============================================
//  SERVICE CARDS — RENDER
// ==============================================
function renderServiceCards() {
  const grid = document.getElementById('servicesGrid');
  grid.innerHTML = '';

  SERVICE_REGISTRY.forEach((svc, i) => {
    const card = document.createElement('div');
    card.className = 'service-card fade-in';
    card.id = `card-${svc.key}`;
    card.style.opacity = '0';
    card.innerHTML = `
      <div class="card-header">
        <div class="card-title">
          <span class="emoji">${svc.emoji}</span>
          <h3>${svc.name}</h3>
        </div>
        <span class="status-pill offline" id="pill-${svc.key}">
          <span class="dot"></span>
          <span id="pill-text-${svc.key}">Offline</span>
        </span>
      </div>
      <div class="card-details">
        <div class="detail-row">
          <span class="detail-label">Port</span>
          <span class="detail-value">${svc.port}</span>
        </div>
        <div class="vram-bar-container">
          <div class="vram-bar-header">
            <span class="vram-bar-label">VRAM</span>
            <span class="vram-bar-value" id="vram-text-${svc.key}">— / —</span>
          </div>
          <div class="vram-bar-track">
            <div class="vram-bar-fill empty" id="vram-fill-${svc.key}" style="width: 0%"></div>
          </div>
        </div>
        <div class="card-actions">
          <button class="btn btn-ghost" id="action-${svc.key}" onclick="handleServiceAction('${svc.key}')" title="">
            Check
          </button>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

// ==============================================
//  SERVICE POLLING
// ==============================================
async function pollServices() {
  try {
    const resp = await fetch(`${API_BASE}/v1/hub/services`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    servicesData = data.services || {};

    let onlineCount = 0;

    SERVICE_REGISTRY.forEach(svc => {
      const info = servicesData[svc.key];
      updateServiceCard(svc.key, info);
      if (info && info.status === 'online') onlineCount++;
    });

    // Update hub badge
    const badge = document.getElementById('hubBadge');
    const badgeText = document.getElementById('hubBadgeText');
    if (onlineCount > 0) {
      badge.className = 'hub-badge online';
      badgeText.textContent = `${onlineCount}/${SERVICE_REGISTRY.length} Online`;
    } else {
      badge.className = 'hub-badge offline';
      badgeText.textContent = 'All Offline';
    }

  } catch (err) {
    // Orchestrator itself is down
    const badge = document.getElementById('hubBadge');
    const badgeText = document.getElementById('hubBadgeText');
    badge.className = 'hub-badge offline';
    badgeText.textContent = 'Hub Offline';

    SERVICE_REGISTRY.forEach(svc => {
      updateServiceCard(svc.key, null);
    });
  }
}

function updateServiceCard(key, info) {
  const pill = document.getElementById(`pill-${key}`);
  const pillText = document.getElementById(`pill-text-${key}`);
  const vramText = document.getElementById(`vram-text-${key}`);
  const vramFill = document.getElementById(`vram-fill-${key}`);
  const actionBtn = document.getElementById(`action-${key}`);

  if (!pill) return;

  if (!info || info.status === 'offline') {
    pill.className = 'status-pill offline';
    pillText.textContent = 'Offline';
    vramText.textContent = '— / —';
    vramFill.style.width = '0%';
    vramFill.className = 'vram-bar-fill empty';
    actionBtn.textContent = 'Offline';
    actionBtn.className = 'btn btn-ghost';
    actionBtn.disabled = true;
    actionBtn.style.opacity = '0.4';
    actionBtn.style.cursor = 'not-allowed';
    actionBtn.title = 'Service not running. Use Start All Servers to launch.';
    return;
  }

  // Re-enable button for online services
  actionBtn.disabled = false;
  actionBtn.style.opacity = '1';
  actionBtn.style.cursor = 'pointer';
  actionBtn.title = '';

  // Online or Idle
  const isIdle = info.model_loaded === false;
  if (isIdle) {
    pill.className = 'status-pill idle';
    pillText.textContent = 'Idle';
    actionBtn.textContent = 'Load';
    actionBtn.className = 'btn btn-ghost';
  } else {
    pill.className = 'status-pill online';
    pillText.textContent = 'Online';
    actionBtn.textContent = 'Unload';
    actionBtn.className = 'btn btn-danger';
  }

  // VRAM bar
  if (info.vram_used_gb !== undefined && info.vram_total_gb !== undefined) {
    const used = info.vram_used_gb.toFixed(1);
    const total = info.vram_total_gb.toFixed(0);
    const pct = Math.min(100, (info.vram_used_gb / info.vram_total_gb) * 100);
    vramText.textContent = `${used} GB / ${total} GB`;
    vramFill.style.width = `${pct}%`;
    vramFill.className = pct > 85 ? 'vram-bar-fill high' : 'vram-bar-fill';
  } else if (isIdle) {
    vramText.textContent = '0 GB (Unloaded)';
    vramFill.style.width = '0%';
    vramFill.className = 'vram-bar-fill empty';
  } else {
    vramText.textContent = 'Model loaded';
    vramFill.style.width = '30%';
    vramFill.className = 'vram-bar-fill';
  }
}

// ==============================================
//  SERVICE ACTIONS (Unload)
// ==============================================
async function handleServiceAction(key) {
  const info = servicesData[key];
  if (!info || info.status === 'offline') {
    addLog(`Cannot reach ${key} — service is offline`, 'warn');
    return;
  }

  const isLoaded = info.model_loaded !== false;

  if (isLoaded) {
    addLog(`Sending unload request to ${key}...`, 'ok');
    const btn = document.getElementById(`action-${key}`);
    if (btn) { btn.disabled = true; btn.textContent = '...'; }
    try {
      const resp = await fetch(`${API_BASE}/v1/hub/unload/${key}`, { method: 'POST' });
      const data = await resp.json();
      addLog(`${key}: ${data.message || 'Unload sent'}`, resp.ok ? 'ok' : 'warn');
      // Refresh after a short delay
      setTimeout(pollServices, 1500);
    } catch (err) {
      addLog(`Failed to unload ${key}: ${err.message}`, 'err');
    }
  } else {
    addLog(`${key} is idle — model will auto-load on next request`, 'warn');
  }
}

// ==============================================
//  TASK POLLING
// ==============================================
async function pollTasks() {
  try {
    const resp = await fetch(`${API_BASE}/v1/hub/tasks`);
    if (!resp.ok) return;
    const data = await resp.json();
    renderTasks(data.tasks || []);
  } catch (err) {
    // silent — tasks are optional
  }
}

function renderTasks(tasks) {
  const list = document.getElementById('taskList');
  const count = document.getElementById('taskCount');
  count.textContent = `${tasks.length} task${tasks.length !== 1 ? 's' : ''}`;

  if (tasks.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">📋</span>
        <span>No recent tasks</span>
      </div>
    `;
    return;
  }

  list.innerHTML = '';
  tasks.forEach(task => {
    const statusClass = task.status === 'COMPLETED' ? 'completed'
      : task.status === 'RUNNING' ? 'running' : 'failed';

    const spinnerHtml = task.status === 'RUNNING' ? '<span class="spinner"></span>' : '';

    const el = document.createElement('div');
    el.className = 'task-item';
    el.innerHTML = `
      <div class="task-info">
        <div class="task-name">
          ${task.workflow || 'unknown'}
          <span class="task-status ${statusClass}">${spinnerHtml}${task.status}</span>
        </div>
        <div class="task-meta">ID: ${task.task_id || '—'}${task.duration ? '  ·  ' + task.duration : ''}</div>
      </div>
      <span class="task-time">${task.time_ago || ''}</span>
    `;
    list.appendChild(el);
  });
}

// ==============================================
//  SYSTEM LOG
// ==============================================
function addLog(msg, level = '') {
  const now = new Date();
  const time = now.toLocaleTimeString('en-US', { hour12: false });

  systemLogs.push({ time, msg, level });
  if (systemLogs.length > 50) systemLogs.shift();

  const logEl = document.getElementById('systemLog');
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `
    <span class="log-time">${time}</span>
    <span class="log-msg ${level}">${escapeHtml(msg)}</span>
  `;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ==============================================
//  QUICK TOOLS — HEALTH CHECK
// ==============================================
async function launchAllServers() {
  addLog('Sending launch-all request to orchestrator...', 'ok');
  try {
    const resp = await fetch(`${API_BASE}/v1/hub/launch-all`, { method: 'POST' });
    const data = await resp.json();
    addLog(data.message || 'Launch request sent', resp.ok ? 'ok' : 'warn');
    // Poll more frequently for a bit to see servers come up
    addLog('Waiting for services to start (this may take 30-60 seconds)...', 'warn');
    let checks = 0;
    const fastPoll = setInterval(() => {
      pollServices();
      checks++;
      if (checks >= 12) clearInterval(fastPoll); // Stop after 60s of fast polling
    }, 5000);
  } catch (err) {
    addLog(`Launch failed: ${err.message}`, 'err');
  }
}

async function runHealthCheck() {
  addLog('Running health check on all services...', 'ok');

  try {
    const resp = await fetch(`${API_BASE}/v1/hub/services`);
    const data = await resp.json();
    const services = data.services || {};

    SERVICE_REGISTRY.forEach(svc => {
      const info = services[svc.key];
      if (info && info.status === 'online') {
        addLog(`  ✅ ${svc.name} (Port ${svc.port}): ONLINE`, 'ok');
      } else {
        addLog(`  ❌ ${svc.name} (Port ${svc.port}): OFFLINE`, 'err');
      }
    });

    addLog('Health check complete.', 'ok');
  } catch (err) {
    addLog('Health check failed — orchestrator unreachable', 'err');
  }
}

// ==============================================
//  QUICK TOOLS — MODALS
// ==============================================
function openModal(type) {
  const overlay = document.getElementById('modalOverlay');
  const content = document.getElementById('modalContent');

  let html = '';

  switch (type) {
    case 'tts':
      html = `
        <div class="modal-header">
          <span class="modal-title">🎙️ Test TTS</span>
          <button class="modal-close" onclick="closeModal()">✕</button>
        </div>
        <div class="form-group">
          <label class="form-label">Text to Synthesize</label>
          <textarea class="form-textarea" id="modalTtsText" placeholder="Hello, welcome to AI-Hub...">Hello, welcome to AI-Hub. The future of open-source AI is here.</textarea>
        </div>
        <div class="form-group">
          <label class="form-label">Voice</label>
          <select class="form-select" id="modalTtsVoice">
            <option value="Vivian">Vivian</option>
            <option value="Echo">Echo</option>
            <option value="Nova">Nova</option>
          </select>
        </div>
        <button class="btn btn-primary" onclick="submitTts()" id="modalSubmitBtn">Synthesize</button>
        <div class="modal-result" id="modalResult"></div>
        <div class="audio-player" id="audioPlayer" style="display:none">
          <audio controls id="audioElement"></audio>
        </div>
      `;
      break;

    case 'image':
      html = `
        <div class="modal-header">
          <span class="modal-title">🖼️ Generate Image</span>
          <button class="modal-close" onclick="closeModal()">✕</button>
        </div>
        <div class="form-group">
          <label class="form-label">Prompt</label>
          <textarea class="form-textarea" id="modalImagePrompt" placeholder="A cyberpunk cityscape at sunset, 4k, cinematic...">A cyberpunk cityscape at sunset, neon lights reflecting on wet streets, cinematic, 4k</textarea>
        </div>
        <div class="form-group">
          <label class="form-label">Size</label>
          <select class="form-select" id="modalImageSize">
            <option value="1024x1024">1024 × 1024</option>
            <option value="768x1024">768 × 1024 (Portrait)</option>
            <option value="1024x768">1024 × 768 (Landscape)</option>
            <option value="512x512">512 × 512 (Fast)</option>
          </select>
        </div>
        <button class="btn btn-primary" onclick="submitImage()" id="modalSubmitBtn">Generate</button>
        <div class="modal-result" id="modalResult"></div>
      `;
      break;

    case 'director':
      html = `
        <div class="modal-header">
          <span class="modal-title">🎬 Run Director Workflow</span>
          <button class="modal-close" onclick="closeModal()">✕</button>
        </div>
        <div class="form-group">
          <label class="form-label">Image Prompt</label>
          <textarea class="form-textarea" id="modalDirImage" placeholder="Detailed visual description for the base frame...">A lone astronaut standing on a red Martian landscape, Earth visible in the sky, dramatic lighting, cinematic</textarea>
        </div>
        <div class="form-group">
          <label class="form-label">Voiceover Script</label>
          <textarea class="form-textarea" id="modalDirVoice" placeholder="The narration text...">Humanity has always looked to the stars. Today, we stand on the surface of a new world.</textarea>
        </div>
        <div class="form-group">
          <label class="form-label">Voice</label>
          <select class="form-select" id="modalDirVoiceProfile">
            <option value="Vivian">Vivian</option>
            <option value="Echo">Echo</option>
            <option value="Nova">Nova</option>
          </select>
        </div>
        <button class="btn btn-primary" onclick="submitDirector()" id="modalSubmitBtn">Launch Pipeline</button>
        <div class="modal-result" id="modalResult"></div>
      `;
      break;
  }

  content.innerHTML = html;
  overlay.classList.add('active');
}

function closeModal() {
  document.getElementById('modalOverlay').classList.remove('active');
}

function closeModalOverlay(e) {
  if (e.target === e.currentTarget) closeModal();
}

function setModalLoading(msg) {
  const result = document.getElementById('modalResult');
  const btn = document.getElementById('modalSubmitBtn');
  result.className = 'modal-result loading';
  result.innerHTML = `<span class="spinner"></span> ${escapeHtml(msg)}`;
  btn.disabled = true;
  btn.style.opacity = '0.5';
}

function setModalResult(text, isError = false) {
  const result = document.getElementById('modalResult');
  const btn = document.getElementById('modalSubmitBtn');
  result.className = isError ? 'modal-result error' : 'modal-result success';
  result.textContent = text;
  btn.disabled = false;
  btn.style.opacity = '1';
}

// --- TTS Submit ---
async function submitTts() {
  const text = document.getElementById('modalTtsText').value.trim();
  const voice = document.getElementById('modalTtsVoice').value;
  if (!text) return;

  setModalLoading('Synthesizing audio...');
  addLog(`TTS request: voice=${voice}, length=${text.length} chars`, 'ok');

  try {
    const resp = await fetch(`${API_BASE}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: text, voice: voice, response_format: 'wav' }),
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(err);
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const audioPlayer = document.getElementById('audioPlayer');
    const audioElement = document.getElementById('audioElement');
    audioElement.src = url;
    audioPlayer.style.display = 'block';

    setModalResult(`✅ Audio generated (${(blob.size / 1024).toFixed(1)} KB)`);
    addLog(`TTS completed: ${(blob.size / 1024).toFixed(1)} KB`, 'ok');
  } catch (err) {
    setModalResult(`Error: ${err.message}`, true);
    addLog(`TTS failed: ${err.message}`, 'err');
  }
}

// --- Image Submit ---
async function submitImage() {
  const prompt = document.getElementById('modalImagePrompt').value.trim();
  const size = document.getElementById('modalImageSize').value;
  if (!prompt) return;

  setModalLoading('Submitting image generation task...');
  addLog(`Image request: "${prompt.substring(0, 40)}...", size=${size}`, 'ok');

  try {
    // Use async endpoint
    const resp = await fetch(`${API_BASE}/v1/images/async_generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, size, cfg_normalization: true }),
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(err);
    }

    const data = await resp.json();
    setModalResult(`✅ Task submitted!\nTask ID: ${data.task_id}\nStatus: ${data.status}\n\nPoll /v1/images/tasks/${data.task_id} for results.`);
    addLog(`Image task created: ${data.task_id}`, 'ok');
  } catch (err) {
    setModalResult(`Error: ${err.message}`, true);
    addLog(`Image generation failed: ${err.message}`, 'err');
  }
}

// --- Director Submit ---
async function submitDirector() {
  const imagePrompt = document.getElementById('modalDirImage').value.trim();
  const voiceText = document.getElementById('modalDirVoice').value.trim();
  const voice = document.getElementById('modalDirVoiceProfile').value;
  if (!imagePrompt || !voiceText) return;

  setModalLoading('Launching Director pipeline... This takes 10-15 minutes.');
  addLog(`Director workflow started`, 'ok');

  try {
    const resp = await fetch(`${API_BASE}/v1/workflows/director`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_prompt: imagePrompt, voiceover_text: voiceText, voice }),
    });

    const data = await resp.json();
    if (resp.ok && data.status === 'COMPLETED') {
      setModalResult(`✅ Director workflow complete!\nTask ID: ${data.task_id}\nVideo: ${data.final_video_path}`);
      addLog(`Director completed: ${data.task_id}`, 'ok');
    } else {
      setModalResult(`⚠️ Response:\n${JSON.stringify(data, null, 2)}`, data.error ? true : false);
      addLog(`Director response: ${data.status || data.error}`, data.error ? 'err' : 'warn');
    }

    // Refresh tasks
    pollTasks();
  } catch (err) {
    setModalResult(`Error: ${err.message}`, true);
    addLog(`Director failed: ${err.message}`, 'err');
  }
}
