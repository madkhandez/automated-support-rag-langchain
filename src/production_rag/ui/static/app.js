// Session State
const state = {
  sessionId: generateUUID(),
  files: [],
  chunksIndexed: 0,
  activeProvider: null,
  settings: { k: 4, threshold: 0.75, temperature: 0.0, maxTokens: 1000 },
  chatHistory: []
};

function generateUUID() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('session-id-display').innerText = `Session ID: ${state.sessionId}`;
  fetchHealthStatus();
});

// Stage Transitions
function showStage(n) {
  document.querySelectorAll('.stage').forEach(el => el.classList.remove('active'));
  document.getElementById(`stage-${n}`).classList.add('active');
  updateBreadcrumb(n);
}

function transitionTo(n) {
  const current = document.querySelector('.stage.active');
  if (current) current.style.opacity = '0';
  
  setTimeout(() => {
    showStage(n);
    setTimeout(() => {
      document.getElementById(`stage-${n}`).style.opacity = '1';
    }, 50);
  }, 300);
}

function updateBreadcrumb(stage) {
  document.querySelectorAll('.breadcrumb-dot').forEach((el, i) => {
    if (i < stage) {
      el.classList.add('active');
    } else {
      el.classList.remove('active');
    }
  });
}

// File Handling
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const ingestBtn = document.getElementById('ingest-btn');

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    addFiles(Array.from(e.dataTransfer.files));
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) {
    addFiles(Array.from(e.target.files));
  }
});

function addFiles(newFiles) {
  // Clear existing files since backend endpoint only takes 1 file in this demo
  state.files = []; 
  
  const validExtensions = ['pdf', 'txt', 'md'];
  newFiles.forEach(f => {
    const ext = f.name.split('.').pop().toLowerCase();
    if (validExtensions.includes(ext)) {
      state.files.push(f);
    } else {
      alert(`Unsupported file format: ${f.name}`);
    }
  });
  renderFileList();
  updateIngestBtn();
}

function removeFile(index) {
  state.files.splice(index, 1);
  renderFileList();
  updateIngestBtn();
}

function renderFileList() {
  const list = document.getElementById('file-list');
  list.innerHTML = '';
  state.files.forEach((file, index) => {
    const size = (file.size / 1024).toFixed(1) + ' KB';
    const div = document.createElement('div');
    div.className = 'file-item';
    div.innerHTML = `
      <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 20px;">📄</span>
        <div>
          <div style="font-weight: 500; font-size: 14px;">${file.name}</div>
          <div style="font-size: 12px; color: var(--color-ink-secondary);">${size}</div>
        </div>
      </div>
      <button class="remove-file" onclick="removeFile(${index})">&times;</button>
    `;
    list.appendChild(div);
  });
}

function updateIngestBtn() {
  ingestBtn.disabled = state.files.length === 0;
}

// Ingestion
ingestBtn.addEventListener('click', ingestDocuments);

async function ingestDocuments() {
  if (!state.files.length) return;
  
  transitionTo(2);
  
  const file = state.files[0];
  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', state.sessionId);
  
  // Fake progress animation
  const fill = document.getElementById('progress-fill');
  const stepTxt = document.getElementById('progress-step');
  const subTxt = document.getElementById('progress-sub');
  
  let progress = 0;
  const progressInterval = setInterval(() => {
    progress += 2;
    if (progress > 90) progress = 90;
    fill.style.width = `${progress}%`;
    
    if (progress < 20) { stepTxt.innerText = "Reading document"; subTxt.innerText = "Parsing contents..."; }
    else if (progress < 50) { stepTxt.innerText = "Splitting into chunks"; subTxt.innerText = "Applying recursive character splitter..."; }
    else if (progress < 80) { stepTxt.innerText = "Generating embeddings"; subTxt.innerText = "Running BGE-M3 model..."; }
    else { stepTxt.innerText = "Indexing to vector store"; subTxt.innerText = "Adding vectors to ChromaDB..."; }
  }, 100);

  try {
    const response = await fetch('/api/v1/ingest', {
      method: 'POST',
      body: formData
    });
    
    clearInterval(progressInterval);
    
    if (!response.ok) {
      throw new Error((await response.json()).detail || "Ingestion failed");
    }
    
    const data = await response.json();
    
    fill.style.width = '100%';
    stepTxt.innerText = "Finalizing...";
    subTxt.innerText = `Indexed ${data.chunk_count} chunks.`;
    
    setTimeout(() => {
      document.getElementById('success-icon').style.display = 'block';
      stepTxt.innerText = "Ready!";
      setTimeout(() => {
        transitionTo(3);
        showGreeting();
      }, 1200);
    }, 600);
    
  } catch (err) {
    clearInterval(progressInterval);
    fill.style.background = '#ff3b30';
    document.getElementById('error-container').style.display = 'block';
    document.getElementById('error-msg').innerText = err.message;
  }
}

// Chat
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatHistoryInner = document.getElementById('chat-history-inner');
const chatHistoryWrap = document.getElementById('chat-history');

chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = (chatInput.scrollHeight) + 'px';
  sendBtn.disabled = chatInput.value.trim().length === 0;
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

function showGreeting() {
  setTimeout(() => {
    appendMessage('bot', "Hi there! 👋 I've just finished reading your document.\nI'm ready to help answer any questions based on its content.\nWhat would you like to know?", [], true);
  }, 400);
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  
  chatInput.value = '';
  chatInput.style.height = 'auto';
  sendBtn.disabled = true;
  
  appendMessage('user', text);
  scrollToBottom();
  
  // Show typing indicator
  const indicator = document.createElement('div');
  indicator.className = 'chat-bubble-wrapper bot';
  indicator.id = 'typing-indicator';
  indicator.innerHTML = `
    <div class="chat-bubble bot">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  chatHistoryInner.appendChild(indicator);
  scrollToBottom();
  
  try {
    const response = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: text,
        session_id: state.sessionId,
        user_id: 'user1'
      })
    });
    
    document.getElementById('typing-indicator').remove();
    
    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || "Failed to get answer");
    }
    
    const data = await response.json();
    appendMessage('bot', data.answer, data.sources, true);
    
  } catch (err) {
    if (document.getElementById('typing-indicator')) document.getElementById('typing-indicator').remove();
    appendMessage('bot', `⚠️ Error: ${err.message}`);
  }
}

function appendMessage(role, text, sources = [], typewrite = false) {
  const wrapper = document.createElement('div');
  wrapper.className = `chat-bubble-wrapper ${role}`;
  
  let sourcesHtml = '';
  if (role === 'bot' && sources && sources.length > 0) {
    const sourcesList = sources.map(s => {
      // Handle format where sources is a list of strings vs list of objects
      let filename = s;
      let excerpt = '';
      if (typeof s === 'object') {
        filename = s.filename || 'Unknown';
        excerpt = s.excerpt || '';
      }
      return `
        <div class="source-item">
          <div class="source-filename">📄 ${filename}</div>
          ${excerpt ? `<div class="source-excerpt">"${excerpt}"</div>` : ''}
        </div>
      `;
    }).join('');
    
    const panelId = 'sources-' + Math.random().toString(36).substr(2, 9);
    sourcesHtml = `
      <button class="sources-toggle" onclick="document.getElementById('${panelId}').style.display = document.getElementById('${panelId}').style.display === 'block' ? 'none' : 'block'">
        📎 ${sources.length} Source${sources.length > 1 ? 's' : ''}
      </button>
      <div class="sources-panel" id="${panelId}">
        ${sourcesList}
      </div>
    `;
  }
  
  if (typewrite) {
    wrapper.innerHTML = `
      <div class="chat-bubble ${role}"></div>
      ${sourcesHtml}
      <div class="bubble-meta">just now</div>
    `;
    chatHistoryInner.appendChild(wrapper);
    typewriterEffect(wrapper.querySelector('.chat-bubble'), text, 15);
  } else {
    // Convert newlines to breaks
    const formattedText = text.replace(/\n/g, '<br>');
    wrapper.innerHTML = `
      <div class="chat-bubble ${role}">${formattedText}</div>
      ${sourcesHtml}
      <div class="bubble-meta">just now</div>
    `;
    chatHistoryInner.appendChild(wrapper);
    scrollToBottom();
  }
}

function typewriterEffect(element, text, speed) {
  let i = 0;
  // Convert newlines for HTML
  const lines = text.split('\n');
  let currentLine = 0;
  let charIdx = 0;
  
  function type() {
    if (currentLine < lines.length) {
      if (charIdx < lines[currentLine].length) {
        element.innerHTML += lines[currentLine].charAt(charIdx);
        charIdx++;
        setTimeout(type, speed);
      } else {
        element.innerHTML += '<br>';
        currentLine++;
        charIdx = 0;
        setTimeout(type, speed);
      }
      scrollToBottom();
    } else {
      // Remove trailing <br> if any
      if (element.innerHTML.endsWith('<br>')) {
        element.innerHTML = element.innerHTML.slice(0, -4);
      }
    }
  }
  type();
}

function scrollToBottom() {
  chatHistoryWrap.scrollTop = chatHistoryWrap.scrollHeight;
}

// Settings
const overlay = document.getElementById('settings-overlay');
const panel = document.getElementById('settings-panel');

document.getElementById('btn-open-settings').addEventListener('click', () => {
  overlay.style.display = 'block';
  setTimeout(() => panel.classList.add('open'), 10);
  fetchHealthStatus();
});

document.getElementById('btn-close-settings').addEventListener('click', closeSettings);
overlay.addEventListener('click', closeSettings);
document.getElementById('btn-apply-settings').addEventListener('click', () => {
  // Apply settings to state
  state.settings.k = parseInt(document.getElementById('inp-k').value);
  state.settings.threshold = parseFloat(document.getElementById('inp-thresh').value);
  state.settings.temperature = parseFloat(document.getElementById('inp-temp').value);
  state.settings.maxTokens = parseInt(document.getElementById('inp-tokens').value);
  closeSettings();
});

function closeSettings() {
  panel.classList.remove('open');
  setTimeout(() => overlay.style.display = 'none', 300);
}

async function fetchHealthStatus() {
  try {
    const res = await fetch('/api/v1/health');
    const data = await res.json();
    
    if (data.active_llm_provider) {
      const badge = document.getElementById('active-provider-badge');
      badge.innerText = data.active_llm_provider.toUpperCase();
      badge.style.background = '#34c759';
      badge.style.color = '#fff';
    }
    
    if (data.provider_statuses) {
      const list = document.getElementById('provider-list');
      list.innerHTML = '';
      data.provider_statuses.forEach(p => {
        const item = document.createElement('div');
        item.className = `provider-item ${p.provider === data.active_llm_provider ? 'active' : ''}`;
        
        let statusBadge = '';
        if (p.status === 'available') statusBadge = '<span style="color: #34c759; font-size: 12px; margin-left: auto;">Available</span>';
        else if (p.status === 'no_api_key') statusBadge = '<span style="color: #ff9500; font-size: 12px; margin-left: auto;">Missing Key</span>';
        else statusBadge = '<span style="color: #ff3b30; font-size: 12px; margin-left: auto;">Error</span>';
        
        item.innerHTML = `<span style="text-transform: capitalize;">${p.provider}</span> ${statusBadge}`;
        list.appendChild(item);
      });
    }
  } catch (e) {
    console.error("Failed to fetch health status", e);
  }
}
