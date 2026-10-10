const status = document.getElementById('status');
const processBtn = document.getElementById('processBtn');
const progressFill = document.getElementById('progressFill');
const results = document.getElementById('results');
const trackList = document.getElementById('trackList');
const backendBadge = document.getElementById('backendBadge');
const cacheBadge = document.getElementById('cacheBadge');
const audioFileName = document.getElementById('audioFileName');
const statusDetail = document.getElementById('statusDetail');
const statsRow = document.getElementById('statsRow');
const statElapsed = document.getElementById('statElapsed');
const statSegment = document.getElementById('statSegment');
const statSpeed = document.getElementById('statSpeed');
const statETA = document.getElementById('statETA');

let trackUrls = {};
let processStartTime = null;

function log(phase, message) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('ru-RU', { hour12: false });
  const logLine = document.createElement('div');
  logLine.className = 'text-zinc-400 py-0.5 border-b border-white/5 last:border-0 font-mono text-[11px]';
  logLine.innerHTML = `<span class="text-pink-400">[${timeStr}]</span> <span class="text-purple-400">[${phase}]</span> ${message}`;
  statusDetail.appendChild(logLine);
  statusDetail.scrollTop = statusDetail.scrollHeight;
}

function formatTime(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '--:--';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

async function init() {
  backendBadge.textContent = 'Colab GPU (Серверный Demucs)';
  backendBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-mono bg-pink-500/20 text-pink-300 border border-pink-500/30';
  
  cacheBadge.textContent = 'Режим: Серверная обработка';
  cacheBadge.classList.remove('hidden');

  const urlParams = new URLSearchParams(window.location.search);
  const audioUrl = urlParams.get('audioUrl');
  if (audioUrl) {
    const rawName = decodeURIComponent(audioUrl.split('/').pop() || 'track.flac');
    audioFileName.textContent = rawName;
    status.textContent = `Загружен: ${rawName} — Готов к разделению на сервере`;
    processBtn.disabled = false;
    processBtn.onclick = () => runServerSeparation(audioUrl);
    setTimeout(() => runServerSeparation(audioUrl), 500);
  } else {
    status.textContent = 'Ожидание аудиофайла...';
  }
}

async function runServerSeparation(audioUrl) {
  processBtn.disabled = true;
  processBtn.textContent = 'Обработка на GPU...';
  statusDetail.classList.remove('hidden');
  statsRow.classList.remove('hidden');
  results.classList.add('hidden');
  progressFill.style.width = '20%';
  processStartTime = Date.now();

  log('GPU-Start', 'Отправка задачи на сервер Colab...');
  status.textContent = 'Нейросетевое разделение на GPU сервера...';

  try {
    const timer = setInterval(() => {
      const elapsed = (Date.now() - processStartTime) / 1000;
      statElapsed.textContent = formatTime(elapsed);
      progressFill.style.width = Math.min(85, 20 + elapsed * 8) + '%';
    }, 500);

    const resp = await fetch('/api/demucs/separate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_url: audioUrl })
    });

    clearInterval(timer);
    const data = await resp.json();

    if (!data.success) {
      throw new Error(data.error || 'Ошибка сервера');
    }

    progressFill.style.width = '100%';
    const totalSec = ((Date.now() - processStartTime) / 1000).toFixed(1);
    statElapsed.textContent = formatTime(parseFloat(totalSec));
    statSegment.textContent = '4/4';
    statSpeed.textContent = 'GPU Ultra';
    statETA.textContent = '0:00';

    log('Complete', `Стэмы успешно сформированы на сервере за ${totalSec} сек!`);
    status.textContent = `Готово за ${totalSec} сек! Все 4 дорожки доступны для прослушивания и загрузки.`;

    displayServerResults(data.stems);
  } catch (err) {
    status.textContent = 'Ошибка: ' + err.message;
    log('Error', err.message);
  } finally {
    processBtn.disabled = false;
    processBtn.textContent = 'Разделить снова';
  }
}

function displayServerResults(stems) {
  trackList.innerHTML = '';
  trackUrls = {};

  const TRACK_CONFIG = {
    drums: { icon: '🥁', label: 'Ударные (Drums)', color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20' },
    bass: { icon: '🎸', label: 'Бас (Bass)', color: 'text-purple-400', bg: 'bg-purple-500/10', border: 'border-purple-500/20' },
    other: { icon: '🎹', label: 'Инструментал (Other)', color: 'text-blue-400', bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
    vocals: { icon: '🎤', label: 'Вокал (Vocals)', color: 'text-pink-400', bg: 'bg-pink-500/10', border: 'border-pink-500/20' }
  };

  for (const [name, url] of Object.entries(stems)) {
    const config = TRACK_CONFIG[name] || { icon: '🎵', label: name, color: 'text-zinc-400', bg: 'bg-white/5', border: 'border-white/10' };
    const trackId = `track-${name}`;
    const baseName = (audioFileName.textContent || 'track').replace(/\.[^/.]+$/, "");
    const saveName = `${baseName}_${name}.wav`;

    trackUrls[name] = { url, filename: saveName };

    const trackDiv = document.createElement('div');
    trackDiv.className = 'bg-suno-card rounded-xl border border-white/5 p-3.5 flex items-center justify-between gap-4 hover:border-white/10 transition-colors';
    trackDiv.innerHTML = `
      <div class="flex items-center gap-3 w-48 flex-shrink-0">
        <div class="w-10 h-10 rounded-lg ${config.bg} border ${config.border} flex items-center justify-center text-lg">
          ${config.icon}
        </div>
        <div class="overflow-hidden">
          <div class="text-xs font-bold ${config.color} truncate">${config.label}</div>
          <div class="text-[10px] text-zinc-500 font-mono">44.1 kHz WAV</div>
        </div>
      </div>

      <div class="flex-1 flex items-center gap-3">
        <button id="play-${trackId}" onclick="togglePlay('${trackId}')" class="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 active:scale-95 transition-transform shadow flex-shrink-0">
          <svg class="w-3.5 h-3.5 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        </button>

        <div id="progress-bg-${trackId}" onclick="seekTrack(event, '${trackId}')" class="flex-1 h-1.5 bg-zinc-800 rounded-lg cursor-pointer overflow-hidden relative">
          <div id="progress-${trackId}" class="h-full bg-gradient-to-r from-pink-500 to-purple-600 w-0 transition-all"></div>
        </div>

        <span id="time-${trackId}" class="text-[11px] font-mono text-zinc-400 w-24 text-right">0:00</span>
      </div>

      <a href="${url}" download="${saveName}" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 hover:bg-white/10 text-zinc-200 border border-white/5 transition-colors flex items-center gap-1.5 flex-shrink-0">
        <span>↓</span> WAV
      </a>

      <audio id="audio-${trackId}" src="${url}" preload="metadata"></audio>
    `;

    trackList.appendChild(trackDiv);

    const audio = document.getElementById(`audio-${trackId}`);
    audio.addEventListener('timeupdate', () => updateProgress(trackId, audio));
    audio.addEventListener('ended', () => resetPlayer(trackId));
  }

  results.classList.remove('hidden');
}

window.downloadAllStems = function() {
  const entries = Object.values(trackUrls);
  let index = 0;

  function downloadNext() {
    if (index >= entries.length) return;
    const item = entries[index];
    const a = document.createElement('a');
    a.href = item.url;
    a.download = item.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    index++;
    setTimeout(downloadNext, 600);
  }

  downloadNext();
};

window.togglePlay = function(trackId) {
  const audio = document.getElementById(`audio-${trackId}`);
  const playBtn = document.getElementById(`play-${trackId}`);

  document.querySelectorAll('audio').forEach(a => {
    if (a.id !== `audio-${trackId}` && !a.paused) {
      a.pause();
      const otherId = a.id.replace('audio-', '');
      resetPlayer(otherId);
    }
  });

  if (audio.paused) {
    audio.play();
    playBtn.innerHTML = `<svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
  } else {
    audio.pause();
    playBtn.innerHTML = `<svg class="w-3.5 h-3.5 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
  }
};

window.seekTrack = function(event, trackId) {
  const audio = document.getElementById(`audio-${trackId}`);
  const progressBg = document.getElementById(`progress-bg-${trackId}`);
  const rect = progressBg.getBoundingClientRect();
  const percent = (event.clientX - rect.left) / rect.width;
  audio.currentTime = percent * audio.duration;
};

function updateProgress(trackId, audio) {
  const progress = document.getElementById(`progress-${trackId}`);
  const timeDisplay = document.getElementById(`time-${trackId}`);
  const percent = (audio.currentTime / audio.duration) * 100;
  progress.style.width = `${percent}%`;
  timeDisplay.textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
}

function resetPlayer(trackId) {
  const playBtn = document.getElementById(`play-${trackId}`);
  const progress = document.getElementById(`progress-${trackId}`);
  if (playBtn) playBtn.innerHTML = `<svg class="w-3.5 h-3.5 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
  if (progress) progress.style.width = '0%';
}

init();