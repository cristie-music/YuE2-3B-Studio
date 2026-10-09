import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.mjs';
import { DemucsProcessor, CONSTANTS } from './src/index.js';

const { SAMPLE_RATE, DEFAULT_MODEL_URL } = CONSTANTS;
const LOCAL_MODEL_URL = './htdemucs_embedded.onnx';

// Утилита дискового кэширования ONNX модели через IndexedDB
const DB_NAME = 'DemucsModelCacheDB';
const STORE_NAME = 'models';
const CACHE_KEY = 'htdemucs_embedded_onnx_v1';

function openModelDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getCachedModelBuffer() {
  try {
    const db = await openModelDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(CACHE_KEY);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch (e) {
    return null;
  }
}

async function saveModelBufferToCache(buffer) {
  try {
    const db = await openModelDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    store.put(buffer, CACHE_KEY);
  } catch (e) {
    console.warn('Не удалось сохранить модель в IndexedDB:', e);
  }
}

let processor = null;
let audioContext = null;
let audioBuffer = null;
let isProcessing = false;

// DOM элементы
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const progressFill = document.getElementById('progressFill');
const status = document.getElementById('status');
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
  let backend = 'wasm';

  if ('gpu' in navigator) {
    try {
      const gpuAdapter = await navigator.gpu.requestAdapter();
      if (gpuAdapter) {
        backend = 'webgpu';
      }
    } catch (e) {
      console.log('WebGPU не доступен:', e);
    }
  }

  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

  if (backend === 'webgpu') {
    ort.env.webgpu = ort.env.webgpu || {};
    ort.env.webgpu.powerPreference = 'high-performance';
    backendBadge.textContent = 'WebGPU (Аппаратный GPU)';
    backendBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-mono bg-pink-500/20 text-pink-300 border border-pink-500/30';
  } else {
    const threads = navigator.hardwareConcurrency || 4;
    backendBadge.textContent = `WASM Multi-thread (${threads} потоков)`;
    backendBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-mono bg-amber-500/20 text-amber-300 border border-amber-500/30';
  }

  processor = new DemucsProcessor({
    ort,
    onProgress: ({ progress, currentSegment, totalSegments }) => {
      progressFill.style.width = (5 + progress * 90) + '%';
      const elapsed = (Date.now() - processStartTime) / 1000;
      statElapsed.textContent = formatTime(elapsed);
      statSegment.textContent = `${currentSegment}/${totalSegments}`;

      if (currentSegment > 0 && audioBuffer) {
        const processedDuration = (currentSegment / totalSegments) * audioBuffer.duration;
        const speed = processedDuration / elapsed;
        statSpeed.textContent = speed.toFixed(2) + 'x';
        const remainingSegments = totalSegments - currentSegment;
        const avgTimePerSegment = elapsed / currentSegment;
        const eta = remainingSegments * avgTimePerSegment;
        statETA.textContent = formatTime(eta);
      }
    },
    onLog: log,
    onDownloadProgress: (loaded, total) => {
      const percent = ((loaded / total) * 100).toFixed(1);
      const loadedMB = (loaded / 1024 / 1024).toFixed(1);
      const totalMB = (total / 1024 / 1024).toFixed(1);
      status.textContent = `Загрузка весов модели... ${loadedMB}MB / ${totalMB}MB (${percent}%)`;
      progressFill.style.width = (loaded / total * 100) + '%';
    }
  });

  status.textContent = 'Проверка кэша модели...';

  try {
    const cachedBuffer = await getCachedModelBuffer();
    if (cachedBuffer) {
      cacheBadge.textContent = 'Кэш: Дисковый (IndexedDB)';
      cacheBadge.classList.remove('hidden');
      status.textContent = 'Инициализация модели из локального кэша...';
      await processor.loadModel(cachedBuffer);
    } else {
      cacheBadge.textContent = 'Кэш: Скачивание (~172MB)...';
      cacheBadge.classList.remove('hidden');
      status.textContent = 'Скачивание модели (~172MB)...';
      
      let fetchedBuffer = null;
      try {
        const resp = await fetch(DEFAULT_MODEL_URL);
        fetchedBuffer = await resp.arrayBuffer();
      } catch (err) {
        const resp = await fetch(LOCAL_MODEL_URL);
        fetchedBuffer = await resp.arrayBuffer();
      }

      await saveModelBufferToCache(fetchedBuffer);
      cacheBadge.textContent = 'Кэш: Сохранен локально';
      await processor.loadModel(fetchedBuffer);
    }

    status.textContent = 'Готово. Ожидание аудиофайла...';
    progressFill.style.width = '0%';
  } catch (e) {
    status.textContent = 'Ошибка загрузки модели: ' + e.message;
    console.error('Ошибка инициализации Demucs:', e);
  }

  audioContext = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: SAMPLE_RATE
  });

  const urlParams = new URLSearchParams(window.location.search);
  const audioUrl = urlParams.get('audioUrl');
  if (audioUrl) {
    await loadAudioFromUrl(audioUrl);
  }
}

async function loadAudioFromUrl(url) {
  try {
    status.textContent = 'Загрузка аудио дорожки с сервера...';
    const fileName = decodeURIComponent(url.split('/').pop() || 'track.flac');
    audioFileName.textContent = fileName;

    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const arrayBuffer = await response.arrayBuffer();
    audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const duration = audioBuffer.duration.toFixed(1);
    status.textContent = `Загружен: ${fileName} (${duration} сек) — Автоматический запуск экстракции...`;
    processBtn.disabled = false;

    setTimeout(() => startProcessing(), 600);
  } catch (e) {
    status.textContent = 'Не удалось загрузить аудио: ' + e.message;
    console.error('Ошибка декодирования аудио:', e);
  }
}

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('border-pink-500', 'bg-pink-500/5');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('border-pink-500', 'bg-pink-500/5');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('border-pink-500', 'bg-pink-500/5');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('audio/')) {
    handleFile(file);
  }
});
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  audioFileName.textContent = file.name;
  status.textContent = 'Чтение аудиофайла...';

  try {
    const arrayBuffer = await file.arrayBuffer();
    audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const duration = audioBuffer.duration.toFixed(1);
    status.textContent = `Загружен файл (${duration} сек) — Готов к разделению`;
    processBtn.disabled = false;
  } catch (e) {
    status.textContent = 'Ошибка чтения аудио: ' + e.message;
  }
}

processBtn.addEventListener('click', startProcessing);

async function startProcessing() {
  if (!audioBuffer || !processor || isProcessing) return;

  isProcessing = true;
  processBtn.disabled = true;
  processBtn.textContent = 'Обработка...';
  results.classList.add('hidden');
  processStartTime = Date.now();
  statusDetail.innerHTML = '';
  statusDetail.classList.remove('hidden');
  statsRow.classList.remove('hidden');

  try {
    log('Init', 'Старт разделения на 4 стэма...');
    status.textContent = 'Подготовка аудиосигнала...';
    progressFill.style.width = '2%';

    let leftChannel = audioBuffer.getChannelData(0);
    let rightChannel = audioBuffer.numberOfChannels > 1
      ? audioBuffer.getChannelData(1)
      : leftChannel;

    if (audioBuffer.sampleRate !== SAMPLE_RATE) {
      log('Resample', `${audioBuffer.sampleRate}Hz → ${SAMPLE_RATE}Hz`);
      const ratio = SAMPLE_RATE / audioBuffer.sampleRate;
      const newLength = Math.floor(leftChannel.length * ratio);
      const newLeft = new Float32Array(newLength);
      const newRight = new Float32Array(newLength);

      for (let i = 0; i < newLength; i++) {
        const srcIdx = i / ratio;
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(idx0 + 1, leftChannel.length - 1);
        const frac = srcIdx - idx0;
        newLeft[i] = leftChannel[idx0] * (1 - frac) + leftChannel[idx1] * frac;
        newRight[i] = rightChannel[idx0] * (1 - frac) + rightChannel[idx1] * frac;
      }

      leftChannel = newLeft;
      rightChannel = newRight;
    }

    status.textContent = 'Синтез стэмов через нейросеть...';
    const separatedTracks = await processor.separate(leftChannel, rightChannel);
    displayResults(separatedTracks);

    const totalTime = ((Date.now() - processStartTime) / 1000).toFixed(1);
    const speedRatio = (audioBuffer.duration / parseFloat(totalTime)).toFixed(2);

    log('Complete', `Успешно завершено за ${totalTime} сек (${speedRatio}x от реального времени)`);
    status.textContent = `Готово! Все 4 стэма успешно извлечены за ${totalTime} сек`;
    progressFill.style.width = '100%';

  } catch (e) {
    status.textContent = 'Сбой обработки: ' + e.message;
    console.error('Ошибка инференса Demucs:', e);
  }

  isProcessing = false;
  processBtn.disabled = false;
  processBtn.textContent = 'Разделить снова';
}

let trackUrls = {};

function displayResults(tracks) {
  trackList.innerHTML = '';
  trackUrls = {};

  const TRACK_CONFIG = {
    drums: { icon: '🥁', label: 'Ударные (Drums)', color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20' },
    bass: { icon: '🎸', label: 'Бас (Bass)', color: 'text-purple-400', bg: 'bg-purple-500/10', border: 'border-purple-500/20' },
    other: { icon: '🎹', label: 'Инструментал (Other)', color: 'text-blue-400', bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
    vocals: { icon: '🎤', label: 'Вокал (Vocals)', color: 'text-pink-400', bg: 'bg-pink-500/10', border: 'border-pink-500/20' }
  };

  for (const [name, track] of Object.entries(tracks)) {
    const config = TRACK_CONFIG[name] || { icon: '🎵', label: name, color: 'text-zinc-400', bg: 'bg-white/5', border: 'border-white/10' };
    const trackBuffer = audioContext.createBuffer(2, track.left.length, SAMPLE_RATE);
    trackBuffer.getChannelData(0).set(track.left);
    trackBuffer.getChannelData(1).set(track.right);

    const audioBlob = audioBufferToWav(trackBuffer);
    const audioUrl = URL.createObjectURL(audioBlob);
    const trackId = `track-${name}`;
    const baseName = (audioFileName.textContent || 'track').replace(/\.[^/.]+$/, "");
    const saveName = `${baseName}_${name}.wav`;

    trackUrls[name] = { url: audioUrl, filename: saveName };

    const trackDiv = document.createElement('div');
    trackDiv.className = 'bg-suno-card rounded-xl border border-white/5 p-3.5 flex items-center justify-between gap-4 hover:border-white/10 transition-colors';
    trackDiv.innerHTML = `
      <div class="flex items-center gap-3 w-48 flex-shrink-0">
        <div class="w-10 h-10 rounded-lg ${config.bg} border ${config.border} flex items-center justify-center text-lg">
          ${config.icon}
        </div>
        <div class="overflow-hidden">
          <div class="text-xs font-bold ${config.color} truncate">${config.label}</div>
          <div class="text-[10px] text-zinc-500 font-mono">${formatTime(trackBuffer.duration)}</div>
        </div>
      </div>

      <div class="flex-1 flex items-center gap-3">
        <button id="play-${trackId}" onclick="togglePlay('${trackId}')" class="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 active:scale-95 transition-transform shadow flex-shrink-0">
          <svg class="w-3.5 h-3.5 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        </button>

        <div id="progress-bg-${trackId}" onclick="seekTrack(event, '${trackId}')" class="flex-1 h-1.5 bg-zinc-800 rounded-lg cursor-pointer overflow-hidden relative">
          <div id="progress-${trackId}" class="h-full bg-gradient-to-r from-pink-500 to-purple-600 w-0 transition-all"></div>
        </div>

        <span id="time-${trackId}" class="text-[11px] font-mono text-zinc-400 w-24 text-right">0:00 / ${formatTime(trackBuffer.duration)}</span>
      </div>

      <a href="${audioUrl}" download="${saveName}" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 hover:bg-white/10 text-zinc-200 border border-white/5 transition-colors flex items-center gap-1.5 flex-shrink-0">
        <span>↓</span> WAV
      </a>

      <audio id="audio-${trackId}" src="${audioUrl}" preload="metadata"></audio>
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

function audioBufferToWav(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const samples = buffer.length;
  const dataSize = samples * blockAlign;
  const bufferSize = 44 + dataSize;

  const arrayBuffer = new ArrayBuffer(bufferSize);
  const view = new DataView(arrayBuffer);

  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, bufferSize - 8, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  const channels = [];
  for (let c = 0; c < numChannels; c++) {
    channels.push(buffer.getChannelData(c));
  }

  let offset = 44;
  for (let i = 0; i < samples; i++) {
    for (let c = 0; c < numChannels; c++) {
      const sample = Math.max(-1, Math.min(1, channels[c][i]));
      const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      view.setInt16(offset, intSample, true);
      offset += 2;
    }
  }

  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

init();