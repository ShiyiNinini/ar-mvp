const demoStage = document.querySelector('#demo-stage');
const video = document.querySelector('#tracking-video');
const liveFrame = document.querySelector('#live-frame');
const playButton = document.querySelector('#screen-play');
const state = document.querySelector('#demo-state');
const detail = document.querySelector('#connection-detail');
const disconnect = document.querySelector('#disconnect-camera');
const recording = document.querySelector('#play-recording');
const pauseRecording = document.querySelector('#pause-recording');
let mode = 'idle';
let generation = 0;
let frameURL;
let busy = false;

async function api(path, method = 'GET') {
  const response = await fetch(`/api/${path}`, {
    method, headers: { 'X-HandTrack': '1' }, cache: 'no-store',
    signal: AbortSignal.timeout(10000),
  });
  if (!(response.headers.get('content-type') || '').includes('application/json')) {
    throw new Error('Open this page through the local camera server: python3 scripts/live_server.py');
  }
  const value = await response.json();
  if (!response.ok) throw new Error(value.message || 'Camera service unavailable');
  return value;
}

function clearScreen() {
  generation++;
  video.pause();
  pauseRecording.disabled = true;
  pauseRecording.textContent = 'Pause';
  demoStage.classList.remove('has-video', 'has-live', 'is-playing');
  liveFrame.removeAttribute('src');
  if (frameURL) URL.revokeObjectURL(frameURL);
  frameURL = undefined;
}

function showError(error) {
  clearScreen();
  mode = 'error';
  state.textContent = 'Camera unavailable';
  detail.textContent = error.message || 'Start the local camera server and check the T265 connection.';
  document.querySelector('#tracking-state').textContent = 'CAMERA OFFLINE';
}

async function pollLive(id) {
  try {
    const info = await api('status');
    if (id !== generation || mode !== 'live') return;
    if (info.state === 'error') throw new Error(info.message);
    if (info.state === 'idle') throw new Error('Camera disconnected. Select Connect camera to restart.');
    if (info.state === 'live') {
      const response = await fetch(`/api/frame?t=${Date.now()}`, { cache: 'no-store', signal: AbortSignal.timeout(10000) });
      if (!response.ok) throw new Error('Camera frames unavailable. Disconnect and reconnect.');
      const blob = await response.blob();
      if (id !== generation || mode !== 'live') return;
      const nextURL = URL.createObjectURL(blob);
      // Decode before swapping, keeping the previous frame visible during download.
      const nextImage = new Image();
      nextImage.src = nextURL;
      try { await nextImage.decode(); } catch (error) { URL.revokeObjectURL(nextURL); throw error; }
      if (id !== generation || mode !== 'live') { URL.revokeObjectURL(nextURL); return; }
      liveFrame.src = nextURL;
      if (frameURL) URL.revokeObjectURL(frameURL);
      frameURL = nextURL;
      demoStage.classList.add('has-live');
      state.textContent = 'Live camera · T265 stereo';
      detail.textContent = `${info.hands} hand${info.hands === 1 ? '' : 's'} tracked · ${info.fps} FPS`;
    } else {
      state.textContent = 'Starting camera and tracking models…';
      detail.textContent = 'Connecting to T265 · Loading models';
    }
    window.setTimeout(() => pollLive(id), info.state === 'live' ? 60 : 700);
  } catch (error) {
    if (id === generation) showError(error);
  }
}

async function startDemo() {
  if (window.location.protocol === 'file:') {
    window.location.assign('http://127.0.0.1:4174/#demo');
    return;
  }
  document.querySelector('#demo').scrollIntoView({ behavior: 'smooth', block: 'center' });
  if (busy || mode === 'live') return;
  busy = true;
  clearScreen();
  mode = 'live';
  demoStage.classList.add('is-playing');
  state.textContent = 'Connecting camera…';
  disconnect.disabled = false;
  try {
    const info = await api('start', 'POST');
    if (info.state === 'error') throw new Error(info.message);
    pollLive(generation);
  } catch (error) { showError(error); }
  finally { busy = false; }
}

async function stopCamera() {
  await api('stop', 'POST');
  clearScreen();
  mode = 'idle';
  disconnect.disabled = true;
  state.textContent = 'Camera disconnected';
  detail.textContent = 'T265 camera · Ready to connect';
}

disconnect.addEventListener('click', async () => {
  if (busy) return;
  busy = true;
  try { await stopCamera(); } catch (error) { showError(error); }
  finally { busy = false; }
});
recording.addEventListener('click', async () => {
  if (busy) return;
  busy = true;
  try {
    if (mode === 'live' || mode === 'error') await stopCamera();
    clearScreen();
    mode = 'recording';
    video.currentTime = 0;
    await video.play();
    demoStage.classList.add('is-playing', 'has-video');
    state.textContent = 'Recorded tracking demo';
    detail.textContent = 'Recorded footage';
    pauseRecording.disabled = false;
  } catch (error) {
    clearScreen();
    mode = 'idle';
    state.textContent = 'Playback unavailable';
    detail.textContent = error.message;
  } finally { busy = false; }
});
pauseRecording.addEventListener('click', async () => {
  if (busy || mode !== 'recording') return;
  busy = true;
  try {
    if (video.paused) {
      await video.play();
      pauseRecording.textContent = 'Pause';
      state.textContent = 'Recorded tracking demo';
      detail.textContent = 'Recorded footage';
    } else {
      video.pause();
      pauseRecording.textContent = 'Resume';
      state.textContent = 'Recording paused';
      detail.textContent = 'Paused · Resume playback or connect camera';
    }
  } catch (error) {
    detail.textContent = error.message;
  } finally { busy = false; }
});
video.addEventListener('ended', () => {
  if (mode !== 'recording') return;
  clearScreen();
  mode = 'idle';
  state.textContent = 'Replay ready';
});
document.querySelectorAll('[data-demo-trigger]').forEach(button => button.addEventListener('click', startDemo));
playButton.addEventListener('click', startDemo);

// Reattach to an already running local session after refreshing the page.
(async () => {
  if (window.location.protocol === 'file:') {
    detail.textContent = 'Connect camera opens the local camera service. Recordings can play here.';
    return;
  }
  const initialGeneration = generation;
  try {
    const info = await api('status');
    if (generation !== initialGeneration || busy || mode !== 'idle') return;
    if (info.state === 'live' || info.state === 'starting') {
      mode = 'live';
      disconnect.disabled = false;
      demoStage.classList.add('is-playing');
      pollLive(generation);
    }
  } catch (_) { /* Static preview remains usable without a camera service. */ }
})();
