const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

// --- UI Constants & Configuration ---
const RULER_HEIGHT = 24;
const BLOCK_HEIGHT = 160; // Increased to make the image timeline area much taller
const AUDIO_TRACK_HEIGHT = 80;
const AUDIO_LANE_HEIGHT = 56;
const CANVAS_HEIGHT = RULER_HEIGHT + BLOCK_HEIGHT + AUDIO_TRACK_HEIGHT;
const HANDLE_HIT_PX = 14;
const MIN_SEGMENT_LENGTH = 6;
const MAX_THUMBNAIL_DIM = 512; // Increased to maintain quality for taller images
const SOURCE_VIDEO_DEFAULT_GUIDE_FRAMES = 9;
const SOURCE_VIDEO_MAX_GUIDE_FRAMES = 65;
const PRIVACY_SCHEMA = "whatdreamscost.ltx-director";
const EMPTY_TIMELINE_JSON = "{\"segments\":[],\"audioSegments\":[]}";

const HIDDEN_WIDGET_NAMES = ["timeline_data", "local_prompts", "segment_lengths", "guide_strength", "audio_data", "use_custom_audio", "use_global_prompt", "hide_timeline_images_prompts", "privacy_mode", "privacy_payload"];

function hideWidget(w) {
  if (!w) return;
  if (!w._origType && w.type !== "hidden") w._origType = w.type;
  w.hidden = true;
  if (!w.options) w.options = {};
  w.options.hidden = true;
  w.computeSize = () => [0, 0];
  if (w.element) w.element.style.display = "none";
}

function widgetBoolValue(value) {
  return value === true || value === "true";
}

function setWidgetBoolValue(widget, value) {
  if (!widget) return;
  widget.value = value ? "true" : "false";
  if (widget.callback) {
    try { widget.callback(widget.value, app.canvas, widget.node, null, null); } catch (e) { }
  }
}

function parseJsonObject(value) {
  if (value && typeof value === "object") return value;
  if (typeof value !== "string" || !value.trim()) return {};
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function isEncryptedPrivacyPayload(value) {
  const parsed = parseJsonObject(value);
  return parsed.encrypted === true && parsed.schema === PRIVACY_SCHEMA && parsed.algorithm === "AES-256-GCM";
}

function timelineValueHasSavedData(value) {
  const parsed = parseJsonObject(value);
  return Array.isArray(parsed.segments) || Array.isArray(parsed.audioSegments);
}

function repairLegacyPrivacyWidgetShift(node) {
  const privacyModeWidget = node.widgets?.find(w => w.name === "privacy_mode");
  const privacyPayloadWidget = node.widgets?.find(w => w.name === "privacy_payload");
  const hideTimelineWidget = node.widgets?.find(w => w.name === "hide_timeline_images_prompts");
  const timelineDataWidget = node.widgets?.find(w => w.name === "timeline_data");

  if (!privacyModeWidget || !privacyPayloadWidget) return;
  if (!widgetBoolValue(privacyModeWidget.value)) return;
  if (isEncryptedPrivacyPayload(privacyPayloadWidget.value)) return;
  if (!timelineValueHasSavedData(timelineDataWidget?.value)) return;

  if (hideTimelineWidget) hideTimelineWidget.value = privacyModeWidget.value;
  privacyModeWidget.value = "false";
  privacyPayloadWidget.value = "";
}

async function fetchPrivacyJson(endpoint, payload = null) {
  const options = payload
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
    : undefined;
  const response = await api.fetchApi(`/wdc_ltx_director/privacy/${endpoint}`, options);
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch (err) {
    throw new Error(text || response.statusText || `HTTP ${response.status}`);
  }
  if (!response.ok || data.ok === false || data.error) throw new Error(data.error || response.statusText);
  return data;
}

function encryptPrivacyStateSync(state) {
  const xhr = new XMLHttpRequest();
  xhr.open("POST", api.apiURL("/wdc_ltx_director/privacy/encrypt"), false);
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.send(JSON.stringify({ state }));
  let data = {};
  try {
    data = xhr.responseText ? JSON.parse(xhr.responseText) : {};
  } catch {
    throw new Error(xhr.responseText || xhr.statusText || `HTTP ${xhr.status}`);
  }
  if (xhr.status < 200 || xhr.status >= 300 || data.ok === false || data.error) {
    throw new Error(data.error || xhr.statusText || `HTTP ${xhr.status}`);
  }
  return data.envelope;
}

function serializedWidgetIndex(node, name) {
  const widgets = node.widgets || [];
  let index = 0;
  for (const widget of widgets) {
    if (widget.type === "button" || widget.serialize === false) continue;
    if (widget.name === name) return index;
    index += 1;
  }
  return -1;
}

function setSerializedWidgetValue(info, node, name, value) {
  if (!Array.isArray(info.widgets_values)) return;
  const index = serializedWidgetIndex(node, name);
  if (index >= 0 && index < info.widgets_values.length) info.widgets_values[index] = value;
}

function applyGlobalPromptWidgetVisibility(globalPromptWidget, isVisible) {
  if (!globalPromptWidget) return;
  if (!globalPromptWidget.options) globalPromptWidget.options = {};
  globalPromptWidget.options.hidden = !isVisible;

  if (isVisible) {
    delete globalPromptWidget.computeSize;
    globalPromptWidget.hidden = false;
    if (globalPromptWidget.element) globalPromptWidget.element.style.display = "";
  } else {
    globalPromptWidget.computeSize = () => [0, 0];
    globalPromptWidget.hidden = true;
    if (globalPromptWidget.element) globalPromptWidget.element.style.display = "none";
  }
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function clampVolume(value) {
  const parsed = parseFloat(value);
  if (!Number.isFinite(parsed)) return 1.0;
  return clamp(parsed, 0, 2);
}

function analyzeAudioBufferLoudness(audioBuffer) {
  const targetRms = Math.pow(10, -18 / 20);
  const peakCeiling = Math.pow(10, -1 / 20);
  let sumSquares = 0;
  let sampleTotal = 0;
  let peak = 0;

  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
    const samples = audioBuffer.getChannelData(channel);
    sampleTotal += samples.length;
    for (let i = 0; i < samples.length; i++) {
      const value = Math.abs(samples[i]);
      sumSquares += value * value;
      if (value > peak) peak = value;
    }
  }

  if (sampleTotal <= 0 || peak <= 0 || sumSquares <= 0) {
    return { rms: 0, peak, volume: 1.0 };
  }

  const rms = Math.sqrt(sumSquares / sampleTotal);
  const rmsGain = targetRms / rms;
  const peakGain = peakCeiling / peak;
  const volume = clampVolume(Math.min(rmsGain, peakGain));

  return { rms, peak, volume };
}

function normalizeAudioLane(value) {
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
}

function audioRangesOverlap(startA, lengthA, startB, lengthB) {
  return startA < startB + lengthB && startB < startA + lengthA;
}

function findFreeAudioLane(audioSegments, start, length, ignoreId = null, preferredLane = null) {
  const isLaneFree = (lane) => !audioSegments.some((seg) => (
    seg.id !== ignoreId
    && normalizeAudioLane(seg.lane) === lane
    && audioRangesOverlap(start, length, seg.start || 0, seg.length || 1)
  ));

  if (preferredLane !== null && preferredLane !== undefined) {
    const lane = normalizeAudioLane(preferredLane);
    if (isLaneFree(lane)) return lane;
  }

  let lane = 0;
  while (!isLaneFree(lane)) lane += 1;
  return lane;
}

function assignMissingAudioLanes(audioSegments) {
  const assigned = [];
  const sorted = [...audioSegments].sort((a, b) => (a.start || 0) - (b.start || 0));

  for (const seg of sorted) {
    const hasLane = Number.isFinite(parseInt(seg.lane, 10));
    if (hasLane) {
      seg.lane = normalizeAudioLane(seg.lane);
    } else {
      seg.lane = findFreeAudioLane(assigned, seg.start || 0, seg.length || 1);
    }
    assigned.push(seg);
  }

  return audioSegments;
}

function timelineNodeInnerWidth(node) {
  return Math.max((node?.size?.[0] || 390) - 28, 320);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[char]);
}

async function fetchTimelineImageJson(url, options) {
  const response = await api.fetchApi(url, options);
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch (err) {
    throw new Error(text || response.statusText || `HTTP ${response.status}`);
  }
  if (!response.ok || data.error) throw new Error(data.error || response.statusText);
  return data;
}

async function clearTimelineThumbnailCache() {
  const data = await fetchTimelineImageJson("/wdc_timeline_images/thumb-cache/clear", { method: "POST" });
  return data.cacheBust || String(Date.now());
}

async function fetchTimelineAudioJson(url, options) {
  const response = await api.fetchApi(url, options);
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch (err) {
    throw new Error(text || response.statusText || `HTTP ${response.status}`);
  }
  if (!response.ok || data.error) throw new Error(data.error || response.statusText);
  return data;
}

// --- Modern Dark/Grey UI CSS (ComfyUI Match) ---
const STYLES = `
  .pr-wrapper {
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 100%;
    height: 100%;
    box-sizing: border-box;
    padding-bottom: 4px;
  }
  .pr-wrapper.drag-active {
    outline: 2px dashed #888;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
  }
  .pr-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2px 0px;
    flex-wrap: wrap;
    gap: 6px;
  }
  .pr-actions {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .pr-btn {
    background: #222;
    color: #e0e0e0;
    border: 1px solid #111;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
  }
  .pr-btn:hover {
    background: #333;
    border-color: #555;
  }
  .pr-btn:disabled {
    opacity: 0.42;
    cursor: not-allowed;
  }
  .pr-btn:disabled:hover {
    background: #222;
    border-color: #111;
  }
  .pr-btn-danger:hover {
    background: #4a1515;
    border-color: #cc4444;
    color: #ffaaaa;
  }
  .pr-canvas {
    border-radius: 6px;
    border: 1px solid #111;
    background: #2a2a2a;
    cursor: pointer;
    width: 100%;
    outline: none;
    display: block; /* Ensure no inline baseline gaps */
  }
  .pr-prop-container {
    display: flex;
    flex-direction: column;
    width: 100%;
    flex-grow: 1; /* Automatically scales to fill node height */
    min-height: 80px;
  }
  .pr-prompt-area {
    width: 100%;
    height: 100%;
    background: #222;
    color: #e0e0e0;
    border: 1px solid #111;
    border-radius: 6px;
    padding: 8px;
    resize: none; /* Removed the manual resize corner handle */
    font-size: 12px;
    line-height: 1.4;
    box-sizing: border-box;
    outline: none;
    transition: border-color 0.2s ease;
  }
  .pr-prompt-area:focus {
    border-color: #888;
  }
  .pr-privacy-hidden-text {
    color: transparent !important;
    caret-color: transparent !important;
    text-shadow: none !important;
    -webkit-text-fill-color: transparent !important;
  }
  .pr-privacy-hidden-text::placeholder {
    color: transparent !important;
  }
  .pr-privacy-status {
    display: none;
    color: #f0c674;
    background: #231f13;
    border: 1px solid #5d4c22;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 11px;
    line-height: 1.35;
  }
  .pr-privacy-status.is-visible {
    display: block;
  }
  .pr-audio-info {
    width: 100%;
    height: 100%;
    background: #181818;
    color: #aaa;
    border: 1px solid #111;
    border-radius: 6px;
    padding: 10px;
    font-size: 12px;
    line-height: 1.6;
    box-sizing: border-box;
    display: none;
  }
  .pr-audio-info span { color: #fff; font-weight: 500; }
  .pr-controls-group {
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 6px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 4px;
    box-sizing: border-box;
    width: 100%;
  }
  .pr-strength-row {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    box-sizing: border-box;
  }
  .pr-height-resizer {
    height: 6px;
    background: #2a2a2a;
    cursor: ns-resize;
    border-radius: 3px;
    margin: 2px 0;
    transition: background 0.15s;
    border: 1px solid #1e1e1e;
  }
  .pr-height-resizer:hover {
    background: #444;
    border-color: #555;
  }
  .pr-strength-label {
    font-size: 11px;
    font-weight: 600;
    color: #fff;
    white-space: nowrap;
    margin-left: auto;
  }
  .pr-strength-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 80px;
    height: 4px;
    background: #444;
    border-radius: 2px;
    outline: none;
    cursor: pointer;
    border: 1px solid #222;
  }
  .pr-strength-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #aaa;
    cursor: pointer;
  }
  .pr-strength-slider:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
  .pr-strength-input {
    font-size: 12px;
    color: #fff;
    background: #222;
    border: 1px solid #444;
    border-radius: 4px;
    width: 52px;
    text-align: center;
    padding: 3px;
  }
  .pr-strength-input::-webkit-outer-spin-button,
  .pr-strength-input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .pr-strength-input[type=number] {
    -moz-appearance: textfield;
  }
  .pr-strength-input:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .pr-gap-menu {
    position: fixed;
    background: #1e1e1e;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 4px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    z-index: 9999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
  }
  .pr-gap-menu-btn {
    background: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 11px;
    font-family: inherit;
    cursor: pointer;
    text-align: left;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: background 0.15s ease;
  }
  .pr-gap-menu-btn:hover {
    background: #3a3a3a;
    border-color: #666;
  }
  .pr-player-controls {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    padding: 2px 0;
    flex-wrap: wrap;
    width: 100%;
  }
  .pr-icon-btn {
    background: #2a2a2a;
    border: 1px solid #444;
    color: #eee;
    cursor: pointer;
    padding: 6px 12px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }
  .pr-icon-btn * {
    pointer-events: none;
  }
  .pr-icon-btn:hover {
    color: #fff;
    background: #3a3a3a;
    border-color: #666;
  }
  .pr-icon-btn.active {
    color: #4fff8f;
    border-color: #4fff8f;
    background: #1a3a2a;
  }
  .pr-seek-bar {
    -webkit-appearance: none;
    appearance: none;
    height: 6px;
    background: #444;
    border-radius: 3px;
    outline: none;
    cursor: pointer;
    border: 1px solid #222;
  }
  .pr-seek-bar::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #ff4444;
    cursor: pointer;
    border: 2px solid #222;
  }
  .pr-timeline-viewport {
    width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
  }
  .pr-timeline-viewport::-webkit-scrollbar {
    height: 10px;
  }
  .pr-timeline-viewport::-webkit-scrollbar-track {
    background: #151515;
    border-radius: 5px;
  }
  .pr-timeline-viewport::-webkit-scrollbar-thumb {
    background: #444
    border-radius: 5px;
    border: 1px solid #000;
  }
  .pr-timeline-viewport::-webkit-scrollbar-thumb:hover {
    background: #666
    border-color: #000;
  }
  .pr-zoom-controls {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-left: 12px;
  }
  .pr-zoom-slider {
    width: 80px;
    -webkit-appearance: none;
    appearance: none;
    height: 4px;
    background: #444;
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  .pr-zoom-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #aaa;
    cursor: pointer;
  }
  .pr-right-group {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .pr-segment-bounds {
    font-size: 12px;
    color: #aaa;
    font-family: monospace;
  }
  .pr-timecode {
    font-size: 14px;
    font-weight: bold;
    color: #e0e0e0;
    font-family: monospace;
  }
  .pr-settings-menu {
    position: fixed;
    background: #1e1e1e;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    z-index: 9999;
    box-shadow: 0 4px 20px rgba(0,0,0,0.7);
    min-width: 220px;
  }
  .pr-settings-title {
    font-size: 11px;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-bottom: 4px;
    border-bottom: 1px solid #333;
    margin-bottom: 2px;
  }
  .pr-settings-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .pr-settings-label {
    font-size: 12px;
    color: #bbb;
    flex: 1;
    white-space: nowrap;
  }
  .pr-number-control {
    display: flex;
    align-items: center;
    border: 1px solid #444;
    border-radius: 4px;
    background: #2a2a2a;
    overflow: hidden;
  }
  .pr-number-btn {
    background: #333;
    color: #aaa;
    border: none;
    width: 20px;
    height: 22px;
    cursor: pointer;
    font-size: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s;
    user-select: none;
  }
  .pr-number-btn:hover {
    background: #444;
    color: #fff;
  }
  .pr-settings-input {
    background: transparent;
    color: #e0e0e0;
    border: none;
    padding: 0 4px;
    font-size: 12px;
    width: 50px;
    height: 22px;
    text-align: center;
    font-family: monospace;
    outline: none;
    -moz-appearance: textfield;
  }
  .pr-settings-input::-webkit-outer-spin-button,
  .pr-settings-input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .pr-settings-select {
    background: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 3px 4px;
    font-size: 12px;
    width: 98px;
    cursor: pointer;
  }
  .pr-settings-divider {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 2px 0;
  }
  .pr-settings-toggle-btn {
    width: 100%;
    background: #252525;
    color: #aaa;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 11px;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
  }
  .pr-settings-toggle-btn:hover {
    background: #2e2e2e;
    color: #ccc;
    border-color: #555;
  }
  .pr-settings-close-btn {
    background: transparent;
    color: #888;
    border: none;
    cursor: pointer;
    padding: 2px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    transition: all 0.15s;
  }
  .pr-settings-close-btn:hover {
    color: #fff;
    background: rgba(255,255,255,0.1);
  }
  .pr-segmented-control {
    display: flex;
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 2px;
    width: 110px;
    height: 22px;
    align-items: center;
    box-sizing: border-box;
  }
  .pr-segment {
    flex: 1;
    text-align: center;
    font-size: 10px;
    font-weight: 500;
    line-height: 18px;
    cursor: pointer;
    border-radius: 4px;
    color: #888;
    transition: all 0.15s ease;
  }
  .pr-segment.active {
    background: #333;
    color: #fff;
  }
  .pr-segment:hover:not(.active) {
    color: #ccc;
  }
  .pr-image-browser-dialog {
    position: fixed;
    z-index: 10001;
    inset: 0;
    background: rgba(0,0,0,.55);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .pr-image-browser-panel {
    width: 720px;
    max-width: 92vw;
    max-height: 86vh;
    overflow: auto;
    background: #222;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 14px;
    color: #ddd;
    font: 12px Arial, sans-serif;
    box-shadow: 0 12px 44px rgba(0,0,0,.55);
  }
  .pr-image-browser-panel h3 {
    margin: 0 0 10px;
    font-size: 15px;
  }
  .pr-image-browser-controls {
    display: grid;
    grid-template-columns: 1fr minmax(150px, 1fr) minmax(108px, 130px) auto auto auto minmax(130px, 180px);
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
  }
  .pr-image-browser-controls select,
  .pr-image-browser-controls input,
  .pr-image-folder-row input,
  .pr-image-folder-row select {
    background: #151515;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px;
    min-width: 0;
  }
  .pr-image-browser-controls button,
  .pr-image-browser-actions button,
  .pr-image-folder-row button {
    background: #333;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 10px;
    cursor: pointer;
  }
  .pr-image-browser-controls button:hover,
  .pr-image-browser-actions button:hover,
  .pr-image-folder-row button:hover {
    background: #444;
  }
  .pr-image-sort-wrap {
    position: relative;
    min-width: 0;
  }
  .pr-image-sort-btn {
    width: 100%;
    min-width: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    background: #333;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 8px;
    cursor: pointer;
    text-align: left;
  }
  .pr-image-sort-btn svg {
    width: 14px;
    height: 14px;
    stroke: currentColor;
    fill: none;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
    flex: 0 0 auto;
  }
  .pr-image-sort-menu {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    z-index: 10002;
    min-width: 144px;
    background: #1d1c25;
    border: 1px solid #3d3c4a;
    border-radius: 6px;
    padding: 6px;
    display: none;
    flex-direction: column;
    gap: 2px;
    box-shadow: 0 8px 22px rgba(0,0,0,.55);
  }
  .pr-image-sort-menu.is-open {
    display: flex;
  }
  .pr-image-sort-option {
    background: transparent !important;
    border: none !important;
    color: #aaa;
    border-radius: 4px !important;
    padding: 7px 10px !important;
    font-size: 12px;
    font-weight: 600;
    text-align: left;
    cursor: pointer;
  }
  .pr-image-sort-option:hover {
    background: #2b2a34 !important;
    color: #ddd;
  }
  .pr-image-sort-option.active {
    color: #f2f2f4;
  }
  .pr-image-icon-btn {
    width: 32px;
    height: 32px;
    padding: 4px !important;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .pr-image-icon-btn svg,
  .pr-image-columns-control svg {
    width: 18px;
    height: 18px;
    stroke: currentColor;
    fill: none;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .pr-image-columns-control {
    display: grid;
    grid-template-columns: 22px 1fr 18px;
    gap: 6px;
    align-items: center;
    color: #ddd;
  }
  .pr-image-columns-control input {
    width: 100%;
    min-width: 0;
  }
  .pr-image-browser-grid {
    --pr-image-columns: 4;
    display: grid;
    grid-template-columns: repeat(var(--pr-image-columns), minmax(0, 1fr));
    gap: 8px;
    max-height: 52vh;
    overflow: auto;
    padding: 2px;
  }
  .pr-image-browser-grid.hide-images .pr-image-tile img {
    opacity: 0;
  }
  .pr-image-browser-panel:hover .pr-image-browser-grid.hide-images .pr-image-tile img,
  .pr-image-browser-grid.show-images .pr-image-tile img {
    opacity: 1;
  }
  .pr-image-tile {
    min-width: 0;
    background: #181818;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 5px;
    color: #ddd;
    cursor: pointer;
    text-align: left;
  }
  .pr-image-tile.selected {
    border-color: #8ab4f8;
    background: #202a36;
  }
  .pr-image-tile img {
    display: block;
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: contain;
    background: #101010;
    border: 1px solid #2d2d2d;
    border-radius: 3px;
    transition: opacity .12s ease;
  }
  .pr-audio-browser-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 52vh;
    overflow: auto;
    padding: 2px;
  }
  .pr-audio-row {
    display: grid;
    grid-template-columns: 34px 1fr;
    gap: 8px;
    align-items: center;
    min-width: 0;
    background: #181818;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 6px;
    color: #ddd;
    cursor: pointer;
    text-align: left;
  }
  .pr-audio-row.selected {
    border-color: #8ab4f8;
    background: #202a36;
  }
  .pr-audio-play {
    width: 30px;
    height: 30px;
    padding: 4px !important;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #333;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    cursor: pointer;
  }
  .pr-audio-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .pr-audio-details {
    min-width: 0;
  }
  .pr-audio-size {
    margin-top: 2px;
    color: #888;
    font-size: 10px;
  }
  .pr-image-browser-meta {
    margin-top: 8px;
    color: #aaa;
    font-size: 11px;
    min-height: 14px;
  }
  .pr-image-browser-actions {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
    margin-top: 12px;
  }
  .pr-image-folder-row {
    display: grid;
    grid-template-columns: 90px 1fr;
    gap: 8px;
    margin-bottom: 8px;
    align-items: center;
  }
  .pr-image-large-preview {
    position: fixed;
    z-index: 10003;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0,0,0,.72);
    padding: 24px;
  }
  .pr-image-large-preview-panel {
    position: relative;
    max-width: 92vw;
    max-height: 92vh;
    background: #151515;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 10px;
    box-shadow: 0 12px 44px rgba(0,0,0,.55);
  }
  .pr-image-large-preview-panel img {
    display: block;
    max-width: calc(92vw - 20px);
    max-height: calc(92vh - 52px);
    object-fit: contain;
    background: #0b0b0b;
  }
  .pr-image-large-preview-close {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 30px;
    height: 30px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(20,20,20,.88);
    color: #ddd;
    border: 1px solid #666;
    border-radius: 4px;
    cursor: pointer;
  }
  .pr-image-large-preview-caption {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: calc(92vw - 20px);
    margin-top: 7px;
    color: #aaa;
    font-size: 12px;
  }
  .pr-prompt-optimizer-panel {
    width: 980px;
  }
  .pr-prompt-optimizer-controls {
    display: grid;
    grid-template-columns: minmax(180px, 1fr) 150px auto auto;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
  }
  .pr-prompt-optimizer-controls button,
  .pr-prompt-optimizer-controls select {
    background: #151515;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px;
    min-width: 0;
  }
  .pr-prompt-optimizer-controls button {
    cursor: pointer;
  }
  .pr-prompt-optimizer-controls button:hover {
    background: #333;
  }
  .pr-prompt-auth-row {
    display: grid;
    grid-template-columns: auto minmax(170px, 1fr) auto auto;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
    background: #181818;
    border: 1px solid #333;
    border-radius: 5px;
    padding: 7px;
  }
  .pr-prompt-auth-row span {
    color: #aaa;
    font-size: 11px;
    white-space: nowrap;
  }
  .pr-prompt-auth-row input {
    background: #151515;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px;
    min-width: 0;
  }
  .pr-prompt-auth-row button {
    background: #333;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 10px;
    cursor: pointer;
  }
  .pr-prompt-auth-row button:hover {
    background: #444;
  }
  .pr-prompt-template-editor {
    display: none;
    margin-bottom: 8px;
    background: #181818;
    border: 1px solid #333;
    border-radius: 5px;
    padding: 8px;
  }
  .pr-prompt-template-editor.is-open {
    display: block;
  }
  .pr-prompt-template-editor textarea {
    width: 100%;
    height: 150px;
    min-height: 110px;
    resize: vertical;
    box-sizing: border-box;
    background: #151515;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 7px;
    font-size: 12px;
    line-height: 1.35;
  }
  .pr-prompt-template-toolbar {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 7px;
  }
  .pr-prompt-template-toolbar span {
    flex: 1;
    min-width: 0;
    color: #aaa;
    font-size: 11px;
  }
  .pr-prompt-template-toolbar button {
    background: #333;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 10px;
    cursor: pointer;
  }
  .pr-prompt-template-toolbar button:hover {
    background: #444;
  }
  .pr-prompt-mode {
    display: flex;
    background: #151515;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 2px;
    min-width: 0;
  }
  .pr-prompt-mode button {
    flex: 1;
    background: transparent;
    color: #aaa;
    border: none;
    border-radius: 3px;
    padding: 5px 8px;
    cursor: pointer;
  }
  .pr-prompt-mode button.active {
    background: #333;
    color: #fff;
  }
  .pr-prompt-optimizer-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 58vh;
    overflow: auto;
    padding: 2px;
  }
  .pr-prompt-optimizer-row {
    display: grid;
    grid-template-columns: 42px 96px minmax(220px, 1fr) minmax(320px, 1.35fr);
    gap: 8px;
    align-items: start;
    background: #181818;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 8px;
    min-width: 0;
  }
  .pr-prompt-optimizer-row-tools {
    align-self: center;
    display: grid;
    gap: 8px;
    justify-items: center;
  }
  .pr-prompt-optimizer-check {
    justify-self: center;
  }
  .pr-prompt-optimizer-height-controls {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .pr-prompt-optimizer-height-controls button {
    width: 24px;
    height: 22px;
    padding: 0;
    line-height: 1;
    border: 1px solid #444;
    border-radius: 4px;
    background: #262626;
    color: #ddd;
    cursor: pointer;
  }
  .pr-prompt-optimizer-height-controls button:hover {
    background: #333;
  }
  .pr-prompt-optimizer-height-controls button:disabled {
    opacity: .45;
    cursor: default;
  }
  .pr-prompt-optimizer-thumb {
    position: relative;
    width: 96px;
    height: 96px;
    min-width: 96px;
    align-self: center;
    background: #101010;
    border: 1px solid #2d2d2d;
    border-radius: 4px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #777;
    font-size: 11px;
  }
  .pr-prompt-optimizer-thumb img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    transition: opacity .12s ease;
  }
  .pr-prompt-optimizer-grid.hide-images .pr-prompt-optimizer-thumb img {
    opacity: 0;
  }
  .pr-prompt-optimizer-grid.hide-images .pr-prompt-optimizer-thumb:hover img,
  .pr-prompt-optimizer-grid.show-images .pr-prompt-optimizer-thumb img {
    opacity: 1;
  }
  .pr-prompt-optimizer-field {
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 0;
    align-self: start;
  }
  .pr-prompt-optimizer-label {
    color: #aaa;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .04em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .pr-prompt-optimizer-field textarea {
    height: 96px;
    min-height: 72px;
    resize: none;
    background: #222;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 7px;
    font-size: 12px;
    line-height: 1.35;
    box-sizing: border-box;
    width: 100%;
  }
  @media (max-width: 880px) {
    .pr-prompt-optimizer-row {
      grid-template-columns: 42px 96px minmax(0, 1fr);
      grid-template-areas:
        "tools thumb direction"
        "tools thumb generated";
    }
    .pr-prompt-optimizer-row-tools {
      grid-area: tools;
    }
    .pr-prompt-optimizer-thumb {
      grid-area: thumb;
    }
    .pr-prompt-optimizer-field:first-of-type {
      grid-area: direction;
    }
    .pr-prompt-optimizer-field:last-of-type {
      grid-area: generated;
    }
  }
  .pr-prompt-optimizer-status {
    color: #aaa;
    font-size: 11px;
    min-height: 16px;
  }
  .pr-prompt-progress {
    display: none;
    gap: 5px;
    margin: 4px 0 8px;
  }
  .pr-prompt-progress.visible {
    display: grid;
  }
  .pr-prompt-progress-track {
    width: 100%;
    height: 7px;
    overflow: hidden;
    background: #111;
    border: 1px solid #333;
    border-radius: 4px;
  }
  .pr-prompt-progress-bar {
    width: 0%;
    height: 100%;
    background: linear-gradient(90deg, #3f7bff, #69d2ff);
    transition: width .25s ease;
  }
  .pr-prompt-progress-text {
    color: #888;
    font-size: 10px;
    min-height: 12px;
  }
`;

if (!document.getElementById("prompt-relay-styles")) {
  const styleEl = document.createElement("style");
  styleEl.id = "prompt-relay-styles";
  styleEl.textContent = STYLES;
  document.head.appendChild(styleEl);
}

// --- Icons ---
const ICONS = {
  upload: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>`,
  audio: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>`,
  trash: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`,
  text: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 7 4 4 20 4 20 7"></polyline><line x1="9" y1="20" x2="15" y2="20"></line><line x1="12" y1="4" x2="12" y2="20"></line></svg>`,
  play: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`,
  pause: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`,
  loop: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12A9 9 0 0 0 6 5.3L3 8"></path><polyline points="3 3 3 8 8 8"></polyline><path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"></path><polyline points="21 21 21 16 16 16"></polyline></svg>`,
  minus: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line></svg>`,
  plus: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>`,
  fit: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><polyline points="8 7 3 12 8 17"></polyline><polyline points="16 7 21 12 16 17"></polyline></svg>`,
  gear: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>`,
  sparkle: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"></path><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"></path><path d="M5 3l.7 1.8L7.5 5.5l-1.8.7L5 8l-.7-1.8-1.8-.7 1.8-.7z"></path></svg>`,
  close: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`
};

// --- Data Models ---
function parseInitial(jsonStr) {
  let parsed = { segments: [], audioSegments: [] };
  try {
    if (jsonStr) {
      const p = JSON.parse(jsonStr);
      if (Array.isArray(p.segments)) parsed.segments = p.segments;
      if (Array.isArray(p.audioSegments)) parsed.audioSegments = p.audioSegments;
    }
  } catch (e) { }

  let currentStart = 0;
  for (let seg of parsed.segments) {
    if (seg.start === undefined) {
      seg.start = currentStart;
      currentStart += seg.length;
    }
    // Guarantee ID assignment to prevent node loading drag breaks
    if (!seg.id) {
      seg.id = Date.now().toString() + Math.random().toString(36).substr(2, 5);
    }
  }

  for (let seg of parsed.audioSegments) {
    if (!seg.id) {
      seg.id = Date.now().toString() + Math.random().toString(36).substr(2, 5);
    }
    if (seg.trimStart === undefined) seg.trimStart = 0;
    seg.volume = clampVolume(seg.volume);
  }
  assignMissingAudioLanes(parsed.audioSegments);

  return parsed;
}

class TimelineEditor {
  constructor(node, container, domWidget) {
    this.node = node;
    this.container = container;
    this.domWidget = domWidget;

    // Track heights (dynamic)
    this.rulerHeight = RULER_HEIGHT;
    this.blockHeight = BLOCK_HEIGHT;
    this.audioTrackHeight = AUDIO_TRACK_HEIGHT;
    this.canvasHeight = CANVAS_HEIGHT;

    // Core data
    this.timeline = { segments: [], audioSegments: [] };
    this.selectionType = "image"; // "image" or "audio"
    this.selectedIndex = -1;

    // Interactions
    this._isDragging = false;
    this._dragType = null;
    this._dragStartX = 0;
    this._dragInitialTimeline = null;
    this.zoomLevel = 1.0;
    this._lastZoom = 1.0;
    this._lastScale = 1.0;
    this._dragTargetId = null;
    this._dragTargetIdRight = null;
    this._previewSegments = null;
    this._lastWidth = 0;
    this._hoveredGapIdx = -1;
    this._isHovering = false;
    this._isRevealAreaHovering = false;
    this._promptOptimizerActive = false;

    // Playback state
    this.currentFrame = 0;
    this.isPlaying = false;
    this.isLooping = false;
    this.audioContext = null;
    this.activeAudioNodes = [];
    this.playbackStartTime = 0;
    this.playbackStartFrame = 0;
    this._playLoopId = null;

    // --- Ghost dragging state ---
    this._ghostSegmentId = null;
    this._ghostTrack = null;
    this._ghostInitialTimeline = null;

    // Attach to Python widgets
    this._gapMenu = null;         // Active gap popup menu element
    this._gapMenuDismisser = null;

    // Attach to Python widgets
    this.durationFramesWidget = this.node.widgets.find(w => w.name === "duration_frames");
    this.durationSecondsWidget = this.node.widgets.find(w => w.name === "duration_seconds");
    this.frameRateWidget = this.node.widgets.find(w => w.name === "frame_rate");
    this.timelineDataWidget = this.node.widgets.find(w => w.name === "timeline_data");
    this.localPromptsWidget = this.node.widgets.find(w => w.name === "local_prompts");
    this.segmentLengthsWidget = this.node.widgets.find(w => w.name === "segment_lengths");
    this.guideStrengthWidget = this.node.widgets.find(w => w.name === "guide_strength");
    this.displayModeWidget = this.node.widgets.find(w => w.name === "display_mode");
    this.useGlobalPromptWidget = this.node.widgets.find(w => w.name === "use_global_prompt");
    this.hideTimelineImagesPromptsWidget = this.node.widgets.find(w => w.name === "hide_timeline_images_prompts");
    this.privacyModeWidget = this.node.widgets.find(w => w.name === "privacy_mode");
    this.privacyPayloadWidget = this.node.widgets.find(w => w.name === "privacy_payload");
    repairLegacyPrivacyWidgetShift(this.node);
    this.privacyBusy = false;
    this.privacyLocked = false;
    this.privacyStatus = "";
    this._privacyEncryptSeq = 0;
    this._privateGlobalPrompt = this.getGlobalPromptWidget()?.value || "";
    this.thumbnailCacheBust = "";

    this.timeline = parseInitial(this.timelineDataWidget?.value);
    this.ensureAudioTrackHeight();
    this.loadImages();

    this.createDOM();
    this.applyGlobalPromptVisibility();
    if (this.isPrivacyModeEnabled() && isEncryptedPrivacyPayload(this.privacyPayloadWidget?.value)) {
      this.timeline = { segments: [], audioSegments: [] };
      this.privacyLocked = true;
      this.privacyStatus = "Decrypting private timeline data...";
      this.updatePrivacyStatus();
      void this.decryptPrivacyPayload();
    }
    if (this.timeline.segments.length > 0) {
      this.selectedIndex = 0;
    }
    this.updateUIFromSelection();
    this.commitChanges(true);
    // Hide settings widgets by default to reduce node clutter.
    // Deferred so all widget types are finalized before we touch them.
    setTimeout(() => this.hideSettingsWidgets(), 0);

    let isSyncing = false;

    const origDurationFramesCallback = this.durationFramesWidget?.callback;
    if (this.durationFramesWidget) {
      this.durationFramesWidget.callback = (...args) => {
        if (origDurationFramesCallback) origDurationFramesCallback.apply(this.durationFramesWidget, args);

        if (!isSyncing && this.durationSecondsWidget) {
          isSyncing = true;
          this.durationSecondsWidget.value = parseFloat((this.getDurationFrames() / this.getFrameRate()).toFixed(3));
          isSyncing = false;
        }

        this.commitChanges();
      };
    }

    const origDurationSecondsCallback = this.durationSecondsWidget?.callback;
    if (this.durationSecondsWidget) {
      this.durationSecondsWidget.callback = (...args) => {
        if (origDurationSecondsCallback) origDurationSecondsCallback.apply(this.durationSecondsWidget, args);

        if (!isSyncing && this.durationFramesWidget) {
          isSyncing = true;
          const newFrames = Math.max(1, Math.round(this.durationSecondsWidget.value * this.getFrameRate()));
          this.durationFramesWidget.value = newFrames;
          if (this.durationFramesWidget.callback) this.durationFramesWidget.callback(newFrames);
          isSyncing = false;
        }
      };
    }

    const origFrameRateCallback = this.frameRateWidget?.callback;
    if (this.frameRateWidget) {
      this.frameRateWidget.callback = (...args) => {
        if (origFrameRateCallback) origFrameRateCallback.apply(this.frameRateWidget, args);
        if (!isSyncing && this.durationSecondsWidget) {
          isSyncing = true;
          this.durationSecondsWidget.value = parseFloat((this.getDurationFrames() / this.getFrameRate()).toFixed(3));
          isSyncing = false;
        }
      };
    }

    const origDisplayModeCallback = this.displayModeWidget?.callback;
    if (this.displayModeWidget) {
      this.displayModeWidget.callback = (...args) => {
        if (origDisplayModeCallback) origDisplayModeCallback.apply(this.displayModeWidget, args);
        this.updateWidgetVisibility();
        this.updateUIFromSelection();
        this.render();
      };
      this.updateWidgetVisibility(); // Initial trigger
    }

    const globalPromptWidget = this.getGlobalPromptWidget();
    const origGlobalPromptCallback = globalPromptWidget?.callback;
    if (globalPromptWidget) {
      globalPromptWidget.callback = (...args) => {
        if (origGlobalPromptCallback) origGlobalPromptCallback.apply(globalPromptWidget, args);
        this._privateGlobalPrompt = globalPromptWidget.value || "";
        if (this.isPrivacyModeEnabled() && !this.privacyLocked) {
          void this.encryptPrivacyState({ renderAfter: false });
        }
      };
    }

    // Polling is much more reliable in Comfy than ResizeObserver due to scale transforms
    this._renderLoop = requestAnimationFrame(this.checkResize.bind(this));
  }

  destroy() {
    cancelAnimationFrame(this._renderLoop);
    this.pauseAudio();
    this.setPromptOptimizerActive(false);
    this.closePromptOptimizer();
    window.removeEventListener("keydown", this.handleKeyDown, true);
    window.removeEventListener("paste", this.handlePaste, true);
    if (this.handleRevealAreaMouseMove) window.removeEventListener("mousemove", this.handleRevealAreaMouseMove, true);
  }

  getDurationFrames() {
    return parseInt((this.durationFramesWidget && this.durationFramesWidget.value > 0) ? this.durationFramesWidget.value : 24, 10);
  }

  getFrameRate() {
    return parseInt((this.frameRateWidget && this.frameRateWidget.value > 0) ? this.frameRateWidget.value : 24, 10);
  }

  isPrivacyModeEnabled() {
    return widgetBoolValue(this.privacyModeWidget?.value);
  }

  setPrivacyStatus(message = "", locked = this.privacyLocked) {
    this.privacyStatus = message;
    this.privacyLocked = locked;
    this.updatePrivacyStatus();
  }

  updatePrivacyStatus() {
    if (!this.privacyStatusEl) return;
    this.privacyStatusEl.textContent = this.privacyStatus || "";
    this.privacyStatusEl.classList.toggle("is-visible", Boolean(this.privacyStatus));
  }

  getTimelinePrivacyState() {
    return {
      global_prompt: this.getGlobalPromptWidget()?.value || this._privateGlobalPrompt || "",
      timeline: this.buildTimelineSaveObject(),
    };
  }

  sanitizeWidgetsForPrivacy() {
    if (this.timelineDataWidget) this.timelineDataWidget.value = EMPTY_TIMELINE_JSON;
    if (this.localPromptsWidget) this.localPromptsWidget.value = "";
    if (this.segmentLengthsWidget) this.segmentLengthsWidget.value = "";
    if (this.guideStrengthWidget) this.guideStrengthWidget.value = "";
  }

  async encryptPrivacyState({ renderAfter = false, markCanvasDirty = true, showStatus = false } = {}) {
    if (!this.isPrivacyModeEnabled() || this.privacyLocked) return false;
    const sequence = ++this._privacyEncryptSeq;
    this.privacyBusy = true;
    if (showStatus) this.setPrivacyStatus("Encrypting private timeline data...", false);
    try {
      const result = await fetchPrivacyJson("encrypt", { state: this.getTimelinePrivacyState() });
      if (sequence !== this._privacyEncryptSeq) return false;
      if (this.privacyPayloadWidget) this.privacyPayloadWidget.value = JSON.stringify(result.envelope);
      this.sanitizeWidgetsForPrivacy();
      if (showStatus || this.privacyStatus.startsWith("Encrypting private timeline data")) {
        this.setPrivacyStatus("", false);
      }
      if (markCanvasDirty && window.app && window.app.graph) window.app.graph.setDirtyCanvas(true, true);
      if (renderAfter) this.render();
      return true;
    } catch (err) {
      if (sequence === this._privacyEncryptSeq) {
        this.setPrivacyStatus(`Privacy encryption failed: ${err.message}`, false);
      }
      return false;
    } finally {
      if (sequence === this._privacyEncryptSeq) this.privacyBusy = false;
    }
  }

  async decryptPrivacyPayload() {
    if (!isEncryptedPrivacyPayload(this.privacyPayloadWidget?.value)) return false;
    this.privacyBusy = true;
    this.setPrivacyStatus("Decrypting private timeline data...", true);
    try {
      const result = await fetchPrivacyJson("decrypt", { payload: parseJsonObject(this.privacyPayloadWidget.value) });
      const state = result.state || {};
      this._privateGlobalPrompt = String(state.global_prompt || "");
      const globalPromptWidget = this.getGlobalPromptWidget();
      if (globalPromptWidget) globalPromptWidget.value = this._privateGlobalPrompt;
      this.timeline = parseInitial(JSON.stringify(state.timeline || {}));
      this.ensureAudioTrackHeight();
      this.loadImages();
      this.selectionType = "image";
      this.selectedIndex = clamp(this.selectedIndex, -1, Math.max(-1, this.timeline.segments.length - 1));
      this.updateUIFromSelection();
      this.applyGlobalPromptVisibility();
      this.commitChanges(true);
      this.setPrivacyStatus("", false);
      this.render();
      return true;
    } catch (err) {
      this.timeline = { segments: [], audioSegments: [] };
      this.setPrivacyStatus(`Private timeline locked: ${err.message}`, true);
      this.updateUIFromSelection();
      this.render();
      return false;
    } finally {
      this.privacyBusy = false;
    }
  }

  async setPrivacyMode(enabled) {
    if (this.privacyBusy && enabled) return;
    if (enabled === this.isPrivacyModeEnabled()) return;
    const wasLocked = this.privacyLocked;

    if (enabled) {
      this.privacyBusy = true;
      try {
        this.setPrivacyStatus("Clearing thumbnail cache...", wasLocked);
        this.thumbnailCacheBust = await clearTimelineThumbnailCache();
      } catch (err) {
        this.setPrivacyStatus(`Thumbnail cache could not be cleared: ${err.message}`, wasLocked);
        return;
      } finally {
        this.privacyBusy = false;
      }
      setWidgetBoolValue(this.privacyModeWidget, true);
      const ok = await this.encryptPrivacyState({ renderAfter: true, showStatus: true });
      if (!ok) {
        setWidgetBoolValue(this.privacyModeWidget, false);
        if (this.privacyPayloadWidget) this.privacyPayloadWidget.value = "";
      }
      return;
    }

    if (!confirm("Disable Privacy mode? This will save the LTX Director timeline and global prompt in clear text inside the workflow.")) {
      return;
    }
    this._privacyEncryptSeq += 1;
    this.privacyBusy = false;
    this.privacyLocked = false;
    this.privacyBusy = true;
    try {
      this.setPrivacyStatus("Clearing thumbnail cache...", false);
      this.thumbnailCacheBust = await clearTimelineThumbnailCache();
    } catch (err) {
      this.thumbnailCacheBust = String(Date.now());
      this.setPrivacyStatus(`Privacy disabled, but thumbnail cache could not be cleared: ${err.message}`, false);
    } finally {
      this.privacyBusy = false;
    }
    setWidgetBoolValue(this.privacyModeWidget, false);
    if (this.privacyPayloadWidget) this.privacyPayloadWidget.value = "";
    if (!this.privacyStatus.startsWith("Privacy disabled, but thumbnail cache could not be cleared")) {
      this.setPrivacyStatus("", false);
    }
    this.commitChanges(true);
    if (window.app && window.app.graph) window.app.graph.setDirtyCanvas(true, true);
  }

  getAudioLaneCount(audioSegments = this.timeline.audioSegments) {
    let maxLane = 0;
    for (const seg of audioSegments || []) {
      maxLane = Math.max(maxLane, normalizeAudioLane(seg.lane));
    }
    return maxLane + 1;
  }

  getRequiredAudioTrackHeight(audioSegments = this.timeline.audioSegments) {
    return Math.max(AUDIO_TRACK_HEIGHT, this.getAudioLaneCount(audioSegments) * AUDIO_LANE_HEIGHT);
  }

  ensureAudioTrackHeight(audioSegments = this.timeline.audioSegments) {
    const requiredHeight = this.getRequiredAudioTrackHeight(audioSegments);
    if (requiredHeight <= this.audioTrackHeight) return false;

    this.audioTrackHeight = requiredHeight;
    this.canvasHeight = this.rulerHeight + this.blockHeight + this.audioTrackHeight;

    if (this.canvas) {
      this.canvas.style.height = `${this.canvasHeight}px`;
      const width = this.canvas.offsetWidth || this._lastWidth || timelineNodeInnerWidth(this.node);
      this.resizeCanvas(width);
    }

    return true;
  }

  getAudioLaneForY(y) {
    const relY = y - (RULER_HEIGHT + this.blockHeight);
    return Math.max(0, Math.min(64, Math.floor(relY / AUDIO_LANE_HEIGHT)));
  }

  getAudioSegmentY(seg) {
    return RULER_HEIGHT + this.blockHeight + normalizeAudioLane(seg.lane) * AUDIO_LANE_HEIGHT;
  }

  getAudioClipHeight() {
    return Math.max(24, AUDIO_LANE_HEIGHT - 6);
  }

  findFreeAudioLane(start, length, ignoreId = null, preferredLane = null, audioSegments = this.timeline.audioSegments) {
    return findFreeAudioLane(audioSegments, start, length, ignoreId, preferredLane);
  }

  hideTimelineImagesPromptsEnabled() {
    const value = this.hideTimelineImagesPromptsWidget?.value;
    return value === true || value === "true";
  }

  shouldHideTimelineImagesPrompts() {
    return this.hideTimelineImagesPromptsEnabled() && (!this._isRevealAreaHovering || this._promptOptimizerActive);
  }

  setPromptOptimizerActive(active) {
    const isActive = !!active;
    if (this._promptOptimizerActive === isActive) return;
    this._promptOptimizerActive = isActive;
    this.updatePromptPrivacyVisibility();
    this.render();
    if (window.app && window.app.graph) window.app.graph.setDirtyCanvas(true, true);
  }

  setElementTextPrivacy(element, shouldHide) {
    if (!element) return;
    element.classList.toggle("pr-privacy-hidden-text", shouldHide);

    const canReadOnly = element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement;
    if (!canReadOnly) return;

    if (shouldHide) {
      if (element.dataset.prPrivacyReadonly === undefined) {
        element.dataset.prPrivacyReadonly = element.readOnly ? "true" : "false";
      }
      element.readOnly = true;
    } else if (element.dataset.prPrivacyReadonly !== undefined) {
      element.readOnly = element.dataset.prPrivacyReadonly === "true";
      delete element.dataset.prPrivacyReadonly;
    }
  }

  getGlobalPromptWidget() {
    return this.node.widgets?.find(w => w.name === "global_prompt");
  }

  globalPromptEnabled() {
    return widgetBoolValue(this.useGlobalPromptWidget?.value);
  }

  applyGlobalPromptVisibility() {
    const globalPromptWidget = this.getGlobalPromptWidget();
    const isVisible = this.globalPromptEnabled();
    if (!isVisible) {
      for (const element of this.getWidgetTextElements(globalPromptWidget)) {
        this.setElementTextPrivacy(element, false);
      }
    }
    applyGlobalPromptWidgetVisibility(globalPromptWidget, isVisible);
  }

  getWidgetTextElements(widget) {
    const elements = [];
    const addElement = (el) => {
      if (el instanceof HTMLElement && !elements.includes(el)) elements.push(el);
    };

    addElement(widget?.inputEl);
    addElement(widget?.textarea);
    addElement(widget?.textArea);
    addElement(widget?.element);

    if (widget?.element instanceof HTMLElement) {
      widget.element.querySelectorAll?.("textarea, input, [contenteditable='true']").forEach(addElement);
    }

    return elements.filter((el) => (
      el instanceof HTMLInputElement
      || el instanceof HTMLTextAreaElement
      || el.getAttribute?.("contenteditable") === "true"
    ));
  }

  updatePromptPrivacyVisibility() {
    const shouldHide = this.shouldHideTimelineImagesPrompts();

    if (this.promptInput) {
      const hideSegmentPrompt = shouldHide && this.promptInput.style.display !== "none";
      this.setElementTextPrivacy(this.promptInput, hideSegmentPrompt);
    }

    const globalPromptWidget = this.getGlobalPromptWidget();
    const globalPromptVisible = !!globalPromptWidget
      && !globalPromptWidget.hidden
      && !(globalPromptWidget.options && globalPromptWidget.options.hidden);
    for (const element of this.getWidgetTextElements(globalPromptWidget)) {
      this.setElementTextPrivacy(element, shouldHide && globalPromptVisible);
    }
  }

  isEventInsideRevealArea(e) {
    if (this.toolbar?.contains?.(e.target)) return false;
    const wrapperRect = this.wrapper?.getBoundingClientRect?.();
    const toolbarRect = this.toolbar?.getBoundingClientRect?.();
    if (!wrapperRect || !toolbarRect) return false;
    const revealTop = toolbarRect.bottom;
    return e.clientX >= wrapperRect.left
      && e.clientX <= wrapperRect.right
      && e.clientY >= revealTop
      && e.clientY <= wrapperRect.bottom;
  }

  isRevealAreaElementHovered() {
    return !!(
      this.viewport?.matches?.(":hover")
      || this.controlsGroup?.matches?.(":hover")
      || this.propContainer?.matches?.(":hover")
      || this.privacyStatusEl?.matches?.(":hover")
    );
  }

  setRevealAreaHovering(isHovering) {
    const hovering = !!isHovering;
    if (this._isRevealAreaHovering === hovering) return;
    this._isRevealAreaHovering = hovering;
    if (this.hideTimelineImagesPromptsEnabled()) {
      this.updatePromptPrivacyVisibility();
      this.render();
    }
  }

  updateRevealAreaHoverState(e) {
    this.setRevealAreaHovering(this.isEventInsideRevealArea(e));
  }

  // Grow the timeline duration to fit `requiredFrames` if it is currently shorter.
  // The timeline only ever grows — never shrinks — through this method.
  growTimelineIfNeeded(requiredFrames) {
    const current = this.getDurationFrames();
    if (requiredFrames <= current) return; // already big enough

    const newFrames = Math.ceil(requiredFrames);
    if (this.durationFramesWidget) {
      this.durationFramesWidget.value = newFrames;
    }
    if (this.durationSecondsWidget) {
      this.durationSecondsWidget.value = parseFloat((newFrames / this.getFrameRate()).toFixed(3));
    }
    // Notify ComfyUI that the widget value changed so it serialises correctly.
    if (window.app && window.app.graph) {
      window.app.graph.setDirtyCanvas(true, true);
    }
  }

  // Returns the maximum allowed zoom level, computed so that at max zoom
  // the viewport shows exactly 4 seconds of the visual timeline.
  getMaxZoom() {
    const visualDurationSecs = this.getVisualDurationFrames() / this.getFrameRate();
    const baseMaxZoom = Math.max(1, visualDurationSecs / 4);

    // Limit max zoom to prevent canvas width from exceeding browser limits (causing crash)
    const viewportWidth = this.viewport ? this.viewport.clientWidth : 1000;
    const MAX_CANVAS_WIDTH = 32768; // Extended limit for modern browsers
    const limitMaxZoom = MAX_CANVAS_WIDTH / Math.max(1, viewportWidth);

    return Math.max(1, Math.min(baseMaxZoom, limitMaxZoom));
  }

  // Returns the visual timeline length in frames:
  // the furthest segment end (across both tracks) × 1.30, with a floor of getDurationFrames().
  // This is used for all rendering/positioning — the actual output duration is getDurationFrames().
  getVisualDurationFrames() {
    let furthest = 0;
    for (const seg of this.timeline.segments) {
      furthest = Math.max(furthest, seg.start + seg.length);
    }
    for (const seg of this.timeline.audioSegments) {
      furthest = Math.max(furthest, seg.start + seg.length);
    }
    const outputDuration = this.getDurationFrames();
    if (furthest <= 0) return outputDuration;
    return Math.max(outputDuration, Math.ceil(furthest * 1.30));
  }

  // Sync the zoom slider's max attribute to the current getMaxZoom() value,
  // clamping zoomLevel if it now exceeds the new max.
  updateZoomSliderMax() {
    if (!this.zoomSlider) return;
    const maxZoom = this.getMaxZoom();
    this.zoomSlider.max = maxZoom.toFixed(2);
    if (this.zoomLevel > maxZoom) {
      this.zoomLevel = maxZoom;
      this.zoomSlider.value = maxZoom;
      // Resize the canvas to match the clamped zoom
      const viewportWidth = this.viewport ? this.viewport.clientWidth : 0;
      if (viewportWidth > 0) {
        const newCanvasWidth = Math.max(viewportWidth, viewportWidth * this.zoomLevel);
        this.canvas.style.width = newCanvasWidth + "px";
        this.resizeCanvas(newCanvasWidth);
      }
    }
  }

  loadImages() {
    for (const seg of this.timeline.segments) {
      if (seg.imageB64 && !seg.imgObj) {
        seg.imgObj = new Image();
        seg.imgObj.onload = () => this.render();
        seg.imgObj.src = seg.imageB64;
      }
    }
  }

  createDOM() {
    this.wrapper = document.createElement("div");
    this.wrapper.className = "pr-wrapper";
    for (const eventName of ["pointerdown", "pointerup", "mousedown", "mouseup", "click", "dblclick", "contextmenu", "wheel"]) {
      this.wrapper.addEventListener(eventName, (e) => {
        if (eventName === "mouseup") this.onMouseUp(e);
        else e.stopPropagation();
      });
    }

    this.wrapper.addEventListener("mouseenter", () => {
      this._isHovering = true;
      this.setRevealAreaHovering(this.isRevealAreaElementHovered());
    });
    this.wrapper.addEventListener("mouseleave", () => {
      this._isHovering = false;
      this.setRevealAreaHovering(false);
    });

    this.handleRevealAreaMouseMove = (e) => this.updateRevealAreaHoverState(e);
    window.addEventListener("mousemove", this.handleRevealAreaMouseMove, true);

    requestAnimationFrame(() => {
      if (!this.wrapper) return;
      this.setRevealAreaHovering(this.isRevealAreaElementHovered());
    });

    this.handleKeyDown = (e) => {
      const activeTag = document.activeElement ? document.activeElement.tagName : "";
      if (activeTag === "INPUT" || activeTag === "TEXTAREA") {
        if (this.wrapper?.contains(document.activeElement)) {
          e.stopPropagation();
          e.stopImmediatePropagation();
        }
        return;
      }

      if ((e.key === "Delete" || e.key === "Backspace") && this.selectedIndex !== -1 && this._isHovering) {
        this.deleteSelectedSegment();
        e.stopPropagation();
        e.stopImmediatePropagation();
        e.preventDefault();
      } else if ((e.key === " " || e.code === "Space") && this._isHovering) {
        this.togglePlay();
        e.stopPropagation();
        e.stopImmediatePropagation();
        e.preventDefault();
      }
    };
    window.addEventListener("keydown", this.handleKeyDown, true);

    this.handlePaste = (e) => {
      if (this._isHovering) {
        const activeTag = document.activeElement ? document.activeElement.tagName : "";
        if (activeTag === "INPUT" || activeTag === "TEXTAREA") return;

        if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
          const imageFiles = Array.from(e.clipboardData.files).filter(f => f.type.startsWith("image/"));
          if (imageFiles.length > 0) {
            this.handleImageUpload(imageFiles, this.currentFrame);
            e.preventDefault();
            e.stopPropagation();
          }
        }
      }
    };
    window.addEventListener("paste", this.handlePaste, true);

    // --- Toolbar ---
    const toolbar = document.createElement("div");
    toolbar.className = "pr-toolbar";
    this.toolbar = toolbar;

    const actionGroup = document.createElement("div");
    actionGroup.className = "pr-actions";

    this.fileInput = document.createElement("input");
    this.fileInput.type = "file";
    this.fileInput.accept = "image/*";
    this.fileInput.multiple = true;
    this.fileInput.style.display = "none";
    this.fileInput.addEventListener("change", (e) => this.handleImageUpload(e.target.files));

    this.audioFileInput = document.createElement("input");
    this.audioFileInput.type = "file";
    this.audioFileInput.accept = "audio/*";
    this.audioFileInput.multiple = true;
    this.audioFileInput.style.display = "none";
    this.audioFileInput.addEventListener("change", (e) => {
      const pending = this._pendingAudioUpload || null;
      this._pendingAudioUpload = null;
      const targetFrame = pending ? pending.targetFrameStart : Math.round(this.currentFrame || 0);
      const targetLane = pending ? pending.targetLane : null;
      this.handleAudioUpload(e.target.files, targetFrame, targetLane);
    });
    this.audioFileInput.addEventListener("cancel", () => {
      this._pendingAudioUpload = null;
    });

    this.videoSourceInput = document.createElement("input");
    this.videoSourceInput.type = "file";
    this.videoSourceInput.accept = "video/*";
    this.videoSourceInput.style.display = "none";
    this.videoSourceInput.addEventListener("change", (e) => this.handleSourceVideoUpload(e.target.files));

    const uploadBtn = document.createElement("button");
    uploadBtn.className = "pr-btn";
    uploadBtn.innerHTML = `${ICONS.upload} Add Image`;
    uploadBtn.addEventListener("click", () => this.showTimelineImageBrowser());

    this.replaceImageBtn = document.createElement("button");
    this.replaceImageBtn.className = "pr-btn";
    this.replaceImageBtn.innerHTML = `${ICONS.upload} Replace Image`;
    this.replaceImageBtn.disabled = true;
    this.replaceImageBtn.title = "Select an image segment to replace its image.";
    this.replaceImageBtn.addEventListener("click", () => {
      const seg = this.getSelectedImageSegment();
      if (!seg) return;
      this.showTimelineImageBrowser(null, null, { mode: "replace", segmentId: seg.id });
    });

    const uploadAudioBtn = document.createElement("button");
    uploadAudioBtn.className = "pr-btn";
    uploadAudioBtn.innerHTML = `${ICONS.audio} Add Audio`;
    uploadAudioBtn.addEventListener("click", () => this.showTimelineAudioBrowser());

    const sourceVideoBtn = document.createElement("button");
    sourceVideoBtn.className = "pr-btn";
    sourceVideoBtn.innerHTML = `${ICONS.play} Add Video Source`;
    sourceVideoBtn.title = "Add a locked source video at the beginning of the timeline.";
    sourceVideoBtn.addEventListener("click", () => this.videoSourceInput.click());

    const addTextBtn = document.createElement("button");
    addTextBtn.className = "pr-btn";
    addTextBtn.innerHTML = `${ICONS.text} Add Text`;
    addTextBtn.addEventListener("click", () => this.addTextSegmentFreeSpace());

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "pr-btn pr-btn-danger";
    deleteBtn.innerHTML = `${ICONS.trash} Delete`;
    deleteBtn.addEventListener("click", () => this.deleteSelectedSegment());

    actionGroup.appendChild(this.fileInput);
    actionGroup.appendChild(this.audioFileInput);
    actionGroup.appendChild(this.videoSourceInput);
    actionGroup.appendChild(uploadBtn);
    actionGroup.appendChild(this.replaceImageBtn);
    actionGroup.appendChild(addTextBtn);
    actionGroup.appendChild(sourceVideoBtn);
    actionGroup.appendChild(uploadAudioBtn);
    actionGroup.appendChild(deleteBtn);
    toolbar.appendChild(actionGroup);

    const rightGroup = document.createElement("div");
    rightGroup.className = "pr-right-group";

    this.segmentBoundsDisplay = document.createElement("div");
    this.segmentBoundsDisplay.className = "pr-segment-bounds";
    this.segmentBoundsDisplay.textContent = "Start: - | End: -";

    this.timeCodeDisplay = document.createElement("div");
    this.timeCodeDisplay.className = "pr-timecode";
    this.timeCodeDisplay.textContent = this.formatTime(0);

    const settingsBtn = document.createElement("button");
    settingsBtn.className = "pr-btn";
    settingsBtn.style.padding = "6px";
    settingsBtn.style.justifyContent = "center";
    settingsBtn.style.width = "28px";
    settingsBtn.style.height = "28px";
    settingsBtn.style.boxSizing = "border-box";
    settingsBtn.innerHTML = ICONS.gear;
    settingsBtn.title = "Settings";
    settingsBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (this._settingsMenu) {
        this.dismissSettingsMenu();
      } else {
        this.showSettingsMenu(settingsBtn);
      }
    });

    const toggleBtn = document.createElement("button");
    toggleBtn.className = "pr-btn";
    toggleBtn.style.padding = "6px 8px";
    toggleBtn.style.fontSize = "11px";
    toggleBtn.style.marginRight = "0px";
    toggleBtn.textContent = "Custom Audio: OFF";
    toggleBtn.title = "Toggle Custom Audio Output";

    const updateToggleStyle = (isOn) => {
      toggleBtn.textContent = isOn ? "Custom Audio: ON" : "Custom Audio: OFF";
      if (isOn) {
        toggleBtn.style.background = "#1c222d";
        toggleBtn.style.borderColor = "#283142";
        toggleBtn.style.color = "#e0e0e0";
      } else {
        toggleBtn.style.background = "#222";
        toggleBtn.style.borderColor = "#111";
        toggleBtn.style.color = "#e0e0e0";
      }
    };

    toggleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const widget = this.node.widgets?.find(w => w.name === "use_custom_audio");
      if (widget) {
        widget.value = !widget.value;
        updateToggleStyle(widget.value);
        this.node.setDirtyCanvas(true, true);
      }
    });

    // Initial state check (widgets might not be ready immediately)
    setTimeout(() => {
      const widget = this.node.widgets?.find(w => w.name === "use_custom_audio");
      if (widget) {
        updateToggleStyle(widget.value);
      }
    }, 100);

    const helpBtn = document.createElement("button");
    helpBtn.className = "pr-btn";
    helpBtn.style.padding = "6px";
    helpBtn.style.justifyContent = "center";
    helpBtn.style.width = "28px";
    helpBtn.style.height = "28px";
    helpBtn.style.boxSizing = "border-box";
    helpBtn.innerHTML = "?";
    helpBtn.title = "Help / Documentation";
    helpBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      window.open("https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI", "_blank");
    });

    const optimizerBtn = document.createElement("button");
    optimizerBtn.className = "pr-btn";
    optimizerBtn.style.padding = "6px";
    optimizerBtn.style.justifyContent = "center";
    optimizerBtn.style.width = "28px";
    optimizerBtn.style.height = "28px";
    optimizerBtn.style.boxSizing = "border-box";
    optimizerBtn.innerHTML = ICONS.sparkle;
    optimizerBtn.title = "Prompt Optimizer";
    optimizerBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.showPromptOptimizer();
    });

    const btnGroup = document.createElement("div");
    btnGroup.style.display = "flex";
    btnGroup.style.gap = "6px";
    btnGroup.style.alignItems = "center";
    btnGroup.appendChild(toggleBtn);
    btnGroup.appendChild(optimizerBtn);
    btnGroup.appendChild(helpBtn);
    btnGroup.appendChild(settingsBtn);
    rightGroup.appendChild(btnGroup);

    toolbar.appendChild(rightGroup);

    // --- Canvas & Viewport ---
    this.viewport = document.createElement("div");
    this.viewport.className = "pr-timeline-viewport";

    this.viewport.addEventListener("wheel", (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        e.stopPropagation();

        let zoomDelta = e.deltaY > 0 ? -0.5 : 0.5;
        this.zoomLevel = Math.max(1, Math.min(this.getMaxZoom(), this.zoomLevel + zoomDelta));
        if (this.zoomSlider) this.zoomSlider.value = this.zoomLevel;

        const oldWidth = this.canvas.offsetWidth;
        const newWidth = this.viewport.clientWidth * this.zoomLevel;
        const mouseX = e.clientX - this.viewport.getBoundingClientRect().left;
        const scrollRatio = (this.viewport.scrollLeft + mouseX) / oldWidth;

        this.canvas.style.width = newWidth + "px";
        this.viewport.scrollLeft = scrollRatio * newWidth - mouseX;
      }
    }, { passive: false, capture: true });

    this.canvas = document.createElement("canvas");
    this.canvas.className = "pr-canvas";
    this.ctx = this.canvas.getContext("2d");
    this.canvas.style.width = "100%";

    this.viewport.appendChild(this.canvas);

    this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.canvas.addEventListener("contextmenu", this.onContextMenu.bind(this));
    this.canvas.style.height = `${this.canvasHeight}px`;

    // --- Content Area Container ---
    const propContainer = document.createElement("div");
    propContainer.className = "pr-prop-container";
    this.propContainer = propContainer;

    // --- Text Area (Image/Text) ---
    this.promptInput = document.createElement("textarea");
    this.promptInput.className = "pr-prompt-area";
    this.promptInput.placeholder = "Enter prompt for selected segment...";
    this.promptInput.addEventListener("input", (e) => {
      e.stopPropagation();
      if (this.privacyLocked) return;
      if (this.shouldHideTimelineImagesPrompts()) {
        this.updateUIFromSelection();
        return;
      }
      if (this.selectionType === "image" && this.timeline.segments[this.selectedIndex]) {
        this.timeline.segments[this.selectedIndex].prompt = this.promptInput.value;
        this._promptEditDirty = true;
      }
    });
    this.promptInput.addEventListener("blur", () => {
      this.flushPromptEdit({ skipNodeResize: true });
    });
    for (const eventName of ["beforeinput", "keydown", "keyup", "keypress", "compositionstart", "compositionupdate", "compositionend"]) {
      this.promptInput.addEventListener(eventName, (e) => e.stopPropagation());
    }

    // --- Audio Info Area ---
    this.audioInfoArea = document.createElement("div");
    this.audioInfoArea.className = "pr-audio-info";

    propContainer.appendChild(this.promptInput);
    propContainer.appendChild(this.audioInfoArea);

    this.wrapper.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.wrapper.classList.add("drag-active");

      const { x, y } = this.getMousePos(e);
      const logicalWidth = this.canvas.offsetWidth;
      const totalFrames = this.getVisualDurationFrames();
      if (!logicalWidth || totalFrames <= 0) return;

      const isAudioTrack = y > RULER_HEIGHT + this.blockHeight;
      const trackType = isAudioTrack ? "audio" : "image";
      const arrToModify = isAudioTrack ? this.timeline.audioSegments : this.timeline.segments;
      const targetAudioLane = isAudioTrack ? this.getAudioLaneForY(y) : 0;

      if (!this._ghostSegmentId || this._ghostTrack !== trackType) {
        this._ghostSegmentId = "GHOST_" + Date.now();
        this._ghostTrack = trackType;
        this._ghostInitialTimeline = JSON.parse(JSON.stringify(arrToModify));

        const frameRate = this.getFrameRate();
        const newLength = Math.max(1, frameRate * 1);

        let mouseFrameX = x * (totalFrames / logicalWidth);
        let startFrame = clamp(Math.round(mouseFrameX - newLength / 2), 0, totalFrames - newLength);

        this._ghostInitialTimeline.push({
          id: this._ghostSegmentId,
          start: startFrame,
          length: newLength,
          lane: targetAudioLane,
          type: "ghost"
        });
      }

      let mouseFrameX = x * (totalFrames / logicalWidth);
      const ghost = this._ghostInitialTimeline.find(s => s.id === this._ghostSegmentId);
      let D_mouse_start = mouseFrameX - ghost.length / 2;

      if (trackType === "audio") {
        const ghostStart = clamp(Math.round(D_mouse_start), 0, totalFrames - ghost.length);
        const ghostLane = this.findFreeAudioLane(ghostStart, ghost.length, ghost.id, targetAudioLane, this._ghostInitialTimeline);
        this._previewSegments = this._ghostInitialTimeline.map((seg) => (
          seg.id === this._ghostSegmentId ? { ...seg, start: ghostStart, resolvedStart: ghostStart, lane: ghostLane } : seg
        ));
        this.ensureAudioTrackHeight(this._previewSegments);
      } else {
        this._previewSegments = this._applyCenterDragPhysics(
          this._ghostInitialTimeline,
          this._ghostSegmentId,
          D_mouse_start,
          mouseFrameX,
          totalFrames,
          totalFrames,
          logicalWidth
        );
      }
      this.render();
    });

    this.wrapper.addEventListener("dragleave", (e) => {
      const rect = this.wrapper.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX >= rect.right ||
        e.clientY < rect.top || e.clientY >= rect.bottom) {
        this.wrapper.classList.remove("drag-active");
        this._ghostSegmentId = null;
        this._ghostTrack = null;
        this._ghostInitialTimeline = null;
        this._previewSegments = null;
        this.render();
      }
    });

    this.wrapper.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.wrapper.classList.remove("drag-active");

      let targetFrameStart = null;
      let targetAudioLane = null;
      let targetTrack = this._ghostTrack || "image";

      if (this._ghostSegmentId && this._previewSegments) {
        const ghost = this._previewSegments.find(s => s.id === this._ghostSegmentId);
        if (ghost) {
          targetFrameStart = ghost.resolvedStart !== undefined ? ghost.resolvedStart : ghost.start;
          targetAudioLane = normalizeAudioLane(ghost.lane);
        }
      }
      this._ghostSegmentId = null;
      this._ghostTrack = null;
      this._ghostInitialTimeline = null;
      this._previewSegments = null;
      this.render();

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const imageFiles = [];
        const audioFiles = [];
        for (let file of e.dataTransfer.files) {
          if (file.type.startsWith("audio/")) audioFiles.push(file);
          if (file.type.startsWith("image/")) imageFiles.push(file);
        }

        // Let implicit intent handle mixing drops: use the track we hovered over
        // for the first type we process, or fallback.
        if (audioFiles.length > 0 && (targetTrack === "audio" || imageFiles.length === 0)) {
          this.handleAudioUpload(audioFiles, targetFrameStart, targetAudioLane);
        } else if (imageFiles.length > 0) {
          this.handleImageUpload(imageFiles, targetFrameStart);
        }
      }
    });

    window.addEventListener("mousemove", this.onMouseMove.bind(this));
    window.addEventListener("mouseup", this.onMouseUp.bind(this));

    // --- Player Controls ---
    const playerControls = document.createElement("div");
    playerControls.className = "pr-player-controls";

    this.playBtn = document.createElement("button");
    this.playBtn.className = "pr-icon-btn";
    this.playBtn.style.padding = "4px";
    this.playBtn.innerHTML = ICONS.play;
    this.playBtn.title = "Play/Pause Audio";
    this.playBtn.addEventListener("click", () => this.togglePlay());

    this.loopBtn = document.createElement("button");
    this.loopBtn.className = "pr-icon-btn";
    this.loopBtn.style.padding = "4px";
    this.loopBtn.innerHTML = ICONS.loop;
    this.loopBtn.title = "Toggle Loop";
    this.loopBtn.addEventListener("click", () => this.toggleLoop());

    this.seekBar = document.createElement("input");
    this.seekBar.type = "range";
    this.seekBar.className = "pr-seek-bar";
    this.seekBar.min = "0";
    this.seekBar.value = "0";
    this.seekBar.style.flex = "1"; // take up remaining space
    this.seekBar.addEventListener("input", (e) => {
      this.currentFrame = parseInt(e.target.value, 10);
      this.render();
      if (this.isPlaying) {
        this.playAudio();
      }
    });

    // --- Zoom Controls ---
    const zoomControls = document.createElement("div");
    zoomControls.className = "pr-zoom-controls";

    const zoomOutBtn = document.createElement("button");
    zoomOutBtn.className = "pr-icon-btn";
    zoomOutBtn.style.padding = "4px";
    zoomOutBtn.innerHTML = ICONS.minus;
    zoomOutBtn.title = "Zoom Out";
    zoomOutBtn.addEventListener("click", () => {
      const currentZoom = parseFloat(this.zoomSlider.value);
      this.zoomSlider.value = Math.max(1, currentZoom - 0.5);
      this.zoomSlider.dispatchEvent(new Event("input"));
    });

    this.zoomSlider = document.createElement("input");
    this.zoomSlider.type = "range";
    this.zoomSlider.className = "pr-zoom-slider";
    this.zoomSlider.min = "1";
    this.zoomSlider.max = "1"; // Updated dynamically via updateZoomSliderMax()
    this.zoomSlider.step = "0.1";
    this.zoomSlider.value = "1";
    this.zoomSlider.title = "Zoom Level";
    this.zoomSlider.addEventListener("input", (e) => {
      this.zoomLevel = parseFloat(e.target.value);

      const viewportWidth = this.viewport.clientWidth;
      const newCanvasWidth = Math.max(viewportWidth, viewportWidth * this.zoomLevel);

      this.canvas.style.width = newCanvasWidth + "px";
      this.resizeCanvas(newCanvasWidth);
      this._lastWidth = viewportWidth;
      this._lastZoom = this.zoomLevel;

      // Keep playhead centered
      const totalFrames = this.getVisualDurationFrames();
      const playheadRatio = this.currentFrame / totalFrames;
      const newPlayheadX = playheadRatio * newCanvasWidth;
      this.viewport.scrollLeft = newPlayheadX - (viewportWidth / 2);
    });

    const zoomInBtn = document.createElement("button");
    zoomInBtn.className = "pr-icon-btn";
    zoomInBtn.style.padding = "4px";
    zoomInBtn.innerHTML = ICONS.plus;
    zoomInBtn.title = "Zoom In";
    zoomInBtn.addEventListener("click", () => {
      const currentZoom = parseFloat(this.zoomSlider.value);
      this.zoomSlider.value = Math.min(this.getMaxZoom(), currentZoom + 0.5);
      this.zoomSlider.dispatchEvent(new Event("input"));
    });

    const zoomFitBtn = document.createElement("button");
    zoomFitBtn.className = "pr-icon-btn";
    zoomFitBtn.style.padding = "4px";
    zoomFitBtn.style.marginLeft = "4px";
    zoomFitBtn.innerHTML = ICONS.fit;
    zoomFitBtn.title = "Zoom to Fit (show full timeline)";
    zoomFitBtn.addEventListener("click", () => {
      this.zoomLevel = 1;
      this.zoomSlider.value = 1;
      const viewportWidth = this.viewport.clientWidth;
      this.canvas.style.width = viewportWidth + "px";
      this.resizeCanvas(viewportWidth);
      this._lastWidth = viewportWidth;
      this._lastZoom = 1;
      this.viewport.scrollLeft = 0;
    });

    zoomControls.appendChild(zoomOutBtn);
    zoomControls.appendChild(this.zoomSlider);
    zoomControls.appendChild(zoomInBtn);
    zoomControls.appendChild(zoomFitBtn);

    playerControls.appendChild(this.playBtn);
    playerControls.appendChild(this.loopBtn);
    playerControls.appendChild(this.seekBar);
    playerControls.appendChild(zoomControls);



    // --- Guide Strength Slider ---
    this.strengthRow = document.createElement("div");
    this.strengthRow.className = "pr-strength-row";

    this.strengthLabel = document.createElement("span");
    this.strengthLabel.className = "pr-strength-label";
    this.strengthLabel.textContent = "Guide Strength:";

    this.strengthValue = document.createElement("input");
    this.strengthValue.type = "text";
    this.strengthValue.className = "pr-strength-input";
    this.strengthValue.value = "1.00";
    this.strengthValue.disabled = true;
    this.strengthValue.style.cursor = "text";

    this.sourceVideoFramesLabel = document.createElement("span");
    this.sourceVideoFramesLabel.className = "pr-strength-label";
    this.sourceVideoFramesLabel.textContent = "Last Frames:";
    this.sourceVideoFramesLabel.style.display = "none";

    this.sourceVideoFramesInput = document.createElement("input");
    this.sourceVideoFramesInput.type = "number";
    this.sourceVideoFramesInput.className = "pr-strength-input";
    this.sourceVideoFramesInput.min = "1";
    this.sourceVideoFramesInput.max = String(SOURCE_VIDEO_MAX_GUIDE_FRAMES);
    this.sourceVideoFramesInput.step = "1";
    this.sourceVideoFramesInput.value = String(SOURCE_VIDEO_DEFAULT_GUIDE_FRAMES);
    this.sourceVideoFramesInput.style.display = "none";
    this.sourceVideoFramesInput.title = "Number of final source video frames used for guidance.";
    this.sourceVideoFramesInput.addEventListener("change", (e) => {
      let value = parseInt(e.target.value, 10);
      if (!Number.isFinite(value)) value = SOURCE_VIDEO_DEFAULT_GUIDE_FRAMES;
      this.setSelectedSourceVideoGuideFrames(value);
    });

    this.strengthValue.addEventListener("change", (e) => {
      let val = parseFloat(e.target.value);
      if (isNaN(val)) val = 1;
      const maxVal = this.selectionType === "audio" ? 2 : 1;
      val = Math.max(0, Math.min(maxVal, val));
      this.strengthValue.value = val.toFixed(2);
      if (this.selectionType === "audio" && this.timeline.audioSegments[this.selectedIndex]) {
        const seg = this.timeline.audioSegments[this.selectedIndex];
        seg.volume = val;
        this.commitChanges();
        this.updateUIFromSelection();
      } else if (this.selectionType === "image" && this.timeline.segments[this.selectedIndex]) {
        const seg = this.timeline.segments[this.selectedIndex];
        if (seg.type !== "text") {
          seg.guideStrength = val;
          this.commitChanges();
        }
      }
    });

    this.strengthRow.appendChild(this.timeCodeDisplay);
    this.strengthRow.appendChild(this.segmentBoundsDisplay);
    this.strengthRow.appendChild(this.strengthLabel);
    this.strengthRow.appendChild(this.strengthValue);
    this.strengthRow.appendChild(this.sourceVideoFramesLabel);
    this.strengthRow.appendChild(this.sourceVideoFramesInput);


    this.wrapper.appendChild(toolbar);
    this.privacyStatusEl = document.createElement("div");
    this.privacyStatusEl.className = "pr-privacy-status";
    this.updatePrivacyStatus();
    this.wrapper.appendChild(this.privacyStatusEl);
    this.wrapper.appendChild(this.viewport);

    const controlsGroup = document.createElement("div");
    controlsGroup.className = "pr-controls-group";
    this.controlsGroup = controlsGroup;
    controlsGroup.appendChild(this.strengthRow);
    controlsGroup.appendChild(playerControls);
    this.wrapper.appendChild(controlsGroup);
    this.wrapper.appendChild(propContainer);

    this.container.appendChild(this.wrapper);
  }

  checkResize() {
    this.syncTimelineWidgetWidth();
    const viewportWidth = this.viewport.clientWidth;
    const currentScale = this.getRenderScale();

    if (viewportWidth > 0 && (this._lastWidth !== viewportWidth || this._lastZoom !== this.zoomLevel || this._lastScale !== currentScale)) {
      this._lastWidth = viewportWidth;
      this._lastZoom = this.zoomLevel;
      this._lastScale = currentScale;

      const newCanvasWidth = Math.max(viewportWidth, viewportWidth * this.zoomLevel);
      this.canvas.style.width = newCanvasWidth + "px";
      this.resizeCanvas(newCanvasWidth);
    }
    this._renderLoop = requestAnimationFrame(this.checkResize.bind(this));
  }

  syncTimelineWidgetWidth() {
    const width = `${timelineNodeInnerWidth(this.node)}px`;
    let changed = false;
    const elements = [
      this.container,
      this.domWidget?.element,
      this.domWidget?.inputEl,
      this.wrapper,
    ];

    for (const element of elements) {
      if (!element?.style) continue;
      if (element.style.width !== width) {
        element.style.width = width;
        element.style.maxWidth = width;
        changed = true;
      }
    }

    if (changed) this._lastWidth = 0;
  }

  getRenderScale() {
    const dpr = window.devicePixelRatio || 1;
    let graphScale = 1;
    try {
      if (window.app && window.app.canvas && window.app.canvas.ds && window.app.canvas.ds.scale) {
        graphScale = window.app.canvas.ds.scale;
      }
    } catch (e) { }
    // Scale up if zoomed in, but don't drop below 1x DPR if zoomed out
    return dpr * Math.max(1, graphScale);
  }

  resizeCanvas(widthPx) {
    const scale = this.getRenderScale();
    const targetWidth = Math.round(widthPx * scale);
    const targetHeight = Math.round(this.canvasHeight * scale);

    this.canvas.width = targetWidth;
    this.canvas.height = targetHeight;
    this.ctx.setTransform(scale, 0, 0, scale, 0, 0);
    this.render();
  }

  // Helper to map mouse events accurately regardless of canvas scaling
  getMousePos(e) {
    const rect = this.canvas.getBoundingClientRect();

    const scaleX = this.canvas.offsetWidth / rect.width;
    const scaleY = this.canvas.offsetHeight / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    return { x, y };
  }

  unloadPromptOptimizerModel(alias) {
    if (!alias) return;
    fetchTimelineImageJson("/wdc_ltx_prompt_optimizer/models/unload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: alias }),
    }).catch((err) => console.warn("Could not unload prompt optimizer model:", err));
  }

  closePromptOptimizer({ unloadModel = true } = {}) {
    const dialog = document.querySelector(".pr-prompt-optimizer-dialog");
    if (dialog && unloadModel) {
      this.unloadPromptOptimizerModel(dialog.dataset.loadedModelAlias || "");
    }
    dialog?.remove();
    this.setPromptOptimizerActive(false);
  }

  async segmentImageDataUrl(seg) {
    const img = seg?.imgObj;
    if (!img || !img.complete || !img.naturalWidth || !img.naturalHeight) return "";
    try {
      const canvas = document.createElement("canvas");
      const maxSide = 768;
      const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
      canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.88));
      if (!blob) return "";
      return await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => resolve("");
        reader.readAsDataURL(blob);
      });
    } catch (err) {
      return "";
    }
  }

  getPromptOptimizerSegments() {
    return [...this.timeline.segments]
      .filter((seg) => seg && seg.type !== "ghost")
      .sort((a, b) => a.start - b.start)
      .map((seg, order) => ({
        id: seg.id,
        order,
        type: seg.type || "image",
        start: Math.round(seg.start || 0),
        length: Math.round(seg.length || 1),
        prompt: seg.prompt || "",
        direction: seg.prompt || "",
        imageFile: seg.imageFile || "",
        imageFolderAlias: seg.imageFolderAlias || "",
        imageB64: seg.imageB64 || "",
        label: seg.imageFile || seg.fileName || (seg.type === "text" ? "Text segment" : "Timeline image"),
      }));
  }

  async loadPromptOptimizerModels(selectEl, statusEl) {
    const data = await fetchTimelineImageJson("/wdc_ltx_prompt_optimizer/models");
    const preferred = this.node.properties?.wdc_prompt_optimizer_model || "qwen3_vl_4b_fast";
    selectEl.innerHTML = "";
    for (const model of data.models || []) {
      const option = document.createElement("option");
      option.value = model.alias;
      const state = model.status === "ready" || model.status === "downloaded" ? "ready" : model.status.replace(/_/g, " ");
      option.textContent = `${model.alias} (${state})`;
      option.title = model.missing_dependencies?.length
        ? `Missing: ${model.missing_dependencies.join(", ")}`
        : model.repo_id;
      selectEl.appendChild(option);
    }
    if ([...selectEl.options].some((option) => option.value === preferred)) {
      selectEl.value = preferred;
    }
    const active = (data.models || []).find((model) => model.alias === selectEl.value);
    const updateStatus = () => {
      const model = (data.models || []).find((item) => item.alias === selectEl.value);
      if (!model) return;
      if (model.missing_dependencies?.length) {
        statusEl.textContent = `Missing optional packages: ${model.missing_dependencies.join(", ")}`;
      } else if (model.downloaded || model.backend === "fallback") {
        statusEl.textContent = `${model.alias} is ready.`;
      } else {
        statusEl.textContent = `${model.alias} will auto-download on Generate.`;
      }
    };
    if (selectEl._promptOptimizerChangeHandler) {
      selectEl.removeEventListener("change", selectEl._promptOptimizerChangeHandler);
    }
    selectEl._promptOptimizerChangeHandler = () => {
      this.node.properties = this.node.properties || {};
      this.node.properties.wdc_prompt_optimizer_model = selectEl.value;
      updateStatus();
    };
    selectEl.addEventListener("change", selectEl._promptOptimizerChangeHandler);
    if (active) updateStatus();
  }

  async loadPromptOptimizerSettings(statusEl, inputEl = null) {
    const data = await fetchTimelineImageJson("/wdc_ltx_prompt_optimizer/settings");
    if (inputEl) inputEl.value = "";
    if (data.tokenConfigured) {
      statusEl.textContent = "HF token: saved locally.";
    } else if (data.envTokenAvailable) {
      statusEl.textContent = "HF token: using environment token.";
    } else {
      statusEl.textContent = "HF token: missing, gated models may fail.";
    }
    statusEl.title = data.configPath || "";
    return data;
  }

  showPromptOptimizer() {
    this.closePromptOptimizer();
    this.setPromptOptimizerActive(true);

    const rows = this.getPromptOptimizerSegments();
    const overlay = document.createElement("div");
    overlay.className = "pr-image-browser-dialog pr-prompt-optimizer-dialog";
    overlay.innerHTML = `
      <div class="pr-image-browser-panel pr-prompt-optimizer-panel">
        <h3>LTX Prompt Optimizer</h3>
        <div class="pr-prompt-optimizer-controls">
          <select class="model" title="Local caption/optimizer model"></select>
          <div class="pr-prompt-mode" role="group" aria-label="Prompt mode">
            <button class="mode active" type="button" data-mode="sfw">SFW</button>
            <button class="mode" type="button" data-mode="nsfw">NSFW</button>
          </div>
          <button class="edit-template" type="button">Prompt</button>
          <button class="generate" type="button">Generate</button>
        </div>
        <div class="pr-prompt-auth-row">
          <span class="auth-status">HF token: checking...</span>
          <input class="hf-token" type="password" autocomplete="off" placeholder="hf_... access token">
          <button class="save-token" type="button">Save</button>
          <button class="clear-token" type="button">Clear</button>
        </div>
        <div class="pr-prompt-template-editor">
          <textarea class="prompt-template" spellcheck="false"></textarea>
          <div class="pr-prompt-template-toolbar">
            <span class="prompt-template-status">Default prompt template.</span>
            <button class="save-template" type="button">Save Prompt</button>
            <button class="reset-template" type="button">Reset Default</button>
          </div>
        </div>
        <div class="pr-prompt-optimizer-status"></div>
        <div class="pr-prompt-progress" aria-hidden="true">
          <div class="pr-prompt-progress-track">
            <div class="pr-prompt-progress-bar"></div>
          </div>
          <div class="pr-prompt-progress-text"></div>
        </div>
        <div class="pr-prompt-optimizer-grid"></div>
        <div class="pr-image-browser-actions">
          <button class="cancel" type="button">Cancel</button>
          <button class="replace" type="button">Replace</button>
        </div>
      </div>`;

    const panel = overlay.querySelector(".pr-prompt-optimizer-panel");
    const modelSelect = overlay.querySelector(".model");
    const statusEl = overlay.querySelector(".pr-prompt-optimizer-status");
    const progressWrap = overlay.querySelector(".pr-prompt-progress");
    const progressBar = overlay.querySelector(".pr-prompt-progress-bar");
    const progressText = overlay.querySelector(".pr-prompt-progress-text");
    const authStatusEl = overlay.querySelector(".auth-status");
    const hfTokenInput = overlay.querySelector(".hf-token");
    const saveTokenBtn = overlay.querySelector(".save-token");
    const clearTokenBtn = overlay.querySelector(".clear-token");
    const editTemplateBtn = overlay.querySelector(".edit-template");
    const promptTemplateEditor = overlay.querySelector(".pr-prompt-template-editor");
    const promptTemplateInput = overlay.querySelector(".prompt-template");
    const promptTemplateStatus = overlay.querySelector(".prompt-template-status");
    const saveTemplateBtn = overlay.querySelector(".save-template");
    const resetTemplateBtn = overlay.querySelector(".reset-template");
    const grid = overlay.querySelector(".pr-prompt-optimizer-grid");
    const generateBtn = overlay.querySelector(".generate");
    const replaceBtn = overlay.querySelector(".replace");
    const cancelBtn = overlay.querySelector(".cancel");
    const modeButtons = [...overlay.querySelectorAll(".mode")];
    let mode = "sfw";
    let loadedModelAlias = "";

    const hideImages = this.hideTimelineImagesPromptsEnabled();
    grid.classList.toggle("hide-images", hideImages);
    grid.classList.toggle("show-images", !hideImages);

    const rowState = new Map();
    const rowTextMinHeight = 72;
    const rowTextMaxHeight = 288;
    const rowTextStep = 24;
    const rowTextDefaultHeight = 96;
    const formatEta = (seconds) => {
      const value = Number(seconds);
      if (!Number.isFinite(value) || value < 0.5) return "";
      if (value < 60) return `about ${Math.ceil(value)}s left`;
      return `about ${Math.ceil(value / 60)}m left`;
    };
    const formatBytes = (bytes) => {
      const value = Number(bytes);
      if (!Number.isFinite(value) || value <= 0) return "";
      const units = ["B", "KB", "MB", "GB", "TB"];
      let scaled = value;
      let unitIndex = 0;
      while (scaled >= 1024 && unitIndex < units.length - 1) {
        scaled /= 1024;
        unitIndex += 1;
      }
      const decimals = scaled >= 10 || unitIndex === 0 ? 0 : 1;
      return `${scaled.toFixed(decimals)} ${units[unitIndex]}`;
    };
    const updateProgressBar = (progress = {}, visible = true) => {
      const percentValue = Number(progress.percent);
      const percent = Number.isFinite(percentValue) ? Math.max(0, Math.min(100, percentValue)) : 0;
      progressWrap.classList.toggle("visible", visible);
      progressWrap.setAttribute("aria-hidden", visible ? "false" : "true");
      progressBar.style.width = `${percent}%`;
      const parts = [];
      if (Number.isFinite(percentValue)) parts.push(`${Math.round(percent)}%`);
      const eta = formatEta(progress.eta_seconds);
      if (eta) parts.push(eta);
      if (progress.estimated && progress.phase === "generating") parts.push("estimated");
      if (progress.phase === "downloading") {
        const currentBytes = formatBytes(progress.download_current_bytes);
        const totalBytes = formatBytes(progress.download_total_bytes);
        if (currentBytes && totalBytes) parts.push(`${currentBytes} / ${totalBytes}`);
        if (progress.download_file_index && progress.download_file_total) {
          parts.push(`file ${progress.download_file_index}/${progress.download_file_total}`);
        }
      }
      progressText.textContent = parts.join(" · ");
    };
    const markLoadedModel = (alias) => {
      loadedModelAlias = alias || "";
      if (loadedModelAlias) {
        overlay.dataset.loadedModelAlias = loadedModelAlias;
      } else {
        delete overlay.dataset.loadedModelAlias;
      }
    };
    const unloadLoadedModel = (alias = loadedModelAlias) => {
      if (!alias) return;
      this.unloadPromptOptimizerModel(alias);
      if (alias === loadedModelAlias) markLoadedModel("");
    };
    const setRowTextareaHeight = (state, height) => {
      if (!state) return;
      const target = Math.max(rowTextMinHeight, Math.min(rowTextMaxHeight, Math.round(height)));
      state.textHeight = target;
      state.direction.style.height = `${target}px`;
      state.generated.style.height = `${target}px`;
      if (state.decreaseHeight) state.decreaseHeight.disabled = this.privacyLocked || target <= rowTextMinHeight;
      if (state.increaseHeight) state.increaseHeight.disabled = this.privacyLocked || target >= rowTextMaxHeight;
    };
    const setBusy = (busy) => {
      generateBtn.disabled = busy || this.privacyLocked;
      replaceBtn.disabled = busy || this.privacyLocked;
      modelSelect.disabled = busy || this.privacyLocked;
      hfTokenInput.disabled = busy || this.privacyLocked;
      saveTokenBtn.disabled = busy || this.privacyLocked;
      clearTokenBtn.disabled = busy || this.privacyLocked;
      editTemplateBtn.disabled = busy || this.privacyLocked;
      promptTemplateInput.disabled = busy || this.privacyLocked;
      saveTemplateBtn.disabled = busy || this.privacyLocked;
      resetTemplateBtn.disabled = busy || this.privacyLocked;
      for (const button of modeButtons) button.disabled = busy || this.privacyLocked;
      for (const state of rowState.values()) {
        if (state.decreaseHeight) state.decreaseHeight.disabled = busy || this.privacyLocked || state.textHeight <= rowTextMinHeight;
        if (state.increaseHeight) state.increaseHeight.disabled = busy || this.privacyLocked || state.textHeight >= rowTextMaxHeight;
      }
    };

    const renderRows = () => {
      grid.innerHTML = "";
      rowState.clear();
      if (!rows.length) {
        const empty = document.createElement("div");
        empty.className = "pr-prompt-optimizer-status";
        empty.textContent = "No timeline segments are available to optimize.";
        grid.appendChild(empty);
        generateBtn.disabled = true;
        replaceBtn.disabled = true;
        return;
      }

      for (const item of rows) {
        const row = document.createElement("div");
        row.className = "pr-prompt-optimizer-row";
        row.dataset.segmentId = item.id;

        const check = document.createElement("input");
        check.type = "checkbox";
        check.className = "pr-prompt-optimizer-check";
        check.checked = true;
        check.title = "Optimize this segment";

        const rowTools = document.createElement("div");
        rowTools.className = "pr-prompt-optimizer-row-tools";
        rowTools.appendChild(check);

        const heightControls = document.createElement("div");
        heightControls.className = "pr-prompt-optimizer-height-controls";
        const decreaseHeight = document.createElement("button");
        decreaseHeight.type = "button";
        decreaseHeight.textContent = "-";
        decreaseHeight.title = "Decrease row height";
        const increaseHeight = document.createElement("button");
        increaseHeight.type = "button";
        increaseHeight.textContent = "+";
        increaseHeight.title = "Increase row height";
        heightControls.appendChild(decreaseHeight);
        heightControls.appendChild(increaseHeight);
        rowTools.appendChild(heightControls);

        const thumb = document.createElement("div");
        thumb.className = "pr-prompt-optimizer-thumb";
        if (item.imageB64) {
          const img = document.createElement("img");
          img.src = item.imageB64;
          img.alt = "";
          thumb.appendChild(img);
        } else {
          thumb.textContent = item.type === "text" ? "Text" : "Image";
        }

        const directionWrap = document.createElement("label");
        directionWrap.className = "pr-prompt-optimizer-field";
        const directionLabel = document.createElement("span");
        directionLabel.className = "pr-prompt-optimizer-label";
        directionLabel.textContent = `${item.order + 1}. ${item.label}`;
        const direction = document.createElement("textarea");
        direction.value = item.prompt;
        direction.placeholder = "Direction or existing segment prompt...";
        directionWrap.appendChild(directionLabel);
        directionWrap.appendChild(direction);

        const generatedWrap = document.createElement("label");
        generatedWrap.className = "pr-prompt-optimizer-field";
        const generatedLabel = document.createElement("span");
        generatedLabel.className = "pr-prompt-optimizer-label";
        generatedLabel.textContent = "Optimized LTX prompt";
        const generated = document.createElement("textarea");
        generated.value = item.prompt;
        generated.placeholder = "Generated prompt will appear here...";
        generatedWrap.appendChild(generatedLabel);
        generatedWrap.appendChild(generated);

        const state = {
          item,
          check,
          direction,
          generated,
          decreaseHeight,
          increaseHeight,
          textHeight: rowTextDefaultHeight,
        };
        rowState.set(item.id, state);
        setRowTextareaHeight(state, rowTextDefaultHeight);

        decreaseHeight.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          setRowTextareaHeight(state, state.textHeight - rowTextStep);
        });
        increaseHeight.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          setRowTextareaHeight(state, state.textHeight + rowTextStep);
        });

        for (const input of [check, decreaseHeight, increaseHeight, direction, generated]) {
          input.disabled = this.privacyLocked;
          for (const eventName of ["click", "keydown", "keyup", "keypress", "beforeinput"]) {
            input.addEventListener(eventName, (event) => event.stopPropagation());
          }
        }
        setRowTextareaHeight(state, state.textHeight);

        row.appendChild(rowTools);
        row.appendChild(thumb);
        row.appendChild(directionWrap);
        row.appendChild(generatedWrap);
        grid.appendChild(row);
      }
    };

    const selectedPayloadRows = async () => {
      const payloadRows = [];
      const selectedRows = rows.filter((item) => rowState.get(item.id)?.check.checked);
      let preparedCount = 0;
      for (const item of rows) {
        const state = rowState.get(item.id);
        const selected = !!state?.check.checked;
        const payload = {
          id: item.id,
          order: item.order,
          type: item.type,
          start: item.start,
          length: item.length,
          selected,
          direction: state?.direction.value || "",
          prompt: state?.direction.value || "",
          imageFile: item.imageFile || "",
          imageFolderAlias: item.imageFolderAlias || "",
          label: item.label || "",
        };

        if (selected && !payload.imageFile && item.imageB64) {
          preparedCount += 1;
          statusEl.textContent = `Preparing image ${preparedCount} of ${selectedRows.length} for upload...`;
          updateProgressBar({ percent: selectedRows.length ? (preparedCount - 1) / selectedRows.length * 100 : 0 }, true);
          await new Promise((resolve) => requestAnimationFrame(resolve));
          const seg = this.timeline.segments.find((candidate) => candidate.id === item.id);
          payload.image_data = await this.segmentImageDataUrl(seg);
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
        payloadRows.push(payload);
      }
      return payloadRows;
    };

    modeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        mode = button.dataset.mode || "sfw";
        modeButtons.forEach((other) => other.classList.toggle("active", other === button));
      });
    });
    modelSelect.addEventListener("change", () => {
      const previous = loadedModelAlias;
      if (previous && previous !== modelSelect.value) unloadLoadedModel(previous);
    });

    generateBtn.addEventListener("click", async () => {
      if (this.privacyLocked) return;
      const selectedCount = [...rowState.values()].filter((state) => state.check.checked).length;
      if (!selectedCount) {
        statusEl.textContent = "Select at least one segment to optimize.";
        return;
      }
      setBusy(true);
      statusEl.textContent = "Preparing selected segments...";
      updateProgressBar({ percent: 0 }, true);
      try {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        const segments = await selectedPayloadRows();
        statusEl.textContent = "Starting prompt optimization...";
        markLoadedModel(modelSelect.value);
        const started = await fetchTimelineImageJson("/wdc_ltx_prompt_optimizer/optimize/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: modelSelect.value,
            mode,
            duration_frames: this.getDurationFrames(),
            frame_rate: this.getFrameRate(),
            segments,
          }),
        });
        let data = null;
        while (true) {
          data = await fetchTimelineImageJson(`/wdc_ltx_prompt_optimizer/optimize/status?job_id=${encodeURIComponent(started.job_id)}`);
          const progress = data.progress || {};
          const suffix = progress.current != null && progress.total != null ? ` (${progress.current} / ${progress.total})` : "";
          statusEl.textContent = `${data.message || "Working..."}${suffix}`;
          updateProgressBar(progress, true);
          if (data.state === "completed") break;
          if (data.state === "failed") throw new Error(data.error || data.message || "Prompt optimization failed.");
          await new Promise((resolve) => setTimeout(resolve, 750));
        }
        for (const result of data.results || []) {
          const state = rowState.get(result.id);
          if (state) state.generated.value = result.prompt || "";
        }
        statusEl.textContent = `Generated ${data.results?.length || 0} prompt${(data.results?.length || 0) === 1 ? "" : "s"}.`;
        updateProgressBar({ ...(data.progress || {}), percent: 100, eta_seconds: 0, estimated: false }, true);
      } catch (err) {
        statusEl.textContent = err.message;
      } finally {
        setBusy(false);
      }
    });

    replaceBtn.addEventListener("click", () => {
      if (this.privacyLocked) return;
      for (const state of rowState.values()) {
        if (!state.check.checked) continue;
        const seg = this.timeline.segments.find((candidate) => candidate.id === state.item.id);
        if (!seg) continue;
        seg.prompt = state.generated.value || "";
      }
      this.flushPromptEdit({ skipNodeResize: true });
      this.updateUIFromSelection();
      this.commitChanges(true);
      this.render();
      if (window.app && window.app.graph) window.app.graph.setDirtyCanvas(true, true);
      unloadLoadedModel();
      this.closePromptOptimizer({ unloadModel: false });
    });

    const refreshOptimizerStatus = async () => {
      const settings = await this.loadPromptOptimizerSettings(authStatusEl, hfTokenInput);
      promptTemplateInput.value = settings.promptTemplate || settings.defaultPromptTemplate || "";
      promptTemplateInput.dataset.defaultTemplate = settings.defaultPromptTemplate || "";
      promptTemplateStatus.textContent = settings.promptTemplateConfigured
        ? "Custom prompt template saved locally."
        : "Using the default motion-only prompt template.";
      await this.loadPromptOptimizerModels(modelSelect, statusEl);
    };

    editTemplateBtn.addEventListener("click", () => {
      promptTemplateEditor.classList.toggle("is-open");
    });

    saveTokenBtn.addEventListener("click", async () => {
      if (this.privacyLocked) return;
      setBusy(true);
      try {
        await fetchTimelineImageJson("/wdc_ltx_prompt_optimizer/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ hf_token: hfTokenInput.value || "" }),
        });
        await refreshOptimizerStatus();
      } catch (err) {
        authStatusEl.textContent = err.message;
      } finally {
        setBusy(false);
      }
    });

    clearTokenBtn.addEventListener("click", async () => {
      if (this.privacyLocked) return;
      setBusy(true);
      try {
        await fetchTimelineImageJson("/wdc_ltx_prompt_optimizer/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ clear: true }),
        });
        await refreshOptimizerStatus();
      } catch (err) {
        authStatusEl.textContent = err.message;
      } finally {
        setBusy(false);
      }
    });

    saveTemplateBtn.addEventListener("click", async () => {
      if (this.privacyLocked) return;
      setBusy(true);
      try {
        await fetchTimelineImageJson("/wdc_ltx_prompt_optimizer/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt_template: promptTemplateInput.value || "" }),
        });
        await refreshOptimizerStatus();
      } catch (err) {
        promptTemplateStatus.textContent = err.message;
      } finally {
        setBusy(false);
      }
    });

    resetTemplateBtn.addEventListener("click", async () => {
      if (this.privacyLocked) return;
      setBusy(true);
      try {
        await fetchTimelineImageJson("/wdc_ltx_prompt_optimizer/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reset_prompt_template: true }),
        });
        await refreshOptimizerStatus();
      } catch (err) {
        promptTemplateStatus.textContent = err.message;
      } finally {
        setBusy(false);
      }
    });

    cancelBtn.addEventListener("click", () => {
      this.closePromptOptimizer();
    });
    for (const eventName of ["pointerdown", "pointerup", "mousedown", "mouseup", "click", "dblclick", "contextmenu", "wheel"]) {
      panel.addEventListener(eventName, (event) => event.stopPropagation());
    }
    for (const input of [hfTokenInput, saveTokenBtn, clearTokenBtn, editTemplateBtn, promptTemplateInput, saveTemplateBtn, resetTemplateBtn, modelSelect]) {
      for (const eventName of ["click", "keydown", "keyup", "keypress", "beforeinput"]) {
        input.addEventListener(eventName, (event) => event.stopPropagation());
      }
    }

    document.body.appendChild(overlay);
    renderRows();
    if (this.privacyLocked) {
      statusEl.textContent = "Private timeline data is locked. Decrypt it before optimizing prompts.";
      setBusy(false);
    } else {
      refreshOptimizerStatus().catch((err) => {
        statusEl.textContent = err.message;
      });
    }
  }

  closeTimelineImageBrowser() {
    if (document.querySelector(".pr-audio-browser-dialog")) {
      this.closeTimelineAudioBrowser();
    }
    document.querySelector(".pr-image-browser-dialog:not(.pr-audio-browser-dialog):not(.pr-prompt-optimizer-dialog)")?.remove();
    document.querySelector(".pr-image-large-preview")?.remove();
  }

  closeTimelineAudioBrowser() {
    this.stopTimelineAudioPreview();
    document.querySelector(".pr-audio-browser-dialog")?.remove();
  }

  showImagePreview(imageUrl, caption = "") {
    document.querySelector(".pr-image-large-preview")?.remove();
    const overlay = document.createElement("div");
    overlay.className = "pr-image-large-preview";
    overlay.innerHTML = `
      <div class="pr-image-large-preview-panel">
        <button class="pr-image-large-preview-close" type="button" title="Close preview" aria-label="Close preview">×</button>
        <img src="${escapeHtml(imageUrl)}" alt="">
        ${caption ? `<div class="pr-image-large-preview-caption">${escapeHtml(caption)}</div>` : ""}
      </div>`;
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay || event.target.closest(".pr-image-large-preview-close")) {
        overlay.remove();
      }
    });
    document.body.appendChild(overlay);
  }

  showTimelineImageLargePreview(alias, image) {
    const imageUrl = `/wdc_timeline_images/image?alias=${encodeURIComponent(alias)}&filename=${encodeURIComponent(image.filename)}&t=${encodeURIComponent(image.mtime || 0)}`;
    this.showImagePreview(imageUrl, `${image.filename} (${image.width || "?"}x${image.height || "?"})`);
  }

  showTimelineSegmentImagePreview(seg) {
    if (!seg?.imageB64) return;
    const caption = seg.imageFile || seg.fileName || "Timeline image";
    this.showImagePreview(seg.imageB64, caption);
  }

  async showTimelineImageFolderDialog(onDone) {
    const alias = prompt("Folder alias");
    if (!alias) return;
    const path = prompt("Folder path");
    if (!path) return;
    await fetchTimelineImageJson("/wdc_timeline_images/folders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alias, path }),
    });
    if (onDone) await onDone(alias);
  }

  async removeTimelineImageFolderDialog(onDone) {
    const data = await fetchTimelineImageJson("/wdc_timeline_images/folders");
    const removable = data.folders.filter((folder) => folder.alias !== "input");
    if (!removable.length) {
      alert("No custom folders to remove.");
      return;
    }
    const alias = prompt(`Folder alias to remove:\n${removable.map((folder) => folder.alias).join("\n")}`);
    if (!alias) return;
    await fetchTimelineImageJson(`/wdc_timeline_images/folders?alias=${encodeURIComponent(alias)}`, { method: "DELETE" });
    if (onDone) await onDone("input");
  }

  getTimelineImageUrl(folderAlias, image) {
    return image.image_url || `/wdc_timeline_images/image?alias=${encodeURIComponent(folderAlias)}&filename=${encodeURIComponent(image.filename)}&t=${encodeURIComponent(image.mtime || 0)}`;
  }

  getTimelineAudioUrl(folderAlias, audio) {
    return audio.audio_url || `/wdc_timeline_audio/audio?alias=${encodeURIComponent(folderAlias)}&filename=${encodeURIComponent(audio.filename)}&t=${encodeURIComponent(audio.mtime || 0)}`;
  }

  getTimelineAudioSegmentUrl(seg) {
    if (!seg?.audioFile) return "";
    if (seg.audioFolderAlias) {
      return api.apiURL(`/wdc_timeline_audio/audio?alias=${encodeURIComponent(seg.audioFolderAlias)}&filename=${encodeURIComponent(seg.audioFile)}`);
    }
    const filename = seg.audioFile.split("/").pop();
    const subfolder = seg.audioFile.includes("/") ? seg.audioFile.split("/").slice(0, -1).join("/") : "";
    return api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}`);
  }

  stopTimelineAudioPreview() {
    if (!this._timelineAudioPreview) return;
    try {
      this._timelineAudioPreview.pause();
      this._timelineAudioPreview.removeAttribute("src");
      this._timelineAudioPreview.load();
    } catch (err) { }
    this._timelineAudioPreview = null;
  }

  openAudioUploadPicker(targetFrameStart = null, targetLane = null) {
    this._pendingAudioUpload = (targetFrameStart !== null || targetLane !== null)
      ? { targetFrameStart, targetLane }
      : null;
    this.audioFileInput.value = "";
    this.audioFileInput.click();
    setTimeout(() => {
      const clearPendingIfCancelled = () => {
        setTimeout(() => {
          if (!this.audioFileInput.files || this.audioFileInput.files.length === 0) {
            this._pendingAudioUpload = null;
          }
        }, 500);
      };
      window.addEventListener("focus", clearPendingIfCancelled, { once: true });
    }, 0);
  }

  async showTimelineAudioFolderDialog(onDone) {
    const alias = prompt("Folder alias");
    if (!alias) return;
    const path = prompt("Folder path");
    if (!path) return;
    await fetchTimelineAudioJson("/wdc_timeline_audio/folders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alias, path }),
    });
    if (onDone) await onDone(alias);
  }

  async removeTimelineAudioFolderDialog(onDone) {
    const data = await fetchTimelineAudioJson("/wdc_timeline_audio/folders");
    const removable = data.folders.filter((folder) => folder.alias !== "input");
    if (!removable.length) {
      alert("No custom folders to remove.");
      return;
    }
    const alias = prompt(`Folder alias to remove:\n${removable.map((folder) => folder.alias).join("\n")}`);
    if (!alias) return;
    await fetchTimelineAudioJson(`/wdc_timeline_audio/folders?alias=${encodeURIComponent(alias)}`, { method: "DELETE" });
    if (onDone) await onDone("input");
  }

  formatAudioFileSize(bytes) {
    const value = Number(bytes || 0);
    if (!Number.isFinite(value) || value <= 0) return "";
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }

  formatAudioDuration(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) return "";
    const totalSeconds = Math.max(0, Math.round(value));
    if (totalSeconds < 60) return `${totalSeconds} s`;
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  getDecodeAudioContext() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return this.audioContext;
  }

  async decodeAudioArrayBuffer(arrayBuffer) {
    const audioCtx = this.getDecodeAudioContext();
    if (audioCtx.state !== "running") {
      try { await audioCtx.resume(); } catch (e) { }
    }
    return await audioCtx.decodeAudioData(arrayBuffer.slice(0));
  }

  getAudioWaveformPeaks(audioBuffer, numPeaks = 200) {
    const channelData = audioBuffer.getChannelData(0);
    const peaks = [];
    const sampleCount = channelData.length;
    if (!sampleCount) return Array(numPeaks).fill(0);

    for (let i = 0; i < numPeaks; i++) {
      const start = Math.floor((i / numPeaks) * sampleCount);
      const end = Math.max(start + 1, Math.floor(((i + 1) / numPeaks) * sampleCount));
      let max = 0;
      for (let j = start; j < end && j < sampleCount; j++) {
        const val = Math.abs(channelData[j]);
        if (val > max) max = val;
      }
      peaks.push(max);
    }
    return peaks;
  }

  addTimelineAudioSegmentFromBuffer(audioBuffer, options = {}) {
    const frameRate = this.getFrameRate();
    const visualDurationFrames = this.getVisualDurationFrames();
    const insertFrame = options.targetFrameStart === null || options.targetFrameStart === undefined
      ? Math.round(this.currentFrame || 0)
      : Math.round(options.targetFrameStart || 0);
    const clipFrames = Math.max(1, Math.ceil(audioBuffer.duration * frameRate));
    const newStart = clamp(insertFrame, 0, Math.max(0, visualDurationFrames - 1));
    const newLength = clipFrames;
    const newLane = this.findFreeAudioLane(newStart, newLength, null, options.targetLane);
    const loudness = analyzeAudioBufferLoudness(audioBuffer);

    const seg = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
      type: "audio",
      start: newStart,
      length: newLength,
      lane: newLane,
      trimStart: 0,
      audioDurationFrames: clipFrames,
      audioFile: options.audioFile,
      fileName: options.fileName || options.audioFile,
      volume: loudness.volume,
      waveformPeaks: this.getAudioWaveformPeaks(audioBuffer)
    };
    if (options.audioFolderAlias) {
      seg.audioFolderAlias = options.audioFolderAlias;
    }

    this.timeline.audioSegments.push(seg);
    this.ensureAudioTrackHeight();
    this.timeline.audioSegments.sort((a, b) => a.start - b.start);
    this.selectionType = "audio";
    this.selectedIndex = this.timeline.audioSegments.findIndex(s => s.id === seg.id);

    this.updateUIFromSelection();
    this.commitChanges(true);
    this.render();
    return seg;
  }

  async addTimelineAudioFromBrowser(folderAlias, audio, targetFrameStart = null, targetLane = null) {
    const audioUrl = this.getTimelineAudioUrl(folderAlias, audio);
    const resp = await fetch(api.apiURL(audioUrl));
    if (!resp.ok) throw new Error(`Could not load audio: ${audio.filename}`);
    const arrayBuffer = await resp.arrayBuffer();
    const audioBuffer = await this.decodeAudioArrayBuffer(arrayBuffer);
    this.addTimelineAudioSegmentFromBuffer(audioBuffer, {
      audioFolderAlias: folderAlias,
      audioFile: audio.filename,
      fileName: audio.filename.split("/").pop(),
      targetFrameStart,
      targetLane,
    });
  }

  async showTimelineAudioBrowser(targetFrameStart = null, targetLane = null) {
    this.closeTimelineImageBrowser();
    this.closeTimelineAudioBrowser();

    const overlay = document.createElement("div");
    overlay.className = "pr-image-browser-dialog pr-audio-browser-dialog";
    overlay.innerHTML = `
      <div class="pr-image-browser-panel">
        <h3>Add Timeline Audio</h3>
        <div class="pr-image-browser-controls" style="grid-template-columns: 1fr minmax(150px, 1fr) auto auto auto;">
          <select class="folder" title="Choose configured audio folder"></select>
          <input class="search" type="search" placeholder="Search audio..." title="Search loaded audio filenames and relative paths">
          <button class="scope pr-image-icon-btn" type="button" title="Recursive folder view" aria-label="Recursive folder view"></button>
          <button class="folder-add pr-image-icon-btn" type="button" title="Add configured audio folder" aria-label="Add configured audio folder">+</button>
          <button class="folder-remove pr-image-icon-btn" type="button" title="Remove configured audio folder" aria-label="Remove configured audio folder">−</button>
        </div>
        <span class="pr-image-browser-meta"></span>
        <div class="pr-audio-browser-list"></div>
        <div class="pr-image-browser-actions">
          <button class="upload-audio" type="button">Upload Audio...</button>
          <button class="cancel" type="button">Cancel</button>
          <button class="ok" type="button">Add Audio</button>
        </div>
      </div>`;

    const panel = overlay.querySelector(".pr-image-browser-panel");
    const folderSelect = overlay.querySelector(".folder");
    const searchInput = overlay.querySelector(".search");
    const scopeButton = overlay.querySelector(".scope");
    const folderAddButton = overlay.querySelector(".folder-add");
    const folderRemoveButton = overlay.querySelector(".folder-remove");
    const list = overlay.querySelector(".pr-audio-browser-list");
    const meta = overlay.querySelector(".pr-image-browser-meta");
    const previewAudio = new Audio();
    this._timelineAudioPreview = previewAudio;

    let availableAudios = [];
    let selectedAudio = null;
    let recursive = true;
    let playingFilename = null;

    const syncScopeButton = () => {
      scopeButton.title = recursive ? "Show audio recursively from subfolders" : "Show only audio directly in this folder";
      scopeButton.innerHTML = recursive
        ? `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h9a2 2 0 0 1 2 2v2"/><path d="M6 12v6a2 2 0 0 0 2 2h5"/><path d="M10 15h4l1.5 1.5H21v3.5H10z"/></svg>`
        : `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h10v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>`;
    };

    const syncPreviewButtons = () => {
      for (const button of list.querySelectorAll(".pr-audio-play")) {
        const isPlaying = button.dataset.filename === playingFilename && !previewAudio.paused;
        button.innerHTML = isPlaying ? ICONS.pause : ICONS.play;
        button.title = isPlaying ? "Pause preview" : "Play preview";
        button.setAttribute("aria-label", button.title);
      }
    };

    const selectAudio = (audio, row) => {
      selectedAudio = audio;
      for (const other of list.querySelectorAll(".pr-audio-row")) other.classList.remove("selected");
      row.classList.add("selected");
      meta.textContent = audio.filename;
    };

    const togglePreview = async (audio) => {
      const audioUrl = api.apiURL(this.getTimelineAudioUrl(folderSelect.value, audio));
      if (playingFilename === audio.filename && !previewAudio.paused) {
        previewAudio.pause();
        syncPreviewButtons();
        return;
      }
      if (playingFilename !== audio.filename) {
        previewAudio.pause();
        previewAudio.src = audioUrl;
        playingFilename = audio.filename;
      }
      try {
        await previewAudio.play();
      } catch (err) {
        playingFilename = null;
        meta.textContent = `Could not preview ${audio.filename}.`;
      }
      syncPreviewButtons();
    };

    const renderAudioList = () => {
      list.innerHTML = "";
      const query = searchInput.value.trim().toLowerCase();
      const visibleAudios = query
        ? availableAudios.filter((audio) => String(audio.filename || "").toLowerCase().includes(query))
        : availableAudios;

      if (selectedAudio && !visibleAudios.some((audio) => audio.filename === selectedAudio.filename)) {
        selectedAudio = null;
      }

      for (const audio of visibleAudios) {
        const row = document.createElement("div");
        row.className = `pr-audio-row${selectedAudio?.filename === audio.filename ? " selected" : ""}`;
        row.title = `${audio.filename}\nClick to select.`;
        row.tabIndex = 0;

        const playButton = document.createElement("button");
        playButton.type = "button";
        playButton.className = "pr-audio-play";
        playButton.dataset.filename = audio.filename;
        playButton.innerHTML = ICONS.play;
        playButton.addEventListener("click", async (event) => {
          event.stopPropagation();
          selectAudio(audio, row);
          await togglePreview(audio);
        });

        const details = document.createElement("div");
        details.className = "pr-audio-details";
        const sizeText = this.formatAudioFileSize(audio.size);
        const durationText = this.formatAudioDuration(audio.duration_seconds);
        const metaText = [sizeText, durationText].filter(Boolean).join(" · ");
        details.innerHTML = `
          <div class="pr-audio-name">${escapeHtml(audio.filename)}</div>
          <div class="pr-audio-size">${escapeHtml(metaText)}</div>`;

        row.appendChild(playButton);
        row.appendChild(details);
        row.addEventListener("click", () => selectAudio(audio, row));
        row.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectAudio(audio, row);
          }
        });
        list.appendChild(row);
      }

      if (!availableAudios.length) {
        meta.textContent = "No audio clips found.";
      } else if (!visibleAudios.length) {
        meta.textContent = `No audio clips match "${searchInput.value.trim()}".`;
      } else if (query) {
        meta.textContent = `${visibleAudios.length} of ${availableAudios.length} audio clips match. Select one to add.`;
      } else if (!selectedAudio) {
        meta.textContent = `${availableAudios.length} audio clips. Select one to add.`;
      }
      syncPreviewButtons();
    };

    const loadFolders = async (preferredAlias = null) => {
      const data = await fetchTimelineAudioJson("/wdc_timeline_audio/folders");
      folderSelect.innerHTML = data.folders.map((folder) => `<option value="${escapeHtml(folder.alias)}">${escapeHtml(folder.alias)}${folder.exists ? "" : " (missing)"}</option>`).join("");
      const lastAlias = preferredAlias || this.node.properties?.wdc_timeline_last_audio_folder_alias;
      if (lastAlias && data.folders.some((folder) => folder.alias === lastAlias)) {
        folderSelect.value = lastAlias;
      }
    };

    const loadAudios = async () => {
      this.node.properties = this.node.properties || {};
      this.node.properties.wdc_timeline_last_audio_folder_alias = folderSelect.value;
      previewAudio.pause();
      playingFilename = null;
      const data = await fetchTimelineAudioJson(`/wdc_timeline_audio/audios?alias=${encodeURIComponent(folderSelect.value)}&recursive=${recursive ? "1" : "0"}`);
      availableAudios = data.audios || [];
      selectedAudio = null;
      renderAudioList();
    };

    previewAudio.addEventListener("pause", syncPreviewButtons);
    previewAudio.addEventListener("ended", () => {
      playingFilename = null;
      syncPreviewButtons();
    });
    previewAudio.addEventListener("error", () => {
      playingFilename = null;
      syncPreviewButtons();
    });

    folderSelect.addEventListener("change", loadAudios);
    scopeButton.addEventListener("click", async () => {
      recursive = !recursive;
      syncScopeButton();
      await loadAudios();
    });
    folderAddButton.addEventListener("click", async () => {
      try {
        await this.showTimelineAudioFolderDialog(async (alias) => {
          await loadFolders(alias);
          await loadAudios();
        });
      } catch (err) {
        alert(err.message);
      }
    });
    folderRemoveButton.addEventListener("click", async () => {
      try {
        await this.removeTimelineAudioFolderDialog(async (alias) => {
          await loadFolders(alias);
          await loadAudios();
        });
      } catch (err) {
        alert(err.message);
      }
    });
    searchInput.addEventListener("input", renderAudioList);
    overlay.querySelector(".upload-audio").addEventListener("click", () => {
      this.closeTimelineAudioBrowser();
      this.openAudioUploadPicker(targetFrameStart, targetLane);
    });
    overlay.querySelector(".cancel").addEventListener("click", () => this.closeTimelineAudioBrowser());
    overlay.querySelector(".ok").addEventListener("click", async () => {
      if (!selectedAudio) {
        alert("Select an audio clip first.");
        return;
      }
      try {
        await this.addTimelineAudioFromBrowser(folderSelect.value, selectedAudio, targetFrameStart, targetLane);
        this.closeTimelineAudioBrowser();
      } catch (err) {
        alert(err.message);
      }
    });
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) this.closeTimelineAudioBrowser();
    });

    document.body.appendChild(overlay);
    syncScopeButton();
    try {
      await loadFolders();
      await loadAudios();
    } catch (err) {
      meta.textContent = err.message;
    }
    panel.focus?.();
  }

  getSelectedImageSegment() {
    if (this.selectionType !== "image" || this.selectedIndex < 0) return null;
    const seg = this.timeline.segments[this.selectedIndex];
    if (!seg || seg.type !== "image" || !seg.imageB64) return null;
    return seg;
  }

  isSourceVideoSegment(seg) {
    return !!seg && seg.type === "source_video";
  }

  getSourceVideoSegment() {
    return this.timeline.segments.find((seg) => this.isSourceVideoSegment(seg)) || null;
  }

  getSourceVideoGuideFrames(seg) {
    return clamp(
      parseInt(seg?.sourceVideoGuideFrames ?? seg?.length ?? SOURCE_VIDEO_DEFAULT_GUIDE_FRAMES, 10) || SOURCE_VIDEO_DEFAULT_GUIDE_FRAMES,
      1,
      SOURCE_VIDEO_MAX_GUIDE_FRAMES
    );
  }

  async captureVideoFinalFrame(file) {
    const url = URL.createObjectURL(file);
    try {
      const video = document.createElement("video");
      video.preload = "metadata";
      video.muted = true;
      video.playsInline = true;
      await new Promise((resolve, reject) => {
        video.onloadedmetadata = resolve;
        video.onerror = () => reject(new Error(`Could not read video metadata: ${file.name}`));
        video.src = url;
      });

      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
      if (duration > 0.05) {
        video.currentTime = Math.max(0, duration - 0.05);
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(resolve, 1200);
          video.onseeked = () => { clearTimeout(timeout); resolve(); };
          video.onerror = () => { clearTimeout(timeout); reject(new Error(`Could not seek video: ${file.name}`)); };
        });
      } else if (video.readyState < 2) {
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(resolve, 1200);
          video.onloadeddata = () => { clearTimeout(timeout); resolve(); };
          video.onerror = () => { clearTimeout(timeout); reject(new Error(`Could not load video frame: ${file.name}`)); };
        });
      }

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 512;
      canvas.height = video.videoHeight || 512;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.9);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async showTimelineImageBrowser(targetFrameStart = null, explicitLength = null, options = {}) {
    this.closeTimelineImageBrowser();
    const isReplace = options.mode === "replace";
    const targetSegmentId = options.segmentId || null;
    const title = isReplace ? "Replace Timeline Image" : "Add Timeline Image";
    const okLabel = isReplace ? "Replace Image" : "Add Image";

    const overlay = document.createElement("div");
    overlay.className = "pr-image-browser-dialog";
    overlay.innerHTML = `
      <div class="pr-image-browser-panel">
        <h3>${escapeHtml(title)}</h3>
        <div class="pr-image-browser-controls">
          <select class="folder" title="Choose configured image folder"></select>
          <input class="search" type="search" placeholder="Search images..." title="Search loaded image filenames and relative paths">
          <div class="pr-image-sort-wrap">
            <button class="sort pr-image-sort-btn" type="button" title="Sort images" aria-haspopup="true" aria-expanded="false">
              <span>Newest</span>
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>
            </button>
            <div class="pr-image-sort-menu" role="menu"></div>
          </div>
          <button class="scope pr-image-icon-btn" type="button" title="Recursive folder view" aria-label="Recursive folder view"></button>
          <button class="folder-add pr-image-icon-btn" type="button" title="Add configured image folder" aria-label="Add configured image folder">+</button>
          <button class="folder-remove pr-image-icon-btn" type="button" title="Remove configured image folder" aria-label="Remove configured image folder">−</button>
          <label class="pr-image-columns-control" title="Thumbnail columns per row">
            <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
            <input class="columns" type="range" min="2" max="8" step="1" value="4">
            <span class="columns-value">4</span>
          </label>
        </div>
        <div class="pr-image-browser-controls" style="grid-template-columns: auto 1fr;">
          <button class="hover-hide pr-image-icon-btn" type="button" title="Hide thumbnails until hovering over window" aria-label="Hide thumbnails until hovering over window"></button>
          <span class="pr-image-browser-meta"></span>
        </div>
        <div class="pr-image-browser-grid hide-images"></div>
        <div class="pr-image-browser-actions">
          <button class="cancel" type="button">Cancel</button>
          <button class="ok" type="button">${escapeHtml(okLabel)}</button>
        </div>
      </div>`;

    const panel = overlay.querySelector(".pr-image-browser-panel");
    const folderSelect = overlay.querySelector(".folder");
    const searchInput = overlay.querySelector(".search");
    const sortButton = overlay.querySelector(".sort");
    const sortButtonLabel = sortButton.querySelector("span");
    const sortMenu = overlay.querySelector(".pr-image-sort-menu");
    const scopeButton = overlay.querySelector(".scope");
    const folderAddButton = overlay.querySelector(".folder-add");
    const folderRemoveButton = overlay.querySelector(".folder-remove");
    const columnsInput = overlay.querySelector(".columns");
    const columnsValue = overlay.querySelector(".columns-value");
    const hoverHideButton = overlay.querySelector(".hover-hide");
    const grid = overlay.querySelector(".pr-image-browser-grid");
    const meta = overlay.querySelector(".pr-image-browser-meta");

    let availableImages = [];
    let selectedImage = null;
    let recursive = true;
    let hideImagesUntilHover = true;
    let sortMode = "newest";
    const sortOptions = [
      { value: "newest", label: "Newest" },
      { value: "oldest", label: "Oldest" },
      { value: "name-asc", label: "Name A-Z" },
      { value: "name-desc", label: "Name Z-A" },
    ];

    const syncScopeButton = () => {
      scopeButton.title = recursive ? "Show images recursively from subfolders" : "Show only images directly in this folder";
      scopeButton.innerHTML = recursive
        ? `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h9a2 2 0 0 1 2 2v2"/><path d="M6 12v6a2 2 0 0 0 2 2h5"/><path d="M10 15h4l1.5 1.5H21v3.5H10z"/></svg>`
        : `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h10v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>`;
    };

    const syncGridVisibility = () => {
      hoverHideButton.title = hideImagesUntilHover ? "Hide thumbnails until the mouse is over this window" : "Always show thumbnails in this window";
      hoverHideButton.innerHTML = hideImagesUntilHover
        ? `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/><circle cx="12" cy="12" r="3"/><path d="M3 3l18 18"/></svg>`
        : `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/><circle cx="12" cy="12" r="3"/></svg>`;
      grid.classList.toggle("hide-images", hideImagesUntilHover);
      grid.classList.toggle("show-images", !hideImagesUntilHover);
    };

    const syncColumns = () => {
      const columns = Number(columnsInput.value || 4);
      grid.style.setProperty("--pr-image-columns", String(columns));
      columnsValue.textContent = String(columns);
    };

    const compareImageNames = (a, b) => String(a.filename || "").localeCompare(String(b.filename || ""), undefined, { sensitivity: "base" });

    const sortedImages = () => availableImages
      .map((image, index) => ({ image, index }))
      .sort((a, b) => {
        let cmp = 0;
        if (sortMode === "newest") {
          cmp = Number(b.image.mtime || 0) - Number(a.image.mtime || 0);
        } else if (sortMode === "oldest") {
          cmp = Number(a.image.mtime || 0) - Number(b.image.mtime || 0);
        } else if (sortMode === "name-desc") {
          cmp = compareImageNames(b.image, a.image);
        } else {
          cmp = compareImageNames(a.image, b.image);
        }
        if (cmp !== 0) return cmp;
        cmp = compareImageNames(a.image, b.image);
        if (cmp !== 0) return cmp;
        return a.index - b.index;
      })
      .map((entry) => entry.image);

    const syncSortMenu = () => {
      const active = sortOptions.find((option) => option.value === sortMode) || sortOptions[0];
      sortButtonLabel.textContent = active.label;
      for (const button of sortMenu.querySelectorAll(".pr-image-sort-option")) {
        button.classList.toggle("active", button.dataset.sortMode === sortMode);
      }
    };

    const closeSortMenu = () => {
      sortMenu.classList.remove("is-open");
      sortButton.setAttribute("aria-expanded", "false");
    };

    const toggleSortMenu = () => {
      const open = !sortMenu.classList.contains("is-open");
      sortMenu.classList.toggle("is-open", open);
      sortButton.setAttribute("aria-expanded", open ? "true" : "false");
    };

    const renderImageGrid = () => {
      grid.innerHTML = "";
      const query = searchInput.value.trim().toLowerCase();
      const images = sortedImages();
      const visibleImages = query
        ? images.filter((image) => String(image.filename || "").toLowerCase().includes(query))
        : images;

      if (selectedImage && !visibleImages.some((image) => image.filename === selectedImage.filename)) {
        selectedImage = null;
      }

      for (const image of visibleImages) {
        const tile = document.createElement("button");
        tile.type = "button";
        tile.className = `pr-image-tile${selectedImage?.filename === image.filename ? " selected" : ""}`;
        tile.title = `${image.filename}\nClick to select. Ctrl-click for large preview.`;
        tile.innerHTML = `<img src="${escapeHtml(image.thumb_url)}" alt="">`;
        tile.addEventListener("click", (event) => {
          if (event.ctrlKey) {
            this.showTimelineImageLargePreview(folderSelect.value, image);
            return;
          }
          selectedImage = image;
          for (const other of grid.querySelectorAll(".pr-image-tile")) other.classList.remove("selected");
          tile.classList.add("selected");
          meta.textContent = `${image.filename} (${image.width || "?"}x${image.height || "?"})`;
        });
        grid.appendChild(tile);
      }

      if (!availableImages.length) {
        meta.textContent = "No images found.";
      } else if (!visibleImages.length) {
        meta.textContent = `No images match "${searchInput.value.trim()}".`;
      } else if (query) {
        meta.textContent = `${visibleImages.length} of ${availableImages.length} images match. Select one to add.`;
      } else if (!selectedImage) {
        meta.textContent = `${availableImages.length} images. Select one to add.`;
      }
      syncGridVisibility();
    };

    const loadFolders = async (preferredAlias = null) => {
      const data = await fetchTimelineImageJson("/wdc_timeline_images/folders");
      folderSelect.innerHTML = data.folders.map((folder) => `<option value="${escapeHtml(folder.alias)}">${escapeHtml(folder.alias)}${folder.exists ? "" : " (missing)"}</option>`).join("");
      const lastAlias = preferredAlias || this.node.properties?.wdc_timeline_last_folder_alias;
      if (lastAlias && data.folders.some((folder) => folder.alias === lastAlias)) {
        folderSelect.value = lastAlias;
      }
    };

    const loadImages = async () => {
      this.node.properties = this.node.properties || {};
      this.node.properties.wdc_timeline_last_folder_alias = folderSelect.value;
      const params = new URLSearchParams({
        alias: folderSelect.value,
        recursive: recursive ? "1" : "0",
      });
      if (this.isPrivacyModeEnabled()) {
        params.set("privacy", "1");
      }
      if (this.thumbnailCacheBust) {
        params.set("cacheBust", this.thumbnailCacheBust);
      }
      const data = await fetchTimelineImageJson(`/wdc_timeline_images/images?${params.toString()}`);
      availableImages = data.images || [];
      selectedImage = null;
      renderImageGrid();
    };

    folderSelect.addEventListener("change", loadImages);
    scopeButton.addEventListener("click", async () => {
      recursive = !recursive;
      syncScopeButton();
      await loadImages();
    });
    folderAddButton.addEventListener("click", async () => {
      try {
        await this.showTimelineImageFolderDialog(async (alias) => {
          await loadFolders(alias);
          await loadImages();
        });
      } catch (err) {
        alert(err.message);
      }
    });
    folderRemoveButton.addEventListener("click", async () => {
      try {
        await this.removeTimelineImageFolderDialog(async (alias) => {
          await loadFolders(alias);
          await loadImages();
        });
      } catch (err) {
        alert(err.message);
      }
    });
    searchInput.addEventListener("input", renderImageGrid);
    columnsInput.addEventListener("input", syncColumns);
    sortButton.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleSortMenu();
    });
    sortMenu.innerHTML = sortOptions.map((option) => (
      `<button class="pr-image-sort-option${option.value === sortMode ? " active" : ""}" type="button" role="menuitem" data-sort-mode="${escapeHtml(option.value)}">${escapeHtml(option.label)}</button>`
    )).join("");
    sortMenu.addEventListener("click", (event) => {
      const option = event.target.closest(".pr-image-sort-option");
      if (!option) return;
      sortMode = option.dataset.sortMode || "newest";
      syncSortMenu();
      closeSortMenu();
      renderImageGrid();
    });
    hoverHideButton.addEventListener("click", () => {
      hideImagesUntilHover = !hideImagesUntilHover;
      syncGridVisibility();
    });
    overlay.addEventListener("click", (event) => {
      if (!event.target.closest(".pr-image-sort-wrap")) closeSortMenu();
    });
    overlay.querySelector(".cancel").addEventListener("click", () => this.closeTimelineImageBrowser());
    overlay.querySelector(".ok").addEventListener("click", async () => {
      if (!selectedImage) {
        alert("Select an image first.");
        return;
      }
      try {
        if (isReplace) {
          await this.replaceTimelineImageFromBrowser(targetSegmentId, folderSelect.value, selectedImage);
        } else {
          await this.addTimelineImageFromBrowser(folderSelect.value, selectedImage, targetFrameStart, explicitLength);
        }
        this.closeTimelineImageBrowser();
      } catch (err) {
        alert(err.message);
      }
    });
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) this.closeTimelineImageBrowser();
    });

    document.body.appendChild(overlay);
    syncColumns();
    syncScopeButton();
    syncSortMenu();
    syncGridVisibility();
    try {
      await loadFolders();
      await loadImages();
    } catch (err) {
      meta.textContent = err.message;
    }
    panel.focus?.();
  }

  async addTimelineImageFromBrowser(folderAlias, image, targetFrameStart = null, explicitLength = null) {
    const frameRate = this.getFrameRate();
    const newLength = explicitLength !== null ? explicitLength : frameRate * 1;
    const imageUrl = this.getTimelineImageUrl(folderAlias, image);

    await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        let newStart = targetFrameStart;
        if (newStart === null) {
          newStart = 0;
          this.timeline.segments.sort((a, b) => a.start - b.start);
          for (let i = 0; i < this.timeline.segments.length; i++) {
            let seg = this.timeline.segments[i];
            if (newStart + newLength <= seg.start) break;
            newStart = Math.max(newStart, seg.start + seg.length);
          }
        }

        const currentDuration = this.getVisualDurationFrames();
        if (targetFrameStart !== null) {
          let tempId = "TEMP_" + Date.now();
          this.timeline.segments.push({ id: tempId, start: newStart, length: newLength, type: "temp" });
          let result = this._applyCenterDragPhysics(this.timeline.segments, tempId, newStart, newStart + newLength / 2, currentDuration, currentDuration, 1);
          for (let shiftedSeg of result) {
            let original = this.timeline.segments.find(s => s.id === shiftedSeg.id);
            if (original) {
              original.start = shiftedSeg.resolvedStart !== undefined ? shiftedSeg.resolvedStart : shiftedSeg.start;
            }
          }
          let tempSeg = this.timeline.segments.find(s => s.id === tempId);
          newStart = tempSeg.start;
          this.timeline.segments = this.timeline.segments.filter(s => s.id !== tempId);
        }

        const seg = {
          id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
          start: newStart,
          length: newLength,
          prompt: "",
          type: "image",
          imageFolderAlias: folderAlias,
          imageFile: image.filename,
          imageB64: imageUrl,
          imgObj: img,
        };

        this.timeline.segments.push(seg);
        this.timeline.segments.sort((a, b) => a.start - b.start);
        this.selectionType = "image";
        this.selectedIndex = this.timeline.segments.findIndex(s => s.id === seg.id);
        this.updateUIFromSelection();
        this.commitChanges(true);
        this.render();
        resolve();
      };
      img.onerror = () => reject(new Error(`Could not load image: ${image.filename}`));
      img.src = imageUrl;
    });
  }

  async replaceTimelineImageFromBrowser(segmentId, folderAlias, image) {
    if (!segmentId) throw new Error("No image segment is selected.");

    const imageUrl = this.getTimelineImageUrl(folderAlias, image);
    await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const idx = this.timeline.segments.findIndex((seg) => seg.id === segmentId);
        const seg = this.timeline.segments[idx];
        if (idx < 0 || !seg || seg.type === "text" || !seg.imageB64) {
          reject(new Error("The selected image segment is no longer available."));
          return;
        }

        seg.imageFolderAlias = folderAlias;
        seg.imageFile = image.filename;
        seg.imageB64 = imageUrl;
        seg.imgObj = img;

        this.selectionType = "image";
        this.selectedIndex = idx;
        this.updateUIFromSelection();
        this.commitChanges(true);
        this.render();
        resolve();
      };
      img.onerror = () => reject(new Error(`Could not load image: ${image.filename}`));
      img.src = imageUrl;
    });
  }

  shiftTimelineSegments(deltaFrames) {
    if (!deltaFrames) return;
    for (const seg of this.timeline.segments) {
      if (!this.isSourceVideoSegment(seg)) {
        seg.start = Math.max(0, Math.round((seg.start || 0) + deltaFrames));
      }
    }
  }

  removeExistingSourceVideoForReplace() {
    const existing = this.getSourceVideoSegment();
    if (!existing) return 0;
    const oldLength = this.getSourceVideoGuideFrames(existing);
    this.timeline.segments = this.timeline.segments.filter((seg) => seg.id !== existing.id);
    this.shiftTimelineSegments(-oldLength);
    return oldLength;
  }

  async handleSourceVideoUpload(files) {
    const file = files?.[0];
    if (!file || !file.type.startsWith("video/")) {
      this.videoSourceInput.value = "";
      return;
    }

    const existing = this.getSourceVideoSegment();
    if (existing && !confirm("Replace the existing source video?")) {
      this.videoSourceInput.value = "";
      return;
    }

    try {
      const guideFrames = SOURCE_VIDEO_DEFAULT_GUIDE_FRAMES;
      const previewDataUrl = await this.captureVideoFinalFrame(file);

      const body = new FormData();
      body.append("image", file);
      const resp = await api.fetchApi("/upload/image", { method: "POST", body });
      if (resp.status !== 200) throw new Error(`Could not upload video: ${file.name}`);

      const data = await resp.json();
      const filename = data.name;
      const subfolder = data.subfolder || "";
      const videoFile = subfolder ? subfolder + "/" + filename : filename;

      this.removeExistingSourceVideoForReplace();
      this.shiftTimelineSegments(guideFrames);

      const sourceSeg = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
        start: 0,
        length: guideFrames,
        prompt: "",
        type: "source_video",
        locked: true,
        videoFile,
        fileName: filename,
        sourceVideoGuideFrames: guideFrames,
        guideStrength: 0.85,
        imageB64: previewDataUrl,
      };

      const img = new Image();
      img.onload = () => {
        sourceSeg.imgObj = img;
        this.render();
      };
      img.src = previewDataUrl;

      this.timeline.segments.push(sourceSeg);
      this.timeline.segments.sort((a, b) => a.start - b.start);
      this.selectionType = "image";
      this.selectedIndex = this.timeline.segments.findIndex((seg) => seg.id === sourceSeg.id);

      const furthest = Math.max(...this.timeline.segments.map((seg) => seg.start + seg.length), 0);
      this.growTimelineIfNeeded(furthest);
      this.updateUIFromSelection();
      this.commitChanges(true);
      this.render();
    } catch (err) {
      alert(err.message);
      console.error("[PromptRelay] Source video upload failed", err);
    } finally {
      this.videoSourceInput.value = "";
    }
  }

  setSelectedSourceVideoGuideFrames(value) {
    if (this.selectionType !== "image" || this.selectedIndex < 0) return;
    const seg = this.timeline.segments[this.selectedIndex];
    if (!this.isSourceVideoSegment(seg)) return;

    const nextFrames = clamp(Math.round(value), 1, SOURCE_VIDEO_MAX_GUIDE_FRAMES);
    const prevFrames = this.getSourceVideoGuideFrames(seg);
    const delta = nextFrames - prevFrames;

    seg.sourceVideoGuideFrames = nextFrames;
    seg.length = nextFrames;
    this.shiftTimelineSegments(delta);
    this.timeline.segments.sort((a, b) => a.start - b.start);
    this.selectedIndex = this.timeline.segments.findIndex((s) => s.id === seg.id);

    const furthest = Math.max(...this.timeline.segments.map((s) => s.start + s.length), 0);
    this.growTimelineIfNeeded(furthest);
    this.updateUIFromSelection();
    this.commitChanges(true);
    this.render();
  }

  // --- Async Image Upload Logic (Handles multiple images simultaneously) ---
  async handleImageUpload(files, targetFrameStart = null, explicitLength = null) {
    const frameRate = this.getFrameRate();
    const durationFrames = this.getDurationFrames();
    const newLength = explicitLength !== null ? explicitLength : frameRate * 1; // Default to 1 second long

    for (let file of files) {
      if (!file.type.startsWith("image/")) continue;

      await new Promise(async (resolve) => {
        try {
          const body = new FormData();
          body.append("image", file);
          const resp = await api.fetchApi("/upload/image", { method: "POST", body });
          if (resp.status !== 200) { resolve(); return; }

          const data = await resp.json();
          const filename = data.name;
          const subfolder = data.subfolder || "";
          const imageFile = subfolder ? subfolder + "/" + filename : filename;
          const imgUrl = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}`);

          const img = new Image();
          img.onload = () => {

            let newStart = targetFrameStart;
            if (newStart === null) {
              // Fallback: find the first free slot, or append past the end
              newStart = 0;
              this.timeline.segments.sort((a, b) => a.start - b.start);
              for (let i = 0; i < this.timeline.segments.length; i++) {
                let seg = this.timeline.segments[i];
                if (newStart + newLength <= seg.start) break;
                newStart = Math.max(newStart, seg.start + seg.length);
              }
            }

            // Use the visual timeline as the physics bound so segments can
            // land anywhere in the padded visual area without touching duration_frames.
            const currentDuration = this.getVisualDurationFrames();

            if (targetFrameStart !== null) {
              // Resolve physics to push existing segments
              let tempId = "TEMP_" + Date.now();
              this.timeline.segments.push({ id: tempId, start: newStart, length: newLength, type: "temp" });
              let result = this._applyCenterDragPhysics(this.timeline.segments, tempId, newStart, newStart + newLength / 2, currentDuration, currentDuration, 1);

              // Update original segments with resolved physics to preserve imgObj
              for (let shiftedSeg of result) {
                let original = this.timeline.segments.find(s => s.id === shiftedSeg.id);
                if (original) {
                  original.start = shiftedSeg.resolvedStart !== undefined ? shiftedSeg.resolvedStart : shiftedSeg.start;
                }
              }

              let tempSeg = this.timeline.segments.find(s => s.id === tempId);
              newStart = tempSeg.start;
              this.timeline.segments = this.timeline.segments.filter(s => s.id !== tempId);
              targetFrameStart = newStart + newLength; // For the next file in batch
            }

            // Use the full intended length — the timeline has already been grown to fit.
            let constrainedLength = newLength;

            const seg = {
              id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
              start: newStart,
              length: constrainedLength,
              prompt: "",
              type: "image",
              imageFile: imageFile,
              imageB64: imgUrl
            };

            const displayImg = new Image();
            displayImg.onload = () => {
              seg.imgObj = displayImg;
              this.render();
              resolve(); // Resolve promise letting next image process
            };
            displayImg.src = imgUrl;

            this.timeline.segments.push(seg);
            this.timeline.segments.sort((a, b) => a.start - b.start);
            this.selectionType = "image";
            this.selectedIndex = this.timeline.segments.findIndex(s => s.id === seg.id);

            this.updateUIFromSelection();
            this.commitChanges(true);
          };
          img.src = imgUrl;
        } catch (err) {
          console.error("[PromptRelay] Image upload failed", err);
          resolve();
        }
      });
    }
    this.fileInput.value = "";
  }

  // --- Async Audio Upload Logic ---
  async handleAudioUpload(files, targetFrameStart = null, targetLane = null) {
    const insertFrame = targetFrameStart === null
      ? Math.round(this.currentFrame || 0)
      : Math.round(targetFrameStart || 0);

    for (let file of files) {
      if (!file.type.startsWith("audio/")) continue;

      await new Promise(async (resolve) => {
        try {
          const body = new FormData();
          body.append("image", file);
          const resp = await api.fetchApi("/upload/image", { method: "POST", body });
          if (resp.status !== 200) { resolve(); return; }

          const data = await resp.json();
          const filename = data.name;
          const subfolder = data.subfolder || "";
          const audioFile = subfolder ? subfolder + "/" + filename : filename;

          const arrayBuffer = await file.arrayBuffer();
          const audioBuffer = await this.decodeAudioArrayBuffer(arrayBuffer);
          this.addTimelineAudioSegmentFromBuffer(audioBuffer, {
            audioFile,
            fileName: file.name,
            targetFrameStart: insertFrame,
            targetLane,
          });
          resolve();
        } catch (err) {
          console.error("[PromptRelay] Audio processing failed", err);
          resolve();
        }
      });
    }
    this.audioFileInput.value = "";
  }

  deleteSelectedSegment() {
    if (this.selectionType === "audio") {
      if (this.timeline.audioSegments.length === 0 || this.selectedIndex === -1) return;
      this.timeline.audioSegments.splice(this.selectedIndex, 1);
      this.selectedIndex = Math.max(-1, this.selectedIndex - 1);
    } else {
      if (this.timeline.segments.length === 0 || this.selectedIndex === -1) return;
      this.timeline.segments.splice(this.selectedIndex, 1);
      this.selectedIndex = Math.max(-1, this.selectedIndex - 1);
    }
    this.updateUIFromSelection();
    this.commitChanges();
    this.render();
  }

  formatTime(frames, dropSuffix = false) {
    const mode = this.displayModeWidget ? this.displayModeWidget.value : "seconds";
    if (mode === "seconds") {
      const secs = frames / this.getFrameRate();
      return dropSuffix ? secs.toFixed(2) : secs.toFixed(2) + "s";
    }
    return dropSuffix ? Math.round(frames).toString() : Math.round(frames) + " frames";
  }

  updateWidgetVisibility() {
    if (this.durationFramesWidget) {
      this.durationFramesWidget.type = "INT";
      if (!this.durationFramesWidget.options) this.durationFramesWidget.options = {};
      this.durationFramesWidget.options.hidden = false;
      this.durationFramesWidget.hidden = false;
      delete this.durationFramesWidget.computeSize;
    }
    if (this.durationSecondsWidget) {
      this.durationSecondsWidget.type = "FLOAT";
      if (!this.durationSecondsWidget.options) this.durationSecondsWidget.options = {};
      this.durationSecondsWidget.options.hidden = false;
      this.durationSecondsWidget.hidden = false;
      delete this.durationSecondsWidget.computeSize;
    }

    // Force node resize and redraw deferred to next tick
    setTimeout(() => {
      if (this.node && this.node.computeSize) {
        const sz = this.node.computeSize();
        this.node.size[1] = sz[1];
        if (window.app && window.app.graph) {
          window.app.graph.setDirtyCanvas(true, true);
        }
      }
    }, 0);
  }

  updateUIFromSelection() {
    let seg = null;
    if (this.selectedIndex >= 0) {
      if (this.selectionType === "audio") {
        const origSeg = this.timeline.audioSegments[this.selectedIndex];
        if (origSeg) {
          const previewIsAudio = this._ghostTrack === 'audio' || (this._previewSegments && this._ghostTrack === null && this.selectionType === 'audio');
          const arr = (this._previewSegments && previewIsAudio) ? this._previewSegments : this.timeline.audioSegments;
          seg = arr.find(s => s.id === origSeg.id) || origSeg;
        }
      } else {
        const origSeg = this.timeline.segments[this.selectedIndex];
        if (origSeg) {
          const previewIsImage = this._ghostTrack === 'image' || (this._previewSegments && this._ghostTrack === null && this.selectionType === 'image');
          const arr = (this._previewSegments && previewIsImage) ? this._previewSegments : this.timeline.segments;
          seg = arr.find(s => s.id === origSeg.id) || origSeg;
        }
      }
    }

    if (this.selectionType === "audio" && seg) {
      this.promptInput.style.display = "none";
      this.strengthRow.style.display = "flex";
      this.audioInfoArea.style.display = "block";
      const volume = clampVolume(seg.volume);
      seg.volume = volume;
      this.audioInfoArea.innerHTML = `
        File: <span>${seg.fileName || "Unknown"}</span><br>
        Length: <span>${this.formatTime(seg.audioDurationFrames)}</span> Output Length: <span>${this.formatTime(seg.length)}</span><br>
        Trim-in: <span>${this.formatTime(Math.round(seg.trimStart))}</span> Trim-Out: <span>${this.formatTime(Math.round(seg.audioDurationFrames - (seg.trimStart + seg.length)))}</span><br>
        Volume: <span>${volume.toFixed(2)}x</span> Lane: <span>${normalizeAudioLane(seg.lane) + 1}</span>
      `;
      if (this.strengthLabel) this.strengthLabel.textContent = "Volume:";
      this.strengthValue.value = volume.toFixed(2);
      this.strengthValue.disabled = false;
      this.sourceVideoFramesLabel.style.display = "none";
      this.sourceVideoFramesInput.style.display = "none";
    } else {
      this.audioInfoArea.style.display = "none";
      this.promptInput.style.display = "block";
      this.strengthRow.style.display = "flex";
      if (this.strengthLabel) this.strengthLabel.textContent = "Guide Strength:";

      if (seg) {
        this.promptInput.value = seg.prompt || "";
        this.promptInput.disabled = this.privacyLocked;

        const isImage = seg.type !== "text";
        const strength = isImage ? (seg.guideStrength ?? 1.0) : 1.0;
        this.strengthValue.value = strength.toFixed(2);
        this.strengthValue.disabled = !isImage;

        const isSourceVideo = this.isSourceVideoSegment(seg);
        this.sourceVideoFramesLabel.style.display = isSourceVideo ? "" : "none";
        this.sourceVideoFramesInput.style.display = isSourceVideo ? "" : "none";
        this.sourceVideoFramesInput.value = String(this.getSourceVideoGuideFrames(seg));
      } else {
        this.promptInput.value = "";
        this.promptInput.disabled = true;
        if (this.strengthLabel) this.strengthLabel.textContent = "Guide Strength:";
        this.strengthValue.value = "1.00";
        this.strengthValue.disabled = true;
        this.sourceVideoFramesLabel.style.display = "none";
        this.sourceVideoFramesInput.style.display = "none";
      }
    }

    if (this.segmentBoundsDisplay) {
      if (seg) {
        const startStr = this.formatTime(seg.start, true);
        const endStr = this.formatTime(seg.start + seg.length, true);
        this.segmentBoundsDisplay.textContent = `Start: ${startStr} | End: ${endStr}`;
      } else {
        this.segmentBoundsDisplay.textContent = "Start: - | End: -";
      }
    }

    if (this.replaceImageBtn) {
      const canReplace = !!this.getSelectedImageSegment();
      this.replaceImageBtn.disabled = !canReplace;
      this.replaceImageBtn.title = canReplace
        ? "Replace the selected segment image."
        : "Select an image segment to replace its image.";
    }

    this.updatePromptPrivacyVisibility();
  }

  // --- Rendering logic ---
  render() {
    const width = this.canvas.offsetWidth || this._lastWidth;
    const height = this.canvasHeight;
    const totalFrames = this.getVisualDurationFrames();

    if (!width || width <= 0) return;

    this.ctx.clearRect(0, 0, width, height);



    // Render Track Backgrounds
    this.ctx.fillStyle = "#111"; // Image track bg
    this.ctx.fillRect(0, RULER_HEIGHT, width, this.blockHeight);
    this.ctx.fillStyle = "#111"; // Audio track bg
    this.ctx.fillRect(0, RULER_HEIGHT + this.blockHeight, width, this.audioTrackHeight);



    // Determine which track the preview belongs to.
    // _ghostTrack is set during HTML file drag-and-drop.
    // During canvas mouse drags, _ghostTrack is null, so fall back to selectionType.
    const previewIsAudio = this._ghostTrack === 'audio' ||
      (this._previewSegments && this._ghostTrack === null && this.selectionType === 'audio');

    let renderSegments = (this._previewSegments && !previewIsAudio)
      ? this._previewSegments : this.timeline.segments;

    let renderAudioSegments = (this._previewSegments && previewIsAudio)
      ? this._previewSegments : this.timeline.audioSegments;



    const activeSegId = this.timeline.segments[this.selectedIndex]?.id;
    const activeAudioSegId = this.timeline.audioSegments[this.selectedIndex]?.id;

    // Sort segments so that the selected one is drawn last (on top)
    const isImageSelection = this.selectionType === "image";
    const sortedSegments = [...renderSegments].sort((a, b) => {
      const aSel = isImageSelection && a.id === activeSegId;
      const bSel = isImageSelection && b.id === activeSegId;
      return aSel - bSel;
    });

    const isAudioSelection = this.selectionType === "audio";
    const sortedAudioSegments = [...renderAudioSegments].sort((a, b) => {
      const aSel = isAudioSelection && a.id === activeAudioSegId;
      const bSel = isAudioSelection && b.id === activeAudioSegId;
      if (aSel !== bSel) return aSel ? 1 : -1;
      const laneDiff = normalizeAudioLane(a.lane) - normalizeAudioLane(b.lane);
      return laneDiff || ((a.start || 0) - (b.start || 0));
    });

    const audioTrackY = RULER_HEIGHT + this.blockHeight;
    const audioLaneCount = this.getAudioLaneCount(renderAudioSegments);
    for (let lane = 0; lane < audioLaneCount; lane++) {
      const laneY = audioTrackY + lane * AUDIO_LANE_HEIGHT;
      this.ctx.fillStyle = lane % 2 === 0 ? "rgba(255,255,255,0.015)" : "rgba(255,255,255,0.035)";
      this.ctx.fillRect(0, laneY, width, AUDIO_LANE_HEIGHT);
      if (lane > 0) {
        this.ctx.fillStyle = "rgba(255,255,255,0.08)";
        this.ctx.fillRect(0, laneY, width, 1);
      }
    }

    // --- Draw Image/Text Segments ---
    const hideTimelineImagesPrompts = this.shouldHideTimelineImagesPrompts();
    for (let i = 0; i < sortedSegments.length; i++) {
      const seg = sortedSegments[i];
      const startX = (seg.start / totalFrames) * width;
      const pxWidth = (seg.length / totalFrames) * width;
      const isSelected = (this.selectionType === "image" && seg.id === activeSegId);

      const originalSeg = this.timeline.segments.find(s => s.id === seg.id);
      const imgObj = originalSeg ? originalSeg.imgObj : seg.imgObj;

      if ((this._isDragging && this.selectionType === "image" && seg.id === this._dragTargetId) || (this._ghostSegmentId && seg.id === this._ghostSegmentId)) {
        this.ctx.globalAlpha = 0.65;
      } else {
        this.ctx.globalAlpha = 1.0;
      }

      if (seg.type === "ghost") {
        this.ctx.fillStyle = "#2a2a2a";
        this.ctx.fillRect(startX, RULER_HEIGHT, pxWidth, this.blockHeight);

        this.ctx.strokeStyle = "#777";
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(startX, RULER_HEIGHT + 1, pxWidth, this.blockHeight - 2);
        this.ctx.setLineDash([]);

        this.ctx.fillStyle = "#aaa";
        this.ctx.textAlign = "center";
        this.ctx.textBaseline = "middle";
        this.ctx.font = "bold 12px sans-serif";
        this.ctx.fillText("Drop to Place", startX + pxWidth / 2, RULER_HEIGHT + this.blockHeight / 2);
      } else {
        this.ctx.fillStyle = seg.type === "text" ? "#000b12" : "#000";
        this.ctx.fillRect(startX, RULER_HEIGHT + 1, pxWidth, this.blockHeight - 2);
      }

      if (!hideTimelineImagesPrompts && imgObj && imgObj.complete && imgObj.naturalWidth > 0 && seg.type !== "ghost") {
        const imgRatio = imgObj.naturalWidth / imgObj.naturalHeight;
        const boxRatio = pxWidth / this.blockHeight;
        let drawW, drawH, drawX, drawY;
        if (imgRatio > boxRatio) {
          drawW = pxWidth; drawH = pxWidth / imgRatio;
          drawX = startX; drawY = RULER_HEIGHT + (this.blockHeight - drawH) / 2;
        } else {
          drawH = this.blockHeight; drawW = this.blockHeight * imgRatio;
          drawY = RULER_HEIGHT; drawX = startX + (pxWidth - drawW) / 2;
        }

        // Clip to segment bounds so tiled images don't bleed into adjacent segments
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.rect(startX, RULER_HEIGHT + 1, pxWidth, this.blockHeight - 2);
        this.ctx.clip();

        if (imgRatio > boxRatio) {
          // Fits width, vertical letterboxing (black bars top/bottom) — keep as is
          this.ctx.drawImage(imgObj, drawX, drawY, drawW, drawH);
        } else {
          // Fits height, horizontal letterboxing (black bars left/right) — tile horizontally
          this.ctx.drawImage(imgObj, drawX, drawY, drawW, drawH);

          // Tile left
          let leftX = drawX - drawW;
          while (leftX + drawW > startX) {
            this.ctx.drawImage(imgObj, leftX, drawY, drawW, drawH);
            leftX -= drawW;
          }

          // Tile right
          let rightX = drawX + drawW;
          while (rightX < startX + pxWidth) {
            this.ctx.drawImage(imgObj, rightX, drawY, drawW, drawH);
            rightX += drawW;
          }
        }
        this.ctx.restore();

        // --- Prompt subtitle overlay ---
        if (seg.prompt && seg.type !== "ghost" && pxWidth > 24) {
          const overlayH = Math.round(this.blockHeight * 0.20);
          const overlayY = RULER_HEIGHT + this.blockHeight - overlayH;

          this.ctx.save();
          this.ctx.beginPath();
          this.ctx.rect(startX, overlayY, pxWidth, overlayH);
          this.ctx.clip();

          // Translucent background
          this.ctx.fillStyle = "rgba(0, 0, 0, 0.70)";
          this.ctx.fillRect(startX, overlayY, pxWidth, overlayH);

          // Text
          const fontSize = Math.min(11, overlayH * 0.58);
          this.ctx.font = `${fontSize}px sans-serif`;
          this.ctx.fillStyle = "#e0e3ed";
          this.ctx.textAlign = "center";
          this.ctx.textBaseline = "middle";

          // Measure and truncate to single line
          const maxTextW = pxWidth - 10;
          let label = seg.prompt;
          if (this.ctx.measureText(label).width > maxTextW) {
            while (label.length > 0 && this.ctx.measureText(label + "…").width > maxTextW) {
              label = label.slice(0, -1);
            }
            label += "…";
          }

          this.ctx.fillText(label, startX + pxWidth / 2, overlayY + overlayH / 2);
          this.ctx.restore();
        }
      } else if (!hideTimelineImagesPrompts && seg.type === "text") {
        const pad = 8;
        const boxW = pxWidth - pad * 2;
        if (boxW > 12) {
          this.ctx.save();
          this.ctx.beginPath();
          this.ctx.rect(startX + pad, RULER_HEIGHT + pad, boxW, this.blockHeight - pad * 2);
          this.ctx.clip();
          this.ctx.fillStyle = "#e0e3ed";
          this.ctx.font = "11px sans-serif";
          this.ctx.textAlign = "center";
          this.ctx.textBaseline = "top";
          const label = seg.prompt || "(no prompt)";
          const words = label.split(" ");
          const lineH = 15;
          let line = "";
          let lines = [];
          for (const word of words) {
            const test = line ? line + " " + word : word;
            if (this.ctx.measureText(test).width > boxW && line) {
              lines.push(line);
              line = word;
            } else {
              line = test;
            }
          }
          if (line) lines.push(line);

          const maxLines = Math.max(1, Math.floor((this.blockHeight - pad * 2) / lineH));
          if (lines.length > maxLines) {
            lines = lines.slice(0, maxLines);
            lines[lines.length - 1] += "…";
          }

          const totalTextHeight = lines.length * lineH;
          let ty = RULER_HEIGHT + (this.blockHeight - totalTextHeight) / 2 + 2;

          for (const l of lines) {
            this.ctx.fillText(l, startX + pxWidth / 2, ty);
            ty += lineH;
          }
          this.ctx.restore();
        }
      }

      if (seg.type === "source_video" && seg.type !== "ghost" && pxWidth > 34) {
        const label = `Source Video · Last ${seg.sourceVideoGuideFrames || seg.length || SOURCE_VIDEO_DEFAULT_GUIDE_FRAMES} frames`;
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.rect(startX, RULER_HEIGHT + 1, pxWidth, this.blockHeight - 2);
        this.ctx.clip();
        this.ctx.fillStyle = "rgba(20, 20, 20, 0.72)";
        this.ctx.fillRect(startX + 6, RULER_HEIGHT + 6, Math.min(pxWidth - 12, 170), 20);
        this.ctx.fillStyle = "#d8e7ff";
        this.ctx.font = "10px sans-serif";
        this.ctx.textAlign = "left";
        this.ctx.textBaseline = "middle";
        this.ctx.fillText(label, startX + 12, RULER_HEIGHT + 16);
        this.ctx.restore();
      }

      if (isSelected) {
        this.ctx.strokeStyle = "#fff";
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(startX, RULER_HEIGHT + 1, pxWidth, this.blockHeight - 2);
        if (!this.isSourceVideoSegment(seg)) {
          this.ctx.fillStyle = "#fff";
          this.ctx.beginPath();
          this.ctx.roundRect(startX, RULER_HEIGHT + this.blockHeight / 2 - 12, 4, 24, 2);
          this.ctx.fill();
          this.ctx.beginPath();
          this.ctx.roundRect(startX + pxWidth - 4, RULER_HEIGHT + this.blockHeight / 2 - 12, 4, 24, 2);
          this.ctx.fill();
        }
      } else {
        this.ctx.strokeStyle = "#000";
        this.ctx.lineWidth = 1.5;
        this.ctx.strokeRect(startX, RULER_HEIGHT + 1, pxWidth, this.blockHeight - 2);
      }
      this.ctx.globalAlpha = 1.0;
    }

    // --- Draw Audio Segments ---
    for (let i = 0; i < sortedAudioSegments.length; i++) {
      const seg = sortedAudioSegments[i];
      const startX = (seg.start / totalFrames) * width;
      const pxWidth = (seg.length / totalFrames) * width;
      const isSelected = (this.selectionType === "audio" && seg.id === activeAudioSegId);
      const trackY = this.getAudioSegmentY(seg);
      const clipHeight = this.getAudioClipHeight();

      if ((this._isDragging && this.selectionType === "audio" && seg.id === this._dragTargetId) || (this._ghostSegmentId && seg.id === this._ghostSegmentId)) {
        this.ctx.globalAlpha = 0.65;
      } else {
        this.ctx.globalAlpha = 1.0;
      }

      if (seg.type === "ghost") {
        this.ctx.fillStyle = "#1a1a1a";
        this.ctx.fillRect(startX, trackY + 3, pxWidth, clipHeight);
        this.ctx.strokeStyle = "#555";
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(startX, trackY + 3, pxWidth, clipHeight);
        this.ctx.setLineDash([]);
        this.ctx.fillStyle = "#888";
        this.ctx.textAlign = "center";
        this.ctx.textBaseline = "middle";
        this.ctx.font = "bold 12px sans-serif";
        this.ctx.fillText("Drop Audio", startX + pxWidth / 2, trackY + AUDIO_LANE_HEIGHT / 2);
      } else {
        this.drawAudioSegmentVisuals(this.ctx, seg, isSelected, trackY, clipHeight, startX, pxWidth);
      }
      this.ctx.globalAlpha = 1.0;
    }

    // --- Draw Ruler & Divider AFTER segments to prevent overlap ---
    // Ruler Background
    this.ctx.fillStyle = "#1e1e1e";
    this.ctx.fillRect(0, 0, width, RULER_HEIGHT);

    // Crisp Ruler Text
    this.ctx.fillStyle = "#aaa";
    this.ctx.textAlign = "center";
    this.ctx.textBaseline = "middle";
    this.ctx.font = "10px sans-serif";

    const frameRate = this.getFrameRate();
    const mode = this.displayModeWidget ? this.displayModeWidget.value : "seconds";

    // Define logical steps for both modes
    let steps;
    if (mode === "seconds") {
      steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    } else {
      steps = [1, 2, 5, 10, 24, 48, 120, 240, 480, 960, 1920];
    }

    const minSpacingPx = 60;
    let majorStep = steps[steps.length - 1];
    for (let i = 0; i < steps.length; i++) {
      const stepFrames = mode === "seconds" ? steps[i] * frameRate : steps[i];
      const spacingPx = (stepFrames / totalFrames) * width;
      if (spacingPx >= minSpacingPx) {
        majorStep = steps[i];
        break;
      }
    }

    const majorStepFrames = mode === "seconds" ? majorStep * frameRate : majorStep;

    let minorStep;
    if (mode === "seconds") {
      if (majorStep <= 0.2) minorStep = majorStep / 2;
      else if (majorStep <= 1) minorStep = majorStep / 5;
      else if (majorStep <= 5) minorStep = 1;
      else if (majorStep <= 15) minorStep = 5;
      else if (majorStep <= 30) minorStep = 10;
      else if (majorStep <= 60) minorStep = 10;
      else minorStep = majorStep / 5;
    } else {
      if (majorStep <= 5) minorStep = 1;
      else if (majorStep <= 10) minorStep = 2;
      else if (majorStep <= 24) minorStep = 6;
      else if (majorStep <= 48) minorStep = 12;
      else minorStep = majorStep / 5;
    }
    const minorStepFrames = mode === "seconds" ? minorStep * frameRate : minorStep;

    this.ctx.fillStyle = "#444";
    const totalMinorTicks = Math.floor(totalFrames / minorStepFrames);
    for (let i = 0; i <= totalMinorTicks; i++) {
      const frameVal = i * minorStepFrames;
      if (Math.abs(frameVal % majorStepFrames) < 0.1) continue;

      const x = (frameVal / totalFrames) * width;
      this.ctx.fillRect(Math.floor(x), RULER_HEIGHT - 3, 1, 3);
    }

    this.ctx.fillStyle = "#aaa";
    const totalMajorTicks = Math.floor(totalFrames / majorStepFrames);
    for (let i = 0; i <= totalMajorTicks; i++) {
      const frameVal = i * majorStepFrames;
      const x = (frameVal / totalFrames) * width;

      this.ctx.fillStyle = "#aaa";
      this.ctx.fillRect(Math.floor(x), RULER_HEIGHT - 6, 1, 6);

      if (frameVal > 0 && frameVal < totalFrames) {
        this.ctx.textAlign = "center";
        this.ctx.fillText(this.formatTime(frameVal, true), x, RULER_HEIGHT / 2);
      }
    }

    this.ctx.textAlign = "left";
    const zeroLabel = mode === "seconds" ? "0" : this.formatTime(0, true);
    this.ctx.fillText(zeroLabel, 4, RULER_HEIGHT / 2);

    // Divider
    this.ctx.fillStyle = "#333";
    this.ctx.fillRect(0, RULER_HEIGHT + this.blockHeight, width, 1);

    // Draw gap "+" buttons
    if (!this._isDragging) {
      const BTN_R = 12;
      const gapRegions = this.getGapRegions();
      for (let i = 0; i < gapRegions.length; i++) {
        const gap = gapRegions[i];
        if (gap.widthPx < BTN_R * 2 + 8) continue;
        const hov = this._hoveredGapIdx === i;
        const BTN_W = 18;
        const BTN_H = 18;
        this.ctx.beginPath();
        this.ctx.roundRect(gap.centerX - BTN_W / 2, gap.centerY - BTN_H / 2, BTN_W, BTN_H, 4);
        this.ctx.fillStyle = hov ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.05)";
        this.ctx.fill();
        this.ctx.fillStyle = hov ? "#fff" : "#888";
        this.ctx.font = "14px sans-serif";
        this.ctx.textAlign = "center";
        this.ctx.textBaseline = "middle";
        this.ctx.fillText("+", gap.centerX, gap.centerY + 1);
      }
    }

    // --- Out-of-duration shadow overlay ---
    // Draw a translucent black mask over the region beyond the actual output duration
    // so the user can clearly see which content will be included in the render.
    const outputFrames = this.getDurationFrames();
    if (outputFrames < totalFrames) {
      const cutoffX = (outputFrames / totalFrames) * width;
      // Semi-transparent black overlay on both tracks
      this.ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
      this.ctx.fillRect(cutoffX, RULER_HEIGHT, width - cutoffX, this.blockHeight + this.audioTrackHeight);
      // Subtle tinted ruler overlay
      this.ctx.fillStyle = "rgba(0, 0, 0, 0.25)";
      this.ctx.fillRect(cutoffX, 0, width - cutoffX, RULER_HEIGHT);
      /*
      // Dashed boundary line at the output duration cutoff
      this.ctx.save();
      this.ctx.strokeStyle = "rgba(255, 80, 80, 0.7)";
      this.ctx.lineWidth = 1.5;
      this.ctx.setLineDash([5, 4]);
      this.ctx.beginPath();
      this.ctx.moveTo(cutoffX, 0);
      this.ctx.lineTo(cutoffX, CANVAS_HEIGHT);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
      this.ctx.restore();
      */
    }

    // --- Draw Playhead ---
    const playheadX = (this.currentFrame / totalFrames) * width;

    // Playhead Line
    this.ctx.beginPath();
    this.ctx.moveTo(playheadX, 14);
    this.ctx.lineTo(playheadX, this.canvasHeight);
    this.ctx.strokeStyle = "#ff4444";
    this.ctx.lineWidth = 1.5;
    this.ctx.stroke();

    // Playhead Handle (Polygon above numbers)
    this.ctx.fillStyle = "#ff4444";
    this.ctx.beginPath();
    this.ctx.moveTo(playheadX - 6, 0);
    this.ctx.lineTo(playheadX + 6, 0);
    this.ctx.lineTo(playheadX + 6, 8);
    this.ctx.lineTo(playheadX, 14);
    this.ctx.lineTo(playheadX - 6, 8);
    this.ctx.fill();

    // Draw vertical grab bar on the right edge of viewport for resizing width
    const grabBarW = 4;
    const grabBarH = 50;
    const grabBarX = this.viewport.scrollLeft + this.viewport.clientWidth - grabBarW - 3;
    const grabBarY = RULER_HEIGHT + (this.blockHeight + this.audioTrackHeight - grabBarH) / 2;
    
    this.ctx.fillStyle = "rgba(40, 40, 40, 0.6)";
    this.ctx.beginPath();
    this.ctx.roundRect(grabBarX, grabBarY, grabBarW, grabBarH, 2);
    this.ctx.fill();

    // Draw horizontal grab bar at the bottom of viewport for resizing height
    const hBarW = 50;
    const hBarH = 4;
    const hBarX = this.viewport.scrollLeft + (this.viewport.clientWidth - hBarW) / 2;
    const hBarY = this.canvasHeight - hBarH - 3; // 3px from the bottom edge
    
    this.ctx.fillStyle = "rgba(20, 20, 20, 0.8)";
    this.ctx.beginPath();
    this.ctx.roundRect(hBarX, hBarY, hBarW, hBarH, 2);
    this.ctx.fill();

    this.updatePlayerUI();
  }

  drawAudioSegmentVisuals(ctx, seg, isSelected, yOffset, trackHeight, startX, pxWidth) {
    ctx.fillStyle = isSelected ? "rgba(42, 74, 58, 0.88)" : "rgba(26, 42, 26, 0.62)";
    ctx.fillRect(startX, yOffset + 2, pxWidth, trackHeight - 3);

    if (seg.waveformPeaks && pxWidth > 0) {
      ctx.fillStyle = isSelected ? "rgba(100, 255, 100, 0.72)" : "rgba(100, 255, 100, 0.38)";
      const audioDurationFrames = Math.max(1, seg.audioDurationFrames || seg.length || 1);
      const startRatio = (seg.trimStart || 0) / audioDurationFrames;
      const endRatio = ((seg.trimStart || 0) + (seg.length || 1)) / audioDurationFrames;
      const peakCount = seg.waveformPeaks.length;
      const centerY = yOffset + trackHeight / 2;
      const volume = clampVolume(seg.volume);

      ctx.beginPath();
      for (let i = 0; i < pxWidth; i++) {
        const pixelRatio = i / pxWidth;
        const globalRatio = startRatio + pixelRatio * (endRatio - startRatio);
        const peakIdx = Math.floor(globalRatio * peakCount);

        if (peakIdx >= 0 && peakIdx < peakCount) {
          const val = clamp(seg.waveformPeaks[peakIdx] * volume, 0, 1);
          const amp = (val * (trackHeight - 12) / 2) * 0.9;
          ctx.fillRect(startX + i, centerY - amp, 1, amp * 2);
        }
      }
    }

    ctx.strokeStyle = isSelected ? "#4fff8f" : "#000";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(startX, yOffset + 2, pxWidth, trackHeight - 3);

    if (isSelected) {
      ctx.fillStyle = "#4fff8f";
      ctx.beginPath();
      ctx.roundRect(startX, yOffset + trackHeight / 2 - 12, 4, 24, 2);
      ctx.fill();
      ctx.beginPath();
      ctx.roundRect(startX + pxWidth - 4, yOffset + trackHeight / 2 - 12, 4, 24, 2);
      ctx.fill();
    }

    ctx.fillStyle = "#ccc";
    ctx.font = "11px sans-serif";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.save();
    ctx.beginPath();
    ctx.rect(startX, yOffset + 2, pxWidth, trackHeight - 3);
    ctx.clip();

    let text = `${seg.fileName || "Audio Track"} · ${clampVolume(seg.volume).toFixed(2)}x`;
    const maxWidth = pxWidth - 12;
    if (ctx.measureText(text).width > maxWidth && maxWidth > 0) {
      while (text.length > 0 && ctx.measureText(text + "...").width > maxWidth) {
        text = text.slice(0, -1);
      }
      text = text + "...";
    }

    ctx.fillText(text, startX + 6, yOffset + 8);
    ctx.restore();
  }


  // --- Interaction Logic ---
  getHitTest(mouseX, mouseY) {
    const width = this.canvas.offsetWidth;
    const totalFrames = this.getVisualDurationFrames();

    // Check Playhead Handle first
    const playheadX = (this.currentFrame / totalFrames) * width;
    if (mouseY <= 24 && Math.abs(mouseX - playheadX) <= 12) {
      return { type: "playhead" };
    }

    if (mouseY <= RULER_HEIGHT) {
      return { type: "ruler" };
    }

    if (mouseY < RULER_HEIGHT || mouseY > this.canvasHeight) return null;

    const isAudioTrack = mouseY > RULER_HEIGHT + this.blockHeight;
    const trackSegments = isAudioTrack ? this.timeline.audioSegments : this.timeline.segments;
    const trackType = isAudioTrack ? "audio" : "image";

    if (trackSegments.length === 0) return null;

    // The variables width and totalFrames are already declared above.

    let sortedSegments = [...trackSegments]
      .map((s, i) => ({ ...s, originalIndex: i }))
      .sort((a, b) => a.start - b.start);

    const HANDLE_CORE = 4;

    for (let i = 0; i < sortedSegments.length; i++) {
      const seg = sortedSegments[i];
      if (trackType === "image" && this.isSourceVideoSegment(seg)) continue;
      const startX = (seg.start / totalFrames) * width;
      const pxWidth = (seg.length / totalFrames) * width;
      const endX = startX + pxWidth;
      if (trackType === "audio") {
        const segY = this.getAudioSegmentY(seg);
        const segH = this.getAudioClipHeight();
        if (mouseY < segY + 3 || mouseY > segY + 3 + segH) continue;
      }

      const prevSeg = sortedSegments[i - 1];
      const nextSeg = sortedSegments[i + 1];

      const isLeftJoint = trackType !== "audio" && prevSeg && prevSeg.start + prevSeg.length === seg.start;
      if (!isLeftJoint) {
        if (Math.abs(mouseX - startX) <= HANDLE_HIT_PX) {
          return { type: "edge", index: seg.originalIndex, dir: "left", track: trackType };
        }
      }

      const isRightJoint = trackType !== "audio" && nextSeg && nextSeg.start === seg.start + seg.length;
      if (isRightJoint) {
        const dx = mouseX - endX;
        if (Math.abs(dx) <= HANDLE_HIT_PX) {
          if (dx < -HANDLE_CORE) {
            return { type: "edge", index: seg.originalIndex, dir: "right", track: trackType };
          } else if (dx > HANDLE_CORE) {
            return { type: "edge", index: nextSeg.originalIndex, dir: "left", track: trackType };
          } else {
            return { type: "joint", leftIndex: seg.originalIndex, rightIndex: nextSeg.originalIndex, track: trackType };
          }
        }
      } else {
        if (Math.abs(mouseX - endX) <= HANDLE_HIT_PX) {
          return { type: "edge", index: seg.originalIndex, dir: "right", track: trackType };
        }
      }
    }

    const centerHitSegments = trackType === "audio" ? [...sortedSegments].reverse() : sortedSegments;
    for (let i = 0; i < centerHitSegments.length; i++) {
      const seg = centerHitSegments[i];
      const startX = (seg.start / totalFrames) * width;
      const pxWidth = (seg.length / totalFrames) * width;
      const endX = startX + pxWidth;
      if (trackType === "audio") {
        const segY = this.getAudioSegmentY(seg);
        const segH = this.getAudioClipHeight();
        if (mouseY < segY + 3 || mouseY > segY + 3 + segH) continue;
      }

      if (mouseX >= startX && mouseX < endX) {
        return { type: "center", index: seg.originalIndex, track: trackType };
      }
    }

    return null;
  }

  onMouseDown(e) {
    e.stopPropagation();
    if (e.button !== 0) return;
    const { x, y } = this.getMousePos(e);

    const isOverDivider = Math.abs(y - (RULER_HEIGHT + this.blockHeight)) <= 4;
    if (isOverDivider) {
      this._isDragging = true;
      this._dragType = "divider";
      this._startBlockHeight = this.blockHeight;
      this._startAudioTrackHeight = this.audioTrackHeight;
      this._startY = y;
      return;
    }

    const isAtBottom = Math.abs(y - this.canvasHeight) <= 15;
    if (isAtBottom) {
      this._isDragging = true;
      this._dragType = "height_resize";
      this._startBlockHeight = this.blockHeight;
      this._startY = y;
      document.body.style.userSelect = "none";
      return;
    }

    const viewRect = this.viewport.getBoundingClientRect();
    const isAtRightEdge = Math.abs(e.clientX - viewRect.right) <= 20;
    if (isAtRightEdge) {
      this._isDragging = true;
      this._dragType = "width_resize";
      this._startNodeWidth = this.node.size[0];
      this._startX = e.clientX;
      document.body.style.userSelect = "none";
      return;
    }

    if (y >= RULER_HEIGHT && y <= this.canvasHeight) {
      const BTN_R = 12;
      const gapRegions = this.getGapRegions();
      for (let i = 0; i < gapRegions.length; i++) {
        const gap = gapRegions[i];
        if (gap.widthPx < BTN_R * 2 + 8) continue;
        const dx = x - gap.centerX, dy2 = y - gap.centerY;
        if (dx * dx + dy2 * dy2 <= BTN_R * BTN_R) {
          if (gap.track === "audio") {
            // Direct to audio upload
            this.promptAddAudioInGap(gap.frameStart, gap.frameEnd, gap.lane);
          } else {
            this.showGapMenu(e.clientX, e.clientY, gap);
          }
          return;
        }
      }
    }

    const hit = this.getHitTest(x, y);
    if (!hit) {
      // Only deselect if they clicked the same track but hit empty space
      const clickedTrack = y > RULER_HEIGHT + this.blockHeight ? "audio" : "image";
      if (this.selectionType === clickedTrack && this.selectedIndex !== -1) {
        this.selectedIndex = -1;
        this.updateUIFromSelection();
        this.render();
      }
      return;
    }

    if (hit.type === "playhead" || hit.type === "ruler") {
      this._isDragging = true;
      this._dragType = "playhead";
      const logicalWidth = this.canvas.offsetWidth;
      const totalFrames = this.getVisualDurationFrames();
      let mouseFrameX = x * (totalFrames / logicalWidth);
      this.currentFrame = clamp(mouseFrameX, 0, totalFrames);
      this.render();
      if (this.isPlaying) {
        this.playAudio();
      }
      return;
    }

    const previousSelectionType = this.selectionType;
    const previousSelectedIndex = this.selectedIndex;
    this.selectionType = hit.track;
    const targetArray = hit.track === "audio" ? this.timeline.audioSegments : this.timeline.segments;

    if ((e.ctrlKey || e.metaKey) && hit.track === "image" && hit.type === "center") {
      const seg = targetArray[hit.index];
      if (seg?.type !== "text" && seg?.imageB64) {
        e.preventDefault();
        e.stopPropagation();
        this.selectedIndex = hit.index;
        this.updateUIFromSelection();
        this.render();
        this.showTimelineSegmentImagePreview(seg);
        return;
      }
    }

    if (hit.track === "image" && this.isSourceVideoSegment(targetArray[hit.index])) {
      if (previousSelectionType !== hit.track || this.selectedIndex !== hit.index) {
        this.selectedIndex = hit.index;
        this.updateUIFromSelection();
        this.render();
      }
      return;
    }

    if (hit.type === "joint") {
      this.selectedIndex = hit.leftIndex;
      if (previousSelectionType !== hit.track || previousSelectedIndex !== hit.leftIndex) {
        this.updateUIFromSelection();
        this.render();
      }
      this._dragType = "joint";
      this._dragTargetId = targetArray[hit.leftIndex].id;
      this._dragTargetIdRight = targetArray[hit.rightIndex].id;
    } else if (hit.type === "center") {
      if (previousSelectionType !== hit.track || this.selectedIndex !== hit.index) {
        this.selectedIndex = hit.index;
        this.updateUIFromSelection();
        this.render();
      }
      this._dragType = "center";
    } else {
      if (previousSelectionType !== hit.track || this.selectedIndex !== hit.index) {
        this.selectedIndex = hit.index;
        this.updateUIFromSelection();
        this.render();
      }
      this._dragType = hit.dir;
    }

    this._isDragging = true;
    this._previewSegments = null;
    this._dragStartX = x;
    this._dragStartY = y;
    this._dragInitialTimeline = JSON.parse(JSON.stringify(targetArray));
    this._dragInitialLane = hit.track === "audio" && hit.index !== undefined
      ? normalizeAudioLane(targetArray[hit.index]?.lane)
      : 0;

    if (hit.type !== "joint") {
      this._dragTargetId = targetArray[hit.index].id;
    }
  }

  onMouseMove(e) {
    const { x: mouseX, y: mouseY } = this.getMousePos(e);

    if (!this._isDragging) {
      let newHoveredGapIdx = -1;
      const BTN_R = 12;
      const gapRegions = this.getGapRegions();
      for (let i = 0; i < gapRegions.length; i++) {
        const gap = gapRegions[i];
        if (gap.widthPx < BTN_R * 2 + 8) continue;
        const dx = mouseX - gap.centerX, dy2 = mouseY - gap.centerY;
        if (dx * dx + dy2 * dy2 <= BTN_R * BTN_R) { newHoveredGapIdx = i; break; }
      }
      if (this._hoveredGapIdx !== newHoveredGapIdx) {
        this._hoveredGapIdx = newHoveredGapIdx;
        this.render();
      }

      const isOverDivider = Math.abs(mouseY - (RULER_HEIGHT + this.blockHeight)) <= 4;
      const isAtBottom = Math.abs(mouseY - this.canvasHeight) <= 15;
      const viewRect = this.viewport.getBoundingClientRect();
      const isAtRightEdge = Math.abs(e.clientX - viewRect.right) <= 20;
      const hit = this.getHitTest(mouseX, mouseY);
      if (isOverDivider || isAtBottom) {
        this.canvas.style.cursor = "ns-resize";
      } else if (isAtRightEdge) {
        this.canvas.style.cursor = "ew-resize";
      } else if (newHoveredGapIdx >= 0) {
        this.canvas.style.cursor = "pointer";
      } else if (hit?.type === "edge") {
        this.canvas.style.cursor = "ew-resize";
      } else if (hit?.type === "joint") {
        this.canvas.style.cursor = "col-resize";
      } else if (hit?.type === "center") {
        const hitArray = hit.track === "audio" ? this.timeline.audioSegments : this.timeline.segments;
        this.canvas.style.cursor = this.isSourceVideoSegment(hitArray[hit.index]) ? "pointer" : "grab";
      } else if (hit?.type === "playhead") {
        this.canvas.style.cursor = "ew-resize";
      } else {
        this.canvas.style.cursor = "default";
      }
      return;
    }

    if (this._dragType === "divider") {
      this.canvas.style.cursor = "ns-resize";
      const deltaY = mouseY - this._startY;

      const minBlockH = 50;
      const minAudioH = Math.max(50, this.getRequiredAudioTrackHeight());

      let newBlockHeight = this._startBlockHeight + deltaY;
      let newAudioTrackHeight = this._startAudioTrackHeight - deltaY;

      if (newBlockHeight < minBlockH) {
        newBlockHeight = minBlockH;
        newAudioTrackHeight = this._startBlockHeight + this._startAudioTrackHeight - minBlockH;
      }
      if (newAudioTrackHeight < minAudioH) {
        newAudioTrackHeight = minAudioH;
        newBlockHeight = this._startBlockHeight + this._startAudioTrackHeight - minAudioH;
      }

      this.blockHeight = newBlockHeight;
      this.audioTrackHeight = newAudioTrackHeight;

      this.render();
      return;
    }

    if (this._dragType === "height_resize") {
      this.canvas.style.cursor = "ns-resize";
      const deltaY = mouseY - this._startY;

      this.blockHeight = Math.max(100, this._startBlockHeight + deltaY);
      this.canvasHeight = this.rulerHeight + this.blockHeight + this.audioTrackHeight;

      this.canvas.style.height = `${this.canvasHeight}px`;

      this.resizeCanvas(this.canvas.offsetWidth);
      this.render();

      if (this.node && this.node.computeSize) {
        const sz = this.node.computeSize();
        this.node.size[1] = sz[1];
        if (window.app && window.app.graph) {
          window.app.graph.setDirtyCanvas(true, true);
        }
      }
      return;
    }

    if (this._dragType === "width_resize") {
      this.canvas.style.cursor = "ew-resize";
      const deltaX = e.clientX - this._startX;

      this.node.size[0] = Math.max(300, this._startNodeWidth + deltaX);

      if (window.app && window.app.graph) {
        window.app.graph.setDirtyCanvas(true, true);
      }
      return;
    }

    if (this._dragType === "playhead") {
      this.canvas.style.cursor = "ew-resize";
      const logicalWidth = this.canvas.offsetWidth;
      const totalFrames = this.getVisualDurationFrames();
      let mouseFrameX = mouseX * (totalFrames / logicalWidth);
      this.currentFrame = clamp(mouseFrameX, 0, totalFrames);
      this.render();
      if (this.isPlaying) {
        this.playAudio(); // Scrub (restart from new position)
      }
      return;
    }

    this.canvas.style.cursor = this._dragType === "center" ? "grabbing" :
      this._dragType === "joint" ? "col-resize" : "ew-resize";

    const logicalWidth = this.canvas.offsetWidth;
    const totalFrames = this.getVisualDurationFrames();
    const durationFrames = totalFrames;
    const dragDelta = Math.round((mouseX - this._dragStartX) * (totalFrames / logicalWidth));

    let t = JSON.parse(JSON.stringify(this._dragInitialTimeline));

    // --- Rolling Edit (Slide Edit) ---
    if (this._dragType === "joint") {
      let leftIdx = t.findIndex(s => s.id === this._dragTargetId);
      let rightIdx = t.findIndex(s => s.id === this._dragTargetIdRight);

      if (leftIdx >= 0 && rightIdx >= 0) {
        let origLeft = this._dragInitialTimeline.find(s => s.id === this._dragTargetId);
        let origRight = this._dragInitialTimeline.find(s => s.id === this._dragTargetIdRight);

        let maxDeltaRight = origRight.length - MIN_SEGMENT_LENGTH;
        let maxDeltaLeft = origLeft.length - MIN_SEGMENT_LENGTH;

        if (this.selectionType === "audio") {
          // Drag LEFT: right clip extends left by un-trimming its head.
          // Can only un-trim as much as the right clip has been trimmed (trimStart >= 0).
          maxDeltaLeft = Math.min(maxDeltaLeft, origRight.trimStart || 0);
          // Drag RIGHT: left clip extends right by consuming its remaining tail audio.
          // Can only extend as far as the left clip's unplayed tail allows.
          let availLeftTail = (origLeft.audioDurationFrames || origLeft.length) - ((origLeft.trimStart || 0) + origLeft.length);
          maxDeltaRight = Math.min(maxDeltaRight, availLeftTail);
        }

        let safeDelta = clamp(dragDelta, -maxDeltaLeft, maxDeltaRight);

        t[leftIdx].length = origLeft.length + safeDelta;
        t[rightIdx].start = origRight.start + safeDelta;
        t[rightIdx].length = origRight.length - safeDelta;

        if (this.selectionType === "audio") {
          t[rightIdx].trimStart = origRight.trimStart + safeDelta;
        }
      }
    }
    // --- Edge & Center Drags ---
    else {
      const targetIdx = t.findIndex((s) => s.id === this._dragTargetId);
      if (targetIdx < 0) return;

      if (this._dragType === "right") {
        let newLen = t[targetIdx].length + dragDelta;
        let maxPossibleLength = totalFrames - t[targetIdx].start;
        if (this.selectionType !== "audio") {
          let nextSeg = t.find(s => s.start >= t[targetIdx].start + t[targetIdx].length && s.id !== t[targetIdx].id);
          if (nextSeg) {
            maxPossibleLength = nextSeg.start - t[targetIdx].start;
          }
        }

        if (this.selectionType === "audio") {
          maxPossibleLength = Math.min(maxPossibleLength, (t[targetIdx].audioDurationFrames || t[targetIdx].length) - (t[targetIdx].trimStart || 0));
          const lane = normalizeAudioLane(t[targetIdx].lane);
          const nextSeg = t
            .filter(s => s.id !== t[targetIdx].id && normalizeAudioLane(s.lane) === lane && s.start >= t[targetIdx].start)
            .sort((a, b) => a.start - b.start)[0];
          if (nextSeg) {
            maxPossibleLength = Math.min(maxPossibleLength, nextSeg.start - t[targetIdx].start);
          }
        }

        t[targetIdx].length = Math.max(MIN_SEGMENT_LENGTH, Math.min(newLen, maxPossibleLength));

      } else if (this._dragType === "left") {
        let newStart = t[targetIdx].start + dragDelta;
        let minPossibleStart = 0;
        if (this.selectionType !== "audio") {
          let prevSeg = t.slice().reverse().find(s => s.start + s.length <= t[targetIdx].start && s.id !== t[targetIdx].id);
          if (prevSeg) {
            minPossibleStart = prevSeg.start + prevSeg.length;
          }
        }

        if (this.selectionType === "audio") {
          minPossibleStart = Math.max(minPossibleStart, t[targetIdx].start - (t[targetIdx].trimStart || 0));
          const lane = normalizeAudioLane(t[targetIdx].lane);
          const prevSeg = t
            .filter(s => s.id !== t[targetIdx].id && normalizeAudioLane(s.lane) === lane && s.start + s.length <= t[targetIdx].start)
            .sort((a, b) => (b.start + b.length) - (a.start + a.length))[0];
          if (prevSeg) {
            minPossibleStart = Math.max(minPossibleStart, prevSeg.start + prevSeg.length);
          }
        }

        let maxStart = t[targetIdx].start + t[targetIdx].length - MIN_SEGMENT_LENGTH;
        newStart = Math.max(minPossibleStart, Math.min(newStart, maxStart));

        let diff = newStart - t[targetIdx].start;
        t[targetIdx].start = newStart;
        t[targetIdx].length -= diff;
        if (this.selectionType === "audio") {
          t[targetIdx].trimStart += diff;
        }

      } else if (this._dragType === "center") {
        let initT = this._dragInitialTimeline;
        let dIdx = initT.findIndex(s => s.id === this._dragTargetId);
        if (dIdx < 0) return;
        let D = JSON.parse(JSON.stringify(initT[dIdx]));

        let D_mouse_start = D.start + dragDelta;
        let mouseFrameX = mouseX * (totalFrames / logicalWidth);

        if (this.selectionType === "audio") {
          t = JSON.parse(JSON.stringify(initT));
          const audioIdx = t.findIndex(s => s.id === D.id);
          if (audioIdx >= 0) {
            const newStart = clamp(Math.round(D_mouse_start), 0, Math.max(0, durationFrames - t[audioIdx].length));
            const laneDelta = Math.round((mouseY - this._dragStartY) / AUDIO_LANE_HEIGHT);
            const preferredLane = Math.max(0, this._dragInitialLane + laneDelta);
            t[audioIdx].start = newStart;
            t[audioIdx].lane = this.findFreeAudioLane(newStart, t[audioIdx].length, t[audioIdx].id, preferredLane, t);
          }
          this.ensureAudioTrackHeight(t);
        } else {
          t = this._applyCenterDragPhysics(initT, D.id, D_mouse_start, mouseFrameX, durationFrames, totalFrames, logicalWidth);
        }
      }
    }

    this._previewSegments = t;
    this.updateUIFromSelection(); // Live update of trim values
    this.render();
  }

  _applyCenterDragPhysics(initT, D_id, D_mouse_start, mouseFrameX, durationFrames, totalFrames, logicalWidth) {
    let t_copy = JSON.parse(JSON.stringify(initT));
    let dIdx = t_copy.findIndex(s => s.id === D_id);
    if (dIdx < 0) return t_copy;

    let D = t_copy[dIdx];
    let D_clamped_start = clamp(D_mouse_start, 0, durationFrames - D.length);

    let baseSegments = t_copy.filter(s => s.id !== D.id);

    let insertIdx = baseSegments.length;
    for (let i = 0; i < baseSegments.length; i++) {
      let centerBase = baseSegments[i].start + baseSegments[i].length / 2;
      if (mouseFrameX < centerBase) {
        insertIdx = i;
        break;
      }
    }

    let leftBound = insertIdx > 0 ? baseSegments[insertIdx - 1].start + baseSegments[insertIdx - 1].length : 0;
    let rightBound = insertIdx < baseSegments.length ? baseSegments[insertIdx].start : durationFrames;

    if (rightBound - leftBound >= D.length) {
      D_clamped_start = clamp(D_clamped_start, leftBound, rightBound - D.length);
    } else {
      let gapCenter = (leftBound + rightBound) / 2;
      D_clamped_start = gapCenter - D.length / 2;
    }

    let t_test = [];
    for (let i = 0; i < insertIdx; i++) {
      t_test.push({ ...baseSegments[i], original_start: baseSegments[i].start });
    }
    t_test.push({ ...D, start: D_clamped_start, original_start: D_clamped_start });
    let D_index = insertIdx;

    for (let i = insertIdx; i < baseSegments.length; i++) {
      t_test.push({ ...baseSegments[i], original_start: baseSegments[i].start });
    }

    for (let i = D_index + 1; i < t_test.length; i++) {
      let prev = t_test[i - 1];
      t_test[i].start = Math.max(t_test[i].original_start, prev.start + prev.length);
    }

    for (let i = D_index - 1; i >= 0; i--) {
      let next = t_test[i + 1];
      t_test[i].start = Math.min(t_test[i].original_start, next.start - t_test[i].length);
    }

    let rightCursor = durationFrames;
    for (let i = t_test.length - 1; i >= 0; i--) {
      if (t_test[i].start + t_test[i].length > rightCursor) {
        t_test[i].start = rightCursor - t_test[i].length;
      }
      rightCursor = t_test[i].start;
    }
    let leftCursor = 0;
    for (let i = 0; i < t_test.length; i++) {
      if (t_test[i].start < leftCursor) {
        t_test[i].start = leftCursor;
      }
      leftCursor = t_test[i].start + t_test[i].length;
    }

    let result = t_test.map(s => {
      let clean = { ...s };
      delete clean.original_start;
      return clean;
    });

    let draggedPreview = result.find(s => s.id === D.id);
    if (draggedPreview) {
      draggedPreview.resolvedStart = draggedPreview.start;
    }

    return result;
  }

  onMouseUp(e) {
    e.stopPropagation();
    document.body.style.userSelect = "";
    if (this._isDragging) {
      const shouldCommit = !!this._previewSegments;
      if (this._previewSegments) {
        const targetArray = this.selectionType === "audio" ? this.timeline.audioSegments : this.timeline.segments;

        const mappedArray = this._previewSegments.map(ps => {
          const orig = targetArray.find(s => s.id === ps.id);
          let finalStart = ps.resolvedStart !== undefined ? ps.resolvedStart : ps.start;
          let newPs = { ...ps, start: finalStart };
          if (orig && orig.imgObj) newPs.imgObj = orig.imgObj;
          delete newPs.resolvedStart;
          return newPs;
        });

        if (this.selectionType === "audio") {
          this.timeline.audioSegments = mappedArray;
          this.ensureAudioTrackHeight();
          if (this._dragTargetId) this.selectedIndex = this.timeline.audioSegments.findIndex(s => s.id === this._dragTargetId);
        } else {
          this.timeline.segments = mappedArray;
          if (this._dragTargetId) this.selectedIndex = this.timeline.segments.findIndex(s => s.id === this._dragTargetId);
        }
      }

      this._isDragging = false;
      this._previewSegments = null;
      this._ghostTrack = null;
      this.canvas.style.cursor = "default";
      if (shouldCommit) this.commitChanges();
    }
  }

  // --- Backend Data Sync ---
  flushPromptEdit({ skipNodeResize = true } = {}) {
    if (!this._promptEditDirty) return;
    this._promptEditDirty = false;
    this.commitChanges(true, skipNodeResize);
  }

  buildTimelineSaveObject() {
    const sortedSegments = [...this.timeline.segments].sort((a, b) => a.start - b.start);
    return {
      segments: sortedSegments.map(s => {
        const { imgObj, ...rest } = s;
        return rest;
      }),
      audioSegments: (this.timeline.audioSegments || []).map(s => ({
        ...s,
        lane: normalizeAudioLane(s.lane),
        volume: clampVolume(s.volume)
      }))
    };
  }

  commitChanges(skipRender = false, skipNodeResize = false) {
    let sortedSegments = [...this.timeline.segments].sort((a, b) => a.start - b.start);
    let contiguousLengths = [];
    let contiguousPrompts = [];
    let currentCursor = 0;
    const durationFrames = this.getDurationFrames();

    // Build segment lengths clipped at the duration cutoff.
    // - Gaps before the first segment, or between segments, are absorbed into the adjacent
    //   segment's length (same as before), but are also clipped at durationFrames.
    // - Segments that start at or past the cutoff are excluded entirely.
    // - Segments that cross the cutoff are trimmed so their end = durationFrames exactly.
    let pendingGap = 0;
    for (let seg of sortedSegments) {
      // Skip segments entirely outside the duration.
      if (seg.start >= durationFrames) break;

      if (seg.start > currentCursor) {
        // Gap between the cursor and this segment — clip it at the cutoff too.
        const gapLength = Math.min(seg.start, durationFrames) - currentCursor;
        if (contiguousLengths.length > 0) {
          contiguousLengths[contiguousLengths.length - 1] += gapLength;
        } else {
          pendingGap += gapLength;
        }
      }

      // Clip segment end at the duration cutoff.
      const clippedEnd = Math.min(seg.start + seg.length, durationFrames);
      const clippedLength = clippedEnd - seg.start;

      contiguousLengths.push(clippedLength + pendingGap);
      let prompt = seg.prompt || "";
      if (this.isSourceVideoSegment(seg) && !prompt.trim()) {
        const following = sortedSegments.find((candidate) => (
          candidate.id !== seg.id
          && candidate.start >= seg.start + seg.length
          && (candidate.prompt || "").trim()
        ));
        prompt = following?.prompt || "";
      }
      contiguousPrompts.push(prompt);
      pendingGap = 0;
      currentCursor = seg.start + seg.length; // advance by the real (unclipped) end for gap detection
    }

    // If segments don't fill to the end of the duration, pad the last segment to reach it.
    const clampedCursor = Math.min(currentCursor, durationFrames);
    if (contiguousLengths.length > 0 && clampedCursor < durationFrames) {
      contiguousLengths[contiguousLengths.length - 1] += durationFrames - clampedCursor;
    }

    const toSave = this.buildTimelineSaveObject();

    const jsonStr = JSON.stringify(toSave);
    if (this.timelineDataWidget) this.timelineDataWidget.value = this.isPrivacyModeEnabled() ? EMPTY_TIMELINE_JSON : jsonStr;

    if (this.localPromptsWidget) {
      this.localPromptsWidget.value = this.isPrivacyModeEnabled() ? "" : contiguousPrompts.join(" | ");
    }
    if (this.segmentLengthsWidget) {
      this.segmentLengthsWidget.value = this.isPrivacyModeEnabled() ? "" : contiguousLengths.join(",");
    }

    if (this.guideStrengthWidget) {
      const imgStrengths = sortedSegments
        .filter(s => s.type !== "text")
        .map(s => (s.guideStrength !== undefined ? s.guideStrength : 1.0).toFixed(2));
      this.guideStrengthWidget.value = this.isPrivacyModeEnabled() ? "" : imgStrengths.join(",");
    }

    if (this.isPrivacyModeEnabled() && !this.privacyLocked) {
      void this.encryptPrivacyState({ renderAfter: false, markCanvasDirty: !skipNodeResize });
    }

    // Keep zoom slider max in sync with the current timeline duration.
    this.updateZoomSliderMax();

    if (!skipNodeResize) {
      setTimeout(() => {
        if (this.node && this.node.computeSize) {
          const sz = this.node.computeSize();
          this.node.size[1] = sz[1];
          if (app.graph) app.graph.setDirtyCanvas(true, true);
        }
      }, 0);
    }

    if (!skipRender) this.render();
  }

  // --- Gap Region Calculation ---
  getGapRegions() {
    const totalFrames = this.getVisualDurationFrames();
    const outputFrames = this.getDurationFrames();
    const width = this.canvas.offsetWidth || this._lastWidth || 0;
    const gaps = [];
    if (!width) return gaps;

    // Image gaps
    let cursor = 0;
    const sortedImg = [...this.timeline.segments].sort((a, b) => a.start - b.start);
    for (const seg of sortedImg) {
      if (seg.start > cursor) {
        const x0 = (cursor / totalFrames) * width;
        const x1 = (seg.start / totalFrames) * width;
        gaps.push({ track: 'image', frameStart: cursor, frameEnd: seg.start, centerX: (x0 + x1) / 2, centerY: RULER_HEIGHT + this.blockHeight / 2, widthPx: x1 - x0 });
      }
      cursor = Math.max(cursor, seg.start + seg.length);
    }
    if (cursor < outputFrames) {
      const x0 = (cursor / totalFrames) * width;
      const x1 = (outputFrames / totalFrames) * width;
      gaps.push({ track: 'image', frameStart: cursor, frameEnd: outputFrames, centerX: (x0 + x1) / 2, centerY: RULER_HEIGHT + this.blockHeight / 2, widthPx: x1 - x0 });
    }

    // Audio gaps
    const visibleAudioLanes = Math.max(this.getAudioLaneCount(), Math.floor(this.audioTrackHeight / AUDIO_LANE_HEIGHT));
    for (let lane = 0; lane < visibleAudioLanes; lane++) {
      cursor = 0;
      const sortedAud = [...this.timeline.audioSegments]
        .filter((seg) => normalizeAudioLane(seg.lane) === lane)
        .sort((a, b) => a.start - b.start);
      const centerY = RULER_HEIGHT + this.blockHeight + lane * AUDIO_LANE_HEIGHT + AUDIO_LANE_HEIGHT / 2;
      for (const seg of sortedAud) {
        if (seg.start > cursor) {
          const x0 = (cursor / totalFrames) * width;
          const x1 = (seg.start / totalFrames) * width;
          gaps.push({ track: 'audio', lane, frameStart: cursor, frameEnd: seg.start, centerX: (x0 + x1) / 2, centerY, widthPx: x1 - x0 });
        }
        cursor = Math.max(cursor, seg.start + seg.length);
      }
      if (cursor < outputFrames) {
        const x0 = (cursor / totalFrames) * width;
        const x1 = (outputFrames / totalFrames) * width;
        gaps.push({ track: 'audio', lane, frameStart: cursor, frameEnd: outputFrames, centerX: (x0 + x1) / 2, centerY, widthPx: x1 - x0 });
      }
    }

    return gaps;
  }

  promptAddAudioInGap(frameStart, frameEnd, lane = 0) {
    this.showTimelineAudioBrowser(frameStart, lane);
  }

  // --- Context Menu ---
  onContextMenu(e) {
    e.preventDefault();
    const { x: mouseX, y: mouseY } = this.getMousePos(e);

    const trackHeight = this.blockHeight;
    const isAudioTrack = mouseY >= RULER_HEIGHT + trackHeight && mouseY <= RULER_HEIGHT + trackHeight + this.audioTrackHeight;
    const isImageTrack = mouseY >= RULER_HEIGHT && mouseY <= RULER_HEIGHT + trackHeight;

    const logicalWidth = this.canvas.offsetWidth || 1;
    const totalFrames = this.getVisualDurationFrames();
    const cursor = mouseX * (totalFrames / logicalWidth);

    let clickedSeg = null;
    let trackType = "";

    if (isAudioTrack) {
      clickedSeg = [...this.timeline.audioSegments].reverse().find(s => {
        const segY = this.getAudioSegmentY(s);
        const segH = this.getAudioClipHeight();
        return cursor >= s.start && cursor <= s.start + s.length && mouseY >= segY + 3 && mouseY <= segY + 3 + segH;
      });
      trackType = "audio";
    } else if (isImageTrack) {
      clickedSeg = this.timeline.segments.find(s => cursor >= s.start && cursor <= s.start + s.length);
      trackType = clickedSeg ? clickedSeg.type : "";
    }

    if (clickedSeg) {
      this.showContextMenu(e.clientX, e.clientY, clickedSeg, trackType);
    } else if (isAudioTrack || isImageTrack) {
      const gapRegions = this.getGapRegions();
      const currentTrack = isAudioTrack ? "audio" : "image";
      const currentLane = isAudioTrack ? this.getAudioLaneForY(mouseY) : 0;
      let gap = gapRegions.find(g => (
        cursor >= g.frameStart
        && cursor <= g.frameEnd
        && g.track === currentTrack
        && (!isAudioTrack || normalizeAudioLane(g.lane) === currentLane)
      ));

      if (!gap) {
        const startFrame = Math.round(cursor);
        gap = {
          track: currentTrack,
          lane: currentLane,
          frameStart: startFrame,
          frameEnd: startFrame + Math.max(1, this.getFrameRate())
        };
      }
      gap.clickedFrame = cursor;

      this.showGapContextMenu(e.clientX, e.clientY, gap);
    }
  }

  showContextMenu(clientX, clientY, seg, trackType) {
    this.dismissContextMenu();
    const menu = document.createElement("div");
    menu.className = "pr-gap-menu";
    menu.style.left = `${clientX + 6}px`;
    menu.style.top = `${clientY - 10}px`;

    const isImage = trackType !== "audio" && trackType !== "text" && seg.imageB64;
    const isSourceVideo = this.isSourceVideoSegment(seg);

    if (isImage) {
      const copyBtn = document.createElement("button");
      copyBtn.className = "pr-gap-menu-btn";
      copyBtn.innerHTML = `Copy Image`;
      copyBtn.onclick = async () => {
        try {
          const res = await fetch(seg.imageB64);
          const blob = await res.blob();
          await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
        } catch (err) {
          console.error("Failed to copy image", err);
        }
        this.dismissContextMenu();
      };
      menu.appendChild(copyBtn);

      const saveBtn = document.createElement("button");
      saveBtn.className = "pr-gap-menu-btn";
      saveBtn.innerHTML = `Save Image`;
      saveBtn.onclick = () => {
        const a = document.createElement("a");
        a.href = seg.imageB64;
        a.download = "timeline_image.jpg";
        a.click();
        this.dismissContextMenu();
      };
      menu.appendChild(saveBtn);

      const openBtn = document.createElement("button");
      openBtn.className = "pr-gap-menu-btn";
      openBtn.innerHTML = `Open Image in New Tab`;
      openBtn.onclick = () => {
        const win = window.open();
        if (win) {
          win.document.write(`<body style="margin:0;display:flex;justify-content:center;align-items:center;background:#0e0e0e;height:100vh;"><img style="max-width:100%;max-height:100%;" src="${seg.imageB64}" /></body>`);
          win.document.close();
        }
        this.dismissContextMenu();
      };
      menu.appendChild(openBtn);
    }

    if (trackType !== "audio") {
      const copyPromptBtn = document.createElement("button");
      copyPromptBtn.className = "pr-gap-menu-btn";
      copyPromptBtn.innerHTML = `Copy Prompt`;
      copyPromptBtn.onclick = async () => {
        try {
          await navigator.clipboard.writeText(seg.prompt || "");
        } catch (err) {
          console.error("Failed to copy prompt", err);
        }
        this.dismissContextMenu();
      };
      menu.appendChild(copyPromptBtn);
    }

    const currentTrack = trackType === "audio" ? "audio" : "image";
    if (!isSourceVideo) {
      const copySegBtn = document.createElement("button");
      copySegBtn.className = "pr-gap-menu-btn";
      copySegBtn.innerHTML = `Copy Segment`;
      copySegBtn.onclick = () => {
        this._copiedSegment = { ...seg, id: Date.now().toString() + Math.random().toString(36).substr(2, 5) };
        this._copiedSegmentTrack = trackType === "audio" ? "audio" : "image";
        this.dismissContextMenu();
      };
      menu.appendChild(copySegBtn);
    }

    if (!isSourceVideo && this._copiedSegment && this._copiedSegmentTrack === currentTrack) {
      const pasteReplaceBtn = document.createElement("button");
      pasteReplaceBtn.className = "pr-gap-menu-btn";
      pasteReplaceBtn.innerHTML = `Paste & Replace`;
      pasteReplaceBtn.onclick = () => {
        const targetArray = currentTrack === "audio" ? this.timeline.audioSegments : this.timeline.segments;
        const replacementLane = currentTrack === "audio"
          ? this.findFreeAudioLane(seg.start, this._copiedSegment.length, seg.id, seg.lane, targetArray)
          : undefined;
        const newSeg = {
          ...this._copiedSegment,
          id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
          start: seg.start,
          length: this._copiedSegment.length,
          ...(currentTrack === "audio" ? { lane: replacementLane } : {})
        };
        const idx = targetArray.findIndex(s => s.id === seg.id);
        if (idx >= 0) targetArray[idx] = newSeg;
        if (currentTrack === "audio") this.ensureAudioTrackHeight();
        this.commitChanges();
        this.dismissContextMenu();
      };
      menu.appendChild(pasteReplaceBtn);
    }

    const delBtn = document.createElement("button");
    delBtn.className = "pr-gap-menu-btn";
    delBtn.innerHTML = `Delete`;
    delBtn.style.color = "#ff4444";
    delBtn.onclick = () => {
      this.selectionType = trackType === "audio" ? "audio" : "image";
      const list = trackType === "audio" ? this.timeline.audioSegments : this.timeline.segments;
      this.selectedIndex = list.findIndex(s => s.id === seg.id);
      this.deleteSelectedSegment();
      this.dismissContextMenu();
    };
    menu.appendChild(delBtn);

    document.body.appendChild(menu);
    this._contextMenu = menu;

    setTimeout(() => {
      this._contextMenuDismisser = (ev) => { if (!menu.contains(ev.target)) this.dismissContextMenu(); };
      document.addEventListener("pointerdown", this._contextMenuDismisser, true);
    }, 0);
  }

  showGapContextMenu(clientX, clientY, gap) {
    this.dismissContextMenu();
    const menu = document.createElement("div");
    menu.className = "pr-gap-menu";
    menu.style.left = `${clientX + 6}px`;
    menu.style.top = `${clientY - 10}px`;

    const currentTrack = gap.track === "audio" ? "audio" : "image";

    if (this._copiedSegment && this._copiedSegmentTrack === currentTrack) {
      const pasteBtn = document.createElement("button");
      pasteBtn.className = "pr-gap-menu-btn";
      pasteBtn.innerHTML = `Paste Segment`;
      pasteBtn.onclick = () => {
        const startFrame = Math.round(gap.clickedFrame !== undefined ? gap.clickedFrame : gap.frameStart);
        const gapLength = gap.frameEnd - startFrame;
        const targetArray = currentTrack === "audio" ? this.timeline.audioSegments : this.timeline.segments;
        const lane = currentTrack === "audio"
          ? this.findFreeAudioLane(startFrame, this._copiedSegment.length, null, gap.lane, targetArray)
          : undefined;

        const newSeg = {
          ...this._copiedSegment,
          id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
          start: startFrame,
          length: currentTrack === "audio" ? this._copiedSegment.length : Math.min(this._copiedSegment.length, gapLength),
          ...(currentTrack === "audio" ? { lane } : {})
        };
        targetArray.push(newSeg);
        if (currentTrack === "audio") this.ensureAudioTrackHeight();
        targetArray.sort((a, b) => a.start - b.start);
        this.commitChanges();
        this.dismissContextMenu();
      };
      menu.appendChild(pasteBtn);
    }

    if (currentTrack === "image") {
      const textBtn = document.createElement("button");
      textBtn.className = "pr-gap-menu-btn";
      textBtn.innerHTML = `${ICONS.text} Text Segment`;
      textBtn.onclick = () => {
        this.addSegmentInGap(gap.frameStart, gap.frameEnd, "text");
        this.dismissContextMenu();
      };
      menu.appendChild(textBtn);

      const imgBtn = document.createElement("button");
      imgBtn.className = "pr-gap-menu-btn";
      imgBtn.innerHTML = `${ICONS.upload} Image Segment`;
      imgBtn.onclick = () => {
        this.dismissContextMenu();
        const gapLength = gap.frameEnd - gap.frameStart;
        this.showTimelineImageBrowser(gap.frameStart, gapLength);
      };
      menu.appendChild(imgBtn);
    }

    document.body.appendChild(menu);
    this._contextMenu = menu;
    setTimeout(() => {
      this._contextMenuDismisser = (ev) => { if (!menu.contains(ev.target)) this.dismissContextMenu(); };
      document.addEventListener("pointerdown", this._contextMenuDismisser, true);
    }, 0);
  }
  dismissContextMenu() {
    if (this._contextMenu) { this._contextMenu.remove(); this._contextMenu = null; }
    if (this._contextMenuDismisser) { document.removeEventListener("pointerdown", this._contextMenuDismisser, true); this._contextMenuDismisser = null; }
  }

  // --- Gap Popup Menu ---
  showGapMenu(clientX, clientY, gap) {
    this.dismissGapMenu();
    const menu = document.createElement("div");
    menu.className = "pr-gap-menu";
    menu.style.left = `${clientX + 6}px`;
    menu.style.top = `${clientY - 10}px`;

    const textBtn = document.createElement("button");
    textBtn.className = "pr-gap-menu-btn";
    textBtn.innerHTML = `${ICONS.text} Text Segment`;
    textBtn.addEventListener("click", () => {
      this.addSegmentInGap(gap.frameStart, gap.frameEnd, "text");
      this.dismissGapMenu();
    });

    const imgBtn = document.createElement("button");
    imgBtn.className = "pr-gap-menu-btn";
    imgBtn.innerHTML = `${ICONS.upload} Image Segment`;
    imgBtn.addEventListener("click", () => {
      this.dismissGapMenu();
      const gapLength = gap.frameEnd - gap.frameStart;
      this.showTimelineImageBrowser(gap.frameStart, gapLength);
    });

    menu.appendChild(textBtn);
    menu.appendChild(imgBtn);
    const currentTrack = gap.track === "audio" ? "audio" : "image";
    if (this._copiedSegment && this._copiedSegmentTrack === currentTrack) {
      const pasteBtn = document.createElement("button");
      pasteBtn.className = "pr-gap-menu-btn";
      pasteBtn.innerHTML = `Paste Segment`;
      pasteBtn.onclick = () => {
        const gapLength = gap.frameEnd - gap.frameStart;

        let finalLength = currentTrack === "audio" ? this._copiedSegment.length : Math.min(this._copiedSegment.length, gapLength);
        if (currentTrack === "image") {
          finalLength = gapLength;
        }

        const newSeg = {
          ...this._copiedSegment,
          id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
          start: gap.frameStart,
          length: finalLength
        };
        const targetArray = currentTrack === "audio" ? this.timeline.audioSegments : this.timeline.segments;
        if (currentTrack === "audio") {
          newSeg.lane = this.findFreeAudioLane(gap.frameStart, finalLength, null, gap.lane, targetArray);
        }
        targetArray.push(newSeg);
        if (currentTrack === "audio") this.ensureAudioTrackHeight();
        targetArray.sort((a, b) => a.start - b.start);
        this.commitChanges();
        this.dismissGapMenu();
      };
      menu.appendChild(pasteBtn);
    }

    document.body.appendChild(menu);
    this._gapMenu = menu;
    setTimeout(() => {
      this._gapMenuDismisser = (ev) => { if (!menu.contains(ev.target)) this.dismissGapMenu(); };
      document.addEventListener("pointerdown", this._gapMenuDismisser, true);
    }, 0);
  }

  dismissGapMenu() {
    if (this._gapMenu) { this._gapMenu.remove(); this._gapMenu = null; }
    if (this._gapMenuDismisser) { document.removeEventListener("pointerdown", this._gapMenuDismisser, true); this._gapMenuDismisser = null; }
  }

  // --- Settings Menu ---
  // Widgets that are managed by the settings menu (hidden from node by default).
  get _settingsWidgetNames() {
    return ["display_mode", "epsilon", "divisible_by", "img_compression"];
  }

  // Hide all settings widgets on the node (called on init).
  hideSettingsWidgets() {
    for (const name of this._settingsWidgetNames) {
      const w = this.node.widgets?.find(w => w.name === name);
      if (w) hideWidget(w);

      if (this.node.inputs) {
        const inputIdx = this.node.inputs.findIndex(i => i.name === name);
        if (inputIdx !== -1) {
          const input = this.node.inputs[inputIdx];
          if (input.link == null) {
            this.node.removeInput(inputIdx);
          }
        }
      }
    }
    this.updateWidgetVisibility();

    // Workaround: toggle display mode to force ComfyUI to refresh the node
    if (this.displayModeWidget) {
      const origVal = this.displayModeWidget.value;
      const otherVal = origVal === "frames" ? "seconds" : "frames";

      this.displayModeWidget.value = otherVal;
      if (this.displayModeWidget.callback) this.displayModeWidget.callback(otherVal);

      this.displayModeWidget.value = origVal;
      if (this.displayModeWidget.callback) this.displayModeWidget.callback(origVal);
    }
  }

  // Restore all settings widgets on the node.
  showSettingsWidgets() {
    for (const name of this._settingsWidgetNames) {
      const w = this.node.widgets?.find(w => w.name === name);
      if (!w) continue;
      const typeMap = {
        display_mode: "combo", epsilon: "FLOAT", divisible_by: "INT",
        img_compression: "INT",
      };
      w.type = typeMap[name] || w._origType || "number";
      w.hidden = false;
      if (w.options) w.options.hidden = false;
      delete w.computeSize;
      if (w.element) w.element.style.display = "";
    }
    this.updateWidgetVisibility();

    // Workaround: toggle display mode to force ComfyUI to refresh the node
    if (this.displayModeWidget) {
      const origVal = this.displayModeWidget.value;
      const otherVal = origVal === "frames" ? "seconds" : "frames";

      this.displayModeWidget.value = otherVal;
      if (this.displayModeWidget.callback) this.displayModeWidget.callback(otherVal);

      this.displayModeWidget.value = origVal;
      if (this.displayModeWidget.callback) this.displayModeWidget.callback(origVal);
    }
  }

  _makeSettingRow(label, inputEl) {
    const row = document.createElement("div");
    row.className = "pr-settings-row";
    const lbl = document.createElement("span");
    lbl.className = "pr-settings-label";
    lbl.textContent = label;
    row.appendChild(lbl);
    row.appendChild(inputEl);
    return row;
  }

  showSettingsMenu(anchorEl) {
    this.dismissSettingsMenu();
    const menu = document.createElement("div");
    menu.className = "pr-settings-menu";

    // Title & Close Button Container
    const titleContainer = document.createElement("div");
    titleContainer.className = "pr-settings-title";
    titleContainer.style.display = "flex";
    titleContainer.style.justifyContent = "space-between";
    titleContainer.style.alignItems = "center";

    const titleText = document.createElement("span");
    titleText.textContent = "Timeline Settings";
    titleContainer.appendChild(titleText);

    const closeBtn = document.createElement("button");
    closeBtn.className = "pr-settings-close-btn";
    closeBtn.innerHTML = ICONS.close;
    closeBtn.title = "Close Settings";
    closeBtn.addEventListener("click", () => this.dismissSettingsMenu());
    titleContainer.appendChild(closeBtn);

    menu.appendChild(titleContainer);

    // Helper: fire a widget's callback safely
    const fireCallback = (w, val) => {
      w.value = val;
      if (w.callback) {
        try { w.callback(val, app.canvas, this.node, null, null); } catch (e) { }
      }
      if (window.app && window.app.graph) window.app.graph.setDirtyCanvas(true, true);
    };

    // --- Display Mode ---
    const dmWidget = this.node.widgets?.find(w => w.name === "display_mode");
    if (dmWidget) {
      const ctrl = document.createElement("div");
      ctrl.className = "pr-segmented-control";

      const framesSeg = document.createElement("div");
      framesSeg.className = "pr-segment";
      framesSeg.textContent = "Frames";

      const secondsSeg = document.createElement("div");
      secondsSeg.className = "pr-segment";
      secondsSeg.textContent = "Seconds";

      const updateActive = (val) => {
        if (val === "frames") {
          framesSeg.classList.add("active");
          secondsSeg.classList.remove("active");
        } else {
          secondsSeg.classList.add("active");
          framesSeg.classList.remove("active");
        }
      };

      updateActive(dmWidget.value);

      const onSegClick = (val) => {
        fireCallback(dmWidget, val);
        updateActive(val);
        // Update ruler/timecode immediately
        if (this.updateWidgetVisibility) this.updateWidgetVisibility();
        if (this.updateUIFromSelection) this.updateUIFromSelection();
        this.render();
      };

      framesSeg.addEventListener("click", () => onSegClick("frames"));
      secondsSeg.addEventListener("click", () => onSegClick("seconds"));

      ctrl.appendChild(secondsSeg);
      ctrl.appendChild(framesSeg);

      menu.appendChild(this._makeSettingRow("Display Mode", ctrl));
    }

    // --- Hide Timeline Images/Prompts ---
    const hideTimelineVisualsWidget = this.node.widgets?.find(w => w.name === "hide_timeline_images_prompts");
    if (hideTimelineVisualsWidget) {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = hideTimelineVisualsWidget.value === true || hideTimelineVisualsWidget.value === "true";
      cb.style.cursor = "pointer";
      cb.title = "Hide timeline images plus global and segment prompt text when the mouse is outside the node.";
      cb.addEventListener("change", () => {
        hideTimelineVisualsWidget.value = cb.checked ? "true" : "false";
        if (hideTimelineVisualsWidget.callback) {
          try { hideTimelineVisualsWidget.callback(hideTimelineVisualsWidget.value, app.canvas, this.node, null, null); } catch (e) { }
        }
        this.updatePromptPrivacyVisibility();
        this.render();
        if (window.app && window.app.graph) window.app.graph.setDirtyCanvas(true, true);
      });
      menu.appendChild(this._makeSettingRow("Hide Images/Prompts", cb));
    }

    if (this.privacyModeWidget) {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = this.isPrivacyModeEnabled();
      cb.disabled = this.privacyBusy && !cb.checked;
      cb.style.cursor = cb.disabled ? "not-allowed" : "pointer";
      cb.title = "Encrypt workflow-saved timeline, media metadata, segment prompts, and global prompt.";
      cb.addEventListener("change", async () => {
        await this.setPrivacyMode(cb.checked);
        cb.checked = this.isPrivacyModeEnabled();
        cb.disabled = this.privacyBusy && !cb.checked;
      });
      menu.appendChild(this._makeSettingRow("Privacy Mode", cb));
    }

    const divider1 = document.createElement("hr");
    divider1.className = "pr-settings-divider";
    menu.appendChild(divider1);

    // Helper to create scrubbable number control with horizontal buttons
    const createScrubbableNumberControl = (w, step, min, max, isFloat = false) => {
      const container = document.createElement("div");
      container.className = "pr-number-control";

      const decBtn = document.createElement("button");
      decBtn.className = "pr-number-btn";
      decBtn.textContent = "-";

      const inp = document.createElement("input");
      inp.type = "number";
      inp.className = "pr-settings-input";
      inp.value = w.value;
      inp.step = step.toString();
      inp.min = min.toString();
      inp.max = max.toString();

      const incBtn = document.createElement("button");
      incBtn.className = "pr-number-btn";
      incBtn.textContent = "+";

      decBtn.addEventListener("click", () => {
        let val = parseFloat(inp.value) - step;
        if (val < min) val = min;
        inp.value = isFloat ? val.toFixed(4) : Math.round(val);
        fireCallback(w, parseFloat(inp.value));
      });

      incBtn.addEventListener("click", () => {
        let val = parseFloat(inp.value) + step;
        if (val > max) val = max;
        inp.value = isFloat ? val.toFixed(4) : Math.round(val);
        fireCallback(w, parseFloat(inp.value));
      });

      inp.addEventListener("change", () => {
        let val = parseFloat(inp.value);
        if (isNaN(val)) val = w.value;
        if (val < min) val = min;
        if (val > max) val = max;
        inp.value = isFloat ? val.toFixed(4) : Math.round(val);
        fireCallback(w, parseFloat(inp.value));
      });

      // Dragging logic
      let isDragging = false;
      let startX = 0;
      let startVal = 0;
      let hasMoved = false;

      inp.style.cursor = "ew-resize";

      inp.addEventListener("mousedown", (e) => {
        startX = e.clientX;
        startVal = parseFloat(inp.value);
        hasMoved = false;

        const onMouseMove = (moveEvent) => {
          const deltaX = moveEvent.clientX - startX;
          if (Math.abs(deltaX) > 3) {
            hasMoved = true;
            isDragging = true;
          }

          if (isDragging) {
            moveEvent.preventDefault();
            const sensitivity = isFloat ? 0.001 : 0.5;
            let newVal = startVal + deltaX * sensitivity;

            if (newVal < min) newVal = min;
            if (newVal > max) newVal = max;

            inp.value = isFloat ? newVal.toFixed(4) : Math.round(newVal);
            fireCallback(w, parseFloat(inp.value));
          }
        };

        const onMouseUp = () => {
          document.removeEventListener("mousemove", onMouseMove);
          document.removeEventListener("mouseup", onMouseUp);

          if (!hasMoved) {
            inp.focus();
            inp.select();
          }
          isDragging = false;
        };

        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("mouseup", onMouseUp);
      });

      container.appendChild(decBtn);
      container.appendChild(inp);
      container.appendChild(incBtn);

      return container;
    };

    // --- Epsilon ---
    const epsWidget = this.node.widgets?.find(w => w.name === "epsilon");
    if (epsWidget) {
      menu.appendChild(this._makeSettingRow("Epsilon", createScrubbableNumberControl(epsWidget, 0.0001, 0.0001, 0.99, true)));
    }

    // --- Divisible By ---
    const divByWidget = this.node.widgets?.find(w => w.name === "divisible_by");
    if (divByWidget) {
      menu.appendChild(this._makeSettingRow("Divisible By", createScrubbableNumberControl(divByWidget, 1, 1, 256, false)));
    }

    // --- Img Compression ---
    const compWidget = this.node.widgets?.find(w => w.name === "img_compression");
    if (compWidget) {
      menu.appendChild(this._makeSettingRow("Img Compression", createScrubbableNumberControl(compWidget, 1, 0, 100, false)));
    }

    // --- Global Prompt Toggle ---
    const globalPromptWidget = this.node.widgets?.find(w => w.name === "global_prompt");
    if (globalPromptWidget) {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = this.globalPromptEnabled();
      cb.style.cursor = "pointer";
      cb.addEventListener("change", () => {
        const isVisible = cb.checked;
        setWidgetBoolValue(this.useGlobalPromptWidget, isVisible);
        this.applyGlobalPromptVisibility();

        // Force refresh via display mode double-toggle trick
        if (this.displayModeWidget) {
          const origVal = this.displayModeWidget.value;
          const otherVal = origVal === "frames" ? "seconds" : "frames";
          this.displayModeWidget.value = otherVal;
          if (this.displayModeWidget.callback) this.displayModeWidget.callback(otherVal);
          this.displayModeWidget.value = origVal;
          if (this.displayModeWidget.callback) this.displayModeWidget.callback(origVal);
        }
        this.updatePromptPrivacyVisibility();
      });
      menu.appendChild(this._makeSettingRow("Use Global Prompt", cb));
    }


    // --- Show/Hide on Node Toggle ---
    const toggleBtn = document.createElement("button");
    toggleBtn.className = "pr-settings-toggle-btn";
    const widgetsVisible = !!(this.node.widgets?.find(w => w.name === "display_mode" && !(w.options && w.options.hidden)));
    toggleBtn.textContent = widgetsVisible ? "Hide Widgets on Node" : "Show Widgets on Node";
    toggleBtn.addEventListener("click", () => {
      const nowVisible = !!(this.node.widgets?.find(w => w.name === "display_mode" && !(w.options && w.options.hidden)));
      if (nowVisible) {
        this.hideSettingsWidgets();
        toggleBtn.textContent = "Show Widgets on Node";
      } else {
        this.showSettingsWidgets();
        toggleBtn.textContent = "Hide Widgets on Node";
      }
    });
    menu.appendChild(toggleBtn);

    // Position the menu below the anchor button (pop down)
    document.body.appendChild(menu);
    const rect = anchorEl.getBoundingClientRect();
    const menuW = menu.offsetWidth || 230;
    const menuH = menu.offsetHeight || 350;
    let left = rect.right - menuW;
    let top = rect.bottom + 6;
    if (left < 4) left = 4;
    // Fallback to top if it overflows the bottom of the screen
    if (top + menuH > window.innerHeight - 4) {
      top = rect.top - menuH - 6;
    }
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;

    this._settingsMenu = menu;
    setTimeout(() => {
      this._settingsDismisser = (ev) => {
        if (!menu.contains(ev.target) && !anchorEl.contains(ev.target)) this.dismissSettingsMenu();
      };
      document.addEventListener("mousedown", this._settingsDismisser);
    }, 0);
  }

  dismissSettingsMenu() {
    if (this._settingsMenu) { this._settingsMenu.remove(); this._settingsMenu = null; }
    if (this._settingsDismisser) { document.removeEventListener("mousedown", this._settingsDismisser); this._settingsDismisser = null; }
  }


  addSegmentInGap(frameStart, frameEnd, type = "text") {
    const seg = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
      start: frameStart, length: frameEnd - frameStart,
      prompt: "", type,
    };
    this.timeline.segments.push(seg);
    this.timeline.segments.sort((a, b) => a.start - b.start);
    this.selectionType = "image";
    this.selectedIndex = this.timeline.segments.findIndex(s => s.id === seg.id);
    this.updateUIFromSelection();
    this.commitChanges();
  }

  addTextSegmentFreeSpace() {
    const frameRate = this.getFrameRate();
    const newLength = Math.max(1, frameRate); // 1 second default
    const sorted = [...this.timeline.segments].sort((a, b) => a.start - b.start);
    let newStart = 0;
    for (const seg of sorted) {
      if (newStart + newLength <= seg.start) break;
      newStart = Math.max(newStart, seg.start + seg.length);
    }
    // Place the segment at the first free slot in the visual timeline (no output duration change).
    const durationFrames = this.getVisualDurationFrames();
    const seg = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
      start: newStart, length: Math.min(newLength, Math.max(newLength, durationFrames - newStart)),
      prompt: "", type: "text",
    };
    this.timeline.segments.push(seg);
    this.timeline.segments.sort((a, b) => a.start - b.start);
    this.selectionType = "image";
    this.selectedIndex = this.timeline.segments.findIndex(s => s.id === seg.id);
    this.updateUIFromSelection();
    this.commitChanges();
  }

  // --- Audio Player Engine ---
  updatePlayerUI() {
    if (!this.playBtn || !this.loopBtn) return;
    this.playBtn.innerHTML = this.isPlaying ? ICONS.pause : ICONS.play;
    if (this.isLooping) {
      this.loopBtn.classList.add("active");
    } else {
      this.loopBtn.classList.remove("active");
    }
    if (this.seekBar) {
      this.seekBar.max = this.getVisualDurationFrames();
      this.seekBar.value = this.currentFrame;
    }
    if (this.timeCodeDisplay) {
      this.timeCodeDisplay.textContent = this.formatTime(this.currentFrame);
    }
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pauseAudio();
    } else {
      if (this.currentFrame >= this.getVisualDurationFrames()) {
        this.currentFrame = 0;
      }
      this.playAudio();
    }
  }

  toggleLoop() {
    this.isLooping = !this.isLooping;
    this.updatePlayerUI();
  }

  async playAudio() {
    this.pauseAudio(true); // clear any existing playback, but don't suspend context if scrubbing

    this._playCounter = (this._playCounter || 0) + 1;
    const playId = this._playCounter;
    this._currentPlayId = playId;
    this.isPlaying = true;

    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.audioContext.state !== 'running') {
      try { await this.audioContext.resume(); } catch (e) { }
    }
    if (this._currentPlayId !== playId || !this.isPlaying) return;

    this.updatePlayerUI();

    const frameRate = this.getFrameRate();
    this.playbackStartFrame = this.currentFrame;
    this.playbackStartTime = this.audioContext.currentTime;

    // Decode and schedule all audio segments that happen AT or AFTER currentFrame
    for (let seg of this.timeline.audioSegments) {
      const segStartFrame = seg.start;
      const segEndFrame = seg.start + seg.length;

      if (segEndFrame <= this.currentFrame) continue;

      try {
        // Build audio buffer: fetch from server URL if audioFile is set, otherwise fall back to audioB64
        let audioBuffer;
        if (seg.audioFile) {
          const audioUrl = this.getTimelineAudioSegmentUrl(seg);
          const resp = await fetch(audioUrl);
          const arrayBuffer = await resp.arrayBuffer();
          audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        } else if (seg.audioB64) {
          const binaryString = window.atob(seg.audioB64);
          const len = binaryString.length;
          const bytes = new Uint8Array(len);
          for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
          }
          audioBuffer = await this.audioContext.decodeAudioData(bytes.buffer);
        } else {
          continue;
        }
        if (this._currentPlayId !== playId || !this.isPlaying) return;

        const framesToSkipInSegment = Math.max(0, this.currentFrame - segStartFrame);
        const waitFrames = Math.max(0, segStartFrame - this.currentFrame);
        const waitTimeSec = waitFrames / frameRate;

        const fileOffsetFrames = seg.trimStart + framesToSkipInSegment;
        const fileOffsetSec = fileOffsetFrames / frameRate;

        const playDurationFrames = seg.length - framesToSkipInSegment;
        const playDurationSec = playDurationFrames / frameRate;

        if (playDurationSec <= 0) continue;

        const source = this.audioContext.createBufferSource();
        const gainNode = this.audioContext.createGain();
        source.buffer = audioBuffer;
        gainNode.gain.value = clampVolume(seg.volume);
        source.connect(gainNode);
        gainNode.connect(this.audioContext.destination);

        const startTime = this.audioContext.currentTime + waitTimeSec;
        source.start(startTime, fileOffsetSec, playDurationSec);

        this.activeAudioNodes.push(source, gainNode);
      } catch (err) {
        console.error("Playback decode error for segment:", err);
      }
    }

    if (this._currentPlayId !== playId || !this.isPlaying) return;

    const loop = () => {
      if (!this.isPlaying || this._currentPlayId !== playId) return;

      const elapsedSec = this.audioContext.currentTime - this.playbackStartTime;
      const elapsedFrames = elapsedSec * frameRate;

      this.currentFrame = this.playbackStartFrame + elapsedFrames;

      const visualDurationFrames = this.getVisualDurationFrames();
      const durationFrames = this.getDurationFrames();

      if (this.isLooping) {
        const loopBound = (this.playbackStartFrame >= durationFrames) ? visualDurationFrames : durationFrames;
        if (this.currentFrame >= loopBound) {
          this.currentFrame = 0;
          this.playAudio(); // Restart playback
          return;
        }
      } else {
        if (this.currentFrame >= visualDurationFrames) {
          this.currentFrame = visualDurationFrames;
          this.pauseAudio();
          this.render();
          return;
        }
      }

      this.render();
      this._playLoopId = requestAnimationFrame(loop);
    };

    this._playLoopId = requestAnimationFrame(loop);
  }

  pauseAudio(isScrubbing = false) {
    this.isPlaying = false;
    this._currentPlayId = null;

    if (!isScrubbing && this.audioContext && this.audioContext.state === 'running') {
      try { this.audioContext.suspend(); } catch (e) { }
    }

    for (let node of this.activeAudioNodes) {
      try { node.stop(); } catch (e) { }
      try { node.disconnect(); } catch (e) { }
    }
    this.activeAudioNodes = [];

    if (this._playLoopId) {
      cancelAnimationFrame(this._playLoopId);
      this._playLoopId = null;
    }
    this.updatePlayerUI();
  }
}

// --- Node Registration Hooks ---
const APPENDED_WIDGET_DEFAULTS = [
  ["timeline_data", "{}"],
  ["local_prompts", ""],
  ["segment_lengths", ""],
  ["use_global_prompt", "false"],
  ["hide_timeline_images_prompts", "false"],
  ["privacy_mode", "false"],
  ["privacy_payload", ""],
];

app.registerExtension({
  name: "LTXDirector",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "LTXDirector") {

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);

        for (const [name, def] of APPENDED_WIDGET_DEFAULTS) {
          if (!this.widgets?.find(w => w.name === name)) {
            this.addWidget("string", name, def, () => { });
          }
        }
        for (const w of this.widgets) {
          if (HIDDEN_WIDGET_NAMES.includes(w.name)) hideWidget(w);
        }

        // Set default width to be wider on creation (approx 2.5x default ~220px)
        this.size[0] = 1000;

        // Force default for img_compression if not set (ComfyUI sometimes skips optional defaults)
        const compWidget = this.widgets?.find(w => w.name === "img_compression");
        if (compWidget && (compWidget.value === undefined || compWidget.value === null || compWidget.value === 0)) {
          compWidget.value = 18;
        }

        const globalPromptWidget = this.widgets?.find(w => w.name === "global_prompt");
        const useGlobalPromptWidget = this.widgets?.find(w => w.name === "use_global_prompt");
        applyGlobalPromptWidgetVisibility(globalPromptWidget, widgetBoolValue(useGlobalPromptWidget?.value));

        const container = document.createElement("div");
        const widget = this.addDOMWidget("timeline_ui", "timeline_ui", container, {
          getValue: () => "",
          setValue: () => { },
        });

        widget.computeSize = function (width) {
          const canvasH = self._timelineEditor ? self._timelineEditor.canvasHeight : CANVAS_HEIGHT;
          return [timelineNodeInnerWidth(self), canvasH + 235];
        };

        const self = this;
        setTimeout(() => {
          try {
            self._timelineEditor = new TimelineEditor(self, container, widget);
          } catch (err) {
            console.error("[PromptRelay] timeline editor init failed:", err);
          }
        }, 0);
      };

      const onRemoved = nodeType.prototype.onRemoved;
      nodeType.prototype.onRemoved = function () {
        this._timelineEditor?.destroy();
        return onRemoved?.apply(this, arguments);
      };

      const onSerialize = nodeType.prototype.onSerialize;
      nodeType.prototype.onSerialize = function (info) {
        const editor = this._timelineEditor;
        editor?.flushPromptEdit({ skipNodeResize: true });
        const privacyModeWidget = this.widgets?.find(w => w.name === "privacy_mode");
        const privacyPayloadWidget = this.widgets?.find(w => w.name === "privacy_payload");
        const privacyEnabled = widgetBoolValue(privacyModeWidget?.value);
        const stashed = [];

        if (privacyEnabled) {
          if (editor && !editor.privacyLocked) {
            try {
              const envelope = encryptPrivacyStateSync(editor.getTimelinePrivacyState());
              if (privacyPayloadWidget) privacyPayloadWidget.value = JSON.stringify(envelope);
              editor.setPrivacyStatus("", false);
            } catch (err) {
              editor.setPrivacyStatus(`Privacy encryption failed while saving: ${err.message}`, false);
            }
          }

          const sanitizedValues = {
            global_prompt: "",
            timeline_data: EMPTY_TIMELINE_JSON,
            local_prompts: "",
            segment_lengths: "",
            guide_strength: "",
            privacy_mode: "true",
            privacy_payload: privacyPayloadWidget?.value || "",
          };
          for (const [name, value] of Object.entries(sanitizedValues)) {
            const widget = this.widgets?.find(w => w.name === name);
            if (!widget) continue;
            stashed.push([widget, widget.value]);
            widget.value = value;
          }
        }

        const out = onSerialize?.apply(this, arguments);

        if (privacyEnabled) {
          setSerializedWidgetValue(info, this, "global_prompt", "");
          setSerializedWidgetValue(info, this, "timeline_data", EMPTY_TIMELINE_JSON);
          setSerializedWidgetValue(info, this, "local_prompts", "");
          setSerializedWidgetValue(info, this, "segment_lengths", "");
          setSerializedWidgetValue(info, this, "guide_strength", "");
          setSerializedWidgetValue(info, this, "privacy_mode", "true");
          setSerializedWidgetValue(info, this, "privacy_payload", privacyPayloadWidget?.value || "");
        }

        for (const [widget, value] of stashed) {
          widget.value = value;
        }

        if (privacyEnabled && editor && !editor.privacyBusy && !editor.privacyLocked) {
          void editor.encryptPrivacyState({ renderAfter: false });
        }

        return out;
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        const out = onConfigure?.apply(this, arguments);
        for (const [name, def] of APPENDED_WIDGET_DEFAULTS) {
          const w = this.widgets.find(x => x.name === name);
          if (w && (w.value == null || w.value === "")) w.value = def;
        }

        const privacyPayloadIndex = serializedWidgetIndex(this, "privacy_payload");
        if (Array.isArray(info?.widgets_values) && privacyPayloadIndex >= 0 && info.widgets_values.length <= privacyPayloadIndex) {
          const privacyModeWidget = this.widgets?.find(w => w.name === "privacy_mode");
          const hideTimelineWidget = this.widgets?.find(w => w.name === "hide_timeline_images_prompts");
          const privacyPayloadWidget = this.widgets?.find(w => w.name === "privacy_payload");
          if (hideTimelineWidget && privacyModeWidget) hideTimelineWidget.value = privacyModeWidget.value || "false";
          if (privacyModeWidget) privacyModeWidget.value = "false";
          if (privacyPayloadWidget) privacyPayloadWidget.value = "";
        }
        repairLegacyPrivacyWidgetShift(this);

        setTimeout(() => {
          const globalPromptWidget = this.widgets?.find(w => w.name === "global_prompt");
          const useGlobalPromptWidget = this.widgets?.find(w => w.name === "use_global_prompt");
          applyGlobalPromptWidgetVisibility(globalPromptWidget, widgetBoolValue(useGlobalPromptWidget?.value));
          if (this._timelineEditor) {
            repairLegacyPrivacyWidgetShift(this);
            this._timelineEditor.useGlobalPromptWidget = useGlobalPromptWidget;
            this._timelineEditor.privacyModeWidget = this.widgets?.find(w => w.name === "privacy_mode");
            this._timelineEditor.privacyPayloadWidget = this.widgets?.find(w => w.name === "privacy_payload");
            if (this._timelineEditor.isPrivacyModeEnabled() && isEncryptedPrivacyPayload(this._timelineEditor.privacyPayloadWidget?.value)) {
              this._timelineEditor.privacyLocked = true;
              this._timelineEditor.setPrivacyStatus("Decrypting private timeline data...", true);
              void this._timelineEditor.decryptPrivacyPayload();
            } else {
              this._timelineEditor.timeline = parseInitial(this._timelineEditor.timelineDataWidget?.value);
              this._timelineEditor.loadImages();
              this._timelineEditor.applyGlobalPromptVisibility();
              this._timelineEditor.selectionType = "image";
              this._timelineEditor.selectedIndex = clamp(
                this._timelineEditor.selectedIndex, -1,
                Math.max(-1, this._timelineEditor.timeline.segments.length - 1)
              );
              this._timelineEditor.updateUIFromSelection();
              this._timelineEditor.render();
            }
          }
        }, 0);
        return out;
      };
    }
  },
});
