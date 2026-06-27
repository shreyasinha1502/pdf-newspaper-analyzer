const form = document.querySelector("#upload-form");
const input = document.querySelector("#pdf-input");
const dropZone = document.querySelector("#drop-zone");
const fileName = document.querySelector("#file-name");
const uploadButton = document.querySelector("#upload-button");
const resetButton = document.querySelector("#reset-button");
const progressBar = document.querySelector("#progress-bar");
const statusText = document.querySelector("#status");
const details = document.querySelector("#details");

const fields = {
  pages: document.querySelector("#pages"),
  headlines: document.querySelector("#headlines"),
  photos: document.querySelector("#photos"),
  processingTime: document.querySelector("#processing-time"),
  headlineConfidence: document.querySelector("#headline-confidence"),
  photoConfidence: document.querySelector("#photo-confidence"),
};

let selectedFile = null;
let progressTimer = null;

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  setFile(event.dataTransfer.files[0]);
});

input.addEventListener("change", () => {
  setFile(input.files[0]);
});

resetButton.addEventListener("click", () => {
  selectedFile = null;
  input.value = "";
  fileName.textContent = "or choose a file to analyze";
  setStatus("Ready");
  setProgress(0);
  details.hidden = true;
  Object.values(fields).forEach((field) => {
    field.textContent = "-";
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) {
    setStatus("Choose a PDF first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  uploadButton.disabled = true;
  setStatus("Uploading and analyzing...");
  startProgress();

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Analysis failed.");
    }
    renderResult(payload);
    setProgress(100);
    setStatus("Analysis complete.");
  } catch (error) {
    setStatus(error.message, true);
    setProgress(0);
  } finally {
    clearInterval(progressTimer);
    uploadButton.disabled = false;
  }
});

function setFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatus("Only PDF files are supported.", true);
    return;
  }
  selectedFile = file;
  fileName.textContent = `${file.name} (${formatBytes(file.size)})`;
  setStatus("Ready to analyze.");
}

function renderResult(result) {
  fields.pages.textContent = result.pages ?? "-";
  fields.headlines.textContent = result.headlines ?? "-";
  fields.photos.textContent = result.photos ?? "-";
  fields.processingTime.textContent = result.processing_time ?? "-";
  fields.headlineConfidence.textContent = formatConfidence(result.headline_confidence);
  fields.photoConfidence.textContent = formatConfidence(result.photo_confidence);
  details.hidden = false;
}

function startProgress() {
  let value = 8;
  setProgress(value);
  clearInterval(progressTimer);
  progressTimer = setInterval(() => {
    value = Math.min(value + Math.max(1, (88 - value) * 0.08), 88);
    setProgress(value);
  }, 350);
}

function setProgress(value) {
  progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`;
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function formatConfidence(value) {
  if (typeof value !== "number") return "-";
  return `${Math.round(value * 100)}%`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
