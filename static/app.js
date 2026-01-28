async function postJson(url, bodyObj) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj ?? {})
  });

  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }

  if (!res.ok) {
    const msg = data?.detail ? JSON.stringify(data.detail) : JSON.stringify(data);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return data;
}

async function getJson(url) {
  const res = await fetch(url);

  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }

  if (!res.ok) {
    const msg = data?.detail ? JSON.stringify(data.detail) : JSON.stringify(data);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return data;
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

document.addEventListener("DOMContentLoaded", () => {
  const uploadBtn = document.getElementById("uploadBtn");
  const trainBtn = document.getElementById("trainBtn");
  const predictBtn = document.getElementById("predictBtn");
  const globalExplainBtn = document.getElementById("globalExplainBtn");
  const localExplainBtn = document.getElementById("localExplainBtn");

  const uploadOut = document.getElementById("uploadOut");
  const trainOut = document.getElementById("trainOut");
  const predictOut = document.getElementById("predictOut");
  const globalExplainOut = document.getElementById("globalExplainOut");
  const localExplainOut = document.getElementById("localExplainOut");
  const datasetsList = document.getElementById("datasetsList");
  const modelsList = document.getElementById("modelsList");

  const csvFileInput = document.getElementById("csvFile");
  const datasetSelect = document.getElementById("datasetSelect");
  const modelTypeSelect = document.getElementById("modelType");
  const targetColumnSelect = document.getElementById("targetColumn");
  const modelIdInput = document.getElementById("modelId");
  const payloadInput = document.getElementById("payload");
  const explainModelIdInput = document.getElementById("explainModelId");
  const explainRowIndexInput = document.getElementById("explainRowIndex");

  async function loadColumns(datasetId) {
    targetColumnSelect.innerHTML = '<option value="">Select target</option>';
    try {
      const data = await getJson(`/datasets/${datasetId}/columns`);
      data.columns.forEach(col => {
        const option = document.createElement("option");
        option.value = col;
        option.textContent = col;
        targetColumnSelect.appendChild(option);
      });
    } catch (err) {
      console.error("Error loading columns:", err);
    }
  }

  async function loadDatasets() {
    datasetsList.innerHTML = "";
    datasetSelect.innerHTML = '<option value="sample">Sample Data</option>';
    try {
      const data = await getJson("/datasets");
      data.datasets.forEach(ds => {
        if (ds.dataset_id !== "sample") {
          const option = document.createElement("option");
          option.value = ds.dataset_id;
          option.textContent = ds.filename;
          datasetSelect.appendChild(option);
        }
        const li = document.createElement("li");
        li.textContent = `${ds.filename} (${ds.rows} rows, ${ds.columns} cols, target: ${ds.has_target ? 'yes' : 'no'})`;
        li.addEventListener("click", () => {
          datasetSelect.value = ds.dataset_id;
          loadColumns(ds.dataset_id);
        });
        datasetsList.appendChild(li);
      });
      // Load columns for sample
      loadColumns("sample");
    } catch (err) {
      datasetsList.innerHTML = `<li>Error: ${err.message}</li>`;
      datasetSelect.innerHTML = '<option value="sample">Sample Data</option>';
    }
  }

  async function loadModels() {
    modelsList.innerHTML = "";
    try {
      const data = await getJson("/models");
      data.models.forEach(model => {
        const li = document.createElement("li");
        li.textContent = `ID: ${model.model_id} | Type: ${model.model_type} | Dataset: ${model.dataset_id} | Score: ${model.score || 'N/A'}`;
        li.addEventListener("click", () => {
          modelIdInput.value = model.model_id;
        });
        modelsList.appendChild(li);
      });
    } catch (err) {
      modelsList.innerHTML = `<li>Error: ${err.message}</li>`;
    }
  }

// Load on page load
window.addEventListener("load", () => {
  loadDatasets();
  loadModels();
});

datasetSelect.addEventListener("change", () => {
  const datasetId = datasetSelect.value;
  loadColumns(datasetId);
});

modelTypeSelect.addEventListener("change", () => {
  const isSupervised = ["logistic", "random_forest"].includes(modelTypeSelect.value);
  targetColumnSelect.disabled = !isSupervised;
});

const uploadBtn = document.getElementById("uploadBtn");
const trainBtn = document.getElementById("trainBtn");
const predictBtn = document.getElementById("predictBtn");
const globalExplainBtn = document.getElementById("globalExplainBtn");
const localExplainBtn = document.getElementById("localExplainBtn");

const uploadOut = document.getElementById("uploadOut");
const trainOut = document.getElementById("trainOut");
const predictOut = document.getElementById("predictOut");
const globalExplainOut = document.getElementById("globalExplainOut");
const localExplainOut = document.getElementById("localExplainOut");
const datasetsList = document.getElementById("datasetsList");
const modelsList = document.getElementById("modelsList");

const csvFileInput = document.getElementById("csvFile");
const datasetSelect = document.getElementById("datasetSelect");
const modelTypeSelect = document.getElementById("modelType");
const targetColumnSelect = document.getElementById("targetColumn");
const modelIdInput = document.getElementById("modelId");
const payloadInput = document.getElementById("payload");
const explainModelIdInput = document.getElementById("explainModelId");
const explainRowIndexInput = document.getElementById("explainRowIndex");

uploadBtn.addEventListener("click", async () => {
  uploadOut.textContent = "Uploading...";
  try {
    const file = csvFileInput.files[0];
    if (!file) throw new Error("Select a CSV file first");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/upload", {
      method: "POST",
      body: formData
    });

    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }

    if (!res.ok) {
      const msg = data?.detail ? JSON.stringify(data.detail) : JSON.stringify(data);
      throw new Error(`HTTP ${res.status}: ${msg}`);
    }

    uploadOut.textContent = `Uploaded: ${data.filename}`;
    csvFileInput.value = "";

    // Reload datasets and select the new one
    await loadDatasets();
    if (datasetSelect.querySelector(`option[value="${data.dataset_id}"]`)) {
      datasetSelect.value = data.dataset_id;
    }
  } catch (err) {
    uploadOut.textContent = `Error: ${err.message}`;
  }
});

trainBtn.addEventListener("click", async () => {
  trainOut.textContent = "Training...";
  try {
    const body = {
      dataset_id: datasetSelect.value,
      model_type: modelTypeSelect.value,
      target_column: targetColumnSelect.value
    };
    const data = await postJson("/train", body);
    trainOut.textContent = `Trained: ${data.model_id}`;

    // Reload models and select the new one
    await loadModels();
    modelIdInput.value = data.model_id;
  } catch (err) {
    trainOut.textContent = `Error: ${err.message}`;
  }
});



predictBtn.addEventListener("click", async () => {
  predictOut.textContent = "Predicting...";
  try {
    const modelId = (modelIdInput.value || "").trim();
    if (!modelId) throw new Error("Model ID is required");

    let payload;
    try {
      payload = JSON.parse(payloadInput.value);
    } catch {
      throw new Error("Payload must be valid JSON");
    }

    const data = await postJson(`/predict/${encodeURIComponent(modelId)}`, payload);
    predictOut.textContent = pretty(data);
  } catch (err) {
    predictOut.textContent = `Error: ${err.message}`;
  }
});

globalExplainBtn.addEventListener("click", async () => {
  globalExplainOut.textContent = "Generating...";
  try {
    const modelId = (explainModelIdInput.value || "").trim();
    if (!modelId) throw new Error("Model ID is required");

    const data = await postJson(`/explain/global/${encodeURIComponent(modelId)}`, {});
    if (data.message) {
      globalExplainOut.textContent = data.message;
    } else {
      globalExplainOut.textContent = pretty(data.global_feature_importance);
    }
  } catch (err) {
    globalExplainOut.textContent = `Error: ${err.message}`;
  }
});

localExplainBtn.addEventListener("click", async () => {
  localExplainOut.textContent = "Generating...";
  try {
    const modelId = (explainModelIdInput.value || "").trim();
    if (!modelId) throw new Error("Model ID is required");

    const rowIndex = parseInt(explainRowIndexInput.value);
    if (isNaN(rowIndex)) throw new Error("Row index must be a number");

    const data = await postJson(`/explain/local/${encodeURIComponent(modelId)}`, rowIndex);
    if (data.message) {
      localExplainOut.textContent = data.message;
    } else {
      localExplainOut.textContent = pretty(data.local_explanation);
    }
  } catch (err) {
    localExplainOut.textContent = `Error: ${err.message}`;
  }
});
