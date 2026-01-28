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

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

const trainBtn = document.getElementById("trainBtn");
const predictBtn = document.getElementById("predictBtn");

const trainOut = document.getElementById("trainOut");
const predictOut = document.getElementById("predictOut");

const modelIdInput = document.getElementById("modelId");
const payloadInput = document.getElementById("payload");

trainBtn.addEventListener("click", async () => {
  trainOut.textContent = "Training...";
  try {
    const data = await postJson("/train", {});
    trainOut.textContent = pretty(data);

    // convenience: auto-fill model id
    if (data.model_id) modelIdInput.value = data.model_id;
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
