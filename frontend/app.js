// Keep the frontend independent: serve this directory separately on port 5500.
const API_BASE = "http://127.0.0.1:8000";
const form = document.querySelector("#prediction-form");
const errorBox = document.querySelector("#error-message");
const submitButton = document.querySelector("#submit-button");
const apiStatus = document.querySelector("#api-status");
const apiPill = document.querySelector(".api-pill");
const resultTitle = document.querySelector("#result-title");
const resultSummary = document.querySelector("#result-summary");
const probabilityValue = document.querySelector("#probability-value");
const probabilityMeter = document.querySelector("#probability-meter");
const resultSymbol = document.querySelector(".result-symbol");
const thresholdLabel = document.querySelector("#threshold-label");
const ageInput = document.querySelector('input[name="Age"]');
const ageOutput = document.querySelector('output[for="Age"]');

function updateAgeOutput() {
  ageOutput.textContent = `${ageInput.value} years`;
}

async function checkApi() {
  try {
    const response = await fetch(`${API_BASE}/`, { method: "GET" });
    if (!response.ok) throw new Error("API unavailable");
    apiStatus.textContent = "API connected";
    apiPill.classList.add("connected");
    apiPill.classList.remove("offline");
  } catch {
    apiStatus.textContent = "API offline";
    apiPill.classList.add("offline");
    apiPill.classList.remove("connected");
  }
}

function clearResult() {
  errorBox.hidden = true;
  resultTitle.textContent = "Your stroke risk estimate will appear here.";
  resultSummary.textContent = "Complete the indicators above, then select “Analyze risk”.";
  probabilityValue.textContent = "—";
  probabilityMeter.style.width = "0%";
  thresholdLabel.textContent = "Threshold —";
  resultSymbol.textContent = "↗";
  resultSymbol.style.color = "var(--accent)";
}

function collectFeatures() {
  const features = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    if (element.type === "checkbox") features[element.name] = element.checked ? 1 : 0;
    else if (element.type === "number" || element.type === "range") features[element.name] = Number(element.value);
  }
  return features;
}

document.querySelector("#randomize-inputs").addEventListener("click", () => {
  const age = document.querySelector('input[name="Age"]');
  ageInput.value = String(Math.floor(Math.random() * 120) + 1);
  updateAgeOutput();
  document.querySelectorAll('input[type="checkbox"]').forEach((input) => {
    input.checked = Math.random() >= 0.5;
  });
  clearResult();
});

document.querySelector("#reset-form").addEventListener("click", () => {
  form.reset();
  updateAgeOutput();
  clearResult();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  const features = collectFeatures();

  if (!Number.isInteger(features.Age) || features.Age < 1 || features.Age > 120) {
    errorBox.textContent = "Please enter an age between 1 and 120.";
    errorBox.hidden = false;
    return;
  }

  submitButton.disabled = true;
  submitButton.innerHTML = "Analyzing… <span>↻</span>";

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features })
    });
    let payload;
    try { payload = await response.json(); } catch { payload = {}; }
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : "The API rejected the request.";
      throw new Error(detail);
    }

    const probability = Number(payload.probability);
    const prediction = Number(payload.prediction);
    const threshold = Number(payload.threshold);
    if (!Number.isFinite(probability) || ![0, 1].includes(prediction) || !Number.isFinite(threshold)) {
      throw new Error("The API returned an invalid prediction response.");
    }

    const pct = Math.max(0, Math.min(100, probability * 100));
    const formattedPct = pct.toFixed(1);
    probabilityValue.textContent = `${formattedPct}%`;
    probabilityMeter.style.width = `${pct}%`;
    thresholdLabel.textContent = `Threshold ${(threshold * 100).toFixed(0)}%`;

    if (probability >= threshold) {
      resultTitle.textContent = "Higher stroke risk indicated.";
      resultSummary.textContent = `The model estimates a ${formattedPct}% probability of stroke risk. This result is not a diagnosis.`;
      resultSymbol.textContent = "!";
      resultSymbol.style.color = "var(--accent-dark)";
    } else {
      resultTitle.textContent = "Lower stroke risk indicated.";
      resultSummary.textContent = `The model estimates a ${formattedPct}% probability of stroke risk. This does not rule out medical risk.`;
      resultSymbol.textContent = "✓";
      resultSymbol.style.color = "var(--success)";
    }
    document.querySelector("#result-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    errorBox.textContent = error instanceof TypeError
      ? "Can't reach the prediction API. Start the backend and check its CORS settings."
      : (error.message || "Something went wrong. Please try again.");
    errorBox.hidden = false;
    await checkApi();
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = "Analyze risk <span>↗</span>";
  }
});

checkApi();
ageInput.addEventListener("input", updateAgeOutput);
updateAgeOutput();
