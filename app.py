"""
Human Activity Recognition — Streamlit App
Tab 1: Predict from UCI test samples (561-feature RBF SVM, 95.5%)
Tab 2: Live prediction from mobile phone sensors — model runs 100% in-browser,
       no background server needed, works on Streamlit Cloud.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIVITIES = {
    1: "🚶 WALKING",
    2: "⬆️ WALKING UPSTAIRS",
    3: "⬇️ WALKING DOWNSTAIRS",
    4: "🪑 SITTING",
    5: "🧍 STANDING",
    6: "🛌 LAYING",
}
ACTIVITY_COLORS = {
    1: "#4f46e5", 2: "#7c3aed", 3: "#db2777",
    4: "#ea580c", 5: "#059669", 6: "#0891b2",
}
BASE = "UCI HAR Dataset/"

# ── Load models (Tab 1) ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    svm    = joblib.load("models/svm_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return svm, scaler

@st.cache_data
def load_test_data():
    X_test  = pd.read_csv(BASE + "test/X_test.txt",  sep=r"\s+", header=None)
    y_test  = pd.read_csv(BASE + "test/y_test.txt",  header=None).squeeze()
    subj_te = pd.read_csv(BASE + "test/subject_test.txt", header=None).squeeze()
    raw_feats = pd.read_csv(BASE + "features.txt", sep=r"\s+", header=None)[1].tolist()
    seen, feats = {}, []
    for f in raw_feats:
        if f in seen: seen[f] += 1; feats.append(f"{f}_{seen[f]}")
        else: seen[f] = 0; feats.append(f)
    X_test.columns = feats
    return X_test, y_test, subj_te, feats

# ── Load browser model JSON (Tab 2) ──────────────────────────────────────────
@st.cache_data
def load_browser_model() -> tuple[str, float]:
    """Return (minified JSON string, accuracy)."""
    path = "models/browser_model.json"
    if not os.path.exists(path):
        return "", 0.0
    with open(path) as f:
        d = json.load(f)
    return json.dumps(d, separators=(',', ':')), d.get("accuracy", 0.0)

try:
    svm, scaler = load_models()
    X_test, y_test, subj_test, feat_names = load_test_data()
    models_ok = True
except Exception as e:
    models_ok = False
    err_msg   = str(e)

browser_model_json, browser_acc = load_browser_model()
browser_ok = bool(browser_model_json)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;background:linear-gradient(135deg,#4f46e5,#7c3aed);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.4rem;'>
🏃 Human Activity Recognition
</h1>
<p style='text-align:center;color:#64748b;margin-top:-0.5rem;'>
UCI HAR Dataset · Samsung Galaxy S II · 30 Subjects · 6 Activities
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.metric("Test Samples Model", "SVM (RBF) — 95.5%")
    if browser_ok:
        st.metric("Live Sensor Model", f"Linear SVM — {browser_acc}%")
    st.metric("Training Samples", "7,352")
    st.markdown("---")
    st.markdown("### 🎯 Activities")
    for k, v in ACTIVITIES.items():
        st.markdown(
            f"<span style='color:{ACTIVITY_COLORS[k]};font-weight:600'>{v}</span>",
            unsafe_allow_html=True,
        )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Test Samples", "📱 Live Sensor (Mobile)"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — UCI Test Samples
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    if not models_ok:
        st.error(f"⚠️ Could not load models. Run `har_notebook.ipynb` first.\n\n`{err_msg}`")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Select a Sample")
        mode = st.radio("Prediction mode",
                        ["Pick a test sample", "Random sample"], key="t1_mode")
        if mode == "Pick a test sample":
            sample_idx = st.slider("Test sample index", 0, len(X_test) - 1, 0)
        else:
            sample_idx = np.random.randint(0, len(X_test))
            st.info(f"Random sample index: **{sample_idx}**")

        true_label = int(y_test.iloc[sample_idx])
        subject_id = int(subj_test.iloc[sample_idx])
        x_sample   = X_test.iloc[[sample_idx]]
        x_scaled   = scaler.transform(x_sample)

        st.markdown(f"**Subject:** #{subject_id}")
        st.markdown(
            f"**True activity:** <span style='color:{ACTIVITY_COLORS[true_label]};"
            f"font-weight:700'>{ACTIVITIES[true_label]}</span>",
            unsafe_allow_html=True,
        )
        predict_btn = st.button("🚀 Predict Activity",
                                use_container_width=True, key="t1_predict")

    with col2:
        st.subheader("🎯 Prediction")
        if predict_btn:
            pred_label = int(svm.predict(x_scaled)[0])
            proba      = svm.predict_proba(x_scaled)[0]
            correct    = pred_label == true_label
            icon       = "✅" if correct else "❌"
            color      = ACTIVITY_COLORS[pred_label]

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{color}22,{color}11);
            border:2px solid {color}44;border-radius:14px;padding:1.2rem;text-align:center;'>
              <div style='font-size:2.5rem;'>{icon}</div>
              <div style='color:{color};font-size:1.3rem;font-weight:800;margin-top:0.3rem;'>
                {ACTIVITIES[pred_label]}
              </div>
              <div style='color:#64748b;font-size:0.85rem;margin-top:0.3rem;'>
                {"Correct prediction!" if correct else f"True: {ACTIVITIES[true_label]}"}
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### Confidence per activity")
            for k, act in ACTIVITIES.items():
                p         = proba[k - 1]
                is_pred   = (k == pred_label)
                bar_color = ACTIVITY_COLORS[k] if is_pred else "#e2e8f0"
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
                  <span style='width:180px;font-size:0.78rem;
                  color:{"#1e293b" if is_pred else "#94a3b8"};
                  font-weight:{"700" if is_pred else "400"}'>{act}</span>
                  <div style='flex:1;background:#f1f5f9;border-radius:99px;
                  height:10px;overflow:hidden;'>
                    <div style='width:{p*100:.1f}%;background:{bar_color};height:100%;
                    border-radius:99px;transition:width 0.5s;'></div>
                  </div>
                  <span style='width:42px;text-align:right;font-size:0.78rem;
                  color:{"#1e293b" if is_pred else "#94a3b8"};
                  font-weight:{"700" if is_pred else "400"}'>{p*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("👈 Select a sample and click **Predict Activity**")

    st.markdown("---")
    with st.expander("📋 Dataset Overview"):
        ca, cb, cc = st.columns(3)
        ca.metric("Total samples", "10,299")
        cb.metric("Train / Test",  "7,352 / 2,947")
        cc.metric("Features",      "561")
        cd, ce, cf = st.columns(3)
        cd.metric("Subjects",      "30 volunteers")
        ce.metric("Sampling rate", "50 Hz")
        cf.metric("Window size",   "2.56 s (128 readings)")

    with st.expander("📈 Model Performance"):
        results_df = pd.DataFrame({
            "Model":    ["Logistic Regression", "SVM (RBF)", "LightGBM",
                         "XGBoost", "Random Forest"],
            "Accuracy": ["95.52%", "95.49%", "93.45%", "93.42%", "92.57%"],
            "Notes":    ["Linear baseline, surprisingly strong",
                         "RBF kernel, C=10 — near-best accuracy",
                         "200 trees, LR=0.1",
                         "200 trees, depth=6",
                         "200 trees — best feature importance"],
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Mobile Sensor (100% in-browser inference, no server needed)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    if not browser_ok:
        st.warning(
            "⚠️ Browser model not found. Run `python train_realtime_model.py` first, "
            "then check that `models/browser_model.json` exists."
        )
    else:
        st.markdown(f"""
        ### 📱 Real-Time Activity Detection on Your Phone
        The ML model runs **entirely inside your phone's browser** — no server connection
        needed. Works on Streamlit Cloud, local network, anywhere.

        **How to use:**
        1. Open this app on your phone's browser
        2. Switch to the **Live Sensor** tab
        3. Tap **▶ Start Recording** — iOS will ask for motion permission, tap *Allow*
        4. Hold phone at your waist, move naturally for ~3 seconds
        5. Prediction appears instantly — **{browser_acc}% accuracy**
        """)

        st.markdown("---")

        # ── Embed the model JSON and full inference logic in JS ───────────────
        component_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0;
     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
body {{ background: transparent; padding: 6px; }}
.card {{
  background: linear-gradient(135deg, #1e1b4b, #312e81);
  border-radius: 18px; padding: 22px; color: white; text-align: center;
}}
h2  {{ font-size: 1.25rem; margin-bottom: 3px; }}
.sub {{ color: #a5b4fc; font-size: 0.82rem; margin-bottom: 18px; }}
#startBtn {{
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white; border: none; border-radius: 99px;
  padding: 15px 32px; font-size: 1rem; font-weight: 700;
  cursor: pointer; width: 100%; margin-bottom: 10px;
  transition: opacity 0.2s; letter-spacing: 0.3px;
}}
#startBtn:disabled {{ opacity: 0.45; cursor: not-allowed; }}
#status {{ font-size: 0.84rem; color: #c7d2fe; min-height: 20px; margin-bottom: 10px; }}
#timer  {{ font-size: 2.6rem; font-weight: 800; color: #fbbf24;
           min-height: 50px; margin-bottom: 6px; }}
.prog-wrap {{ background: #1e1b4b; border-radius: 99px; height: 8px;
              overflow: hidden; margin-bottom: 18px; }}
#progressBar {{ height: 100%; width: 0%;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  border-radius: 99px; transition: width 0.12s; }}
#result {{
  background: rgba(0,0,0,0.25); border-radius: 14px; padding: 16px;
  margin-top: 6px; display: none;
}}
#actName {{ font-size: 1.5rem; font-weight: 800; margin-bottom: 12px; }}
.bar-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }}
.bar-label {{ font-size: 0.73rem; color: #a5b4fc; width: 148px; text-align: left; }}
.bar-track {{ flex: 1; background: #1e1b4b; border-radius: 99px;
              height: 9px; overflow: hidden; }}
.bar-fill  {{ height: 100%; border-radius: 99px; transition: width 0.4s; }}
.bar-pct   {{ font-size: 0.73rem; color: #c7d2fe; width: 38px; text-align: right; }}
#errMsg    {{ color: #f87171; font-size: 0.82rem; margin-top: 10px; line-height: 1.5; }}
</style>
</head>
<body>
<div class="card">
  <h2>🎯 Live Activity Detector</h2>
  <p class="sub">Model runs in your browser · needs accelerometer + gyroscope · ~3 s window</p>
  <button id="startBtn" onclick="startRecording()">▶ Start Recording</button>
  <div id="status">Ready — tap Start Recording</div>
  <div id="timer"></div>
  <div class="prog-wrap"><div id="progressBar"></div></div>
  <div id="result">
    <div id="actName"></div>
    <div id="bars"></div>
  </div>
  <div id="errMsg"></div>
</div>

<script>
// ── Embedded model (LinearSVC + sigmoid calibration, trained on UCI raw signals) ──
const MODEL = {browser_model_json};

// ── Activity labels & colours ──────────────────────────────────────────────
const ACTIVITIES = {{
  1:"🚶 WALKING", 2:"⬆️ WALKING UPSTAIRS", 3:"⬇️ WALKING DOWNSTAIRS",
  4:"🪑 SITTING", 5:"🧍 STANDING", 6:"🛌 LAYING"
}};
const COLORS = {{
  1:"#4f46e5", 2:"#7c3aed", 3:"#db2777",
  4:"#ea580c", 5:"#059669", 6:"#0891b2"
}};
const TARGET = 128;

let readings = [];
let motionListener = null;

// ── Helpers ───────────────────────────────────────────────────────────────
function setStatus(msg) {{ document.getElementById("status").textContent = msg; }}
function setError(msg)  {{ document.getElementById("errMsg").innerHTML  = msg; }}

// ── Feature extraction: (9 channels × 128 steps) → 45 features ──────────
// channels order: body_acc_xyz, body_gyro_xyz, total_acc_xyz  (matches training)
function extractFeatures(channelData) {{
  // channelData: Float32Array of length 9*128, row-major: channel[i] = channelData[i*128 ... i*128+127]
  const N_CH = 9, N_T = 128;
  const feats = [];
  for (const stat of ['mean','std','min','max','rms']) {{
    for (let ch = 0; ch < N_CH; ch++) {{
      const row = channelData.slice(ch * N_T, ch * N_T + N_T);
      let val;
      if (stat === 'mean') {{
        val = row.reduce((a,b)=>a+b, 0) / N_T;
      }} else if (stat === 'std') {{
        const m = row.reduce((a,b)=>a+b, 0) / N_T;
        val = Math.sqrt(row.reduce((a,b)=>a+(b-m)**2, 0) / N_T);
      }} else if (stat === 'min') {{
        val = Math.min(...row);
      }} else if (stat === 'max') {{
        val = Math.max(...row);
      }} else {{  // rms
        val = Math.sqrt(row.reduce((a,b)=>a+b*b, 0) / N_T);
      }}
      feats.push(val);
    }}
  }}
  return feats;  // length 45
}}

// ── Standardise using scaler params ──────────────────────────────────────
function standardise(feats) {{
  return feats.map((v, i) => (v - MODEL.scaler_mean[i]) / MODEL.scaler_std[i]);
}}

// ── Linear decision function: coef (n_classes×45) · features + intercept ─
function decisionFunction(features, coef, intercept) {{
  return coef.map((row, k) => {{
    let s = intercept[k];
    for (let j = 0; j < features.length; j++) s += row[j] * features[j];
    return s;
  }});
}}

// ── Sigmoid calibration: P = 1 / (1 + exp(A*f + B)) ─────────────────────
function sigmoidCalibrate(decision, calibrators) {{
  // calibrators: array of {{a, b}} per class (OvR)
  const proba = calibrators.map((cal, k) => 1 / (1 + Math.exp(cal.a * decision[k] + cal.b)));
  // Normalise to sum=1
  const total = proba.reduce((a,b)=>a+b, 0);
  return proba.map(p => p / total);
}}

// ── Average over CV folds ─────────────────────────────────────────────────
function predict(channelData) {{
  const feats  = extractFeatures(channelData);
  const scaled = standardise(feats);

  // Average probabilities across folds
  const avgProba = new Array(MODEL.n_classes).fill(0);
  for (let fold = 0; fold < MODEL.n_folds; fold++) {{
    const dec   = decisionFunction(scaled, MODEL.coef[fold], MODEL.intercept[fold]);
    const proba = sigmoidCalibrate(dec, MODEL.calibrators[fold]);
    proba.forEach((p, k) => avgProba[k] += p);
  }}
  avgProba.forEach((_, k) => avgProba[k] /= MODEL.n_folds);

  const predIdx  = avgProba.indexOf(Math.max(...avgProba));
  const predClass = MODEL.classes[predIdx];  // 1-indexed
  return {{ predClass, proba: avgProba }};
}}

// ── Recording flow ────────────────────────────────────────────────────────
async function startRecording() {{
  document.getElementById("startBtn").disabled = true;
  document.getElementById("result").style.display = "none";
  setError("");
  readings = [];

  // iOS 13+ permission gate
  if (typeof DeviceMotionEvent !== "undefined" &&
      typeof DeviceMotionEvent.requestPermission === "function") {{
    try {{
      const perm = await DeviceMotionEvent.requestPermission();
      if (perm !== "granted") {{
        setError("❌ Motion permission denied. Tap Start Recording again and allow access.");
        document.getElementById("startBtn").disabled = false;
        return;
      }}
    }} catch(e) {{
      setError("❌ " + e.message);
      document.getElementById("startBtn").disabled = false;
      return;
    }}
  }} else if (typeof DeviceMotionEvent === "undefined") {{
    setError(
      "❌ <b>DeviceMotion not available</b> on this browser/device.<br>" +
      "Open this page on a <b>real phone</b> (Android Chrome or iPhone Safari).<br>" +
      "Laptops and desktop browsers do not have motion sensors."
    );
    document.getElementById("startBtn").disabled = false;
    return;
  }}

  setStatus("📡 Recording sensor data…");
  const prog  = document.getElementById("progressBar");
  const timer = document.getElementById("timer");

  motionListener = (e) => {{
    const ag = e.accelerationIncludingGravity || {{}};
    const g  = e.acceleration || {{}};
    const gy = e.rotationRate || {{}};
    // row: [body_acc_x, y, z,  body_gyro_x, y, z,  total_acc_x, y, z]
    readings.push([
      g.x  || 0, g.y  || 0, g.z  || 0,
      gy.alpha || 0, gy.beta || 0, gy.gamma || 0,
      ag.x || 0, ag.y || 0, ag.z || 0
    ]);
  }};
  window.addEventListener("devicemotion", motionListener);

  let elapsed = 0;
  const totalMs = TARGET * 20;   // 128 × 20 ms = 2.56 s
  const tick = setInterval(() => {{
    elapsed += 80;
    const pct = Math.min(100, (readings.length / TARGET) * 100);
    prog.style.width = pct + "%";
    const secsLeft = Math.max(0, (totalMs - elapsed) / 1000).toFixed(1);
    timer.textContent = secsLeft > 0 ? secsLeft + "s" : "";
    if (readings.length >= TARGET) {{
      clearInterval(tick);
      window.removeEventListener("devicemotion", motionListener);
      runPrediction();
    }}
  }}, 80);

  // Fallback timeout after 7 s
  setTimeout(() => {{
    clearInterval(tick);
    window.removeEventListener("devicemotion", motionListener);
    if (readings.length < 16) {{
      setError("❌ Almost no sensor data received.<br>Make sure you're on a phone and " +
               "motion access is allowed in your browser settings.");
      document.getElementById("startBtn").disabled = false;
      timer.textContent = "";
    }} else if (readings.length < TARGET) {{
      runPrediction();  // use whatever we have
    }}
  }}, 7000);
}}

function runPrediction() {{
  setStatus("🤖 Predicting…");
  document.getElementById("timer").textContent = "";
  document.getElementById("progressBar").style.width = "100%";

  // Pad / trim to exactly 128 rows
  let r = readings.slice(0, TARGET);
  while (r.length < TARGET) r.push(r[r.length - 1]);

  // Reshape: (128 rows × 9 cols) → row-major (9 channels × 128 timesteps)
  // channelData[ch * 128 + t] = r[t][ch]
  const N_CH = 9;
  const channelData = new Float32Array(N_CH * TARGET);
  for (let t = 0; t < TARGET; t++)
    for (let ch = 0; ch < N_CH; ch++)
      channelData[ch * TARGET + t] = r[t][ch];

  try {{
    const {{ predClass, proba }} = predict(channelData);
    showResult(predClass, proba);
    setStatus("✅ Done — tap Start Recording to try again");
  }} catch(e) {{
    setError("❌ Prediction error: " + e.message);
    setStatus("Error");
  }}
  document.getElementById("startBtn").disabled = false;
  document.getElementById("progressBar").style.width = "0%";
}}

function showResult(predClass, proba) {{
  const color = COLORS[predClass] || "#6366f1";
  const name  = ACTIVITIES[predClass] || "Unknown";
  const conf  = (proba[predClass - 1] * 100).toFixed(1);

  const actEl = document.getElementById("actName");
  actEl.textContent = name + "  (" + conf + "%)";
  actEl.style.color = color;

  const barsEl = document.getElementById("bars");
  barsEl.innerHTML = "";
  for (let i = 1; i <= 6; i++) {{
    const p = (proba[i - 1] * 100).toFixed(1);
    const c = (i === predClass) ? COLORS[i] : "#4338ca";
    barsEl.innerHTML += `
      <div class="bar-row">
        <span class="bar-label">${{ACTIVITIES[i]}}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width:${{p}}%;background:${{c}}"></div>
        </div>
        <span class="bar-pct">${{p}}%</span>
      </div>`;
  }}
  document.getElementById("result").style.display = "block";
}}
</script>
</body>
</html>
"""

        st.components.v1.html(component_html, height=660, scrolling=False)

        st.markdown("---")
        st.markdown("""
        **Tips for best results**

        - Hold phone at **waist height** (as the UCI dataset was recorded)
        - Walk/sit/stand for the full ~3 seconds before the timer ends
        - **iOS (iPhone/iPad):** Use **Safari** — Chrome on iOS blocks DeviceMotion
        - **Android:** Chrome works perfectly, no permission needed
        - LAYING (flat on back) is detected with ~100% accuracy
        - SITTING vs STANDING is the hardest pair — both static

        > The browser model uses 45 simplified features vs 561 for the test-sample model,
        > giving 86% vs 95.5% accuracy — the trade-off for zero-latency in-browser inference.
        """)

st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:0.8rem;margin-top:1.5rem;'>"
    "UCI HAR Dataset · Davide Anguita et al. (2012) · Rajneesh Babu · IISc Bengaluru</p>",
    unsafe_allow_html=True,
)
