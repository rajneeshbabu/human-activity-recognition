"""
Human Activity Recognition — Streamlit App
Continuous real-time prediction from mobile phone sensors using a sliding window.
LinearSVC model runs 100% in-browser. No pkl files needed. Works on Streamlit Cloud.
"""

import streamlit as st
import json, os

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="centered",
)

@st.cache_data
def load_browser_model():
    path = "models/browser_model.json"
    if not os.path.exists(path):
        return "", 0.0
    with open(path) as f:
        d = json.load(f)
    return json.dumps(d, separators=(",", ":")), float(d.get("accuracy", 86.29))

browser_model_json, browser_acc = load_browser_model()

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

with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.metric("Model",    "Linear SVM (in-browser)")
    st.metric("Accuracy", f"{browser_acc}%")
    st.metric("Update rate", "Every ~0.64 s")
    st.metric("Window",   "2.56 s · 50 Hz · 128 steps")
    st.markdown("---")
    st.markdown("### 🎯 Activities")
    ACTS = [
        ("🚶 WALKING",            "#4f46e5"),
        ("⬆️ WALKING UPSTAIRS",   "#7c3aed"),
        ("⬇️ WALKING DOWNSTAIRS", "#db2777"),
        ("🪑 SITTING",            "#ea580c"),
        ("🧍 STANDING",           "#059669"),
        ("🛌 LAYING",             "#0891b2"),
    ]
    for name, color in ACTS:
        st.markdown(
            f"<span style='color:{color};font-weight:600'>{name}</span>",
            unsafe_allow_html=True,
        )

st.markdown(f"""
### 📱 Continuous Real-Time Activity Detection

Open on your **phone** → tap **▶ Start** → move naturally.
The model predicts your activity **continuously**, updating every ~0.6 seconds.
Tap **⏹ Stop** to pause.

> **iPhone:** use Safari · **Android:** use Chrome · {browser_acc}% accuracy
""")

if not browser_model_json:
    st.error("Model file not found. Run `python train_realtime_model.py` and push `models/browser_model.json`.")
    st.stop()

component_html = f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{{box-sizing:border-box;margin:0;padding:0;
   font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}}
body{{background:transparent;padding:6px;}}
.card{{background:linear-gradient(135deg,#1e1b4b,#312e81);
       border-radius:18px;padding:20px;color:white;text-align:center;}}
h2{{font-size:1.2rem;margin-bottom:2px;}}
.sub{{color:#a5b4fc;font-size:0.8rem;margin-bottom:16px;}}

/* buttons */
.btn-row{{display:flex;gap:10px;margin-bottom:12px;}}
#startBtn{{flex:1;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;
  border:none;border-radius:99px;padding:14px;font-size:1rem;font-weight:700;
  cursor:pointer;transition:opacity 0.2s;}}
#stopBtn{{flex:1;background:linear-gradient(135deg,#dc2626,#ef4444);color:white;
  border:none;border-radius:99px;padding:14px;font-size:1rem;font-weight:700;
  cursor:pointer;display:none;transition:opacity 0.2s;}}
#startBtn:disabled{{opacity:0.4;cursor:not-allowed;}}

/* status bar */
#status{{font-size:0.82rem;color:#c7d2fe;min-height:18px;margin-bottom:8px;}}

/* pulse ring around big activity card */
@keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(99,102,241,0.5);}}
                  70%{{box-shadow:0 0 0 12px rgba(99,102,241,0);}}
                  100%{{box-shadow:0 0 0 0 rgba(99,102,241,0);}}}}

/* big activity display */
#actCard{{background:rgba(0,0,0,0.3);border-radius:16px;padding:20px 16px;
  margin:8px 0 14px;display:none;animation:pulse 1.5s infinite;
  border:2px solid rgba(255,255,255,0.08);}}
#actEmoji{{font-size:3rem;line-height:1;margin-bottom:6px;}}
#actName{{font-size:1.4rem;font-weight:800;margin-bottom:4px;}}
#actConf{{font-size:0.9rem;color:#c7d2fe;}}

/* live indicator dot */
#liveDot{{display:inline-block;width:8px;height:8px;background:#22c55e;
  border-radius:50%;margin-right:6px;
  animation:blink 1s step-start infinite;}}
@keyframes blink{{50%{{opacity:0;}}}}

/* confidence bars */
#bars{{text-align:left;}}
.bar-row{{display:flex;align-items:center;gap:6px;margin-bottom:5px;}}
.bar-label{{font-size:0.72rem;color:#a5b4fc;width:144px;flex-shrink:0;}}
.bar-track{{flex:1;background:#1e1b4b;border-radius:99px;height:10px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:99px;transition:width 0.5s ease;}}
.bar-pct{{font-size:0.72rem;color:#c7d2fe;width:36px;text-align:right;flex-shrink:0;}}

/* update counter */
#counter{{font-size:0.72rem;color:#6366f1;margin-top:10px;}}
#errMsg{{color:#f87171;font-size:0.82rem;margin-top:10px;line-height:1.6;}}
</style>
</head>
<body>
<div class="card">
  <h2>🎯 Real-Time Activity Detector</h2>
  <p class="sub">Sliding window · updates every ~0.6 s · runs fully in browser</p>

  <div class="btn-row">
    <button id="startBtn" onclick="startContinuous()">▶ Start</button>
    <button id="stopBtn"  onclick="stopContinuous()">⏹ Stop</button>
  </div>
  <div id="status">Tap Start to begin continuous detection</div>

  <div id="actCard">
    <div id="actEmoji">❓</div>
    <div id="actName">—</div>
    <div id="actConf"></div>
  </div>

  <div id="bars"></div>
  <div id="counter"></div>
  <div id="errMsg"></div>
</div>

<script>
const ACTIVITIES={{1:"🚶 WALKING",2:"⬆️ WALKING UPSTAIRS",3:"⬇️ WALKING DOWNSTAIRS",
  4:"🪑 SITTING",5:"🧍 STANDING",6:"🛌 LAYING"}};
const EMOJIS={{1:"🚶",2:"⬆️",3:"⬇️",4:"🪑",5:"🧍",6:"🛌"}};
const COLORS={{1:"#4f46e5",2:"#7c3aed",3:"#db2777",4:"#ea580c",5:"#059669",6:"#0891b2"}};

const MODEL = {browser_model_json};
const WINDOW=128, STEP=32;   // predict every 32 new samples (~0.64 s at 50 Hz)

let buffer=[];         // rolling buffer, keep last WINDOW readings
let motionListener=null;
let predCount=0;
let running=false;

function setStatus(m){{document.getElementById("status").innerHTML=m;}}
function setError(m){{document.getElementById("errMsg").innerHTML=m;}}

// ── Feature extraction ───────────────────────────────────────────────────
function extractFeatures(cd){{
  const N=9,T=WINDOW,f=[];
  for(const s of['mean','std','min','max','rms']){{
    for(let c=0;c<N;c++){{
      const r=cd.slice(c*T,c*T+T); let v;
      if(s==='mean')      {{v=r.reduce((a,b)=>a+b,0)/T;}}
      else if(s==='std')  {{const m=r.reduce((a,b)=>a+b,0)/T;
                            v=Math.sqrt(r.reduce((a,b)=>a+(b-m)**2,0)/T);}}
      else if(s==='min')  {{v=Math.min(...r);}}
      else if(s==='max')  {{v=Math.max(...r);}}
      else                {{v=Math.sqrt(r.reduce((a,b)=>a+b*b,0)/T);}}
      f.push(v);
    }}
  }}
  return f;
}}
function standardise(f){{return f.map((v,i)=>(v-MODEL.scaler_mean[i])/MODEL.scaler_std[i]);}}
function decisionFn(f,coef,bias){{return coef.map((row,k)=>{{
  let s=bias[k];for(let j=0;j<f.length;j++)s+=row[j]*f[j];return s;}});}}
function sigmoidCal(dec,cals){{
  const p=cals.map((c,k)=>1/(1+Math.exp(c.a*dec[k]+c.b)));
  const t=p.reduce((a,b)=>a+b,0);return p.map(v=>v/t);}}
function predict(cd){{
  const f=extractFeatures(cd),sc=standardise(f);
  const avg=new Array(MODEL.n_classes).fill(0);
  for(let fold=0;fold<MODEL.n_folds;fold++){{
    const dec=decisionFn(sc,MODEL.coef[fold],MODEL.intercept[fold]);
    const p=sigmoidCal(dec,MODEL.calibrators[fold]);
    p.forEach((v,k)=>avg[k]+=v);
  }}
  avg.forEach((_,k)=>avg[k]/=MODEL.n_folds);
  const idx=avg.indexOf(Math.max(...avg));
  return{{predClass:MODEL.classes[idx],proba:avg}};
}}

// ── Predict from current buffer ──────────────────────────────────────────
function runPrediction(){{
  if(buffer.length < WINDOW) return;
  const recent = buffer.slice(-WINDOW);          // last 128 readings
  const cd = new Float32Array(9*WINDOW);
  for(let t=0;t<WINDOW;t++)
    for(let c=0;c<9;c++)
      cd[c*WINDOW+t]=recent[t][c];

  try{{
    const{{predClass,proba}}=predict(cd);
    updateDisplay(predClass,proba);
    predCount++;
    document.getElementById("counter").textContent=
      "Predictions made: "+predCount;
  }}catch(e){{console.error(e);}}
}}

// ── Update the UI ────────────────────────────────────────────────────────
function updateDisplay(predClass,proba){{
  const color=COLORS[predClass]||"#6366f1";
  const card=document.getElementById("actCard");
  card.style.display="block";
  card.style.borderColor=color+"66";
  card.style.animationName="pulse";  // re-trigger pulse
  card.style.animationName="none";
  requestAnimationFrame(()=>{{card.style.animationName="pulse";}});

  document.getElementById("actEmoji").textContent=EMOJIS[predClass]||"❓";
  document.getElementById("actName").textContent=ACTIVITIES[predClass]||"Unknown";
  document.getElementById("actName").style.color=color;
  const conf=(proba[predClass-1]*100).toFixed(1);
  document.getElementById("actConf").textContent="Confidence: "+conf+"%";

  const barsEl=document.getElementById("bars");
  barsEl.innerHTML="";
  for(let i=1;i<=6;i++){{
    const p=(proba[i-1]*100).toFixed(1);
    const c=(i===predClass)?COLORS[i]:"#4338ca";
    const bold=(i===predClass)?"700":"400";
    const tc=(i===predClass)?"#e0e7ff":"#a5b4fc";
    barsEl.innerHTML+=`<div class="bar-row">
      <span class="bar-label" style="color:${{tc}};font-weight:${{bold}}">${{ACTIVITIES[i]}}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${{p}}%;background:${{c}}"></div>
      </div>
      <span class="bar-pct" style="color:${{tc}};font-weight:${{bold}}">${{p}}%</span>
    </div>`;
  }}
}}

// ── Start continuous detection ───────────────────────────────────────────
async function startContinuous(){{
  document.getElementById("startBtn").disabled=true;
  setError("");

  // iOS permission
  if(typeof DeviceMotionEvent!=="undefined"&&
     typeof DeviceMotionEvent.requestPermission==="function"){{
    try{{
      const perm=await DeviceMotionEvent.requestPermission();
      if(perm!=="granted"){{
        setError("❌ Permission denied. Tap Start again and allow motion access.");
        document.getElementById("startBtn").disabled=false; return;
      }}
    }}catch(e){{
      setError("❌ "+e.message);
      document.getElementById("startBtn").disabled=false; return;
    }}
  }}else if(typeof DeviceMotionEvent==="undefined"){{
    setError("❌ <b>Motion sensors not available.</b><br>"+
      "Open this on a <b>real phone</b> — iPhone Safari or Android Chrome.<br>"+
      "Laptops do not have motion sensors.");
    document.getElementById("startBtn").disabled=false; return;
  }}

  buffer=[]; running=true; predCount=0;
  document.getElementById("stopBtn").style.display="block";
  document.getElementById("startBtn").style.display="none";

  let sinceLastPred=0;
  motionListener=(e)=>{{
    if(!running) return;
    const ag=e.accelerationIncludingGravity||{{}};
    const g=e.acceleration||{{}};
    const gy=e.rotationRate||{{}};
    buffer.push([
      g.x||0, g.y||0, g.z||0,
      gy.alpha||0, gy.beta||0, gy.gamma||0,
      ag.x||0, ag.y||0, ag.z||0
    ]);
    // Keep buffer from growing forever
    if(buffer.length > WINDOW*4) buffer=buffer.slice(-WINDOW*2);
    sinceLastPred++;

    if(buffer.length >= WINDOW && sinceLastPred >= STEP){{
      sinceLastPred=0;
      runPrediction();
    }}
  }};
  window.addEventListener("devicemotion", motionListener);

  setStatus('<span id="liveDot"></span>Live — detecting continuously…');
}}

// ── Stop ─────────────────────────────────────────────────────────────────
function stopContinuous(){{
  running=false;
  if(motionListener) window.removeEventListener("devicemotion", motionListener);
  motionListener=null;
  document.getElementById("stopBtn").style.display="none";
  document.getElementById("startBtn").style.display="block";
  document.getElementById("startBtn").disabled=false;
  setStatus("Stopped — tap Start to resume");
}}
</script>
</body>
</html>"""

st.components.v1.html(component_html, height=680, scrolling=False)

st.markdown("---")
st.markdown("""
**Tips**
- **iPhone:** Safari only (Chrome on iOS blocks motion sensors)
- **Android:** Chrome works, no extra steps
- Hold phone at **waist height** for best accuracy
- Walk/sit/stand for at least 2–3 seconds for the first prediction to appear
- LAYING (flat) → ~100% confidence · SITTING vs STANDING is hardest pair
""")

st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:0.8rem;margin-top:1rem;'>"
    "UCI HAR Dataset · Davide Anguita et al. (2012) · Rajneesh Babu · IISc Bengaluru</p>",
    unsafe_allow_html=True,
)
