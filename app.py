"""
Human Activity Recognition — Streamlit App
Live prediction from mobile phone sensors.
LinearSVC model runs 100% in-browser via embedded JSON weights.
No pkl files needed — works on Streamlit Cloud.
"""

import streamlit as st
import json, os

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="centered",
)

# ── Load browser model JSON (committed to repo, not gitignored) ───────────────
@st.cache_data
def load_browser_model():
    path = "models/browser_model.json"
    if not os.path.exists(path):
        return "", 0.0
    with open(path) as f:
        d = json.load(f)
    return json.dumps(d, separators=(",", ":")), float(d.get("accuracy", 86.29))

browser_model_json, browser_acc = load_browser_model()

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
    st.metric("Live Sensor Model",  f"Linear SVM — {browser_acc}%")
    st.metric("Training Samples",   "7,352")
    st.metric("Features",           "45 (raw inertial signals)")
    st.metric("Window",             "2.56 s · 50 Hz · 128 steps")
    st.markdown("---")
    st.markdown("### 🎯 Activities")
    ACTS = [
        ("🚶 WALKING",              "#4f46e5"),
        ("⬆️ WALKING UPSTAIRS",     "#7c3aed"),
        ("⬇️ WALKING DOWNSTAIRS",   "#db2777"),
        ("🪑 SITTING",              "#ea580c"),
        ("🧍 STANDING",             "#059669"),
        ("🛌 LAYING",               "#0891b2"),
    ]
    for name, color in ACTS:
        st.markdown(
            f"<span style='color:{color};font-weight:600'>{name}</span>",
            unsafe_allow_html=True,
        )

# ── Instructions ──────────────────────────────────────────────────────────────
st.markdown(f"""
### 📱 Real-Time Activity Detection on Your Phone

The ML model runs **entirely inside your phone's browser** —
no server connection needed. Works on Streamlit Cloud, any network,
any device with motion sensors.

**How to use:**
1. Open this app on your **phone's browser** (not laptop)
2. Tap **▶ Start Recording** — iOS will ask for motion permission, tap *Allow*
3. Hold phone at your **waist**, move naturally for ~3 seconds
4. Prediction appears instantly — **{browser_acc}% accuracy**
""")

# ── Sensor + inference component (model weights injected as JS constant) ──────
if not browser_model_json:
    st.error("Model file not found. Run `python train_realtime_model.py` first.")
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
       border-radius:18px;padding:22px;color:white;text-align:center;}}
h2{{font-size:1.25rem;margin-bottom:3px;}}
.sub{{color:#a5b4fc;font-size:0.82rem;margin-bottom:18px;}}
#startBtn{{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;
  border:none;border-radius:99px;padding:15px 32px;font-size:1rem;
  font-weight:700;cursor:pointer;width:100%;margin-bottom:10px;transition:opacity 0.2s;}}
#startBtn:disabled{{opacity:0.45;cursor:not-allowed;}}
#status{{font-size:0.84rem;color:#c7d2fe;min-height:20px;margin-bottom:10px;}}
#timer{{font-size:2.6rem;font-weight:800;color:#fbbf24;min-height:50px;margin-bottom:6px;}}
.prog-wrap{{background:#1e1b4b;border-radius:99px;height:8px;overflow:hidden;margin-bottom:18px;}}
#progressBar{{height:100%;width:0%;background:linear-gradient(90deg,#6366f1,#8b5cf6);
  border-radius:99px;transition:width 0.12s;}}
#result{{background:rgba(0,0,0,0.25);border-radius:14px;padding:16px;
  margin-top:6px;display:none;}}
#actName{{font-size:1.5rem;font-weight:800;margin-bottom:12px;}}
.bar-row{{display:flex;align-items:center;gap:6px;margin-bottom:4px;}}
.bar-label{{font-size:0.73rem;color:#a5b4fc;width:148px;text-align:left;}}
.bar-track{{flex:1;background:#1e1b4b;border-radius:99px;height:9px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:99px;transition:width 0.4s;}}
.bar-pct{{font-size:0.73rem;color:#c7d2fe;width:38px;text-align:right;}}
#errMsg{{color:#f87171;font-size:0.82rem;margin-top:10px;line-height:1.6;}}
</style>
</head>
<body>
<div class="card">
  <h2>🎯 Live Activity Detector</h2>
  <p class="sub">Model runs in your browser · accelerometer + gyroscope · ~3 s window</p>
  <button id="startBtn" onclick="startRecording()">▶ Start Recording</button>
  <div id="status">Ready — tap Start Recording</div>
  <div id="timer"></div>
  <div class="prog-wrap"><div id="progressBar"></div></div>
  <div id="result"><div id="actName"></div><div id="bars"></div></div>
  <div id="errMsg"></div>
</div>
<script>
const ACTIVITIES={{1:"🚶 WALKING",2:"⬆️ WALKING UPSTAIRS",3:"⬇️ WALKING DOWNSTAIRS",
  4:"🪑 SITTING",5:"🧍 STANDING",6:"🛌 LAYING"}};
const COLORS={{1:"#4f46e5",2:"#7c3aed",3:"#db2777",4:"#ea580c",5:"#059669",6:"#0891b2"}};
const TARGET=128;
const MODEL={browser_model_json};
let readings=[];
function setStatus(m){{document.getElementById("status").textContent=m;}}
function setError(m){{document.getElementById("errMsg").innerHTML=m;}}

function extractFeatures(cd){{
  const N=9,T=128,f=[];
  for(const s of['mean','std','min','max','rms']){{
    for(let c=0;c<N;c++){{
      const r=cd.slice(c*T,c*T+T);let v;
      if(s==='mean'){{v=r.reduce((a,b)=>a+b,0)/T;}}
      else if(s==='std'){{const m=r.reduce((a,b)=>a+b,0)/T;
        v=Math.sqrt(r.reduce((a,b)=>a+(b-m)**2,0)/T);}}
      else if(s==='min'){{v=Math.min(...r);}}
      else if(s==='max'){{v=Math.max(...r);}}
      else{{v=Math.sqrt(r.reduce((a,b)=>a+b*b,0)/T);}}
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

async function startRecording(){{
  document.getElementById("startBtn").disabled=true;
  document.getElementById("result").style.display="none";
  setError("");readings=[];
  if(typeof DeviceMotionEvent!=="undefined"&&
     typeof DeviceMotionEvent.requestPermission==="function"){{
    try{{
      const perm=await DeviceMotionEvent.requestPermission();
      if(perm!=="granted"){{
        setError("❌ Permission denied. Tap Start again and allow motion access.");
        document.getElementById("startBtn").disabled=false;return;}}
    }}catch(e){{setError("❌ "+e.message);
      document.getElementById("startBtn").disabled=false;return;}}
  }}else if(typeof DeviceMotionEvent==="undefined"){{
    setError("❌ <b>Motion sensors not available.</b><br>"+
      "Open this page on a real phone — Android Chrome or iPhone Safari.<br>"+
      "Laptops and desktops do not have motion sensors.");
    document.getElementById("startBtn").disabled=false;return;}}
  setStatus("📡 Recording…");
  const prog=document.getElementById("progressBar");
  const timer=document.getElementById("timer");
  const ml=(e)=>{{
    const ag=e.accelerationIncludingGravity||{{}};
    const g=e.acceleration||{{}};const gy=e.rotationRate||{{}};
    readings.push([g.x||0,g.y||0,g.z||0,
      gy.alpha||0,gy.beta||0,gy.gamma||0,ag.x||0,ag.y||0,ag.z||0]);
  }};
  window.addEventListener("devicemotion",ml);
  let elapsed=0;
  const tick=setInterval(()=>{{
    elapsed+=80;
    prog.style.width=Math.min(100,(readings.length/TARGET)*100)+"%";
    const s=Math.max(0,(TARGET*20-elapsed)/1000).toFixed(1);
    timer.textContent=s>0?s+"s":"";
    if(readings.length>=TARGET){{
      clearInterval(tick);window.removeEventListener("devicemotion",ml);runPred();}}
  }},80);
  setTimeout(()=>{{
    clearInterval(tick);window.removeEventListener("devicemotion",ml);
    if(readings.length<16){{
      setError("❌ Too few readings. Check browser permissions and retry.");
      document.getElementById("startBtn").disabled=false;timer.textContent="";
    }}else if(readings.length<TARGET){{runPred();}}
  }},7000);
}}

function runPred(){{
  setStatus("🤖 Predicting…");
  document.getElementById("timer").textContent="";
  document.getElementById("progressBar").style.width="100%";
  let r=readings.slice(0,TARGET);
  while(r.length<TARGET)r.push(r[r.length-1]);
  const cd=new Float32Array(9*TARGET);
  for(let t=0;t<TARGET;t++)for(let c=0;c<9;c++)cd[c*TARGET+t]=r[t][c];
  try{{
    const{{predClass,proba}}=predict(cd);
    showResult(predClass,proba);
    setStatus("✅ Done — tap Start to record again");
  }}catch(e){{setError("❌ Prediction error: "+e.message);setStatus("Error");}}
  document.getElementById("startBtn").disabled=false;
  document.getElementById("progressBar").style.width="0%";
}}

function showResult(predClass,proba){{
  const col=COLORS[predClass]||"#6366f1";
  const el=document.getElementById("actName");
  el.textContent=ACTIVITIES[predClass]+"  ("+(proba[predClass-1]*100).toFixed(1)+"%)";
  el.style.color=col;
  const barsEl=document.getElementById("bars");barsEl.innerHTML="";
  for(let i=1;i<=6;i++){{
    const p=(proba[i-1]*100).toFixed(1);
    const c=(i===predClass)?COLORS[i]:"#4338ca";
    barsEl.innerHTML+=`<div class="bar-row">
      <span class="bar-label">${{ACTIVITIES[i]}}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${{p}}%;background:${{c}}"></div></div>
      <span class="bar-pct">${{p}}%</span></div>`;
  }}
  document.getElementById("result").style.display="block";
}}
</script>
</body>
</html>"""

st.components.v1.html(component_html, height=660, scrolling=False)

st.markdown("---")
st.markdown("""
**Tips for best results**
- Hold phone at **waist height** while recording (as the dataset was collected)
- **iPhone:** Use **Safari** — Chrome on iOS blocks DeviceMotion
- **Android:** Chrome works automatically, no extra steps
- LAYING (flat on back) → ~100% accuracy
- SITTING vs STANDING is hardest — hold still for the full 3 seconds

> LinearSVC · 45 features (mean, std, min, max, rms per channel) · 86.3% accuracy
""")

st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:0.8rem;margin-top:1.5rem;'>"
    "UCI HAR Dataset · Davide Anguita et al. (2012) · Rajneesh Babu · IISc Bengaluru</p>",
    unsafe_allow_html=True,
)
