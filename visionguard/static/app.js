const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const API=location.protocol.startsWith('http')?'':null;let file=null,csv=null,live=false;
const go=$('#go'),drop=$('#drop');
const setMode=ok=>{live=ok;$('#mode').className='pill'+(ok?' live':'');$('#mode span').textContent=ok?'Live model connected':'Demo mode'};
fetch('/api/health').then(r=>r.json()).then(j=>setMode(!!j.model)).catch(()=>setMode(false));

function pick(f){if(!f||!f.type.startsWith('image/'))return;file=f;$('#prev').src=URL.createObjectURL(f);$('#prev').hidden=false;$('#hint').hidden=true;go.disabled=false;$('#clear').hidden=false}
$('#file').onchange=e=>pick(e.target.files[0]);$('#csv').onchange=e=>csv=e.target.files[0]||null;
drop.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#file').click()}};
['dragenter','dragover'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',e=>pick(e.dataTransfer.files[0]));
$('#clear').onclick=()=>{file=null;$('#file').value='';$('#prev').hidden=true;$('#hint').hidden=false;go.disabled=true;$('#clear').hidden=true;$('#report').hidden=true};
$('#tog').onchange=e=>$('#boxes').classList.toggle('off',!e.target.checked);
$('#copy').onclick=async e=>{try{await navigator.clipboard.writeText($('#text').textContent);e.target.textContent='Copied'}catch{e.target.textContent='Copy failed'}setTimeout(()=>e.target.textContent='Copy text',1500)};

go.onclick=async()=>{
  go.disabled=true;go.textContent='Analyzing…';$('#scan').hidden=false;
  let r;
  try{
    if(!live)throw 0;
    const fd=new FormData();fd.append('image',file);if(csv)fd.append('csv',csv);
    const res=await fetch('/api/analyze',{method:'POST',body:fd});
    if(!res.ok)throw new Error((await res.json().catch(()=>({}))).error||'Server error');
    r=await res.json();r.demo=false;
  }catch(e){
    if(e instanceof Error)alert('Analysis failed: '+e.message);
    r=await demo();
  }
  await render(r);
  $('#scan').hidden=true;go.disabled=false;go.textContent='Analyze image';
  $('#report').scrollIntoView({behavior:'smooth'});
};

async function demo(){
  await new Promise(r=>setTimeout(r,1400));
  const img=await new Promise(r=>{const i=new Image();i.onload=()=>r(i);i.src=$('#prev').src});
  const w=img.naturalWidth,h=img.naturalHeight,p=0.12+((file.size*7919)%780)/1000;
  const B=(x,y,a,b)=>[[x*w,y*h],[(x+a)*w,y*h],[(x+a)*w,(y+b)*h],[x*w,(y+b)*h]];
  return{demo:true,p_real:p,width:w,height:h,ms:1400,
    ocr:[{text:'ExampleCare',conf:93.4,box:B(.12,.12,.4,.1)},{text:'Daily Moisturizing Lotion',conf:88.1,box:B(.12,.28,.7,.1)},{text:'200 ml',conf:96.7,box:B(.12,.72,.25,.09)}],
    matches:[{product_name:'Daily Moisturizing Lotion',brand:'ExampleCare',net_quantity:'200 ml',overall:98,name:100,brand_s:100,qty:92},{product_name:'Herbal Shampoo',brand:'LeafAndRoot',net_quantity:'250 ml',overall:41,name:35,brand_s:40,qty:62},{product_name:'Vitamin C Face Serum',brand:'GlowSample',net_quantity:'30 ml',overall:33,name:30,brand_s:28,qty:44}]};
}

async function render(r){
  $('#report').hidden=false;$('#banner').hidden=!r.demo;
  $('#orig').src=$('#prev').src;
  const real=r.p_real>=.5,col=real?'var(--real)':'var(--ai)',conf=Math.max(r.p_real,1-r.p_real)*100;
  $('#vlabel').textContent=real?'Looks like a real photo':'Looks AI-generated';$('#vlabel').style.color=col;
  $('#meta').textContent=`${r.width}×${r.height} · ${(file.size/1024).toFixed(0)} KB · ${(r.ms/1000).toFixed(1)} s`;
  const ring=$('#ring');ring.style.stroke=col;ring.style.strokeDashoffset=364.4;
  ['fReal','fAi'].forEach(i=>$('#'+i).style.width=0);
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    ring.style.strokeDashoffset=364.4*(1-conf/100);
    $('#fReal').style.width=r.p_real*100+'%';$('#fReal').style.background='var(--real)';
    $('#fAi').style.width=(1-r.p_real)*100+'%';$('#fAi').style.background='var(--ai)';}));
  $('#pct').textContent=conf.toFixed(0)+'%';
  $('#bReal').textContent=(r.p_real*100).toFixed(1)+'%';$('#bAi').textContent=((1-r.p_real)*100).toFixed(1)+'%';
  const svg=$('#boxes');svg.setAttribute('viewBox',`0 0 ${r.width} ${r.height}`);
  const sw=Math.max(2,r.width/260);
  svg.innerHTML=r.ocr.map(o=>`<polygon points="${o.box.map(p=>p.join(',')).join(' ')}" fill="rgba(120,220,190,.12)" stroke="#78dcbe" stroke-width="${sw}"/>`).join('');
  $('#text').textContent=r.ocr.map(o=>o.text).join('\n')||'No readable text detected.';
  $('#conf').innerHTML=r.ocr.slice(0,8).map(o=>`<div class="bar"><span>${esc(o.text.slice(0,36))}</span><b>${o.conf.toFixed(0)}%</b><i><u style="width:${o.conf}%;background:var(--ink)"></u></i></div>`).join('');
  $('#matches').innerHTML=r.matches.length?r.matches.slice(0,3).map(m=>{
    const c=m.overall>=75?'var(--real)':m.overall>=50?'var(--warn)':'var(--ai)',t=m.overall>=75?'Strong':m.overall>=50?'Partial':'Weak';
    return`<div class="m"><div class="t"><b>${esc(m.product_name)}</b><span class="tag" style="color:${c};background:color-mix(in srgb,${c} 14%,transparent)">${t} · ${m.overall.toFixed(0)}%</span></div><div class="sub">${esc(m.brand)} · ${esc(m.net_quantity)}</div><div class="chips"><span>Name ${m.name.toFixed(0)}%</span><span>Brand ${m.brand_s.toFixed(0)}%</span><span>Quantity ${m.qty.toFixed(0)}%</span></div></div>`}).join(''):'<p class="sub">No text to match.</p>';
}


// Load saved evaluation results. No model re-run is needed.
const pct = v => Number.isFinite(v) ? (v * 100).toFixed(1) + '%' : '—';
async function loadMetrics() {
  const status = $('#metricsStatus');
  try {
    const response = await fetch('/evaluation/metrics.json');
    if (!response.ok) throw new Error('Saved evaluation metrics could not be loaded.');
    const m = await response.json();
    $('#mAccuracy').textContent = pct(m.accuracy);
    $('#mPrecision').textContent = pct(m.classes.REAL.precision);
    $('#mRecall').textContent = pct(m.classes.REAL.recall);
    $('#mF1').textContent = pct(m.classes.REAL.f1);
    $('#mAuc').textContent = Number(m.roc_auc).toFixed(4);
    $('#mCount').textContent = Number(m.test_count).toLocaleString();
    status.textContent = 'Saved evaluation results loaded';
    status.classList.add('ready');
  } catch (e) {
    status.textContent = 'Saved evaluation data unavailable';
    status.title = e.message;
  }
}
loadMetrics();
