/* ===== Shared core: language, formatting, Keel connection, tooltip, charts ===== */
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
/* Light by default, whatever the OS setting; the toggle switches and is remembered. */
let THEME='light';try{const t=localStorage.getItem('keel-theme');if(t==='dark'||t==='light')THEME=t}catch(e){}
document.documentElement.dataset.theme=THEME;
function bindTheme(){const b=document.getElementById('themeBtn');if(!b)return;const lab=()=>{b.textContent=THEME==='light'?(LANG==='ja'?'ダーク':'Dark'):(LANG==='ja'?'ライト':'Light');b.setAttribute('aria-label',THEME==='light'?'Switch to dark':'Switch to light')};lab();
  b.onclick=()=>{THEME=THEME==='light'?'dark':'light';document.documentElement.dataset.theme=THEME;try{localStorage.setItem('keel-theme',THEME)}catch(e){}lab();window.redraw&&window.redraw()};bindTheme.lab=lab}
let LANG='en';
try{const l=localStorage.getItem('keel-lang');if(l==='ja'||l==='en')LANG=l}catch(e){}
const JA=()=>LANG==='ja', T=(en,ja)=>JA()?ja:en;
function setLang(l){LANG=l;try{localStorage.setItem('keel-lang',l)}catch(e){}document.documentElement.lang=l;
  $$('[data-ja]').forEach(e=>{if(e.dataset.en==null)e.dataset.en=e.innerHTML;e.innerHTML=JA()?e.dataset.ja:e.dataset.en});
  $$('#langSeg button').forEach(b=>b.classList.toggle('on',b.dataset.l===l));if(bindTheme.lab)bindTheme.lab();}
const num=(v,d=2)=>v==null||!isFinite(v)?'—':Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=(v,d=1,sign=true)=>v==null||!isFinite(v)?'—':(sign&&v>0?'+':v<0?'−':'')+Math.abs(v).toFixed(d)+'%';
const bp=v=>{if(v==null||!isFinite(v))return'—';const r=Math.round(v*100);return(r>0?'+':r<0?'−':'')+Math.abs(r)+'bp'};
const cls=v=>v==null?'':v>0?'up':v<0?'dn':'';
const px=v=>{if(v==null||!isFinite(v))return'—';const a=Math.abs(v);return num(v,a>=1000?0:a>=100?1:a>=10?2:3)};
const big=v=>{if(v==null||!isFinite(v))return'—';const a=Math.abs(v);return a>=1e12?(v/1e12).toFixed(1)+'T':a>=1e9?(v/1e9).toFixed(0)+'B':a>=1e6?(v/1e6).toFixed(0)+'M':num(v,0)};
const fdate=(d,o={day:'numeric',month:'short',year:'numeric'})=>{if(!d)return'—';const x=new Date(String(d).length<=10?d+'T00:00:00':d);return isNaN(x)?String(d):x.toLocaleDateString(JA()?'ja-JP':'en-GB',o)};

/* ----- Keel: market data through the viewer's "Keel" connector ----- */
const KEEL='keel'; /* exact display name of the custom connector (a different 'Keel' exists in the connector directory) */
let MCP=null, MCPREADY=null;
function mcpReady(){
  if(!MCPREADY)MCPREADY=(window.claude&&window.claude.use?window.claude.use('mcp'):Promise.resolve(null)).then(m=>(MCP=m)).catch(()=>null);
  return MCPREADY;
}
const unwrap=p=>{if(p&&typeof p==='object'&&!Array.isArray(p)&&Object.keys(p).length===1&&p.result&&typeof p.result==='object')return p.result;return p};
/* Returns {ok:true,data} or {ok:false,code,message}. Retries once on retryable errors (Render waking up). */
async function keel(tool,input,{fresh=false}={}){
  const m=await mcpReady();
  if(!m)return{ok:false,code:'no_mcp'};
  const opts={cache:fresh?{refresh:true,staleTime:300000,gcTime:3600000}:{staleTime:300000,gcTime:3600000}};
  for(let attempt=0;attempt<2;attempt++){
    try{const r=await m.callTool(KEEL,tool,input,opts);
      let d=unwrap(r.payload);
      if(typeof d==='string'){try{d=JSON.parse(d)}catch(e){return{ok:false,code:'tool_error',message:d.slice(0,200)}}}
      return{ok:true,data:d,storedAt:r.cache?.storedAt};
    }catch(e){
      if(e&&e.retryable&&attempt===0){await new Promise(ok=>setTimeout(ok,Math.min(e.retryAfterMs||4000+Math.random()*3000,20000)));continue}
      return{ok:false,code:e?.code||'upstream_error',message:e?.message||''};
    }
  }
  return{ok:false,code:'upstream_error'};
}
/* Status pill + explanation for each failure code */
function keelMsg(code,message){
  switch(code){
    case 'no_mcp':case 'not_granted':case 'capability_disabled':case 'capability_removed':
      return T('Live data runs inside claude.ai. This view shows sample data.','ライブデータはclaude.ai内で動作します。この表示はサンプルデータです。');
    case 'server_not_connected':case 'selection_required':case 'server_not_found':
      return T('Add your keel connector in claude.ai Settings → Connectors, then reload.','claude.aiの設定 → コネクタでKeelを追加し、再読み込みしてください。');
    case 'needs_reauth':return T('Reconnect Keel in claude.ai Settings → Connectors.','claude.aiの設定 → コネクタでKeelを再接続してください。');
    case 'not_in_manifest':return T('Keel is not allowed for this page. Allow it from the page’s Permissions menu.','このページではKeelが許可されていません。ページの権限メニューから許可してください。');
    case 'blocked_by_policy':case 'approval_required':return T('This page reached a connector that cannot serve it. Make sure your custom connector is named exactly “keel” (lowercase), not the directory’s “Keel”.','このページは別のコネクタに接続されています。カスタムコネクタ名が小文字の「keel」であることを確認してください（ディレクトリの「Keel」とは別物です）。');
    case 'server_unavailable':return T('Keel did not answer. The free server may be waking up; try Refresh in a minute.','Keelが応答しません。無料サーバーの起動中の可能性があります。1分後に更新してください。');
    case 'tool_error':return T('Keel reported an error: ','Keelのエラー：')+(message||'');
    default:return T('Live data unavailable','ライブデータを取得できません')+(message?': '+message:'.');
  }
}
let STAT=null;
function restatus(){if(STAT)setStatus(...STAT)}
function setStatus(state,code,message){
  STAT=[state,code,message];const p=$('#status');if(!p)return;
  p.className='pill '+state;
  p.innerHTML='<i></i>'+(state==='live'?T('Live · Keel','ライブ · Keel'):state==='busy'?T('Loading…','読み込み中…'):state==='sample'?T('Sample data','サンプルデータ'):T('Sample data','サンプルデータ'));
  const b=$('#banner');if(!b)return;
  if(state==='live'||state==='busy'){b.hidden=true;return}
  b.hidden=false;
  b.innerHTML=`<span>${state==='sample'&&code==='no_mcp'?'':'<b>'+T('Showing sample figures.','サンプル数値を表示中。')+'</b> '}${esc(keelMsg(code,message))}</span><button class="lnk" id="howBtn">${T('How to connect','接続方法')}</button>`;
  $('#howBtn').onclick=()=>{const h=$('#how');h.hidden=!h.hidden};
}
function howHtml(){return `<b>${T('Connect live data (one time)','ライブデータの接続（初回のみ）')}</b><ol>
<li>${T('Deploy the Keel repo on Render (Blueprint). It creates <code>keel-api</code> and <code>keel-mcp</code>.','KeelリポジトリをRenderにデプロイ（Blueprint）。<code>keel-api</code>と<code>keel-mcp</code>が作成されます。')}</li>
<li>${T('In claude.ai, Settings → Connectors → Add custom connector. Name it exactly <code>keel</code> (lowercase; a different “Keel” connector exists); URL: your <code>keel-mcp</code> service address on Render + <code>/mcp</code>.','claude.aiの設定 → コネクタ → カスタムコネクタを追加。名前は必ず小文字の<code>keel</code>（別の「Keel」コネクタがあるため）、URLはRenderの<code>keel-mcp</code>のアドレス＋<code>/mcp</code>。')}</li>
<li>${T('Reload this page and allow Keel when asked.','このページを再読み込みし、確認が出たらKeelを許可します。')}</li></ol>
<p class="note" style="margin-top:8px">${T('Only ticker symbols and parameters are sent. Public market data from OpenBB providers; free sources can be delayed.','送信されるのはティッカーとパラメータのみ。データはOpenBB経由の公開市場データで、無料ソースのため遅延する場合があります。')}</p>`}

/* ----- tooltip ----- */
let TIP=null;
function tip(html,ev){if(!TIP){TIP=document.createElement('div');TIP.id='tip';document.body.appendChild(TIP)}
  if(html==null){TIP.hidden=true;return}TIP.hidden=false;TIP.innerHTML=html;
  const r=TIP.getBoundingClientRect(),W=innerWidth,H=innerHeight;let x=ev.clientX+14,y=ev.clientY+14;
  if(x+r.width>W-8)x=ev.clientX-r.width-14;if(y+r.height>H-8)y=ev.clientY-r.height-14;TIP.style.left=Math.max(8,x)+'px';TIP.style.top=Math.max(8,y)+'px'}

/* ----- scales ----- */
function niceTicks(lo,hi,n=5){if(!(hi>lo)){hi=lo+1}const raw=(hi-lo)/n,p=Math.pow(10,Math.floor(Math.log10(raw))),f=raw/p,st=(f<=1?1:f<=2?2:f<=2.5?2.5:f<=5?5:10)*p;
  const a=Math.floor(lo/st)*st,b=Math.ceil(hi/st)*st,t=[];for(let v=a;v<=b+st*1e-9;v+=st)t.push(+v.toFixed(10));return t}
const W=el=>Math.max(260,el.clientWidth||el.parentNode.clientWidth||600);

/* Line chart. series:[{name,color,dash,pts:[{x:Number|Date,y}],width}]
   opts:{h,xType:'date'|'num',yFmt,xFmt,yLabel,zero,tipX,area} */
function lineChart(svg,series,o={}){
  const w=W(svg),h=o.h||260,m={l:o.ml||44,r:o.mr||14,t:12,b:26};
  const all=series.flatMap(s=>s.pts).filter(p=>p.y!=null&&isFinite(p.y));
  if(!all.length){svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.innerHTML=`<text x="${w/2}" y="${h/2}" text-anchor="middle">${T('No data','データなし')}</text>`;return}
  const xs=all.map(p=>+p.x),x0=Math.min(...xs),x1=Math.max(...xs);
  let y0=Math.min(...all.map(p=>p.y)),y1=Math.max(...all.map(p=>p.y));if(o.zero){y0=Math.min(0,y0);y1=Math.max(0,y1)}
  const pad=(y1-y0)*.06||1;const yt=niceTicks(y0-pad,y1+pad,o.yt||5);y0=yt[0];y1=yt[yt.length-1];
  const X=v=>m.l+(x1>x0?(v-x0)/(x1-x0):.5)*(w-m.l-m.r),Y=v=>m.t+(1-(v-y0)/(y1-y0))*(h-m.t-m.b);
  const yf=o.yFmt||(v=>num(v,Math.abs(y1-y0)<2?2:Math.abs(y1-y0)<20?1:0));
  let g='';
  yt.forEach(v=>g+=`<line class="grd" x1="${m.l}" x2="${w-m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l-6}" y="${Y(v)+4}" text-anchor="end">${yf(v)}</text>`);
  if(o.zero&&y0<0&&y1>0)g+=`<line class="zero" x1="${m.l}" x2="${w-m.r}" y1="${Y(0)}" y2="${Y(0)}"/>`;
  /* x ticks */
  if(o.xType==='date'){
    const span=(x1-x0)/864e5,step=span>1500?12:span>700?6:span>300?3:span>90?1:0;
    const d0=new Date(x0),ticks=[];
    if(step){let d=new Date(d0.getFullYear(),d0.getMonth()+1,1);while(+d<=x1){if(d.getMonth()%step===0)ticks.push(+d);d=new Date(d.getFullYear(),d.getMonth()+1,1)}}
    else{let d=new Date(x0);while(+d<=x1){ticks.push(+d);d=new Date(+d+7*864e5)}}
    const maxT=Math.max(2,Math.floor((w-m.l-m.r)/70)),every=Math.ceil(ticks.length/maxT);
    ticks.forEach((t,i)=>{if(i%every)return;const d=new Date(t);const lab=step>=12||(step&&d.getMonth()===0)?String(d.getFullYear()):d.toLocaleDateString(JA()?'ja-JP':'en-GB',step?{month:'short'}:{day:'numeric',month:'short'});
      g+=`<line class="grd" x1="${X(t)}" x2="${X(t)}" y1="${h-m.b}" y2="${h-m.b+4}"/><text x="${X(t)}" y="${h-8}" text-anchor="middle">${lab}</text>`});
  }else{
    const xt=o.xTicks||niceTicks(x0,x1,6);const xf=o.xFmt||(v=>v);
    let lastX=-1e9;xt.filter(v=>v>=x0&&v<=x1).forEach(v=>{if(X(v)-lastX<26)return;lastX=X(v);g+=`<text x="${X(v)}" y="${h-8}" text-anchor="middle">${xf(v)}</text>`});
  }
  g+=`<line class="ax" x1="${m.l}" x2="${w-m.r}" y1="${h-m.b}" y2="${h-m.b}"/>`;
  series.forEach(s=>{const pts=s.pts.filter(p=>p.y!=null&&isFinite(p.y));if(!pts.length)return;
    const d=pts.map((p,i)=>(i?'L':'M')+X(+p.x).toFixed(1)+' '+Y(p.y).toFixed(1)).join('');
    if(s.area){const yb=Y(y0<=0&&y1>=0?0:y0);g+=`<path d="${d}L${X(+pts[pts.length-1].x).toFixed(1)} ${yb}L${X(+pts[0].x).toFixed(1)} ${yb}Z" fill="${s.color}" fill-opacity=".08" stroke="none"/>`}
    g+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width||1.6}" ${s.dash?'stroke-dasharray="5 4"':''} stroke-linejoin="round" stroke-linecap="round"/>`;
    if(s.dots)pts.forEach(p=>g+=`<circle cx="${X(+p.x)}" cy="${Y(p.y)}" r="3" fill="${s.color}"/>`);
    const L=pts[pts.length-1];if(!s.dots)g+=`<circle cx="${X(+L.x)}" cy="${Y(L.y)}" r="2.6" fill="${s.color}"/>`;
  });
  g+=`<line id="${svg.id}-hv" class="ax" x1="0" x2="0" y1="${m.t}" y2="${h-m.b}" visibility="hidden"/><rect x="${m.l}" y="${m.t}" width="${w-m.l-m.r}" height="${h-m.t-m.b}" fill="transparent" class="hit"/>`;
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('height',h);svg.innerHTML=g;
  const hit=svg.querySelector('.hit'),hv=svg.querySelector(`[id="${svg.id}-hv"]`);
  const xsAll=[...new Set(series.flatMap(s=>s.pts.map(p=>+p.x)))].sort((a,b)=>a-b);
  hit.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect(),vx=(ev.clientX-r.left)/r.width*w;
    const xv=x0+(vx-m.l)/(w-m.l-m.r)*(x1-x0);let best=xsAll[0];for(const v of xsAll)if(Math.abs(v-xv)<Math.abs(best-xv))best=v;
    hv.setAttribute('x1',X(best));hv.setAttribute('x2',X(best));hv.setAttribute('visibility','visible');
    const head=o.tipX?o.tipX(best):o.xType==='date'?fdate(new Date(best).toISOString().slice(0,10)):best;
    const rows=series.map(s=>{const p=s.pts.find(q=>+q.x===best);return p&&p.y!=null?`<div><span style="color:${s.color}">■</span> ${esc(s.name)} <b>${(o.tipFmt||yf)(p.y)}</b></div>`:''}).join('');
    tip(`<div class="m">${head}</div>${rows}`,ev)});
  hit.addEventListener('mouseleave',()=>{hv.setAttribute('visibility','hidden');tip(null)});
}
function legend(el,series){el.innerHTML=series.map(s=>`<span><i class="${s.dash?'d':''}" style="background:${s.color};border-color:${s.color}"></i>${esc(s.name)}</span>`).join('')}
function spark(vals,w=120,h=28,color='var(--ink)'){const v=vals.filter(x=>x!=null);if(v.length<2)return'';const lo=Math.min(...v),hi=Math.max(...v);
  const d=v.map((y,i)=>(i?'L':'M')+(i/(v.length-1)*w).toFixed(1)+' '+(h-2-(hi>lo?(y-lo)/(hi-lo):.5)*(h-4)).toFixed(1)).join('');
  const lx=w,ly=h-2-(hi>lo?(v[v.length-1]-lo)/(hi-lo):.5)*(h-4);
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="display:block;overflow:visible"><path d="${d}" fill="none" stroke="${color}" stroke-width="1.3"/><circle cx="${lx}" cy="${ly}" r="2.2" fill="${color}"/></svg>`}
const cssv=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
let RZ;addEventListener('resize',()=>{clearTimeout(RZ);RZ=setTimeout(()=>window.redraw&&window.redraw(),150)});

document.addEventListener('DOMContentLoaded',bindTheme);if(document.readyState!=='loading')setTimeout(bindTheme,0);
