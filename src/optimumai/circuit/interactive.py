"""Interactive circuits — drag the inputs and watch the math update in real time.

Each builder emits a single self-contained ``.html`` (no server, no build step):
move a slider and the forward values *and* the gradients recompute live in the
browser, so you feel the cause↔effect of softmax or backprop directly.
"""

from __future__ import annotations

import json

_SOFTMAX_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OptimumAI — interactive softmax</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#111}
 .sl{display:flex;align-items:center;gap:10px;margin:6px 0}
 .sl label{width:70px;font-family:monospace}
 .bar{height:26px;background:#2563eb;border-radius:4px;transition:width .08s}
 .row{display:flex;align-items:center;gap:10px;margin:4px 0}
 .row .name{width:70px;font-family:monospace}
 .val{width:70px;font-family:monospace;color:#2563eb;font-weight:700}
 #eq{font-family:monospace;color:#555;margin-top:12px}
 h3{margin:0 0 6px}
</style></head><body>
<h2 class="sr-only">Interactive softmax: drag the logit sliders to see the output
probability distribution recompute live.</h2>
<h3>Interactive softmax — drag the logits</h3>
<div id="sliders"></div>
<h3 style="margin-top:16px">softmax output (probabilities)</h3>
<div id="bars"></div>
<div id="eq">softmax(zᵢ) = e^{zᵢ} / Σⱼ e^{zⱼ}</div>
<script>
let Z = __LOGITS__;
const sliders = document.getElementById('sliders');
const bars = document.getElementById('bars');
Z.forEach((z,i)=>{
  const d=document.createElement('div'); d.className='sl';
  d.innerHTML = `<label>z${i} = <span id="zv${i}">${z.toFixed(2)}</span></label>`;
  const s=document.createElement('input'); s.type='range'; s.min=-5; s.max=5; s.step=0.1; s.value=z;
  s.oninput=()=>{ Z[i]=+s.value; document.getElementById('zv'+i).textContent=(+s.value).toFixed(2); render(); };
  d.appendChild(s); sliders.appendChild(d);
  const b=document.createElement('div'); b.className='row';
  b.innerHTML=`<span class="name">p${i}</span><div class="bar" id="bar${i}" style="width:0"></div><span class="val" id="pv${i}"></span>`;
  bars.appendChild(b);
});
function render(){
  const m=Math.max(...Z); const ex=Z.map(z=>Math.exp(z-m)); const s=ex.reduce((a,b)=>a+b,0);
  const p=ex.map(e=>e/s);
  p.forEach((pi,i)=>{ document.getElementById('bar'+i).style.width=(pi*400)+'px';
    document.getElementById('pv'+i).textContent=pi.toFixed(3); });
}
render();
</script></body></html>
"""

_BACKPROP_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OptimumAI — interactive backprop</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#111}
 .sl{display:flex;align-items:center;gap:10px;margin:6px 0}
 .sl label{width:60px;font-family:monospace}
 table{border-collapse:collapse;margin-top:14px}
 td,th{border:1px solid #ddd;padding:6px 14px;font-family:monospace;text-align:right}
 .data{color:#2563eb;font-weight:700}.grad{color:#ea7317;font-weight:700}
 caption{font-family:monospace;color:#555;margin-bottom:8px}
</style></head><body>
<h2 class="sr-only">Interactive backprop: drag the leaf sliders to see forward
values (blue) and gradients (orange) recompute live for L = (a*b + c) * f.</h2>
<h3>Interactive backprop — drag a, b, c, f</h3>
<div id="sliders"></div>
<table><caption>L = (a·b + c) · f</caption>
<tr><th>node</th><th>data</th><th>grad ∂L/∂·</th></tr>
<tbody id="rows"></tbody></table>
<script>
let V = __VARS__;   // {a,b,c,f}
const names=['a','b','c','f'];
const sliders=document.getElementById('sliders');
names.forEach(n=>{
  const d=document.createElement('div'); d.className='sl';
  d.innerHTML=`<label>${n} = <span id="v_${n}">${V[n].toFixed(2)}</span></label>`;
  const s=document.createElement('input'); s.type='range'; s.min=-5; s.max=5; s.step=0.1; s.value=V[n];
  s.oninput=()=>{ V[n]=+s.value; document.getElementById('v_'+n).textContent=(+s.value).toFixed(2); render(); };
  d.appendChild(s); sliders.appendChild(d);
});
function render(){
  const {a,b,c,f}=V;
  const e=a*b, splusc=e+c, L=splusc*f;                 // forward
  const dL=1, df=ssc(), ds=f, de=ds, dc=ds, da=de*b, db=de*a;  // backward
  function ssc(){return splusc;}
  const rows=[['a',a,da],['b',b,db],['c',c,dc],['f',f,df],
              ['e=a·b',e,de],['a·b+c',splusc,ds],['L',L,dL]];
  document.getElementById('rows').innerHTML = rows.map(r=>
    `<tr><td>${r[0]}</td><td class="data">${r[1].toFixed(3)}</td><td class="grad">${r[2].toFixed(3)}</td></tr>`
  ).join('');
}
render();
</script></body></html>
"""


def interactive_softmax(logits=(2.0, 1.0, 0.1, -0.5), out: str | None = None) -> str:
    """Live softmax: drag the logits, watch the probability bars move."""
    html = _SOFTMAX_TEMPLATE.replace("__LOGITS__", json.dumps([float(z) for z in logits]))
    if out is None:
        return html
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


def interactive_backprop(a=2.0, b=-3.0, c=10.0, f=-2.0, out: str | None = None) -> str:
    """Live backprop through L = (a·b + c)·f: drag a/b/c/f, watch data & gradients."""
    html = _BACKPROP_TEMPLATE.replace(
        "__VARS__", json.dumps({"a": float(a), "b": float(b), "c": float(c), "f": float(f)})
    )
    if out is None:
        return html
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


def interactive(concept: str = "softmax", out: str | None = None) -> str:
    """Build an interactive circuit for ``concept`` ("softmax" or "backprop")."""
    builders = {"softmax": interactive_softmax, "backprop": interactive_backprop}
    if concept not in builders:
        raise ValueError(f"unknown interactive concept {concept!r}; choose from {list(builders)}")
    return builders[concept](out=out)


def demo(out: str | None = None) -> str:
    """The interactive softmax circuit."""
    return interactive_softmax(out=out or "interactive_softmax.html")
