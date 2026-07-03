"""An editable, bidirectional equation ↔ graph, as a self-contained HTML file.

No server and no Python at view time: the page uses Plotly + math.js from CDNs.
Edit the equation → the curve replots live. Drag a parameter slider → the curve
*and* the displayed equation update. That's the "change the equation, see the
graph, and vice versa" loop, in one shareable ``.html``.
"""

from __future__ import annotations

import re

_MATH_NAMES = {"x", "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "pi", "e", "pow"}

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OptimumAI — editable equation ↔ graph</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/12.4.1/math.min.js"></script>
<style>
 body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#111}
 .row{margin:10px 0}
 input[type=text]{width:360px;font-family:monospace;font-size:15px;padding:6px}
 .sl{display:flex;align-items:center;gap:10px;margin:4px 0}
 .sl label{width:20px;font-family:monospace;font-weight:700}
 #eqline{font-family:monospace;font-size:15px;color:#2563eb;margin-top:8px}
 .err{color:#c0392b}
</style></head>
<body>
<h2 class="sr-only">Interactive editor: edit an equation to update its plot, or drag parameter
sliders to update both the curve and the equation.</h2>
<h3>OptimumAI — editable equation ↔ graph</h3>
<div class="row">f(x) = <input id="eq" type="text" value="__EQUATION__"></div>
<div id="sliders" class="row"></div>
<div id="eqline"></div>
<div id="plot" style="width:100%;max-width:760px;height:420px"></div>
<script>
const PARAMS = __PARAMS_JSON__;      // {name: value}
const XMIN = __XMIN__, XMAX = __XMAX__;
const box = document.getElementById('sliders');
Object.keys(PARAMS).forEach(name => {
  const wrap = document.createElement('div'); wrap.className = 'sl';
  const lab = document.createElement('label'); lab.textContent = name;
  const s = document.createElement('input');
  s.type='range'; s.min=-5; s.max=5; s.step=0.1; s.value=PARAMS[name]; s.id='p_'+name;
  const val = document.createElement('span'); val.id='v_'+name; val.textContent=(+PARAMS[name]).toFixed(1);
  s.addEventListener('input', () => { PARAMS[name]=+s.value; val.textContent=(+s.value).toFixed(1); render(); });
  wrap.appendChild(lab); wrap.appendChild(s); wrap.appendChild(val); box.appendChild(wrap);
});
document.getElementById('eq').addEventListener('input', render);
function render(){
  const expr = document.getElementById('eq').value;
  const eqline = document.getElementById('eqline');
  let compiled;
  try { compiled = math.compile(expr); } catch(e){ eqline.innerHTML='<span class="err">parse error</span>'; return; }
  const xs=[], ys=[]; const N=200;
  for(let k=0;k<=N;k++){ const x=XMIN+(XMAX-XMIN)*k/N; let y;
    try{ y=compiled.evaluate(Object.assign({x}, PARAMS)); }catch(e){ y=NaN; }
    xs.push(x); ys.push(y); }
  const shown = Object.entries(PARAMS).map(([k,v])=>k+'='+(+v).toFixed(2)).join(', ');
  eqline.textContent = 'f(x) = ' + expr + (shown ? '   with  '+shown : '');
  Plotly.react('plot', [{x:xs,y:ys,mode:'lines',line:{color:'#2563eb',width:2}}],
    {margin:{t:10,r:10,b:40,l:40}, xaxis:{title:'x'}, yaxis:{title:'f(x)'}}, {displayModeBar:false});
}
render();
</script>
</body></html>
"""


def _detect_params(expression: str) -> list[str]:
    names = re.findall(r"[A-Za-z_]\w*", expression)
    seen: list[str] = []
    for n in names:
        if n not in _MATH_NAMES and n not in seen:
            seen.append(n)
    return seen


def editable_plot(
    expression: str = "a*x^2 + b*x + c",
    params: dict[str, float] | None = None,
    xrange: tuple[float, float] = (-5.0, 5.0),
    out: str | None = None,
) -> str:
    """Build the interactive editor HTML.

    If ``out`` is given, write the file and return its path; otherwise return the
    HTML string. Parameters (single-letter names other than ``x``) get sliders
    unless you pass an explicit ``params`` mapping.
    """
    if params is None:
        params = {name: 1.0 for name in _detect_params(expression)}
    import json

    html = (
        _TEMPLATE.replace("__EQUATION__", expression)
        .replace("__PARAMS_JSON__", json.dumps(params))
        .replace("__XMIN__", repr(float(xrange[0])))
        .replace("__XMAX__", repr(float(xrange[1])))
    )
    if out is None:
        return html
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


def demo(out: str | None = None) -> str:
    """The default editable quadratic ``a·x² + b·x + c``."""
    return editable_plot(out=out or "editable_plot.html")
