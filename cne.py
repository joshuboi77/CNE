import sys
import re
from collections import Counter
from pathlib import Path
from math import sqrt
import numpy as np

from bokeh.io import show
from bokeh.models import ColumnDataSource, CustomJS, TapTool, HoverTool, FactorRange
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Div, TextInput, Button

# ----- Semantic collapse parameters -----
# Probability thresholds computed as count / total_tokens (global or per-context)
SATURATE_TAU = 0.01   # optional plateau threshold (not used for color)
COLLAPSE_TAU = 0.03   # at/above this share, node collapses (turns black)
COLLAPSE_SHRINK = 0.55  # collapsed node diameter = base_min_px * this factor
RED_GAMMA = 0.75      # bell-curve peak threshold for red (0..1); higher => fewer reds

# -----------------------
# 1) Load / tokenize text
# -----------------------
DEFAULT_TEXT = """
Shall I compare thee to a summer’s day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer’s lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm’d;
And every fair from fair sometime declines,
By chance or nature’s changing course untrimm’d;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow’st;
Nor shall Death brag thou wander’st in his shade,
When in eternal lines to time thou grow’st:
   So long as men can breathe or eyes can see,
   So long lives this, and this gives life to thee.
""".strip()

def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="ignore")

WORD_RE = re.compile(r"[A-Za-z’']+")

def tokenize(text: str):
    # Lowercase; keep simple English word-like tokens (keeps Shakespeare’s curly apostrophes ’)
    words = [w.lower() for w in WORD_RE.findall(text)]
    return words

# -----------------------
# 2) Compute frequencies
# -----------------------
def word_frequencies(words):
    return Counter(words)

# -----------------------
# 2b) Context windows (±window)
# -----------------------
def build_context_map(tokens: list[str], window: int = 10) -> dict[str, Counter]:
    """
    For each word, count neighbors that appear within ±window positions
    across the whole token sequence (excluding the center token itself).
    """
    ctx: dict[str, Counter] = {}
    n = len(tokens)
    for i, w in enumerate(tokens):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        # exclude the center index i
        for j in range(start, end):
            if j == i:
                continue
            v = tokens[j]
            d = ctx.setdefault(w, Counter())
            d[v] += 1
    return ctx

# -----------------------
# 3) Layout: grid packing
#    (Robust & fast; zoom to explore)
# -----------------------
def grid_layout(n, base_step=1.0):
    # Place items on a square grid; return x,y arrays
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    xs, ys = [], []
    for i in range(n):
        r = i // cols
        c = i % cols
        xs.append(c * base_step)
        ys.append(-r * base_step)
    # center them around (0,0)
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    return xs, ys, cols, rows

# -----------------------
# 4) Size mapping
# -----------------------
def size_from_count(counts, base_min_px: float, extra_px: float = 60.0):
    """
    Map counts to circle diameters (screen px). Ensures a minimum diameter that can fit
    the longest word with padding. Frequencies add up to `extra_px` on top of that.
    """
    arr = np.array(counts, dtype=float)
    if arr.size == 0:
        return arr
    if arr.max() == arr.min():
        return np.full_like(arr, base_min_px + extra_px * 0.5)
    s = (np.sqrt(arr) - np.sqrt(arr.min())) / (np.sqrt(arr.max()) - np.sqrt(arr.min()))
    return base_min_px + s * extra_px

# -----------------------
# 5) Build interactive plot
# -----------------------
def build_plot(word_counts: Counter, token_sequence: list[str], ctx_window: int = 10):
    # sort by frequency desc, then alpha
    items = sorted(word_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ctx_map = build_context_map(token_sequence, window=ctx_window)
    # make JSON-serializable for CustomJS args (Bokeh can only take models or JSON)
    ctx_map_js = {w: dict(cnt) for w, cnt in ctx_map.items()}
    # Global counts and totals for PPMI/cosine in JS
    global_counts_js = {w: int(c) for w, c in word_counts.items()}
    total_tokens = int(len(token_sequence))
    # total co-occurrence mass (sum of all context counts)
    cooc_total = int(sum(sum(d.values()) for d in ctx_map.values()))
    words = [w for w, _ in items]
    counts = [c for _, c in items]
    n = len(words)

    # baseline minimum diameter based on the longest word label + padding
    max_word_len = max((len(w) for w in words), default=1)
    char_px = 7.2   # approx px per character at 9pt in-browser rendering
    pad_px = 12.0   # buffer space between text and circle edge (each side)
    base_min_px = char_px * max_word_len + 2.0 * pad_px

    # layout
    # choose a grid step relative to the largest circle diameter to avoid heavy overlap at top-left
    sizes = size_from_count(counts, base_min_px=base_min_px, extra_px=70.0)

    # ---- Global bell-curve salience (log-space) to determine "red" once for the whole corpus
    log_counts = np.log(np.array(counts, dtype=float) + 1.0)
    mu = float(log_counts.mean()) if len(log_counts) else 0.0
    sigma = float(log_counts.std(ddof=0)) if len(log_counts) else 1.0
    if sigma == 0.0:
        sigma = 1.0
    # Gaussian peak score
    grav_scores = np.exp(-0.5 * ((log_counts - mu) / sigma) ** 2)
    is_red_flags = [bool(g >= RED_GAMMA) for g in grav_scores]
    # For JS: map of word -> 1 if red else 0
    red_map_js = {w: (1 if is_red_flags[idx] else 0) for idx, w in enumerate(words)}

    # Precompute color tiers
    idx_normals = [i for i, f in enumerate(is_red_flags) if not f]
    idx_giants = [i for i, f in enumerate(is_red_flags) if f]

    # Normals: green / blue / purple by count terciles (within non-red set)
    if idx_normals:
        normal_counts = np.array([counts[i] for i in idx_normals], dtype=float)
        t1_n = float(np.percentile(normal_counts, 33))
        t2_n = float(np.percentile(normal_counts, 66))
        normal_tier = {}
        for i in idx_normals:
            c = counts[i]
            if c <= t1_n:
                normal_tier[i] = 'green'   # low normal
            elif c <= t2_n:
                normal_tier[i] = 'blue'    # mid normal
            else:
                normal_tier[i] = 'purple'  # high normal (but not red)
    else:
        normal_tier = {}

    # Giants: yellow / orange / red by grav-score terciles (within red set)
    if idx_giants:
        giant_scores = np.array([grav_scores[i] for i in idx_giants], dtype=float)
        g1 = float(np.percentile(giant_scores, 33))
        g2 = float(np.percentile(giant_scores, 66))
        giant_tier = {}
        for i in idx_giants:
            g = grav_scores[i]
            if g <= g1:
                giant_tier[i] = 'yellow'
            elif g <= g2:
                giant_tier[i] = 'orange'
            else:
                giant_tier[i] = 'red'
    else:
        giant_tier = {}

    # Map of giant words to their assigned global color (no recompute in subgraphs)
    giant_color_map_js = {words[i]: giant_tier[i] for i in idx_giants}

    # Apply semantic collapse in the global environment using probabilities p = count/total_tokens
    total_tokens = float(len(token_sequence)) if len(token_sequence) else 1.0
    fill_colors = []
    line_colors = []
    line_widths = []
    alphas = []
    for i, c in enumerate(counts):
        p_share = c / total_tokens
        if p_share >= COLLAPSE_TAU:
            # collapse: shrink toward a fixed fraction of the baseline and turn black
            sizes[i] = max(base_min_px * COLLAPSE_SHRINK, 1.0)
            fill_colors.append('black')
            line_colors.append('#000000')
            line_widths.append(1.5)
            alphas.append(1.0)
        else:
            if is_red_flags[i]:
                # giants palette: yellow / orange / red
                color = giant_tier.get(i, 'red')
                if color == 'yellow':
                    fill_colors.append('yellow'); line_colors.append('#b38600'); alphas.append(0.9)
                elif color == 'orange':
                    fill_colors.append('orange'); line_colors.append('#cc7000'); alphas.append(0.9)
                else:
                    fill_colors.append('red');    line_colors.append('#7f1f1f'); alphas.append(0.9)
            else:
                # normals palette: green / blue / purple
                color = normal_tier.get(i, 'blue')
                if color == 'green':
                    fill_colors.append('green');  line_colors.append('#0f5132'); alphas.append(0.8)
                elif color == 'purple':
                    fill_colors.append('purple'); line_colors.append('#4b2e83'); alphas.append(0.8)
                else:
                    fill_colors.append('blue');   line_colors.append('#1f3b7f'); alphas.append(0.8)
            line_widths.append(1.0)

    max_diam = sizes.max()
    step = max_diam * 1.2  # spacing multiplier
    xs, ys, cols, rows = grid_layout(n, base_step=step)

    # data source
    source = ColumnDataSource(data=dict(
        x=xs,
        y=ys,
        size=sizes,
        word=words,
        count=counts,
        fill=fill_colors,
        alpha=alphas,
        line_color=line_colors,
        line_width=line_widths,
        # label positions (word on center, count slightly below)
        text_word=words,
        text_count=[str(c) for c in counts],
        text_word_y=ys + sizes * 0.00,     # center
        text_count_y=ys - sizes * 0.45,    # below the circle center
    ))

    source_ctx = ColumnDataSource(data=dict(
        x=[], y=[], size=[], word=[], count=[], fill=[], alpha=[],
        line_color=[], line_width=[], text_word=[], text_count=[],
        text_word_y=[], text_count_y=[],
    ))

    # figure with tools
    TOOLS = "pan,wheel_zoom,reset,save,tap,box_zoom"
    p = figure(
        title=f"Word Frequency Nodes (n={n}, grid={cols}×{rows}) — click a node",
        tools=TOOLS,
        active_scroll="wheel_zoom",
        match_aspect=False,            # allow free stretch
        sizing_mode="stretch_both",    # fill available container space
    )

    # --- Search controls (main graph only)
    search_input = TextInput(title="Search word (exact, case-insensitive):", placeholder="e.g., love")
    search_btn = Button(label="Find", button_type="primary")

    search_cb = CustomJS(args=dict(
        src=source,
        p=p,
        ti=search_input,
    ), code="""
const q = (ti.value || "").trim().toLowerCase();
if (!q) { return; }
const words = src.data['word'];
let idx = -1;
for (let i = 0; i < words.length; i++) {
    if ((words[i] || "").toLowerCase() === q) { idx = i; break; }
}
if (idx === -1) {
    alert(`Not found: “${q}”`);
    return;
}
// Select in the main source; this will trigger the existing selection callback
src.selected.indices = [idx];
src.change.emit();

// Zoom/pan to center on the found node (preserve ~30% of current view width/height)
const xr = p.x_range, yr = p.y_range;
const x = src.data['x'][idx];
const y = src.data['y'][idx];
const newW = (xr.end - xr.start) * 0.30;
const newH = (yr.end - yr.start) * 0.30;
xr.start = x - newW / 2; xr.end = x + newW / 2;
yr.start = y - newH / 2; yr.end = y + newH / 2;
""")
    search_btn.js_on_event("button_click", search_cb)

    # circles
    r_circ = p.scatter(
        x="x", y="y",
        marker="circle",
        size="size",
        fill_color="fill",
        fill_alpha="alpha",
        line_color="line_color",
        line_width="line_width",
        source=source,
    )

    # word label (fixed screen font for readability when zooming)
    p.text(
        x="x", y="y",
        text="text_word",
        source=source,
        text_align="center",
        text_baseline="middle",
        text_font_size="9pt",
        text_color="white",
    )

    # count label (below word)
    p.text(
        x="x", y="text_count_y",
        text="text_count",
        source=source,
        text_align="center",
        text_baseline="top",
        text_font_size="8pt",
        text_color="#e6e6ff",
    )

    # hover (optional)
    hover = HoverTool(tooltips=[
        ("word", "@word"),
        ("count", "@count"),
    ], renderers=[r_circ])
    p.add_tools(hover)

    # Secondary figure: context graph (populated on click)
    p2 = figure(
        title="Context (±%d) — click a word above" % ctx_window,
        tools=TOOLS,
        active_scroll="wheel_zoom",
        match_aspect=False,
        sizing_mode="stretch_both",
    )

    r_circ_ctx = p2.scatter(
        x="x", y="y",
        marker="circle",
        size="size",
        fill_color="fill",
        fill_alpha="alpha",
        line_color="line_color",
        line_width="line_width",
        source=source_ctx,
    )

    p2.text(
        x="x", y="y",
        text="text_word",
        source=source_ctx,
        text_align="center",
        text_baseline="middle",
        text_font_size="9pt",
        text_color="white",
    )
    p2.text(
        x="x", y="text_count_y",
        text="text_count",
        source=source_ctx,
        text_align="center",
        text_baseline="top",
        text_font_size="8pt",
        text_color="#e6e6ff",
    )
    hover2 = HoverTool(tooltips=[("word", "@word"), ("count", "@count")], renderers=[r_circ_ctx])
    p2.add_tools(hover2)
    p2.grid.visible = False
    p2.axis.visible = False
    p2.outline_line_color = None
    p2.min_border = 0

    # --- Side panel: ranked modal cohort per section (updates on click)
    div_ctx = Div(text="<em>Click a node to see ranked neighbors by tier.</em>",
                  width=420)

    # --- Comparison figure: cosine(shape) ranking for purple modal candidates
    source_comp = ColumnDataSource(data=dict(word=[], score=[]))
    p3 = figure(
        title="Shape similarity — top candidates (purple modal band)",
        tools="pan,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=FactorRange(factors=[]),  # will be populated dynamically via .factors
        height=260,
        sizing_mode="stretch_both",
    )
    p3.vbar(x="word", top="score", width=0.9, source=source_comp)
    p3.xaxis.major_label_orientation = 0.95
    p3.yaxis.axis_label = "cosine(shape)"
    p3.grid.grid_line_alpha = 0.3
    p3.min_border = 0

    # click behavior: highlight + alert word/count + populate context plot
    callback = CustomJS(args=dict(
        src=source,
        src_ctx=source_ctx,
        p2=p2,
        src_comp=source_comp,
        p3=p3,
        ctx_map=ctx_map_js,
        ctx_window=ctx_window,
        COLLAPSE_TAU=COLLAPSE_TAU,
        COLLAPSE_SHRINK=COLLAPSE_SHRINK,
        red_map=red_map_js,
        giant_colors=giant_color_map_js,
        panel=div_ctx,
        global_counts=global_counts_js,
        total_tokens=total_tokens,
        cooc_total=cooc_total,
    ), code="""
        function size_from_count_js(counts, base_min_px, extra_px) {
            if (counts.length === 0) return [];
            let minv = Infinity, maxv = -Infinity;
            for (const c of counts) { if (c < minv) minv = c; if (c > maxv) maxv = c; }
            if (minv === maxv) {
                return counts.map(_ => base_min_px + extra_px * 0.5);
            }
            const smin = Math.sqrt(minv), smax = Math.sqrt(maxv);
            const denom = (smax - smin);
            return counts.map(c => {
                const sc = (Math.sqrt(c) - smin) / denom;
                return base_min_px + sc * extra_px;
            });
        }
        function grid_layout_js(n, step) {
            const cols = Math.ceil(Math.sqrt(n));
            const rows = Math.ceil(n / cols);
            const xs = new Array(n);
            const ys = new Array(n);
            for (let i = 0; i < n; i++) {
                const r = Math.floor(i / cols);
                const c = i % cols;
                xs[i] = c * step;
                ys[i] = -r * step;
            }
            // center around (0,0)
            const xmean = xs.reduce((a,b)=>a+b,0)/n;
            const ymean = ys.reduce((a,b)=>a+b,0)/n;
            for (let i = 0; i < n; i++) { xs[i] -= xmean; ys[i] -= ymean; }
            return {xs, ys};
        }

        const inds = cb_obj.indices;
        if (!inds.length) { return; }
        const i = inds[0];
        const w = src.data['word'][i];
        const c = src.data['count'][i];

        // soft-reset any previous highlight
        const N = src.get_length();
        for (let k=0; k<N; k++) {
            src.data['line_width'][k] = 1.0;
            src.data['alpha'][k] = 0.8;
        }
        // highlight selection
        src.data['line_width'][i] = 3.0;
        src.data['alpha'][i] = 1.0;
        src.change.emit();

        // Build context data for the clicked word
        const entry = ctx_map[w];
        if (!entry) {
            // clear secondary if nothing
            src_ctx.data = {x:[],y:[],size:[],word:[],count:[],fill:[],alpha:[],line_color:[],line_width:[],text_word:[],text_count:[],text_word_y:[],text_count_y:[]};
            src_ctx.change.emit();
            p2.title.text = "Context (±" + ctx_window + ") — no neighbors";
            return;
        }
        const words2 = Object.keys(entry);
        const counts2 = words2.map(k => entry[k]);

        // baseline min diameter from longest neighbor word
        let maxLen = 1;
        for (const s of words2) { if (s.length > maxLen) maxLen = s.length; }
        const char_px = 7.2;
        const pad_px  = 12.0;
        const base_min_px = char_px * maxLen + 2.0 * pad_px;

        // initial sizes by frequency (sqrt scale)
        const sizes2 = size_from_count_js(counts2, base_min_px, 70.0);

        // apply semantic collapse within this context window
        const total2 = counts2.reduce((a,b)=>a+b, 0) || 1;
        const fill = new Array(words2.length);
        const alpha = new Array(words2.length);
        const line_color = new Array(words2.length);
        const line_width = new Array(words2.length);

        // helpers for percentiles
        function percentile(arr, p) {
            if (!arr || arr.length === 0) return 0;
            const a = arr.slice().sort((x,y)=>x-y);
            const idx = (p/100) * (a.length - 1);
            const lo = Math.floor(idx), hi = Math.ceil(idx);
            if (lo === hi) return a[lo];
            const w = idx - lo;
            return a[lo]*(1-w) + a[hi]*w;
        }
        // compute tiers for NORMAL (non-giant) words within this context window
        const normal_idx = [];
        const normal_counts = [];
        for (let k = 0; k < words2.length; k++) {
            if (red_map[words2[k]] !== 1) { // not a giant globally
                normal_idx.push(k);
                normal_counts.push(counts2[k]);
            }
        }
        const t1_n = percentile(normal_counts, 33);
        const t2_n = percentile(normal_counts, 66);
        for (let idx = 0; idx < words2.length; idx++) {
            const p_share = counts2[idx] / total2;
            if (p_share >= COLLAPSE_TAU) {
                sizes2[idx] = Math.max(base_min_px * COLLAPSE_SHRINK, 1.0);
                fill[idx] = 'black';
                alpha[idx] = 1.0;
                line_color[idx] = '#000000';
                line_width[idx] = 1.5;
            } else {
                // Giants use their GLOBAL assigned color (yellow/orange/red) from the main corpus
                if (red_map[words2[idx]] === 1) {
                    const gc = giant_colors[words2[idx]] || 'red';
                    if (gc === 'yellow') {
                        fill[idx] = 'yellow'; alpha[idx] = 0.9; line_color[idx] = '#b38600';
                    } else if (gc === 'orange') {
                        fill[idx] = 'orange'; alpha[idx] = 0.9; line_color[idx] = '#cc7000';
                    } else {
                        fill[idx] = 'red';    alpha[idx] = 0.9; line_color[idx] = '#7f1f1f';
                    }
                } else {
                    // Recompute NORMAL tiers locally for the subgraph: green / blue / purple by count terciles
                    const c_local = counts2[idx];
                    if (normal_counts.length === 0) {
                        fill[idx] = 'blue'; alpha[idx] = 0.8; line_color[idx] = '#1f3b7f';
                    } else if (c_local <= t1_n) {
                        fill[idx] = 'green'; alpha[idx] = 0.8; line_color[idx] = '#0f5132';
                    } else if (c_local <= t2_n) {
                        fill[idx] = 'blue';  alpha[idx] = 0.8; line_color[idx] = '#1f3b7f';
                    } else {
                        fill[idx] = 'purple'; alpha[idx] = 0.8; line_color[idx] = '#4b2e83';
                    }
                }
                line_width[idx] = 1.0;
            }
        }

        const max_d = Math.max(...sizes2);
        const step = max_d * 1.2;
        const grid = grid_layout_js(words2.length, step);

        // labels
        const text_word = words2.slice();
        const text_count = counts2.map(x => String(x));
        const text_word_y = grid.ys.map((y,idx) => y + sizes2[idx]*0.00);
        const text_count_y = grid.ys.map((y,idx) => y - sizes2[idx]*0.45);

        src_ctx.data = {
            x: grid.xs,
            y: grid.ys,
            size: sizes2,
            word: words2,
            count: counts2,
            fill: fill,
            alpha: alpha,
            line_color: line_color,
            line_width: line_width,
            text_word: text_word,
            text_count: text_count,
            text_word_y: text_word_y,
            text_count_y: text_count_y
        };
        src_ctx.change.emit();

        // ----- Build side panel: modal cohort per section -----
        function median(arr) {
            if (!arr.length) return 0;
            const a = arr.slice().sort((x,y)=>x-y);
            const m = Math.floor(a.length/2);
            return a.length % 2 ? a[m] : 0.5*(a[m-1]+a[m]);
        }
        function quartiles(arr) {
            if (!arr.length) return {q1:0,q3:0};
            const a = arr.slice().sort((x,y)=>x-y);
            const mid = Math.floor(a.length/2);
            const lower = a.slice(0, mid);
            const upper = a.length % 2 ? a.slice(mid+1) : a.slice(mid);
            const q1 = median(lower);
            const q3 = median(upper);
            return {q1,q3};
        }
        function within_band(val, center, delta) {
            return val >= center - delta && val <= center + delta;
        }
        // ----- Shape vectors via PPMI and cosine ranking -----
        function build_shape(word, K) {
            // returns sparse map: neighbor -> PPMI(word, neighbor)
            const co = ctx_map[word];
            if (!co) return {};
            const entries = Object.entries(co);
            const pw = (global_counts[word] || 0) / (total_tokens || 1);
            const out = [];
            for (const [nbr, cnt] of entries) {
                const pu = (global_counts[nbr] || 0) / (total_tokens || 1);
                if (pu <= 0 || pw <= 0) continue;
                const pwu = (cnt || 0) / (cooc_total || 1);
                if (pwu <= 0) continue;
                // optional: drop globally-collapsed features
                const pu_global = (global_counts[nbr] || 0) / (total_tokens || 1);
                if (pu_global >= COLLAPSE_TAU) continue;
                const pmi = Math.log(pwu / (pw * pu));
                const ppmi = Math.max(0, pmi);
                if (ppmi > 0) out.push([nbr, ppmi]);
            }
            // keep top-K by weight
            out.sort((a,b)=> b[1]-a[1]);
            const top = (K && K>0) ? out.slice(0, K) : out;
            const sparse = {};
            for (const [nbr, w] of top) sparse[nbr] = w;
            return sparse;
        }
        function cosine_sparse(a, b) {
            // a,b are maps: key -> weight
            let dot = 0.0, na = 0.0, nb = 0.0;
            for (const k in a) { const v = a[k]; na += v*v; if (b[k] !== undefined) dot += v * b[k]; }
            for (const k in b) { const v = b[k]; nb += v*v; }
            if (na === 0 || nb === 0) return 0.0;
            return dot / (Math.sqrt(na) * Math.sqrt(nb));
        }
        function sectionLists(words2, counts2, colors2, focusWord, purple_override) {
            const sections = {
                green: [], blue: [], purple: [],
                yellow: [], orange: [], red: []
            };
            for (let i=0;i<words2.length;i++){
                const col = colors2[i];
                if (sections[col] !== undefined) {
                    sections[col].push([words2[i], counts2[i]]);
                }
            }
            const order = ["green","blue","purple","yellow","orange","red"];
            const html = [];
            for (const key of order) {
                const items = sections[key];
                if (!items.length) continue;
                // If this is the PURPLE section, use the unified pre-ranked list
                if (key === "purple" && purple_override && purple_override.length) {
                    // use provided ranked list (word, count, cosine, mP)
                    const MAXK = 30;
                    const topk = purple_override.slice(0, MAXK);
                    const title = key.charAt(0).toUpperCase() + key.slice(1);
                    const m = purple_override[0][3]; // shared median used earlier
                    const delta = 0; // shown implicitly via cosine; omit ± here or keep prior computation if desired
                    html.push(`<div style="margin:8px 0;"><div style="font-weight:600;">${title} — modal band (cosine-ranked)</div><ul style="margin:4px 0 0 20px;">` +
                        topk.map(t=>`<li><span style="font-family:monospace;">${t[0]}</span> <span style="opacity:0.7;">(${t[1]})</span> <span style="opacity:0.6;">cos=${t[2].toFixed(3)}</span></li>`).join("") +
                        `</ul></div>`);
                    continue; // move to next section; skip default rendering
                }
                // counts for this section
                const vals = items.map(t=>t[1]);
                const m = median(vals);
                const {q1,q3} = quartiles(vals);
                // delta from IQR; require at least ±1 band
                const delta = Math.max(1, Math.round(0.25 * Math.max(1, (q3 - q1))));
                // filter to modal band
                const cohort = items.filter(t => within_band(t[1], m, delta));
                // sort by |c - m| asc, then count desc, then alpha asc
                cohort.sort((a,b)=>{
                    const da = Math.abs(a[1]-m), db = Math.abs(b[1]-m);
                    if (da !== db) return da - db;
                    if (b[1] !== a[1]) return b[1] - a[1];
                    return a[0] < b[0] ? -1 : (a[0] > b[0] ? 1 : 0);
                });
                // cap the list
                const MAXK = 30;
                const topk = cohort.slice(0, MAXK);
                // render
                const title = key.charAt(0).toUpperCase() + key.slice(1);
                html.push(`<div style="margin:8px 0;"><div style="font-weight:600;">${title} — modal band (center=${m}, ±${delta})</div><ul style="margin:4px 0 0 20px;">` +
                    topk.map(t=>`<li><span style="font-family:monospace;">${t[0]}</span> <span style="opacity:0.7;">(${t[1]})</span></li>`).join("") +
                    `</ul></div>`);
            }
            return html.join("");
        }
        // colors2 array mirrors fill[] chosen for subgraph nodes
        const colors2 = fill.slice();

        // ----- Unified PURPLE cohort (modal band) computed once with cosine(shape) -----
        // gather purple items from the subgraph coloring
        const itemsPurpleAll = [];
        for (let i2=0;i2<words2.length;i2++){
            if (colors2[i2] === 'purple') {
                itemsPurpleAll.push([words2[i2], counts2[i2]]);
            }
        }
        // compute modal band for purple
        let purple_ranked = [];
        if (itemsPurpleAll.length > 0) {
            const valsP = itemsPurpleAll.map(t=>t[1]);
            const mP = median(valsP);
            const {q1: q1P, q3: q3P} = quartiles(valsP);
            const deltaP = Math.max(1, Math.round(0.25 * Math.max(1, (q3P - q1P))));
            const cohortP = itemsPurpleAll.filter(t => within_band(t[1], mP, deltaP));

            // cosine re-ranking using PPMI shapes
            const Kshape = 128;
            const shapeF_unified = build_shape(w, Kshape);
            const ranked = cohortP.map(t => {
                const sh = build_shape(t[0], Kshape);
                const cs = cosine_sparse(shapeF_unified, sh);
                return [t[0], t[1], cs, mP]; // include median for later display
            });
            ranked.sort((a,b)=>{
                if (b[2] !== a[2]) return b[2] - a[2];         // cosine desc
                const da = Math.abs(a[1]-mP), db = Math.abs(b[1]-mP); // proximity to median
                if (da !== db) return da - db;
                if (b[1] !== a[1]) return b[1] - a[1];          // count desc
                return a[0] < b[0] ? -1 : (a[0] > b[0] ? 1 : 0);
            });
            purple_ranked = ranked; // array of [word, count, cosine, mP]
        }

        const panelHTML = sectionLists(words2, counts2, colors2, w, purple_ranked);
        panel.text = panelHTML || "<em>No neighbors in tiers to display.</em>";

        p2.title.text = `Context (±${ctx_window}) for “${w}” (n=${words2.length})`;

        // ----- Comparison chart from unified purple_ranked -----
        let comp_words = [];
        let comp_scores = [];
        if (purple_ranked.length > 0) {
            const TOPK = 20;
            const top = purple_ranked.slice(0, TOPK);
            comp_words = top.map(x=>x[0]);
            comp_scores = top.map(x=>x[2]); // cosine
        }
        // Update bar chart source and x_range
        src_comp.data = { word: comp_words, score: comp_scores };
        src_comp.change.emit();
        p3.x_range.factors = comp_words.slice();
        p3.title.text = `Shape similarity — top ${comp_words.length} purple candidates for “${w}”`;
    """)
    # Attach to the data source selection
    source.selected.js_on_change("indices", callback)

    # aesthetics
    p.grid.visible = False
    p.axis.visible = False
    p.outline_line_color = None
    p.min_border = 0

    controls = row(search_input, search_btn, sizing_mode="stretch_width")
    bottom = row(p2, div_ctx, sizing_mode="stretch_both")
    return column(controls, p, bottom, p3, sizing_mode="stretch_both")

# -----------------------
# 6) Main
# -----------------------
def main():
    text_path = sys.argv[1] if len(sys.argv) > 1 else None
    text = load_text(text_path)
    words = tokenize(text)
    counts = word_frequencies(words)
    plot = build_plot(counts, words, ctx_window=10)
    show(plot)

if __name__ == "__main__":
    main()