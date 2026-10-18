"""README 그림을 직접 만든다(발표 자료 이미지를 쓰지 않음).

    python docs/figures/make_figures.py      # 저장소 루트에서 실행 (Edge 또는 Chrome 필요)

  fig_algorithm.png  Cluster-Coupled Adam 한 스텝 (code/optimizers.py ClusterCoupledAdam.step 기준)
  fig_mixing.png     gradient 섞기 개념도 (2차원 예시 값으로 실제 계산)
  fig_versions.png   v1 → v2 → v3 → Final 개선 과정 (README 4절의 실험 기록 요약)
  repro_curves.png   결과 곡선 (docs/repro/*.csv)
  e30_curves.png     30에폭 확인 실험 (docs/experiments/2026-10_e30/*.csv)
스타일·렌더러: figstyle.py
"""
import csv, glob
from pathlib import Path

import numpy as np

from figstyle import page, render

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "img"
W = 1240


def _marker(mid, color, size=7):
    return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="{size}" markerHeight="{size}" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')


# ---------------------------------------------------------------- 1. 한 스텝 흐름
def fig_algorithm():
    y = 150
    nodes = [  # x, 제목, 설명 줄들, 모양
        (70, "gradient <tspan class='m'>g</tspan>", ["임베딩 행렬 (N행 × d)"], "in"),
        (205, "burn-in 확인", ["1024스텝이 지났으면", "아래 경로로"], "split"),
        (385, "EMA gradient", ["ema ← 0.9·ema + 0.1·<tspan class='m'>g</tspan>"], "dot"),
        (580, "K-means로 행 묶기", ["128스텝마다 다시 묶음", "K는 처음 한 번 Elbow로"], "dot"),
        (775, "클러스터 평균 <tspan class='m'>ḡ</tspan>", ["이번 배치에서 gradient가", "생긴 행만 평균"], "dot"),
        (962, "섞기", ["α = 0.25, 비활성 행은 0"], "core"),
        (1100, "Adam 갱신", ["m, v 모멘트 → θ"], "out"),
    ]
    s = ['<svg width="1152" height="322" viewBox="0 0 1152 322">',
         '<defs>' + _marker("ah", "#2B50E0") + _marker("aa", "#E0700B") + '</defs>']
    s.append(f'<path d="M205,{y} C205,70 225,46 290,46 L1020,46 C1080,46 1100,70 1100,{y-24}" fill="none" '
             'stroke="#E0700B" stroke-width="3" stroke-dasharray="9 7" marker-end="url(#aa)"/>')
    s.append('<rect x="392" y="30" width="520" height="30" fill="#FFFFFF"/>')
    s.append('<text x="652" y="51" text-anchor="middle" font-size="15" font-weight="700" fill="#E0700B">'
             '처음 1024스텝(약 5에폭)은 섞지 않고 바로 Adam으로</text>')
    s.append(f'<line x1="70" y1="{y}" x2="205" y2="{y}" stroke="#14202B" stroke-width="4"/>')
    s.append(f'<line x1="205" y1="{y}" x2="1078" y2="{y}" stroke="#2B50E0" stroke-width="4" marker-end="url(#ah)"/>')
    for x, title, lines, kind in nodes:
        if kind == "split":
            s.append(f'<rect x="{x-12}" y="{y-12}" width="24" height="24" transform="rotate(45 {x} {y})" '
                     'fill="#FFFFFF" stroke="#E0700B" stroke-width="3.5"/>')
        elif kind == "core":
            s.append(f'<circle cx="{x}" cy="{y}" r="15" fill="#2B50E0"/>')
        elif kind == "out":
            s.append(f'<circle cx="{x}" cy="{y}" r="16" fill="#14202B"/>')
        elif kind == "in":
            s.append(f'<circle cx="{x}" cy="{y}" r="10" fill="#14202B"/>')
        else:
            s.append(f'<circle cx="{x}" cy="{y}" r="10" fill="#FFFFFF" stroke="#2B50E0" stroke-width="3.5"/>')
        s.append(f'<text x="{x}" y="{y+48}" text-anchor="middle" font-size="17" font-weight="700" fill="#14202B">{title}</text>')
        for k, ln in enumerate(lines):
            s.append(f'<text x="{x}" y="{y+72+k*19}" text-anchor="middle" font-size="13.5" fill="#5A6876">{ln}</text>')
    s.append(f'<rect x="867" y="{y+86}" width="190" height="38" rx="6" fill="#E8EDFD"/>')
    s.append(f'<text x="962" y="{y+112}" text-anchor="middle" font-size="19" fill="#14202B" class="m">'
             'g̃ = (1−α) g + α ḡ</text>')
    s.append('<path d="M360,304 L360,312 L860,312 L860,304" fill="none" stroke="#8C98A4" stroke-width="1.5"/>')
    s.append('<text x="610" y="298" text-anchor="middle" font-size="13" fill="#8C98A4">'
             '임베딩 두 테이블(embedding_user, embedding_item)에만 적용, 나머지 파라미터는 일반 Adam</text>')
    s.append('</svg>')
    body = ('<div class="frame"><div class="title">비슷한 행끼리 묶고, 묶음 평균 쪽으로 gradient를 조금 당긴 뒤 Adam에 넘긴다</div>'
            '<div class="lede">Cluster-Coupled Adam 한 스텝, 기준 코드 <span class="code">code/optimizers.py</span> '
            'ClusterCoupledAdam.step</div><div style="margin-top:18px">' + "".join(s) + '</div></div>')
    render(page(body, W), OUT / "fig_algorithm.png", W, 460)


# ---------------------------------------------------------------- 2. 섞기 개념도
def _vec_panel(pid, title, vecs, ghosts=None, mean=None, color="#2B50E0"):
    sc, ox, oy = 124, 60, 318          # 1단위 = 128px, 원점 위치
    P = lambda v: (ox + v[0] * sc, oy - v[1] * sc)
    s = ['<svg width="440" height="430" viewBox="0 0 440 430"><defs>',
         _marker(f"v{pid}", color, 6), _marker(f"g{pid}", "#B9C6F5", 6), _marker(f"a{pid}", "#E0700B", 5), '</defs>']
    for gx in np.arange(0, 2.51, 0.5):
        x, _ = P((gx, 0))
        s.append(f'<line x1="{x}" y1="40" x2="{x}" y2="408" stroke="#EEF1F5"/>')
        s.append(f'<text x="{x}" y="424" text-anchor="middle" font-size="11" fill="#8C98A4">{gx:g}</text>')
    for gy in np.arange(-0.5, 2.51, 0.5):
        _, yy = P((0, gy))
        s.append(f'<line x1="20" y1="{yy}" x2="430" y2="{yy}" stroke="#EEF1F5"/>')
    s.append(f'<line x1="20" y1="{oy}" x2="430" y2="{oy}" stroke="#C7CFD8" stroke-width="1.5"/>')
    s.append(f'<line x1="{ox}" y1="40" x2="{ox}" y2="408" stroke="#C7CFD8" stroke-width="1.5"/>')
    if ghosts is not None:
        for gv, v in zip(ghosts, vecs):
            x, yy = P(gv)
            s.append(f'<line x1="{ox}" y1="{oy}" x2="{x}" y2="{yy}" stroke="#B9C6F5" stroke-width="2" marker-end="url(#g{pid})"/>')
            x2, y2 = P(v)
            s.append(f'<line x1="{x}" y1="{yy}" x2="{x2}" y2="{y2}" stroke="#E0700B" stroke-width="2" '
                     f'stroke-dasharray="3 4" marker-end="url(#a{pid})"/>')
    if mean is not None:
        x, yy = P(mean)
        s.append(f'<line x1="{ox}" y1="{oy}" x2="{x}" y2="{yy}" stroke="#E0700B" stroke-width="5" stroke-linecap="round"/>')
        s.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="#E0700B"/>')
    for k, v in enumerate(vecs):
        x, yy = P(v)
        s.append(f'<line x1="{ox}" y1="{oy}" x2="{x}" y2="{yy}" stroke="{color}" stroke-width="3.5" marker-end="url(#v{pid})"/>')
        s.append(f'<text x="{x+10}" y="{yy-6}" font-size="14" font-weight="700" fill="{color}">행 {k+1}</text>')
    if mean is not None:
        x, yy = P(mean)
        s.append(f'<text x="{x-6}" y="{yy-22}" font-size="15" font-weight="700" fill="#E0700B">평균 <tspan class="m">ḡ</tspan></text>')
    s.append('</svg>')
    return f'<div><div class="ptitle">{title}</div>{"".join(s)}</div>'


def fig_mixing():
    g = np.array([[1.0, 2.0], [2.0, 0.5], [1.5, -0.5]])
    gbar = g.mean(axis=0)
    alpha = 0.25
    mixed = (1 - alpha) * g + alpha * gbar
    left = _vec_panel("L", "기존 Adam: 각 행이 자기 gradient만 따라간다", g)
    right = _vec_panel("R", f"CCA (α = {alpha}): 각 행을 평균 쪽으로 25% 당긴다", mixed, ghosts=g, mean=gbar)
    body = ('<style>.row{display:flex;gap:40px;margin-top:20px}.ptitle{font-size:15px;font-weight:700;margin:0 0 4px 20px}'
            '.legend{font-size:13px;color:#5A6876;margin-top:4px;margin-left:6px}.legend i{display:inline-block;width:22px;height:0;'
            'vertical-align:middle;margin:0 6px 0 14px;border-top:3px solid}</style>'
            '<div class="frame"><div class="title">같은 묶음의 행들은 평균 방향으로 조금씩 끌려가 함께 움직인다</div>'
            f'<div class="lede">2차원 예시 값으로 실제 계산: <span class="m">g</span>₁=(1, 2), <span class="m">g</span>₂=(2, 0.5), '
            f'<span class="m">g</span>₃=(1.5, −0.5), <span class="m">ḡ</span>=({gbar[0]:.2f}, {gbar[1]:.2f}), '
            '<span class="m">g̃ = (1−α) g + α ḡ</span></div>'
            f'<div class="row">{left}{right}</div>'
            '<div class="legend"><i style="border-color:#B9C6F5"></i>원래 gradient'
            '<i style="border-color:#E0700B;border-top-style:dashed"></i>평균 쪽으로 당겨진 만큼'
            '<i style="border-color:#2B50E0"></i>섞은 gradient</div></div>')
    render(page(body, 1010), OUT / "fig_mixing.png", 1010, 620)


# ---------------------------------------------------------------- 3. 버전별 개선
def fig_versions():
    vers = [
        ("v1", "파라미터 값으로 클러스터링", "Adam ≥ CCA<br>α↑ 성능↓, K↑ 성능↑", "파라미터 값은 학습 방향을 담지 못한다", False),
        ("v2", "EMA gradient로 클러스터링<br>Elbow로 K 자동 선택", "Loss는 Adam과 비슷<br>초반 지표가 소폭 뒤처짐",
         "초반 gradient가 불안정하고 재클러스터링이 너무 잦다<br>(주기 sweep: 64~128에서 안정)", False),
        ("v3", "burn-in 확대<br>재클러스터링 주기 64~128<br>활성 행 Masking", "v2보다 세 지표 개선<br>후반 변동 감소", "", False),
        ("Final", "ema_grad 클러스터링 확정", "최종 지표 Adam과 동등<br>전 구간 우위는 입증 못 함", "다음 검증: 여러 시드, 조밀한 데이터셋", True),
    ]
    head = "".join(f'<div class="vh{" fin" if fin else ""}"><span class="dot"></span><span class="vn">{v}</span></div>' for v, *_, fin in vers)

    def row(label, idx, cls=""):
        cells = "".join(f'<div class="c {cls}{"" if v[idx] else " empty"}">{v[idx] or "―"}</div>' for v in vers)
        return f'<div class="lab">{label}</div>' + cells
    css = ('<style>.grid{display:grid;grid-template-columns:110px repeat(4,1fr);margin-top:26px;position:relative}'
           '.track{position:absolute;left:118px;right:40px;top:14px;height:3px;background:#2B50E0}'
           '.vh{position:relative;font-size:22px;font-weight:800;padding:0 0 14px 0;display:flex;align-items:center;gap:10px}'
           '.vh .dot{width:18px;height:18px;border-radius:50%;background:#fff;border:3.5px solid #2B50E0;z-index:1}'
           '.vh .vn{background:#fff;padding:0 10px 0 4px;z-index:1;line-height:1}.vh.fin{color:#2B50E0}.vh.fin .dot{background:#2B50E0}'
           '.lab{font-size:13px;font-weight:700;color:#5A6876;padding:12px 0;border-top:1px solid #D9DFE6}'
           '.c{font-size:14.5px;line-height:1.55;padding:12px 18px 12px 0;border-top:1px solid #D9DFE6}'
           '.c.dg{color:#A9520A;font-weight:600;border-top:2px solid #E0700B}'
           '.c.dg.empty{border-top:1px solid #D9DFE6}.c.empty{color:#C7CFD8;font-weight:400}</style>')
    body = (css + '<div class="frame"><div class="title">실패할 때마다 원인을 한 문장으로 정의했고, 그 문장이 다음 버전의 설계가 됐다</div>'
            '<div class="lede">내 담당: SGD 비교 실험, α·K·재클러스터링 주기 sweep 3종. 개선 방향은 팀 회의와 AI 질의 결과를 함께 검토해 결정</div>'
            f'<div class="grid"><div class="track"></div><div></div>{head}'
            + row("바꾼 것", 1) + row("실험 결과", 2) + row("진단 · 다음", 3, "dg") +
            '</div><div class="src">재현(노트북 최종 설정, 10에폭): Adam Recall@20 0.1218, CCA 0.1203. 실험 기록은 README 4절</div></div>')
    render(page(body, W), OUT / "fig_versions.png", W, 470)


# ---------------------------------------------------------------- 4. 재현 곡선
def _read(pattern):
    f = sorted(glob.glob(str(ROOT / "docs" / "repro" / pattern)))[-1]
    with open(f, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("epoch", "loss", "recall@20", "ndcg@20")}


def _line_panel(title, ep, a, c, fmt, burn=5, first=False):
    efmt = fmt
    w, h, l, r, t, b = 380, 250, 54, 66, 14, 32
    lo, hi = min(a.min(), c.min()), max(a.max(), c.max())
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    X = lambda e: l + (e - ep.min()) / (ep.max() - ep.min()) * (w - l - r)
    Y = lambda v: t + (hi - v) / (hi - lo) * (h - t - b)
    s = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    x0, x1 = X(ep.min()), X(burn + 0.5)
    s.append(f'<rect x="{x0}" y="{t}" width="{x1-x0}" height="{h-t-b}" fill="#FDF1E4"/>')
    if first:
        s.append(f'<text x="{x0+8}" y="{h-b-10}" font-size="12" font-weight="700" fill="#E0700B">burn-in (섞기 꺼짐)</text>')
    raw = (hi - lo) / 4
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw))
    ticks = np.arange(np.ceil(lo / step) * step, hi, step)
    dec = max(0, int(-np.floor(np.log10(step))) + (1 if step / mag == 2.5 else 0))
    fmt = lambda v, d=dec: f"{abs(v) if abs(v) < step / 1e6 else v:.{d}f}"
    for v in ticks:
        s.append(f'<line x1="{l}" y1="{Y(v)}" x2="{w-r}" y2="{Y(v)}" stroke="#EEF1F5"/>')
        s.append(f'<text x="{l-8}" y="{Y(v)+4}" text-anchor="end" font-size="11" fill="#8C98A4">{fmt(v)}</text>')
    for e in (1, 5, 10):
        s.append(f'<text x="{X(e)}" y="{h-10}" text-anchor="middle" font-size="11" fill="#8C98A4">{e}</text>')
    for arr, col, wd in ((a, "#14202B", 2.2), (c, "#2B50E0", 2.8)):
        pts = " ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in zip(ep, arr))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wd}" stroke-linejoin="round"/>')
        s.append(f'<circle cx="{X(ep[-1])}" cy="{Y(arr[-1])}" r="4" fill="{col}"/>')
    ya, yc = Y(a[-1]), Y(c[-1])
    if abs(ya - yc) < 16:
        mid = (ya + yc) / 2
        ya, yc = (mid - 8, mid + 8) if a[-1] >= c[-1] else (mid + 8, mid - 8)
    s.append(f'<text x="{w-r+8}" y="{ya+4}" font-size="12" font-weight="700" fill="#14202B">{efmt(a[-1])}</text>')
    s.append(f'<text x="{w-r+8}" y="{yc+4}" font-size="12" font-weight="700" fill="#2B50E0">{efmt(c[-1])}</text>')
    s.append('</svg>')
    return f'<div><div class="ptitle">{title}</div>{"".join(s)}</div>'


def fig_repro():
    a, c = _read("logs_gowalla_adam_*.csv"), _read("logs_gowalla_cluster_*.csv")
    ep = a["epoch"]
    panels = [_line_panel("BPR Loss", ep, a["loss"], c["loss"], lambda v: f"{v:.3f}", first=True),
              _line_panel("Recall@20", ep, a["recall@20"], c["recall@20"], lambda v: f"{v:.4f}"),
              _line_panel("NDCG@20", ep, a["ndcg@20"], c["ndcg@20"], lambda v: f"{v:.4f}")]
    body = ('<style>.row{display:flex;gap:4px;margin-top:18px}.ptitle{font-size:14px;font-weight:700;margin:0 0 2px 54px}'
            '.key{font-size:13px;color:#5A6876;margin-top:4px;margin-left:40px}.key i{display:inline-block;width:22px;border-top:3px solid;'
            'vertical-align:middle;margin:0 6px 0 14px}</style>'
            '<div class="frame"><div class="title">burn-in 동안 CCA는 Adam과 같았고, 섞기가 켜진 뒤에는 소폭 낮아 최종 성능은 비슷했다</div>'
            '<div class="lede">Gowalla, LightGCN(dim 32, 1층), α 0.25, 주기 128, warmup 1024, 10에폭, CPU, 시드 2020. '
            '로그: <span class="code">docs/repro/*.csv</span></div>'
            f'<div class="row">{"".join(panels)}</div>'
            '<div class="key"><i style="border-color:#14202B"></i>Adam<i style="border-color:#2B50E0"></i>Cluster-Coupled Adam'
            '<span style="margin-left:18px">가로축: 에폭</span></div></div>')
    render(page(body, W), OUT / "repro_curves.png", W, 420)


# ---------------------------------------------------------------- 5. 30에폭 확인 실험
E30 = ROOT / "docs" / "experiments" / "2026-10_e30"
E30_LINES = (("shrink", "shrink (직접 설계)", "#2B50E0", 3.0, ""),
             ("vshare", "vshare (보폭 공유)", "#0F8A5F", 2.2, ""),
             ("cca", "cca (팀 방식)", "#8C98A4", 2.2, "6 4"))


def _e30_rel(ds, v):
    """같은 시드의 Adam 대비 Recall@20 변화율(%)을 에폭별로 구해 시드 평균."""
    rel = []
    for s in (2020, 2021, 2022):
        a = np.array([float(r["recall@20"]) for r in csv.DictReader(open(E30 / f"{ds}_adam_s{s}.csv", encoding="utf-8"))])
        x = np.array([float(r["recall@20"]) for r in csv.DictReader(open(E30 / f"{ds}_{v}_s{s}.csv", encoding="utf-8"))])
        rel.append(100 * (x / a - 1))
    return np.mean(rel, axis=0)


def _e30_panel(title, ds, burn):
    w, h, l, r, t, b = 560, 300, 50, 70, 14, 34
    lo, hi = -3.0, 2.5
    X = lambda e: l + (e - 2) / 28 * (w - l - r)
    Y = lambda v: t + (hi - v) / (hi - lo) * (h - t - b)
    s = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    s.append(f'<rect x="{X(2)}" y="{t}" width="{X(burn)-X(2)}" height="{h-t-b}" fill="#FDF1E4"/>')
    for v in (-3, -2, -1, 0, 1, 2):
        s.append(f'<line x1="{l}" y1="{Y(v)}" x2="{w-r}" y2="{Y(v)}" stroke="{"#14202B" if v == 0 else "#EEF1F5"}" '
                 f'stroke-width="{1.4 if v == 0 else 1}"/>')
        s.append(f'<text x="{l-8}" y="{Y(v)+4}" text-anchor="end" font-size="11" fill="#8C98A4">{v:+d}%</text>'.replace("+0%", "0"))
    s.append(f'<text x="{X(burn)+6}" y="{Y(0)-6}" font-size="12" font-weight="700" fill="#14202B">Adam</text>')
    for e in (2, 10, 20, 30):
        s.append(f'<text x="{X(e)}" y="{h-12}" text-anchor="middle" font-size="11" fill="#8C98A4">{e}</text>')
    ends = []
    for v, _, col, wd, dash in E30_LINES:
        y = _e30_rel(ds, v)
        pts = " ".join(f"{X(e):.1f},{Y(max(lo, min(hi, q))):.1f}" for e, q in zip(range(2, 31), y[1:]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wd}" stroke-linejoin="round"'
                 f'{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>')
        s.append(f'<circle cx="{X(30)}" cy="{Y(y[-1])}" r="3.8" fill="{col}"/>')
        ends.append([Y(y[-1]), f"{y[-1]:+.1f}%".replace("-", "−"), col])
    ends.sort()
    for i in range(1, len(ends)):  # 끝 숫자 겹침 방지
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    for yy, txt, col in ends:
        s.append(f'<text x="{w-r+8}" y="{yy+4}" font-size="12" font-weight="700" fill="{col}">{txt}</text>')
    s.append('</svg>')
    return f'<div><div class="ptitle">{title}</div>{"".join(s)}</div>'


def fig_e30():
    # burn-in 1024 스텝 = 학습 상호작용 수 / 배치 4096 으로 환산한 에폭
    panels = [_e30_panel("Gowalla", "gowalla", 1 + 1024 / (810128 / 4096)), _e30_panel("Yelp2018", "yelp2018", 1 + 1024 / (1237259 / 4096))]
    key = "".join(f'<i style="border-color:{c};{"border-top-style:dashed;" if d else ""}"></i>{n}' for _, n, c, _, d in E30_LINES)
    body = ('<style>.row{display:flex;gap:16px;margin-top:18px}.ptitle{font-size:14px;font-weight:700;margin:0 0 2px 50px}'
            '.key{font-size:13px;color:#5A6876;margin-top:4px;margin-left:36px}.key i{display:inline-block;width:22px;border-top:3px solid;'
            'vertical-align:middle;margin:0 6px 0 14px}</style>'
            '<div class="frame"><div class="title">보폭 공유는 초반에만 앞섰고, 행별 수축은 늦게 출발해 Yelp2018에서 Adam을 앞질렀다</div>'
            '<div class="lede">세로축: 같은 시드의 Adam 대비 Recall@20 변화율(시드 3개 평균, 0 = Adam). LightGCN(dim 32, 1층), Colab T4. '
            '1에폭은 Recall이 0.005 수준이라 비율이 크게 흔들려 2에폭부터 그림. 30에폭에서도 모든 방식의 Recall이 계속 오르는 중이라 최종 수렴값은 아님. 로그: <span class="code">docs/experiments/2026-10_e30/</span></div>'
            f'<div class="row">{"".join(panels)}</div>'
            f'<div class="key">{key}<span style="margin-left:18px">가로축: 에폭 · 주황 띠: burn-in(섞기 꺼짐)</span></div></div>')
    render(page(body, W), OUT / "e30_curves.png", W, 520)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_algorithm(); fig_mixing(); fig_versions(); fig_repro(); fig_e30()
    print("ok")
