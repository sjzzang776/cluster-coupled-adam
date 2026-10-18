"""README 그림 공통 스타일과 렌더러.

그림은 HTML/SVG로 그리고 헤드리스 Edge(또는 Chrome)로 2배 해상도 PNG를 찍는다.
디자인 원칙 (실험 노트):
  - 흰 종이 + 옅은 점 격자. 제목은 그림이 말하는 결론 한 문장.
  - 주색은 코발트 하나. 주황(burn-in·주의), 초록(일치·익절), 빨강(불일치·손절)은 의미가 있을 때만.
  - 글꼴: Noto Sans KR(본문), Cambria Math(수식 기호), Consolas(코드 이름에만).
"""
import shutil, subprocess, tempfile
from pathlib import Path

CSS = """
:root{--ink:#14202B;--sub:#5A6876;--faint:#8C98A4;--hair:#D9DFE6;--paper:#FFFFFF;--dot:#E6EBF0;
 --cobalt:#2B50E0;--cobalt-soft:#E8EDFD;--amber:#E0700B;--amber-soft:#FDF1E4;--green:#0F8A5F;--green-soft:#E3F4EC;
 --red:#D23F37;--red-soft:#FBE9E7}
*{box-sizing:border-box}
html,body{margin:0}
body{font-family:"Noto Sans KR","Malgun Gothic",sans-serif;color:var(--ink);background:var(--paper);
 background-image:radial-gradient(var(--dot) 1.1px,transparent 1.3px);background-size:18px 18px;
 -webkit-font-smoothing:antialiased}
.frame{padding:34px 44px 26px}
.title{font-size:25px;font-weight:800;letter-spacing:-.02em;line-height:1.3}
.lede{font-size:14.5px;color:var(--sub);margin-top:6px;line-height:1.5}
.src{font-size:12px;color:var(--faint);margin-top:14px}
.m{font-family:"Cambria Math","Times New Roman",serif;font-style:italic}
.code{font-family:Consolas,"Cascadia Mono",monospace;font-size:.92em}
svg text{font-family:"Noto Sans KR","Malgun Gothic",sans-serif}
svg .m{font-family:"Cambria Math","Times New Roman",serif;font-style:italic}
svg .code{font-family:Consolas,monospace}
"""

_BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


def _browser():
    for c in _BROWSERS:
        if Path(c).exists():
            return c
    for n in ("msedge", "google-chrome", "chromium", "chrome"):
        if shutil.which(n):
            return shutil.which(n)
    raise SystemExit("Edge/Chrome을 찾지 못했습니다.")


def page(body, w):
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{CSS}'
            f'body{{width:{w}px}}</style></head><body>{body}</body></html>')


def render(html, out, w, h):
    """html 문자열을 w×h(CSS px) 화면으로 열어 2배 해상도 PNG로 저장."""
    out = Path(out)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "fig.html"
        src.write_text(html, encoding="utf-8")
        subprocess.run([_browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", f"--window-size={w},{h}",
                        "--virtual-time-budget=3000", f"--screenshot={out}", src.as_uri()],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    if not out.exists():
        raise SystemExit(f"렌더 실패: {out}")
    return out


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
