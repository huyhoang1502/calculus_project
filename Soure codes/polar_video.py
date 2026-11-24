"""
Polar Curves - Simulation Video (Calculus 1, MT1003, Group 5, HCMUT)
Render one scene:   py -m manim -qm polar_video.py TangentProof
Render everything:  .\render_all.ps1 m        (l = low, m = medium, h = high)
"""
from manim import *
import numpy as np

config.background_color = "#0e1124"

# ------------------------------------------------------------------ palette
C_CARD = "#ff6b6b"   # cardioid
C_ROSE = "#4dd0e1"   # rose
C_SPIR = "#ffd166"   # spiral / highlighted point
C_TAN = "#7bed9f"    # tangent line
C_AREA = "#6c8cff"   # area / sectors
C_WARN = "#ff9f43"   # warnings, limits
GRID = "#6b7394"

MEMBERS = ["Âu Hoàng Huy", "Trần Đức Nhân", "Yìu Văn Thăng", "Hoàng Bảo Sơn", "Nguyễn Tuệ Minh"]  
SUPERVISOR = "Trương Văn Trí"

_trap = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------------ the curves used in the video
def card(t):  return 1 + np.cos(t)          # cardioid r = 1 + cos(theta)
def dcard(t): return -np.sin(t)
def rose(t):  return np.cos(2 * t)          # rose r = cos(2 theta)
def drose(t): return -2 * np.sin(2 * t)

A_S = 0.5                                    # nautilus spiral r = a e^(b theta), a in cm
B_S = np.log((1 + np.sqrt(5)) / 2) / (np.pi / 2)   # golden ratio per quarter turn
def nautilus(t): return A_S * np.exp(B_S * t)


def slope(f, df, th):
    """dy/dx of r = f(theta) at theta (the formula proved in Result 1)."""
    num = df(th) * np.sin(th) + f(th) * np.cos(th)
    den = df(th) * np.cos(th) - f(th) * np.sin(th)
    return float(np.clip(num / den, -99, 99)) if abs(den) > 1e-12 else 99.0


def trap_n(fun, a, b, n):
    t = np.linspace(a, b, n + 1)
    return float(_trap(fun(t), t))


# ------------------------------------------------------------------ helpers
class Frame:
    """A polar grid placed anywhere on screen. pt(r, theta) -> screen point."""

    def __init__(self, R=2.0, size=5.0, center=ORIGIN, step=0.5, spokes=24):
        self.R, self.u = R, size / (2 * R)       # u = screen units per unit of r
        self.c = np.array(center, dtype=float)
        self.step, self.spokes = step, spokes

    def pt(self, r, th):
        return self.c + self.u * r * np.array([np.cos(th), np.sin(th), 0.0])

    def grid(self):
        g = VGroup()
        for k in range(1, int(round(self.R / self.step)) + 1):
            g.add(Circle(radius=k * self.step * self.u, stroke_width=1,
                         stroke_color=GRID, stroke_opacity=0.55).move_to(self.c))
        for j in range(self.spokes):
            g.add(Line(self.c, self.pt(self.R, TAU * j / self.spokes), stroke_width=1,
                       stroke_color=GRID, stroke_opacity=0.4))
        g.add(Line(self.pt(self.R, PI), self.pt(self.R, 0), stroke_width=2, stroke_color=GREY_B))
        g.add(Line(self.pt(self.R, -PI / 2), self.pt(self.R, PI / 2), stroke_width=2, stroke_color=GREY_B))
        return g

    def curve(self, f, t0, t1, color=WHITE, width=4):
        t1 = max(t1, t0 + 0.03)
        return ParametricFunction(lambda t: self.pt(f(t), t), t_range=[t0, t1, 0.01],
                                  color=color, stroke_width=width)

    def wedge(self, r, a, b, color=C_AREA, opacity=0.6):
        """Thin circular sector of radius r between angles a and b."""
        pts = [self.c] + [self.pt(r, t) for t in np.linspace(a, b, 6)]
        return Polygon(*pts, stroke_width=1, stroke_color=WHITE, fill_color=color, fill_opacity=opacity)

    def region(self, f, a, b, color=C_AREA, opacity=0.4):
        """Region between the pole and the curve for theta in [a, b]."""
        pts = [self.c] + [self.pt(f(t), t) for t in np.linspace(a, b, 90)]
        return Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=opacity)

    def tangent(self, f, df, th, half=1.3, color=C_TAN):
        r, dr = f(th), df(th)
        dx = dr * np.cos(th) - r * np.sin(th)
        dy = dr * np.sin(th) + r * np.cos(th)
        n = np.hypot(dx, dy)
        d = np.array([dx, dy, 0.0]) / n if n > 1e-9 else np.array([-np.sin(th), np.cos(th), 0.0])
        p = self.pt(r, th)
        return Line(p - half * d, p + half * d, color=color, stroke_width=4)


def make_sectors(F, f, n, a, b, c1=C_AREA, c2=C_ROSE):
    cols = color_gradient([c1, c2], n)
    d = (b - a) / n
    return VGroup(*[F.wedge(abs(f(a + (i + 0.5) * d)), a + i * d, a + (i + 1) * d, cols[i]) for i in range(n)])


def heading(text):
    return Text(text, font_size=28, color=GREY_B).to_corner(UL, buff=0.35)


def fit(m, w):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def stack(items, width, anchor):
    """Arrange mobjects in a left-aligned column, fit to width, put top-left at anchor."""
    g = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
    fit(g, width)
    g.move_to(anchor, aligned_edge=UL)
    return g


class TimedScene(Scene):
    DURATION = 10

    def pad_to(self, seconds=None):
        left = (seconds or self.DURATION) - self.renderer.time
        if left > 0.01:
            self.wait(left)

    def wipe(self, keep=(), run_time=0.8):
        gone = [m for m in self.mobjects if all(m is not k for k in keep)]
        for m in gone:
            m.clear_updaters()
        if gone:
            self.play(FadeOut(*gone), run_time=run_time)


# ================================================================== 0:00-0:20  HOOK
class Hook(TimedScene):
    DURATION = 20

    def construct(self):
        F = Frame(R=2.0, size=5.6, center=ORIGIN, step=0.5)
        grid = F.grid()
        self.play(FadeIn(grid), run_time=1.5)
        specs = [
            (F.curve(card, 0, TAU, C_CARD), "Cardioid", C_CARD),
            (F.curve(lambda t: 1.9 * np.cos(2 * t), 0, TAU, C_ROSE), "Rose", C_ROSE),
            (F.curve(lambda t: 0.15 * t, 0, 4 * PI, C_SPIR), "Spiral", C_SPIR),
        ]
        curves = []
        for cv, name, col in specs:
            lab = Text(name, font_size=32, color=col).move_to(RIGHT * 4.6)
            self.play(Create(cv), FadeIn(lab), run_time=3.2, rate_func=smooth)
            self.play(FadeOut(lab), run_time=0.4)
            curves.append(cv)
        title = Text("Polar Curves", font_size=72, weight=BOLD)
        sub = Text("Tangents · Area · Arc length", font_size=30, color=GREY_B).next_to(title, DOWN, buff=0.35)
        self.play(FadeOut(*curves), FadeOut(grid), Write(title), run_time=2)
        self.play(FadeIn(sub), run_time=1)
        self.pad_to()


# ================================================================== 0:20-0:40  BACKGROUND
class Background(TimedScene):
    DURATION = 20

    def construct(self):
        self.add(heading("Background and problem"))
        F = Frame(R=2.0, size=5.0, center=LEFT * 3.0, step=0.5)
        r0, t0 = 1.6, 0.9
        P = F.pt(r0, t0)
        self.play(FadeIn(F.grid()), run_time=1)

        dot = Dot(P, color=C_SPIR, radius=0.09)
        rad = Line(F.c, P, color=C_SPIR, stroke_width=4)
        arc = Arc(radius=0.55, start_angle=0, angle=t0, arc_center=F.c, color=WHITE, stroke_width=3)
        lr = MathTex("r", color=C_SPIR).scale(0.8).move_to(F.pt(r0 / 2, t0) + 0.3 * np.array([-np.sin(t0), np.cos(t0), 0]))
        lt = MathTex(r"\theta").scale(0.7).move_to(F.c + 0.9 * np.array([np.cos(t0 / 2), np.sin(t0 / 2), 0]))
        lp = MathTex(r"P(r,\theta)").scale(0.7).next_to(dot, UR, buff=0.1)
        self.play(Create(rad), FadeIn(dot), FadeIn(lr), run_time=1.5)
        self.play(Create(arc), FadeIn(lt), FadeIn(lp), run_time=1.5)

        fx, fy = F.pt(r0 * np.cos(t0), 0), F.pt(r0 * np.sin(t0), PI / 2)
        dx_, dy_ = DashedLine(P, fx, color=GREY_B), DashedLine(P, fy, color=GREY_B)
        lx = MathTex("x").scale(0.7).next_to(fx, DOWN, buff=0.1)
        ly = MathTex("y").scale(0.7).next_to(fy, LEFT, buff=0.1)
        self.play(Create(dx_), Create(dy_), FadeIn(lx), FadeIn(ly), run_time=2)

        e1 = MathTex(r"x=r\cos\theta", color=C_TAN)
        e2 = MathTex(r"y=r\sin\theta", color=C_TAN)
        e3 = MathTex(r"r^{2}=x^{2}+y^{2},\quad \tan\theta=\tfrac{y}{x}")
        eqs = VGroup(e1, e2, e3).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(RIGHT * 3.4 + UP * 0.6)
        for e in (e1, e2, e3):
            self.play(Write(e), run_time=1.4)
        q = Text("How do we get slope, area and length in polar form?", font_size=30, color=C_SPIR)
        q.move_to(DOWN * 3.2)
        self.play(FadeIn(q, shift=UP * 0.2), run_time=1.5)
        self.pad_to()


# ================================================================== 0:40-1:05  DEFINITIONS
class Definitions(TimedScene):
    DURATION = 25

    def construct(self):
        self.add(heading("Definitions and hypotheses"))
        F = Frame(R=2.0, size=5.0, center=LEFT * 3.2, step=0.5)
        self.play(FadeIn(F.grid()), run_time=1)
        th = ValueTracker(0.02)

        path = always_redraw(lambda: F.curve(card, 0.0, th.get_value(), C_CARD, 4))
        dot = always_redraw(lambda: Dot(F.pt(card(th.get_value()), th.get_value()), color=C_SPIR, radius=0.09))
        rad = always_redraw(lambda: Line(F.c, F.pt(card(th.get_value()), th.get_value()), color=C_SPIR, stroke_width=3))

        eq = MathTex(r"r=f(\theta),\qquad \alpha\le\theta\le\beta").move_to(RIGHT * 3.4 + UP * 2.2)
        self.play(Write(eq), run_time=1.5)

        # slider
        nl = NumberLine(x_range=[0, TAU, PI / 2], length=5.2, include_ticks=True, color=GREY_B).move_to(RIGHT * 3.4 + DOWN * 1.6)
        marks = VGroup(*[MathTex(s).scale(0.6).next_to(nl.n2p(v), DOWN, buff=0.15)
                         for s, v in [("0", 0), (r"\pi", PI), (r"2\pi", TAU)]])
        sl = always_redraw(lambda: Dot(nl.n2p(th.get_value()), color=C_SPIR, radius=0.1))
        lab = MathTex(r"\theta=").scale(0.9).move_to(RIGHT * 2.4 + DOWN * 0.8)
        num = DecimalNumber(0.0, num_decimal_places=2, font_size=36)
        num.add_updater(lambda m: m.set_value(th.get_value()).next_to(lab, RIGHT, buff=0.15))
        self.add(path, rad, dot, sl, num)
        self.play(FadeIn(nl), FadeIn(marks), FadeIn(lab), run_time=1)

        c1 = Text("f continuous: the curve has no jumps", font_size=24, color=C_TAN).move_to(RIGHT * 3.4 + UP * 1.2)
        c2 = Text("f differentiable: a tangent exists", font_size=24, color=C_WARN).move_to(RIGHT * 3.4 + UP * 0.5)
        self.play(FadeIn(c1), run_time=1)
        self.play(th.animate.set_value(PI), run_time=4, rate_func=linear)
        self.play(FadeIn(c2), run_time=1)
        self.play(th.animate.set_value(TAU - 0.02), run_time=4, rate_func=linear)
        num.clear_updaters()
        self.pad_to()


# ================================================================== 1:05-2:05  RESULT 1: TANGENT
class TangentProof(TimedScene):
    DURATION = 40

    def construct(self):
        head = heading("Result 1: slope of the tangent")
        self.add(head)
        F = Frame(R=2.0, size=4.6, center=RIGHT * 3.7, step=0.5)
        grid = F.grid()
        curve = F.curve(card, 0, TAU, C_CARD)
        self.play(FadeIn(grid), Create(curve), run_time=2.5)

        steps = [
            ("Step 1. Parametrise by θ (f must be differentiable)",
             r"x=f(\theta)\cos\theta,\qquad y=f(\theta)\sin\theta"),
            ("Step 2. Product rule",
             r"\frac{dx}{d\theta}=f'\cos\theta-f\sin\theta,\qquad \frac{dy}{d\theta}=f'\sin\theta+f\cos\theta"),
            ("Step 3. Chain rule: dy/dx = (dy/dθ) / (dx/dθ), if dx/dθ ≠ 0",
             r"\frac{dy}{dx}=\frac{f'(\theta)\sin\theta+f(\theta)\cos\theta}{f'(\theta)\cos\theta-f(\theta)\sin\theta}"),
        ]
        blocks = []
        for lab, tex in steps:
            l = Text(lab, font_size=21, color=GREY_B)
            m = MathTex(tex)
            fit(m, 7.0)
            blocks.append(VGroup(l, m).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        col = VGroup(*blocks).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        fit(col, 7.2)
        col.move_to(LEFT * 6.9 + UP * 2.6, aligned_edge=UL)

        for b in blocks:
            self.play(FadeIn(b[0]), run_time=0.5)
            self.play(Write(b[1]), run_time=2.0)
            self.wait(1.2)
        box = SurroundingRectangle(blocks[2][1], color=C_TAN, buff=0.1)
        self.play(Create(box), run_time=0.6)

        # tangent line slides along the curve
        tr = ValueTracker(0.3)
        tang = always_redraw(lambda: F.tangent(card, dcard, tr.get_value()))
        pdot = always_redraw(lambda: Dot(F.pt(card(tr.get_value()), tr.get_value()), color=C_SPIR, radius=0.09))
        sl_lab = MathTex(r"\frac{dy}{dx}=").scale(0.8).move_to(RIGHT * 2.9 + DOWN * 3.0)
        sl_num = DecimalNumber(0.0, num_decimal_places=2, font_size=34)
        sl_num.add_updater(lambda m: m.set_value(slope(card, dcard, tr.get_value())).next_to(sl_lab, RIGHT, buff=0.15))
        self.add(tang, pdot, sl_num)
        self.play(FadeIn(sl_lab), run_time=0.5)
        self.play(tr.animate.set_value(6.0), run_time=6, rate_func=linear)
        self.play(tr.animate.set_value(PI / 2), run_time=1.5)
        sl_num.clear_updaters()

        # ---- Example 1: cardioid at theta = pi/2
        self.play(FadeOut(col), FadeOut(box), run_time=0.7)
        fin = MathTex(r"\frac{dy}{dx}=\frac{f'\sin\theta+f\cos\theta}{f'\cos\theta-f\sin\theta}", color=C_TAN).scale(0.75)
        fin.move_to(LEFT * 6.9 + UP * 2.6, aligned_edge=UL)
        h1 = Text("Example 1: cardioid r = 1 + cos θ at θ = π/2", font_size=24, color=C_CARD)
        a1 = MathTex(r"f=1,\qquad f'=-\sin\tfrac{\pi}{2}=-1")
        a2 = MathTex(r"\frac{dy}{dx}=\frac{(-1)(1)+(1)(0)}{(-1)(0)-(1)(1)}=\frac{-1}{-1}=1", color=C_SPIR)
        ex1 = stack([h1, a1, a2], 7.0, LEFT * 6.9 + UP * 1.2)
        self.play(FadeIn(fin), run_time=0.6)
        self.play(FadeIn(h1), run_time=0.6)
        self.play(Write(a1), run_time=1.5)
        self.play(Write(a2), run_time=2.5)
        self.wait(2.0)

        # ---- Example 2: rose r = cos 2theta at theta = pi/6
        self.play(FadeOut(ex1), run_time=0.6)
        self.wipe(keep=[head, fin], run_time=0.8)
        F2 = Frame(R=1.0, size=4.6, center=RIGHT * 3.7, step=0.25)
        rcurve = F2.curve(rose, 0, TAU, C_ROSE)
        self.play(FadeIn(F2.grid()), Create(rcurve), run_time=2)
        t2 = PI / 6
        tg2 = F2.tangent(rose, drose, t2, half=0.9)
        d2 = Dot(F2.pt(rose(t2), t2), color=C_SPIR, radius=0.09)
        h2 = Text("Example 2: rose r = cos 2θ at θ = π/6", font_size=24, color=C_ROSE)
        b1 = MathTex(r"f=\tfrac12,\qquad f'=-2\sin\tfrac{\pi}{3}=-\sqrt3")
        b2 = MathTex(r"\frac{dy}{dx}=\frac{-\frac{\sqrt3}{4}}{-\frac74}=\frac{\sqrt3}{7}\approx 0.247", color=C_SPIR)
        ex2 = stack([h2, b1, b2], 7.0, LEFT * 6.9 + UP * 1.2)
        self.play(FadeIn(h2), run_time=0.6)
        self.play(Write(b1), run_time=1.5)
        self.play(Write(b2), Create(tg2), FadeIn(d2), run_time=2.5)
        self.pad_to()


# ================================================================== 2:05-3:05  RESULT 2: AREA
class AreaProof(TimedScene):
    DURATION = 40

    def construct(self):
        head = heading("Result 2: area in polar form")
        self.add(head)
        F = Frame(R=2.0, size=4.8, center=LEFT * 3.4, step=0.5)
        outline = F.curve(card, 0, TAU, C_CARD)
        self.play(FadeIn(F.grid()), Create(outline), run_time=2)

        # sectors: N = 6 -> 18 -> 60
        s6 = make_sectors(F, card, 6, 0, TAU)
        s18 = make_sectors(F, card, 18, 0, TAU)
        s60 = make_sectors(F, card, 60, 0, TAU)
        nlab = Text("N = 6", font_size=28, color=C_SPIR).move_to(LEFT * 3.4 + DOWN * 2.9)
        self.play(LaggedStart(*[FadeIn(s) for s in s6], lag_ratio=0.15), FadeIn(nlab), run_time=2.5)
        self.play(Transform(s6, s18), Transform(nlab, Text("N = 18", font_size=28, color=C_SPIR).move_to(nlab)), run_time=1.8)
        self.play(Transform(s6, s60), Transform(nlab, Text("N = 60", font_size=28, color=C_SPIR).move_to(nlab)), run_time=1.8)

        steps = [
            ("One sector (radius r, angle Δθ):", r"\Delta A\approx\tfrac12\,r^{2}\,\Delta\theta"),
            ("Add all N sectors (f continuous):", r"A\approx\sum_{i=1}^{N}\tfrac12\,[f(\theta_i^{*})]^{2}\,\Delta\theta"),
            ("Let N → ∞ (Riemann sum → integral):", r"A=\tfrac12\int_{\alpha}^{\beta}[f(\theta)]^{2}\,d\theta"),
        ]
        blocks = []
        for lab, tex in steps:
            blocks.append(VGroup(Text(lab, font_size=22, color=GREY_B), MathTex(tex)).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        col = VGroup(*blocks).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        fit(col, 6.6)
        col.move_to(RIGHT * 0.2 + UP * 2.4, aligned_edge=UL)
        for b in blocks:
            self.play(FadeIn(b[0]), Write(b[1]), run_time=2.2)
            self.wait(0.8)
        box = SurroundingRectangle(blocks[2][1], color=C_TAN, buff=0.1)
        self.play(Create(box), run_time=0.6)
        self.wait(1.0)

        # ---- Example 1: cardioid
        self.play(FadeOut(col), FadeOut(box), run_time=0.7)
        h1 = Text("Example 1: cardioid r = 1 + cos θ, 0 ≤ θ ≤ 2π", font_size=24, color=C_CARD)
        l1 = MathTex(r"A=\tfrac12\int_0^{2\pi}(1+2\cos\theta+\cos^{2}\theta)\,d\theta")
        l2 = MathTex(r"=\tfrac12\,(2\pi+0+\pi)=\tfrac{3\pi}{2}\approx 4.712", color=C_SPIR)
        ex1 = stack([h1, l1, l2], 6.2, RIGHT * 0.1 + UP * 2.0)
        self.play(FadeIn(h1), run_time=0.6)
        self.play(Write(l1), run_time=2.2)
        self.play(Write(l2), run_time=2.2)
        self.wait(2.5)

        # ---- Example 2: one rose petal
        self.play(FadeOut(ex1), run_time=0.6)
        self.wipe(keep=[head], run_time=0.8)
        F2 = Frame(R=1.0, size=4.8, center=LEFT * 3.4, step=0.25)
        rose_all = F2.curve(rose, 0, TAU, C_ROSE, 3)
        self.play(FadeIn(F2.grid()), Create(rose_all), run_time=2)
        petal = make_sectors(F2, rose, 40, -PI / 4, PI / 4)
        self.play(LaggedStart(*[FadeIn(s) for s in petal], lag_ratio=0.05), run_time=2.5)
        h2 = Text("Example 2: one petal of r = cos 2θ, −π/4 ≤ θ ≤ π/4", font_size=24, color=C_ROSE)
        m1 = MathTex(r"A=\tfrac12\int_{-\pi/4}^{\pi/4}\cos^{2}2\theta\,d\theta=\tfrac14\int_{-\pi/4}^{\pi/4}(1+\cos4\theta)\,d\theta")
        m2 = MathTex(r"=\tfrac14\cdot\tfrac{\pi}{2}=\tfrac{\pi}{8}\approx 0.393", color=C_SPIR)
        ex2 = stack([h2, m1, m2], 6.2, RIGHT * 0.1 + UP * 2.0)
        self.play(FadeIn(h2), run_time=0.6)
        self.play(Write(m1), run_time=2.5)
        self.play(Write(m2), run_time=2.0)
        self.pad_to()


# ================================================================== 3:05-3:25  ARC LENGTH (extra)
class ArcLength(TimedScene):
    DURATION = 20

    def construct(self):
        self.add(heading("Extra: arc length"))
        F = Frame(R=2.0, size=4.8, center=LEFT * 3.4, step=0.5)
        self.play(FadeIn(F.grid()), run_time=1)

        f1 = MathTex(r"ds=\sqrt{dx^{2}+dy^{2}}=\sqrt{r^{2}+(r')^{2}}\,d\theta")
        f2 = MathTex(r"L=\int_{\alpha}^{\beta}\sqrt{f(\theta)^{2}+f'(\theta)^{2}}\,d\theta", color=C_TAN)
        top = stack([f1, f2], 6.6, RIGHT * 0.2 + UP * 2.6)
        self.play(Write(f1), run_time=2)
        self.play(Write(f2), run_time=2)

        th = ValueTracker(0.02)
        path = always_redraw(lambda: F.curve(card, 0.0, th.get_value(), C_CARD, 4))
        dot = always_redraw(lambda: Dot(F.pt(card(th.get_value()), th.get_value()), color=C_SPIR, radius=0.09))

        def L_of(t):
            return 4 * np.sin(t / 2) if t <= PI else 8 - 4 * np.sin(t / 2)

        lab = MathTex(r"L(\theta)=").scale(0.9).move_to(RIGHT * 1.6 + DOWN * 0.6)
        num = DecimalNumber(0.0, num_decimal_places=3, font_size=40, color=C_SPIR)
        num.add_updater(lambda m: m.set_value(L_of(th.get_value())).next_to(lab, RIGHT, buff=0.15))
        self.add(path, dot, num)
        self.play(FadeIn(lab), run_time=0.4)
        self.play(th.animate.set_value(TAU - 0.001), run_time=5, rate_func=linear)
        num.clear_updaters()

        g1 = MathTex(r"r^{2}+r'^{2}=2+2\cos\theta=4\cos^{2}\tfrac{\theta}{2}")
        g2 = MathTex(r"L=\int_0^{2\pi}2\left|\cos\tfrac{\theta}{2}\right|d\theta=8", color=C_SPIR)
        bot = stack([g1, g2], 6.6, RIGHT * 0.2 + DOWN * 1.2)
        self.play(Write(g1), run_time=1.5)
        self.play(Write(g2), run_time=1.8)
        self.pad_to()


# ================================================================== 3:25-3:55  COUNTEREXAMPLE
class Counterexample(TimedScene):
    DURATION = 30

    def construct(self):
        self.add(heading("Counterexample: the curve must be traced once"))
        F = Frame(R=1.2, size=5.0, center=LEFT * 3.2, step=0.3)
        self.play(FadeIn(F.grid()), run_time=1)
        f = np.cos

        th = ValueTracker(0.02)
        seg1 = always_redraw(lambda: F.curve(f, 0.0, min(th.get_value(), PI), C_ROSE, 5))
        seg2 = always_redraw(lambda: F.curve(f, PI, th.get_value(), C_CARD, 3) if th.get_value() > PI + 0.05 else VGroup())
        dot = always_redraw(lambda: Dot(F.pt(f(th.get_value()), th.get_value()), color=C_SPIR, radius=0.09))
        eq = MathTex(r"r=\cos\theta,\quad 0\le\theta\le 2\pi").move_to(RIGHT * 3.4 + UP * 2.4)
        self.add(seg1, seg2, dot)
        self.play(Write(eq), run_time=1.2)
        p1 = Text("pass 1: 0 ≤ θ ≤ π", font_size=22, color=C_ROSE).move_to(RIGHT * 3.4 + UP * 1.6)
        p2 = Text("pass 2: π ≤ θ ≤ 2π (same circle again!)", font_size=22, color=C_CARD).move_to(RIGHT * 3.4 + UP * 1.1)
        self.play(FadeIn(p1), th.animate.set_value(PI), run_time=3.5, rate_func=linear)
        self.play(FadeIn(p2), th.animate.set_value(TAU - 0.01), run_time=3.5, rate_func=linear)

        n1 = MathTex(r"\tfrac12\int_0^{2\pi}\cos^{2}\theta\,d\theta=\tfrac{\pi}{2}", color=C_WARN).scale(0.7)
        n1l = Text("Formula as stated:", font_size=19, color=GREY_B)
        t1 = MathTex(r"\text{circle of radius }\tfrac12:\quad A=\pi\cdot\tfrac14=\tfrac{\pi}{4}", color=C_TAN).scale(0.7)
        t1l = Text("True area:", font_size=19, color=GREY_B)
        k1 = MathTex(r"\tfrac12\int_0^{\pi}\cos^{2}\theta\,d\theta=\tfrac{\pi}{4}", color=C_TAN).scale(0.7)
        k1l = Text("Trace once (θ from 0 to π):", font_size=19, color=GREY_B)
        rows = [VGroup(label, formula).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
                for label, formula in [(n1l, n1), (t1l, t1), (k1l, k1)]]
        VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(
            RIGHT * 0.2 + UP * 0.72, aligned_edge=UL
        )
        self.play(FadeIn(n1l), Write(n1), run_time=2)
        self.play(FadeIn(t1l), Write(t1), run_time=2)
        self.play(FadeIn(k1l), Write(k1), run_time=2)
        warn = Text("Hypothesis needed: traced exactly once on [α, β]", font_size=24, color=C_WARN).move_to(DOWN * 3.3)
        self.play(FadeIn(warn), run_time=1)
        self.pad_to()


# ================================================================== 3:55-4:30  CASE 1: ANTENNA
class CaseAntenna(TimedScene):
    DURATION = 35

    def construct(self):
        self.add(heading("Case study 1: antenna radiation pattern"))
        F = Frame(R=2.0, size=4.8, center=LEFT * 3.2, step=0.5)
        th0 = float(np.arccos(np.sqrt(2) - 1))              # half-power angle
        t = np.linspace(-PI, PI, 20001)
        A_tot = 0.5 * float(_trap(card(t) ** 2, t))
        tb = np.linspace(-th0, th0, 20001)
        A_beam = 0.5 * float(_trap(card(tb) ** 2, tb))
        s0 = slope(card, dcard, th0)

        lobe = F.curve(card, -PI, PI, C_CARD)
        self.play(FadeIn(F.grid()), Create(lobe), run_time=3)

        model = VGroup(
            Text("Model: 2-D far-field cut, normalised field E(θ) = 1 + cos θ", font_size=21),
            Text("Power per unit length ∝ ∫E² dθ = 2A (polar area)", font_size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        fit(model, 6.8)
        model.move_to(RIGHT * 0.2 + UP * 2.8, aligned_edge=UL)
        self.play(FadeIn(model), run_time=1.5)

        # beam wedge, rays and tangent at the half-power angle
        beam = F.region(card, -th0, th0, C_AREA, 0.4)
        ray_a = DashedLine(F.c, F.pt(2.0, th0), color=C_SPIR)
        ray_b = DashedLine(F.c, F.pt(2.0, -th0), color=C_SPIR)
        tg = F.tangent(card, dcard, th0, half=0.9)
        pd = Dot(F.pt(card(th0), th0), color=C_SPIR, radius=0.09)
        self.play(FadeIn(beam), Create(ray_a), Create(ray_b), run_time=2)
        self.play(Create(tg), FadeIn(pd), run_time=1.2)

        lines = [
            MathTex(r"E(\theta_0)=\tfrac{E_{\max}}{\sqrt2}\;\Rightarrow\;1+\cos\theta_0=\sqrt2"),
            MathTex(rf"\theta_0={np.degrees(th0):.1f}^\circ,\quad \mathrm{{HPBW}}=2\theta_0={2 * np.degrees(th0):.1f}^\circ", color=C_SPIR),
            MathTex(rf"\left.\frac{{dy}}{{dx}}\right|_{{\theta_0}}={s0:.3f}\quad(\text{{tangent formula}})", color=C_TAN),
            MathTex(rf"\frac{{P_{{\mathrm{{beam}}}}}}{{P_{{\mathrm{{tot}}}}}}=\frac{{{A_beam:.3f}}}{{{A_tot:.3f}}}={100 * A_beam / A_tot:.1f}\%", color=C_SPIR),
        ]
        col = stack(lines, 6.2, RIGHT * 0.4 + UP * 1.7)
        for m in lines:
            self.play(Write(m), run_time=2.0)
        warn = Text("Valid: far field, one frequency, lossless. Real antennas are 3-D.",
                    font_size=19, color=C_WARN).move_to(RIGHT * 3.4 + DOWN * 3.3)
        self.play(FadeIn(warn), run_time=1)
        self.pad_to()


# ================================================================== 4:30-5:05  CASE 2: ORBIT
def kepler_theta(M, e):
    """True anomaly theta (0..2pi) from mean anomaly M (Kepler's equation by Newton)."""
    E = M
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return float(2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2)))


class CaseOrbit(TimedScene):
    DURATION = 35

    def construct(self):
        self.add(heading("Case study 2: Mars orbit (Kepler's second law)"))
        a, e_vis = 1.524, 0.55                      # a in AU; e exaggerated only for the picture
        p_vis = a * (1 - e_vis ** 2)
        rr = lambda t: p_vis / (1 + e_vis * np.cos(t))
        F = Frame(R=2.5, size=5.0, center=LEFT * 2.6, step=0.5)

        e_m = 0.0934                                # real Mars eccentricity
        p_m = a * (1 - e_m ** 2)
        tt = np.linspace(0, TAU, 40001)
        A_num = 0.5 * float(_trap((p_m / (1 + e_m * np.cos(tt))) ** 2, tt))
        A_ex = np.pi * a * a * np.sqrt(1 - e_m ** 2)
        T_yr = a ** 1.5

        thetas = [0.0] + [kepler_theta(TAU * k / 8, e_vis) for k in range(1, 8)] + [TAU]
        areas = []
        for k in range(8):
            x = np.linspace(thetas[k], thetas[k + 1], 4001)
            areas.append(0.5 * float(_trap(rr(x) ** 2, x)))

        sun = Dot(F.c, color=C_SPIR, radius=0.14)
        sun_l = Text("Sun", font_size=20, color=C_SPIR).next_to(sun, DOWN, buff=0.1)
        orbit = F.curve(rr, 0, TAU, C_CARD)
        self.play(FadeIn(F.grid()), Create(orbit), FadeIn(sun), FadeIn(sun_l), run_time=3)

        it = [
            MathTex(r"r=\dfrac{p}{1+e\cos\theta},\quad p=a(1-e^{2})"),
            Text(f"Mars: a = 1.524 AU, e = 0.0934  →  p = {p_m:.3f} AU", font_size=20),
            Text("(picture drawn with e = 0.55 so the effect is visible)", font_size=17, color=GREY_B),
        ]
        col = stack(it, 5.7, RIGHT * 0.7 + UP * 2.7)
        for m in it:
            self.play(FadeIn(m), run_time=1.2)

        M = ValueTracker(0.0)
        pos = lambda: kepler_theta(M.get_value(), e_vis)
        planet = always_redraw(lambda: Dot(F.pt(rr(pos()), pos()), color=C_ROSE, radius=0.1))
        link = always_redraw(lambda: Line(F.c, F.pt(rr(pos()), pos()), color=C_ROSE, stroke_width=2))
        self.add(link, planet)
        cols = color_gradient([C_AREA, C_ROSE], 8)
        for k in range(8):
            sec = F.region(rr, thetas[k], thetas[k + 1], cols[k], 0.45)
            self.play(M.animate.set_value(TAU * (k + 1) / 8), FadeIn(sec), run_time=0.9, rate_func=linear)

        r1 = MathTex(r"\frac{dA}{dt}=\tfrac12 r^{2}\dot\theta=\frac h2=\text{const}", color=C_TAN)
        r2 = MathTex(rf"\Delta t=\tfrac T8:\quad A_1={areas[0]:.3f},\ A_5={areas[4]:.3f}\ \mathrm{{AU}}^2", color=C_SPIR)
        r3 = MathTex(rf"\tfrac12\int_0^{{2\pi}}r^{{2}}d\theta={A_num:.5f}\ \approx\ \pi ab={A_ex:.5f}\ \mathrm{{AU}}^2")
        col2 = stack([r1, r2, r3], 5.7, RIGHT * 0.7 + UP * 0.6)
        self.play(Write(r1), run_time=1.8)
        self.play(Write(r2), run_time=1.8)
        self.play(Write(r3), run_time=1.8)
        note = Text(f"Kepler III: T = a^1.5 = {T_yr:.2f} yr.  Valid: 2-body, no perturbations.",
                    font_size=18, color=C_WARN)
        fit(note, 6.0).move_to(RIGHT * 3.4 + DOWN * 3.3)
        self.play(FadeIn(note), run_time=0.8)
        self.pad_to()


# ================================================================== 5:05-5:40  CASE 3: NAUTILUS SPIRAL
class CaseSpiral(TimedScene):
    DURATION = 35

    def construct(self):
        self.add(heading("Case study 3: natural spiral (nautilus)"))
        T1 = 3 * PI
        F = Frame(R=10.0, size=5.0, center=LEFT * 3.2, step=2.5)       # radius in cm
        r_end = nautilus(T1)
        L_closed = np.sqrt(1 + B_S ** 2) / B_S * (r_end - A_S)
        tt = np.linspace(0, T1, 200001)
        L_num = float(_trap(np.sqrt(nautilus(tt) ** 2 + (B_S * nautilus(tt)) ** 2), tt))
        psi = np.degrees(np.arctan(1 / B_S))

        spiral = F.curve(nautilus, 0, T1, C_SPIR, 4)
        spokes = VGroup(*[DashedLine(F.c, F.pt(nautilus(k * PI / 2), k * PI / 2), color=GREY_B, stroke_width=1.5)
                          for k in range(0, 7)])
        self.play(FadeIn(F.grid()), run_time=1)
        self.play(Create(spiral), run_time=5)
        self.play(FadeIn(spokes), run_time=1)

        it = [
            MathTex(rf"r=a\,e^{{b\theta}},\quad a=0.5\,\mathrm{{cm}},\ b=\tfrac{{\ln\varphi}}{{\pi/2}}={B_S:.4f}"),
            Text("Each quarter turn multiplies r by φ ≈ 1.618 (golden-ratio model)", font_size=19, color=GREY_B),
            MathTex(rf"r'=br\ \Rightarrow\ \tan\psi=\tfrac{{r}}{{r'}}=\tfrac1b\ \Rightarrow\ \psi={psi:.1f}^\circ\ (\text{{constant}})", color=C_TAN),
            MathTex(r"L=\int_0^{3\pi}\!\sqrt{r^{2}+r'^{2}}\,d\theta=\frac{\sqrt{1+b^{2}}}{b}\,[r(3\pi)-r(0)]"),
            MathTex(rf"L={L_closed:.2f}\ \mathrm{{cm}}\quad(\text{{numeric: }}{L_num:.2f}\ \mathrm{{cm}})", color=C_SPIR),
        ]
        col = stack(it, 6.1, RIGHT * 0.2 + UP * 2.7)
        for m in it:
            self.play(Write(m) if isinstance(m, MathTex) else FadeIn(m), run_time=2.0)
        warn = Text("Idealised: real shells vary in b and stop growing. Fit a, b from measurements.",
                    font_size=18, color=C_WARN).move_to(DOWN * 3.3)
        self.play(FadeIn(warn), run_time=1)
        self.pad_to()


# ================================================================== 5:40-6:10  NUMERICAL VERIFICATION
class Verification(TimedScene):
    DURATION = 30

    def construct(self):
        self.add(heading("Numerical verification (trapezoid rule, N subintervals)"))
        T1 = 3 * PI
        L_naut = np.sqrt(1 + B_S ** 2) / B_S * (nautilus(T1) - A_S)
        quantities = [
            ("Cardioid L", 8.0, lambda n: trap_n(lambda t: np.sqrt(card(t) ** 2 + dcard(t) ** 2), 0, TAU, n), C_CARD),
            ("Spiral r=θ, A", 4 * np.pi ** 3 / 3, lambda n: trap_n(lambda t: 0.5 * t ** 2, 0, TAU, n), C_ROSE),
            ("Nautilus L", L_naut,
             lambda n: trap_n(lambda t: np.sqrt(nautilus(t) ** 2 + (B_S * nautilus(t)) ** 2), 0, T1, n), C_SPIR),
        ]

        rows = []
        for name, exact, fn, _ in quantities:
            for n in (10, 100, 1000):
                val = fn(n)
                err = abs(val - exact)
                rows.append([name, str(n), f"{exact:.6f}", f"{val:.6f}", f"{err:.2e}", f"{err / abs(exact):.2e}"])
        labels = [Text(s, font_size=22, weight=BOLD) for s in ["Quantity", "N", "Exact", "Numeric", "Abs. error", "Rel. error"]]
        table = Table(rows, col_labels=labels, include_outer_lines=True,
                      element_to_mobject=lambda s: Text(s, font_size=22))
        table.scale_to_fit_width(12.4)
        if table.height > 5.3:
            table.scale_to_fit_height(5.3)
        table.move_to(DOWN * 0.3)
        self.play(Create(table.get_horizontal_lines()), Create(table.get_vertical_lines()), run_time=2)
        self.play(FadeIn(table.get_entries_without_labels()), FadeIn(table.get_col_labels()), run_time=2.5)
        self.wait(4)
        self.play(FadeOut(table), run_time=0.8)

        # convergence plot
        Ns = [10, 20, 40, 80, 160, 320, 640, 1280]
        axes = Axes(x_range=[1, 3.2, 0.5], y_range=[-11, 0, 2], x_length=8, y_length=4.2, tips=False,
                    axis_config={"include_numbers": True, "font_size": 22}).move_to(LEFT * 1.6 + DOWN * 0.4)
        xl = MathTex(r"\log_{10}N").scale(0.7).next_to(axes, DOWN, buff=0.2)
        yl = MathTex(r"\log_{10}(\text{relative error})").scale(0.6).next_to(axes, UP, buff=0.15)
        self.play(Create(axes), FadeIn(xl), FadeIn(yl), run_time=1.5)
        for i, (name, exact, fn, col) in enumerate(quantities):
            pts = [axes.c2p(np.log10(n), max(np.log10(max(abs(fn(n) - exact) / abs(exact), 1e-16)), -10.5)) for n in Ns]
            line = VMobject(color=col, stroke_width=4).set_points_as_corners(pts)
            dots = VGroup(*[Dot(p, color=col, radius=0.06) for p in pts])
            leg = Text(name, font_size=22, color=col).move_to(RIGHT * 5.3 + UP * (1.2 - 0.6 * i))
            self.play(Create(line), FadeIn(dots), FadeIn(leg), run_time=1.6)
        note = Text("Straight lines on log-log axes: error shrinks as a power of 1/N", font_size=22, color=C_TAN)
        note.move_to(DOWN * 3.3)
        self.play(FadeIn(note), run_time=1)
        self.pad_to()


# ================================================================== 6:10-6:30  CONCLUSION AND CREDITS
class Conclusion(TimedScene):
    DURATION = 20

    def construct(self):
        title = Text("Conclusion", font_size=40, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        rec = VGroup(
            VGroup(Text("Tangent", font_size=24, color=C_TAN),
                   MathTex(r"\frac{dy}{dx}=\frac{f'\sin\theta+f\cos\theta}{f'\cos\theta-f\sin\theta}")).arrange(RIGHT, buff=0.5),
            VGroup(Text("Area", font_size=24, color=C_AREA),
                   MathTex(r"A=\tfrac12\int_{\alpha}^{\beta}f(\theta)^{2}\,d\theta")).arrange(RIGHT, buff=0.5),
            VGroup(Text("Length", font_size=24, color=C_SPIR),
                   MathTex(r"L=\int_{\alpha}^{\beta}\sqrt{f^{2}+f'^{2}}\,d\theta")).arrange(RIGHT, buff=0.5),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        fit(rec, 11.5)
        rec.move_to(UP * 2.35, aligned_edge=UP)
        for r in rec:
            self.play(FadeIn(r, shift=RIGHT * 0.3), run_time=1.0)
        lim = Text("Limits: f continuous / differentiable, curve traced once, idealised models.",
                   font_size=22, color=C_WARN)
        fit(lim, 11.5).move_to(DOWN * 2.75)
        self.play(FadeIn(lim), run_time=1)
        self.wait(2)
        self.play(FadeOut(rec), FadeOut(lim), run_time=0.8)

        member_lines = VGroup(
            Text(" · ".join(MEMBERS[:3]), font_size=24, font="Arial", color=C_SPIR),
            Text(" · ".join(MEMBERS[3:]), font_size=24, font="Arial", color=C_SPIR),
        ).arrange(DOWN, buff=0.12)
        fit(member_lines, 11.5)
        credits = VGroup(
            Text("Group 5 · Calculus 1 (MT1003)", font_size=34, weight=BOLD),
            Text("Ho Chi Minh City University of Technology (HCMUT) – VNU-HCM", font_size=24, color=GREY_B),
            Text("Supervisor: " + SUPERVISOR, font_size=24, font="Arial"),
            member_lines,
            Text("Animation: Manim Community", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.3).move_to(DOWN * 0.2)
        self.play(FadeIn(credits, shift=UP * 0.2), run_time=1.5)
        self.pad_to()
