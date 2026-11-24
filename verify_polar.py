"""Numerical verification for the polar curves report (Group 5).
Exact value vs composite trapezoid rule with N subintervals."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trap = getattr(np, "trapezoid", None) or np.trapz
TAU = 2 * np.pi

def trap_n(fun, a, b, n):
    t = np.linspace(a, b, n + 1)
    return float(trap(fun(t), t))

card  = lambda t: 1 + np.cos(t)
dcard = lambda t: -np.sin(t)
rose  = lambda t: np.cos(2 * t)
B = np.log((1 + np.sqrt(5)) / 2) / (np.pi / 2)
A0 = 0.5
naut = lambda t: A0 * np.exp(B * t)

quantities = [
    ("Cardioid area, $r=1+\\cos\\theta$", 1.5 * np.pi,
     lambda n: trap_n(lambda t: 0.5 * card(t) ** 2, 0, TAU, n)),
    ("Cardioid length", 8.0,
     lambda n: trap_n(lambda t: np.sqrt(card(t) ** 2 + dcard(t) ** 2), 0, TAU, n)),
    ("Rose petal area, $r=\\cos 2\\theta$", np.pi / 8,
     lambda n: trap_n(lambda t: 0.5 * rose(t) ** 2, -np.pi / 4, np.pi / 4, n)),
    ("Archimedean spiral area, $r=\\theta$", 4 * np.pi ** 3 / 3,
     lambda n: trap_n(lambda t: 0.5 * t ** 2, 0, TAU, n)),
    ("Nautilus spiral length", np.sqrt(1 + B ** 2) / B * (naut(3 * np.pi) - A0),
     lambda n: trap_n(lambda t: np.sqrt(naut(t) ** 2 + (B * naut(t)) ** 2), 0, 3 * np.pi, n)),
]

rows = []
for name, ex, fn in quantities:
    for n in (10, 100, 1000):
        v = fn(n); e = abs(v - ex)
        rows.append((name, n, ex, v, e, e / abs(ex)))
        print(f"{name:45s} N={n:5d} exact={ex:.6f} num={v:.6f} abs={e:.2e} rel={e/abs(ex):.2e}")

Ns = [10, 20, 40, 80, 160, 320, 640, 1280]
plt.figure(figsize=(6.2, 3.8))
for (name, ex, fn), m in zip(quantities, "osd^v"):
    errs = [abs(fn(n) - ex) / abs(ex) for n in Ns]
    if max(errs) < 1e-12:      # exact to machine precision (periodic integrand): skip on log plot
        continue
    plt.loglog(Ns, errs, marker=m, label=name)
plt.loglog(Ns, [30 / n ** 2 for n in Ns], "k--", lw=1, label=r"slope $-2$ guide")
plt.xlabel("N (subintervals)"); plt.ylabel("relative error")
plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig("convergence.png", dpi=200)
