# main code
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Optical and crystal parameters
# ============================================================

L = 10e-3                    # Crystal length [m]

lambda_p = 775e-9            # Pump wavelength [m]
lambda_s = 1550e-9           # Signal wavelength [m]
lambda_i = 1550e-9           # Idler wavelength [m]

T_C = 32                     # Crystal temperature [°C]，另一套干涉儀溫度大概在 32°C
MFD1 = 5e-6                  # 出光 fiber MFD = 5 um
MFD2 = 10.4e-6               # 收光 fiber MFD = 10.4 um
f1 = 225e-3                  # PBS 前的 lens，聚焦用
f2 = 300e-3                  # PBS 後的 lens，用成平行光
f_coupler1 = 11e-3           # 出光 coupler f = 11mm
f_coupler2 = 18.4e-3         # 收光 coupler f = 18.4mm

wp0 = 2 * lambda_p * f_coupler1 / (np.pi * MFD1)
wp = lambda_p * f1 / (np.pi * wp0)

# 收光 coupler 逆推：此 coupler 要求的準直光 beam waist (radius)
# w_coupler = lambda * f / (pi * w0_fiber)，w0_fiber = MFD/2
w_coupler = 2 * lambda_s * f_coupler2 / (np.pi * MFD2)   # ≈ 1.746 mm


# ============================================================
# 2. Sellmeier coefficients
#
# Reference:
# O. Gayer, Z. Sacks, E. Galun, A. Arie,
# "Temperature and wavelength dependent refractive index
# equations for MgO-doped congruent and stoichiometric LiNbO3",
# Appl. Phys. B 91, 343–348 (2008).
#
# 5% MgO-doped congruent LiNbO3
# ============================================================

coeff_e = {
    "a1": 5.756,
    "a2": 0.0983,
    "a3": 0.2020,
    "a4": 189.32,
    "a5": 12.52,
    "a6": 1.32e-2,
    "b1": 2.860e-6,
    "b2": 4.700e-8,
    "b3": 6.113e-8,
    "b4": 1.516e-4
}

coeff_o = {
    "a1": 5.653,
    "a2": 0.1185,
    "a3": 0.2091,
    "a4": 89.61,
    "a5": 10.85,
    "a6": 1.97e-2,
    "b1": 7.941e-7,
    "b2": 3.134e-8,
    "b3": -4.641e-9,
    "b4": -2.188e-6
}


def refractive_index(lambda_um, T_C, coeff):
    """Refractive index from Gayer et al. Eq. (2)."""
    f = (T_C - 24.5) * (T_C + 570.82)
    n_squared = coeff["a1"] + coeff["b1"] * f + (coeff["a2"] + coeff["b2"] * f) / (lambda_um**2 - (coeff["a3"] + coeff["b3"] * f)**2) + (coeff["a4"] + coeff["b4"] * f) / (lambda_um**2 - coeff["a5"]**2) - coeff["a6"] * lambda_um**2
    return np.sqrt(n_squared)


def wave_number(n, wavelength):
    """Wave number inside the crystal: k = 2*pi*n/lambda."""
    return 2.0 * np.pi * n / wavelength


# ============================================================
# 4. Refractive indices
#
# Current polarization assignment:
# pump   : extraordinary
# signal : extraordinary
# idler  : ordinary
# ============================================================

n_p = refractive_index(lambda_p * 1e6, T_C, coeff_e)
n_s = refractive_index(lambda_s * 1e6, T_C, coeff_e)
n_i = refractive_index(lambda_i * 1e6, T_C, coeff_o)

ZRp = np.pi * wp * wp * n_p / lambda_p                 # Pump Rayleigh range inside the crystal [m]

def xi_from_ZR(L, ZR):
    """Bennink focal parameter: xi = L/(2*Z_R)."""
    return L / (2.0 * ZR)

print(n_p)
print(n_s)
print(n_i)


# ============================================================
# 5. Wave numbers inside the crystal
#
# k_j = 2*pi*n_j/lambda_j
# ============================================================

k_p = wave_number(n_p, lambda_p)
k_s = wave_number(n_s, lambda_s)
k_i = wave_number(n_i, lambda_i)


# ============================================================
# 6. Pump focusing parameter
#
# Bennink Eq. (11):
# xi_p = L/(k_p*w_p^2)
#
# Using Z_R = k*w^2/2:
# xi_p = L/(2*Z_Rp)
# ============================================================

xi_p = xi_from_ZR(L, ZRp)

# test (8.8, 54, 146)
# xi_p = 0.5*1/8.8


# ============================================================
# 7. Bennink equations
# ============================================================

def A_plus(k_p, k_s, k_i, xi_p, xi_s, xi_i):
    """Bennink Eq. (12): A_+"""
    return 1 + k_s * xi_s / (k_p * xi_p) + k_i * xi_i / (k_p * xi_p)


def B_plus(delta_k, k_p, k_s, k_i, xi_p, xi_s, xi_i):
    """Bennink Eq. (13): B_+"""
    return (1 - delta_k / k_p) * (1 + (k_s + delta_k) * xi_p / ((k_p - delta_k) * xi_s) + (k_i + delta_k) * xi_p / ((k_p - delta_k) * xi_i))


def A_s(k_p, k_s, k_i, xi_p, xi_s):
    """Bennink Eq. (51): A_s"""
    return 2 * np.sqrt((1 + k_s * xi_s / (k_p * xi_p)) * k_i / k_p)


def B_s(delta_k, k_p, k_s, k_i, xi_p, xi_s):
    """Bennink Eq. (52): B_s"""
    return 2 * (1 - delta_k / k_p) * np.sqrt((1 + (k_s + delta_k) * xi_p / ((k_p - delta_k) * xi_s)) * (k_i + delta_k) / (k_p - delta_k))


def A_i(k_p, k_s, k_i, xi_p, xi_i):
    """Bennink Eq. (51) with s <-> i: A_i"""
    return 2 * np.sqrt((1 + k_i * xi_i / (k_p * xi_p)) * k_s / k_p)


def B_i(delta_k, k_p, k_s, k_i, xi_p, xi_i):
    """Bennink Eq. (52) with s <-> i: B_i"""
    return 2 * (1 - delta_k / k_p) * np.sqrt((1 + (k_i + delta_k) * xi_p / ((k_p - delta_k) * xi_i)) * (k_s + delta_k) / (k_p - delta_k))


def aggregate_xi(A_plus_value, B_plus_value, xi_p, xi_s, xi_i):
    """Bennink Eq. (15): aggregate focal parameter xi"""
    return B_plus_value * xi_s * xi_i / (A_plus_value * xi_p)


# ============================================================
# 8. Sweep collection focal parameter
#
# Fixed pump focal parameter xi_p
# Variable collection focal parameter xi_c
#
# Assume:
# xi_s = xi_i = xi_c
# ============================================================

ZRc_over_L = np.linspace(0.05, 10.0, 10000)

xi_c = 1.0 / (2.0 * ZRc_over_L)

xi_s = xi_c
xi_i = xi_c

delta_k = k_p - k_s - k_i

A_plus_value = A_plus(k_p, k_s, k_i, xi_p, xi_s, xi_i)
B_plus_value = B_plus(delta_k, k_p, k_s, k_i, xi_p, xi_s, xi_i)

A_s_value = A_s(k_p, k_s, k_i, xi_p, xi_s)
B_s_value = B_s(delta_k, k_p, k_s, k_i, xi_p, xi_s)

A_i_value = A_i(k_p, k_s, k_i, xi_p, xi_i)
B_i_value = B_i(delta_k, k_p, k_s, k_i, xi_p, xi_i)

xi = aggregate_xi(A_plus_value, B_plus_value, xi_p, xi_s, xi_i)


# ============================================================
# 9. Reduced probabilities
#
# Common prefactors in Eqs. (40) and (53) are cancelled
# ============================================================

P_si = np.arctan(xi) / (A_plus_value * B_plus_value)
P_s = np.arctan(B_s_value * xi_s / A_s_value) / (A_s_value * B_s_value)
P_i = np.arctan(B_i_value * xi_i / A_i_value) / (A_i_value * B_i_value)


# ============================================================
# 10. Correlated mode coupling efficiency
# ============================================================

eta_s = P_si / P_s
eta_i = P_si / P_i
eta_si = P_si / np.sqrt(P_s * P_i)


# ============================================================
# 11. Find maxima
# ============================================================

max_idx_s = np.nanargmax(eta_s)
max_idx_i = np.nanargmax(eta_i)
max_idx_si = np.nanargmax(eta_si)

eta_s_max = eta_s[max_idx_s]
eta_i_max = eta_i[max_idx_i]
eta_si_max = eta_si[max_idx_si]

xi_c_opt_s = xi_c[max_idx_s]
xi_c_opt_i = xi_c[max_idx_i]
xi_c_opt_si = xi_c[max_idx_si]

ZRc_opt_s = L / (2 * xi_c_opt_s)
ZRc_opt_i = L / (2 * xi_c_opt_i)
ZRc_opt_si = L / (2 * xi_c_opt_si)

ZRc_over_L_opt_s = ZRc_opt_s / L
ZRc_over_L_opt_i = ZRc_opt_i / L
ZRc_over_L_opt_si = ZRc_opt_si / L

# calculate beam waist size at the center of crystal
w_c_opt_s = np.sqrt(2 * ZRc_opt_s / k_s)
w_c_opt_i = np.sqrt(2 * ZRc_opt_i / k_i)

# At the symmetric optimum eta_si, xi_s = xi_i = xi_c,
# so Z_Rs = Z_Ri = ZRc_opt_si, but w_s and w_i are slightly different because k_s != k_i.
w_s_opt_si = np.sqrt(2 * ZRc_opt_si / k_s)
w_i_opt_si = np.sqrt(2 * ZRc_opt_si / k_i)


# ============================================================
# 11b. Beam waist after f2 lens and expander magnification M
#
# 流程:
#  (1) 由最佳 ZRc 求晶體內的 beam waist w0 (平面介面時，出晶體後 waist 大小不變)
#  (2) 晶體中心位於 f2 前焦面 -> f2 後焦面得到準直光
#      w_f2 = lambda * f2 / (pi * w0)
#  (3) coupler 需要的準直光為 w_coupler
#      M = w_coupler / w_f2   (M > 1: 放大; M < 1: 縮小)
# ============================================================

def collimated_waist(lam, n, ZR, f):
    """晶體內 Rayleigh range ZR -> 經焦距 f 的透鏡後的準直光 beam waist (radius)."""
    w0 = np.sqrt(lam * ZR / (np.pi * n))     # 晶體內 waist (ZR = pi*n*w0^2/lambda)
    w_f = lam * f / (np.pi * w0)             # 透鏡後準直光半徑
    return w0, w_f

results_M = {}
for name, lam, n, ZR in [
    ("eta_s  (signal optimum)", lambda_s, n_s, ZRc_opt_s),
    ("eta_i  (idler optimum)",  lambda_i, n_i, ZRc_opt_i),
    ("eta_si (symmetric opt.)", lambda_s, n_s, ZRc_opt_si),
]:
    w0, w_f2 = collimated_waist(lam, n, ZR, f2)
    M = w_coupler / w_f2
    results_M[name] = (w0, w_f2, M)

# 主要使用 eta_si 的結果
wc_0, wc_f2, M_si = results_M["eta_si (symmetric opt.)"]


# ============================================================
# 12. Print results
# ============================================================

print("========================================")
print("Pump focusing")
print("========================================")
print(f"ZRp = {ZRp * 1e3:.4f} mm")
print(f"xi_p = {xi_p:.6f}")

print("\n========================================")
print("Maximum eta_s = P_si / P_s")
print("========================================")
print(f"Maximum eta_s = {eta_s_max:.6f}")
print(f"Optimal xi_c = {xi_c_opt_s:.6f}")
print(f"Optimal ZRc = {ZRc_opt_s * 1e3:.6f} mm")
print(f"Optimal ZRc / L = {ZRc_over_L_opt_s:.6f}")

print("\n========================================")
print("Maximum eta_i = P_si / P_i")
print("========================================")
print(f"Maximum eta_i = {eta_i_max:.6f}")
print(f"Optimal xi_c = {xi_c_opt_i:.6f}")
print(f"Optimal ZRc = {ZRc_opt_i * 1e3:.6f} mm")
print(f"Optimal ZRc / L = {ZRc_over_L_opt_i:.6f}")

print("\n========================================")
print("Maximum eta_si = P_si / sqrt(P_s * P_i)")
print("========================================")
print(f"Maximum eta_si = {eta_si_max:.6f}")
print(f"Optimal xi_c = {xi_c_opt_si:.6f}")
print(f"Optimal ZRc = {ZRc_opt_si * 1e3:.6f} mm")
print(f"Optimal ZRc / L = {ZRc_over_L_opt_si:.6f}")

print("\n========================================")
print("Collection beam waist at optimum")
print("========================================")
print(f"w_c for maximum eta_s = {w_c_opt_s * 1e6:.3f} um")
print(f"w_c for maximum eta_i = {w_c_opt_i * 1e6:.3f} um")

print("\nAt maximum eta_si:")
print(f"Signal waist w_s = {w_s_opt_si * 1e6:.3f} um")
print(f"Idler waist  w_i = {w_i_opt_si * 1e6:.3f} um")

print("\n========================================")
print("Beam waist after f2 and expander magnification M")
print("========================================")
print(f"Coupler target waist w_coupler = {w_coupler * 1e3:.4f} mm")
print(f"f2 = {f2 * 1e3:.1f} mm")
for name, (w0, w_f2, M) in results_M.items():
    print(f"\n[{name}]")
    print(f"  Crystal waist w0          = {w0 * 1e6:.3f} um")
    print(f"  Waist after f2 (w_f2)     = {w_f2 * 1e3:.4f} mm")
    print(f"  M = w_coupler / w_f2      = {M:.4f}")
    print(f"  (1/M = w_f2 / w_coupler   = {1 / M:.4f})")


# ============================================================
# 13. Plot helper
#
# Horizontal axis follows Schwaller Fig. 4:
# ZRc / L
# ============================================================

def plot_metric(ZRc_over_L, y, ZRc_over_L_opt, xi_c_opt, y_opt, metric, title_text):

    plt.figure(figsize=(7, 5))

    plt.plot(ZRc_over_L, y, linewidth=2)

    plt.scatter(ZRc_over_L_opt, y_opt, zorder=3)

    plt.annotate(
        rf"$\eta_{{{metric},\max}} = {y_opt:.4f}$" + "\n"
        + rf"$Z_{{Rc}}/L = {ZRc_over_L_opt:.4f}$" + "\n"
        + rf"$\xi_c = {xi_c_opt:.4f}$",
        xy=(ZRc_over_L_opt, y_opt),
        xytext=(100, -50),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->")
    )

    plt.xlabel(r"$Z_{Rc}/L$")
    plt.ylabel(rf"$\eta_{{{metric}}}$")
    plt.title(title_text)

    # Same x-axis range and tick spacing as Schwaller Fig. 4
    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 11, 1))

    plt.ylim(0, 1.05)

    plt.grid()
    plt.tight_layout()

    plt.show()


# ============================================================
# 14. Plot eta_s
# ============================================================

plot_metric(
    ZRc_over_L,
    eta_s,
    ZRc_over_L_opt_s,
    xi_c_opt_s,
    eta_s_max,
    metric="s",
    title_text=r"Signal heralding ratio $\eta_s = P_{si}/P_s$"
)


# ============================================================
# 15. Plot eta_i
# ============================================================

plot_metric(
    ZRc_over_L,
    eta_i,
    ZRc_over_L_opt_i,
    xi_c_opt_i,
    eta_i_max,
    metric="i",
    title_text=r"Idler heralding ratio $\eta_i = P_{si}/P_i$"
)


# ============================================================
# 16. Plot eta_si
# ============================================================

plot_metric(
    ZRc_over_L,
    eta_si,
    ZRc_over_L_opt_si,
    xi_c_opt_si,
    eta_si_max,
    metric="si",
    title_text=r"Symmetric heralding ratio $\eta_{si} = P_{si}/\sqrt{P_s P_i}$"
)
