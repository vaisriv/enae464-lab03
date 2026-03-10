import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# -- File paths ---------------------------------------------------------------
PRESSURE_FILE = os.path.join("data", "pressure_taps_vs_aoa.csv")
COORDS_FILE   = os.path.join("data", "naca_4412_airfoil_coords_and_taps.csv")
OUT_TEXT    = os.path.join("outputs", "text")
OUT_FIGURES = os.path.join("outputs", "figures")
os.makedirs(OUT_TEXT,    exist_ok=True)
os.makedirs(OUT_FIGURES, exist_ok=True)

# -- Load data ----------------------------------------------------------------
df_p = pd.read_csv(PRESSURE_FILE)
df_p.columns = df_p.columns.str.strip()

df_c = pd.read_csv(COORDS_FILE)
df_c.columns = df_c.columns.str.strip()

# -- Parse tap metadata from coords file --------------------------------------
# Channel 1  (x/c=1, first row)  = open to atmosphere  -> P_atm reference
# Channel 16 (x/c=1, last row)   = tunnel static port   -> P_inf for Cp
# Only rows with a real integer Tap # are actual surface taps (Channels 2-15).
# Dropping rows without a Tap # correctly excludes both reference channels.
tap_rows = df_c.copy()
tap_rows["Tap #"]         = pd.to_numeric(tap_rows["Tap #"],         errors="coerce")
tap_rows["DSA Channel #"] = pd.to_numeric(tap_rows["DSA Channel #"], errors="coerce")
tap_rows = tap_rows.dropna(subset=["Tap #", "DSA Channel #"]).copy()
tap_rows["Tap #"]         = tap_rows["Tap #"].astype(int)
tap_rows["DSA Channel #"] = tap_rows["DSA Channel #"].astype(int)
tap_rows["x/c"]           = tap_rows["x/c"].astype(float)
tap_rows["y/c"]           = tap_rows["y/c"].astype(float)

# Surface classification: y/c >= 0 -> upper, y/c < 0 -> lower
# (leading edge tap at x/c=0, y/c=0 is upper by convention)
tap_rows["surface"] = tap_rows["y/c"].apply(lambda y: "upper" if y >= 0 else "lower")

# Sort: upper surface nose->trailing, then lower surface nose->trailing
upper = tap_rows[tap_rows["surface"] == "upper"].sort_values("x/c")
lower = tap_rows[tap_rows["surface"] == "lower"].sort_values("x/c")
taps_ordered = pd.concat([upper, lower]).reset_index(drop=True)

# Map channel number -> column name in pressure DataFrame
def ch_col(n):
    return f"Channel {n}"

# -- Flow conditions ----------------------------------------------------------
# Channel 16 = tunnel static pressure -> correct P_inf for Cp definition
# Channel 1  = open to atmosphere     -> independent check / P_atm
#
# Dynamic pressure:
#   q_inf = P_stagnation - P_inf
#   At AoA = 0 the leading-edge tap (Tap 8, Channel 9) is at the stagnation
#   point, so P_stag ~ Channel 9 at AoA=0. Tunnel speed is fixed -> q_inf constant.
#
#   Cp = (P_tap - P_inf) / q_inf   where P_inf = Channel 16

aoa_vals     = df_p["AoA"].values
P_inf_series = df_p[ch_col(16)].values    # tunnel static pressure (Pa gauge)

# q_inf from pitot: Channel 1 (atmosphere) - Channel 16 (tunnel static)
# This is consistent across all AoA (~630 Pa), confirming it is the correct source.
# Use the mean value for Cp normalisation (tunnel speed is nominally constant).
q_inf = (df_p["Channel 1"] - df_p["Channel 16"]).mean()

print(f"Freestream dynamic pressure q_inf (pitot mean) = {q_inf:.2f} Pa")

# -- 1. Build pressures table -------------------------------------------------
pressure_records = []
for _, prow in df_p.iterrows():
    rec = {"AoA": prow["AoA"]}
    for _, trow in taps_ordered.iterrows():
        col_name = f"Tap{int(trow['Tap #'])}_{trow['surface']}_xc{trow['x/c']:.3f}"
        rec[col_name] = prow[ch_col(int(trow["DSA Channel #"]))]
    pressure_records.append(rec)

df_pressures = pd.DataFrame(pressure_records)
pressure_path = os.path.join(OUT_TEXT, "pressures_table.csv")
df_pressures.to_csv(pressure_path, index=False)
print(f"Saved: {pressure_path}")

# -- 2. Build Cp table --------------------------------------------------------
cp_records = []
for i, prow in df_p.iterrows():
    rec = {"AoA": prow["AoA"]}
    P_inf_i = prow[ch_col(16)]             # tunnel static for this row
    for _, trow in taps_ordered.iterrows():
        P_tap = prow[ch_col(int(trow["DSA Channel #"]))]
        Cp    = (P_tap - P_inf_i) / q_inf
        col_name = f"Cp_Tap{int(trow['Tap #'])}_{trow['surface']}_xc{trow['x/c']:.3f}"
        rec[col_name] = round(Cp, 4)
    cp_records.append(rec)

df_cp = pd.DataFrame(cp_records)
cp_path = os.path.join(OUT_TEXT, "cp_table.csv")
df_cp.to_csv(cp_path, index=False)
print(f"Saved: {cp_path}")


# -- 3. Integrated Cp table (trapezoidal rule) ---------------------------------
# Integrate Cp over x/c separately for upper and lower surfaces:
#   C_P_upper = integral of Cp_upper dx/c  from x/c=0 to 1  (leading -> trailing)
#   C_P_lower = integral of Cp_lower dx/c  from x/c=0 to 1
#
# The upper surface Cp is conventionally negative (suction), so C_P_upper will
# be negative; the net normal force coefficient cn = C_P_lower - C_P_upper.
# Both integrals use np.trapz with the tap x/c values as the abscissa.

xc_u_ref = upper["x/c"].values   # fixed tap x/c locations, upper
xc_l_ref = lower["x/c"].values   # fixed tap x/c locations, lower

int_cp_records = []
for i, prow in df_p.iterrows():
    P_inf_i = prow[ch_col(16)]

    cp_u_vals = np.array([
        (prow[ch_col(int(t["DSA Channel #"]))] - P_inf_i) / q_inf
        for _, t in upper.iterrows()
    ])
    cp_l_vals = np.array([
        (prow[ch_col(int(t["DSA Channel #"]))] - P_inf_i) / q_inf
        for _, t in lower.iterrows()
    ])

    Cp_upper = np.trapezoid(cp_u_vals, xc_u_ref)
    Cp_lower = np.trapezoid(cp_l_vals, xc_l_ref)
    Cn       = Cp_lower - Cp_upper   # net normal force coefficient

    int_cp_records.append({
        "AoA":      prow["AoA"],
        "C_P_upper": round(Cp_upper, 6),
        "C_P_lower": round(Cp_lower, 6),
        "Cn":        round(Cn,       6),
    })

df_int_cp = pd.DataFrame(int_cp_records)
int_cp_path = os.path.join(OUT_TEXT, "integrated_cp_table.csv")
df_int_cp.to_csv(int_cp_path, index=False)
print(f"Saved: {int_cp_path}")


# -- 4. Flow conditions: U_inf and Reynolds number ----------------------------
# Pitot tube reading: q_inf = P_atm - P_static = Channel 1 - Channel 16
#   Channel 1  = atmosphere (total pressure port of pitot)
#   Channel 16 = tunnel static pressure
# This gives a consistent q_inf (~630 Pa) across all AoA, confirming it is
# the correct dynamic pressure source.
#
# Physical constants (standard sea-level air):
RHO     = 1.225       # kg/m^3, air density
MU      = 1.789e-5    # Pa·s,   dynamic viscosity
CHORD   = 0.3         # m,      airfoil chord length (set to your model's chord)
SPAN    = 1.0         # m,      airfoil span (per-unit-span if 1.0)

q_inf_per_aoa = (df_p["Channel 1"] - df_p["Channel 16"]).values   # Pa, one per AoA row
U_inf_per_aoa = np.sqrt(2.0 * q_inf_per_aoa / RHO)                # m/s
Re_per_aoa    = RHO * U_inf_per_aoa * CHORD / MU

# Use the mean q_inf (tunnel speed is nominally constant) for Cp normalisation
q_inf = q_inf_per_aoa.mean()
U_inf = np.sqrt(2.0 * q_inf / RHO)
Re    = RHO * U_inf * CHORD / MU

print(f"\nMean dynamic pressure  q_inf = {q_inf:.2f} Pa")
print(f"Freestream velocity    U_inf = {U_inf:.3f} m/s")
print(f"Reynolds number        Re    = {Re:.4e}  (chord = {CHORD} m)")

# -- 5. Pressure drag and lift at each AoA ------------------------------------
# From the integrated Cp distributions:
#
#   Normal force coefficient (already computed):
#     Cn = C_P_lower - C_P_upper   (positive = upward)
#
#   Axial (chord-wise) force coefficient via leading-edge pressure distribution.
#   For pressure-only drag, integrate Cp over the airfoil surface projected onto
#   the chord axis.  Using the surface y/c coordinates of the taps:
#
#     Ca = -( integral Cp_upper d(y/c) - integral Cp_lower d(y/c) )
#
#   Then resolve into lift and drag in the wind axis:
#     CL = Cn * cos(alpha) - Ca * sin(alpha)
#     CD = Cn * sin(alpha) + Ca * cos(alpha)
#
#   Forces per unit span:
#     L = CL * q_inf * CHORD * SPAN
#     D = CD * q_inf * CHORD * SPAN

xc_u_ref = upper["x/c"].values
xc_l_ref = lower["x/c"].values
yc_u_ref = upper["y/c"].values
yc_l_ref = lower["y/c"].values

flow_records = []
for i, prow in df_p.iterrows():
    aoa_deg = float(prow["AoA"])
    alpha   = np.radians(aoa_deg)
    q_i     = prow["Channel 1"] - prow["Channel 16"]   # per-AoA dynamic pressure
    U_i     = np.sqrt(2.0 * q_i / RHO)
    Re_i    = RHO * U_i * CHORD / MU

    P_inf_i = prow[ch_col(16)]

    cp_u_vals = np.array([
        (prow[ch_col(int(t["DSA Channel #"]))] - P_inf_i) / q_inf
        for _, t in upper.iterrows()
    ])
    cp_l_vals = np.array([
        (prow[ch_col(int(t["DSA Channel #"]))] - P_inf_i) / q_inf
        for _, t in lower.iterrows()
    ])

    # Normal force coefficient (perpendicular to chord)
    Cn = np.trapezoid(cp_l_vals, xc_l_ref) - np.trapezoid(cp_u_vals, xc_u_ref)

    # Axial force coefficient (along chord, positive toward trailing edge)
    # Upper surface: Cp acts inward (negative y direction) → contributes +Ca
    # Lower surface: Cp acts inward (positive y direction) → contributes -Ca
    Ca = -(np.trapezoid(cp_u_vals, yc_u_ref) - np.trapezoid(cp_l_vals, yc_l_ref))

    # Wind-axis coefficients
    CL = Cn * np.cos(alpha) - Ca * np.sin(alpha)
    CD = Cn * np.sin(alpha) + Ca * np.cos(alpha)

    # Dimensional forces per unit span (N/m)
    L = CL * q_i * CHORD * SPAN
    D = CD * q_i * CHORD * SPAN

    flow_records.append({
        "AoA (deg)":   aoa_deg,
        "q_inf (Pa)":  round(q_i,    4),
        "U_inf (m/s)": round(U_i,    4),
        "Re":          round(Re_i,   1),
        "Cn":          round(Cn,     6),
        "Ca":          round(Ca,     6),
        "CL":          round(CL,     6),
        "CD":          round(CD,     6),
        "Lift (N/m)":  round(L,      4),
        "Drag (N/m)":  round(D,      4),
    })

df_flow = pd.DataFrame(flow_records)
flow_path = os.path.join(OUT_TEXT, "flow_and_forces_table.csv")
df_flow.to_csv(flow_path, index=False)
print(f"Saved: {flow_path}")

# -- 6. Cp vs x/c plots -------------------------------------------------------
# Full airfoil coordinates for shape subplot
xc_all = df_c["x/c"].astype(float).values
yc_all = df_c["y/c"].astype(float).values

for _, prow in df_p.iterrows():
    aoa     = int(prow["AoA"])
    P_inf_i = prow[ch_col(16)]             # tunnel static for this AoA

    xc_u, cp_u, yc_u_tap = [], [], []
    xc_l, cp_l, yc_l_tap = [], [], []

    for _, trow in taps_ordered.iterrows():
        P_tap = prow[ch_col(int(trow["DSA Channel #"]))]
        Cp    = (P_tap - P_inf_i) / q_inf
        if trow["surface"] == "upper":
            xc_u.append(trow["x/c"])
            cp_u.append(Cp)
            yc_u_tap.append(trow["y/c"])
        else:
            xc_l.append(trow["x/c"])
            cp_l.append(Cp)
            yc_l_tap.append(trow["y/c"])

    xc_u, cp_u = np.array(xc_u), np.array(cp_u)
    xc_l, cp_l = np.array(xc_l), np.array(cp_l)
    yc_u_tap   = np.array(yc_u_tap)
    yc_l_tap   = np.array(yc_l_tap)

    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

    # Top: Cp plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(xc_u, cp_u, "bo-", label="Upper surface", markersize=6)
    ax1.plot(xc_l, cp_l, "rs-", label="Lower surface", markersize=6)
    ax1.axhline(0, color="k", linewidth=0.7, linestyle="--")
    ax1.invert_yaxis()   # convention: Cp decreases upward
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_xlabel(r"$x/c$", fontsize=13)
    ax1.set_ylabel(r"$C_p$", fontsize=13)
    ax1.set_title(rf"Pressure Coefficient $C_p$ vs $x/c$ -- AoA = {aoa} deg", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bottom: airfoil shape with taps at actual y/c positions
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(xc_all, yc_all, "k-", linewidth=1.5)
    ax2.scatter(xc_u, yc_u_tap, color="blue", s=40, zorder=3, label="Upper taps")
    ax2.scatter(xc_l, yc_l_tap, color="red",  s=40, zorder=3, label="Lower taps")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$x/c$", fontsize=11)
    ax2.set_ylabel(r"$y/c$", fontsize=11)
    ax2.set_title("NACA 4412 Airfoil Profile with Tap Locations", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fname = os.path.join(OUT_FIGURES, f"cp_vs_xc_AoA_{aoa:+03d}deg.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# -- 7. CL and CD vs AoA plots + stall estimate -------------------------------
# Pull CL, CD, and AoA directly from the already-computed df_flow.
aoa_plot = df_flow["AoA (deg)"].values
CL_plot  = df_flow["CL"].values
CD_plot  = df_flow["CD"].values

# -- Stall estimate -----------------------------------------------------------
# Stall is identified as the angle of attack at which CL reaches its maximum
# before the onset of flow separation (the first peak when scanning from low
# to high AoA, i.e. considering only AoA >= 0 to avoid the negative-alpha peak).
positive_mask  = aoa_plot >= 0
CL_pos         = CL_plot[positive_mask]
aoa_pos        = aoa_plot[positive_mask]
stall_idx      = int(np.argmax(CL_pos))
CL_max         = CL_pos[stall_idx]
aoa_stall      = aoa_pos[stall_idx]

print(f"\nStall estimate:")
print(f"  CL_max   = {CL_max:.4f}")
print(f"  AoA_stall = {aoa_stall:.1f} deg")

# Write stall summary to text file
stall_path = os.path.join(OUT_TEXT, "stall_estimate.csv")
pd.DataFrame([{"AoA_stall (deg)": aoa_stall, "CL_max": round(CL_max, 6)}]).to_csv(
    stall_path, index=False
)
print(f"Saved: {stall_path}")

# -- Plot: CL vs AoA ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(aoa_plot, CL_plot, "bo-", markersize=6, linewidth=1.5, label=r"$C_L$")
ax.axvline(aoa_stall, color="red", linestyle="--", linewidth=1.2,
           label=rf"Stall  $\alpha = {aoa_stall:.0f}\circ$,  $C_{{L,\max}} = {CL_max:.3f}$")
ax.scatter([aoa_stall], [CL_max], color="red", zorder=5, s=60)

ax.set_xlabel(r"Angle of Attack $\alpha$ (deg)", fontsize=13)
ax.set_ylabel(r"Lift Coefficient $C_L$", fontsize=13)
ax.set_title(r"Lift Coefficient $C_L$ vs Angle of Attack", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
ax.axhline(0, color="k", linewidth=0.7)
ax.axvline(0, color="k", linewidth=0.7)

cl_path = os.path.join(OUT_FIGURES, "CL_vs_AoA.png")
plt.savefig(cl_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {cl_path}")

# -- Plot: CD vs AoA ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(aoa_plot, CD_plot, "rs-", markersize=6, linewidth=1.5, label=r"$C_D$")

ax.set_xlabel(r"Angle of Attack $\alpha$ (deg)", fontsize=13)
ax.set_ylabel(r"Drag Coefficient $C_D$", fontsize=13)
ax.set_title(r"Drag Coefficient $C_D$ vs Angle of Attack", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
ax.axhline(0, color="k", linewidth=0.7)
ax.axvline(0, color="k", linewidth=0.7)

cd_path = os.path.join(OUT_FIGURES, "CD_vs_AoA.png")
plt.savefig(cd_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {cd_path}")

print("\nDone! CSVs -> ./outputs/text/   |   Plots -> ./outputs/figures/")
