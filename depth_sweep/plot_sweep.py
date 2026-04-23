import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

df = pd.read_csv("../results/noise_catalog/sweep_exp.csv")

CHOSEN_DEG  = 7
USE_LOG_LAT = True

# global style settings for better readability
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "lines.linewidth": 2.0
})

fig, ax_time = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)


# latency curve
lat, = ax_time.plot(df.degree, df.total_ms, marker="o", lw=2,
                    color="tab:blue", label="FHE latency (ms)")
ax_time.set_xlabel("Polynomial degree  (≈ multiplicative depth)")
ax_time.set_ylabel("Single eval latency (ms)")
if USE_LOG_LAT:
    ax_time.set_yscale("log")
ax_time.grid(True, which="both", alpha=0.25)

# poly MAE curve
ax_poly = ax_time.twinx()
poly, = ax_poly.plot(df.degree, df.poly_mae, marker="s", ls="--", lw=2,
                     color="tab:red", label="Polynomial MAE")
ax_poly.set_ylabel("Polynomial MAE")

# CKKS MAE inset plot
ax_in = inset_axes(
    ax_time,
    width="38%",
    height="42%",
    loc="lower right",
    bbox_to_anchor=(0.0, 0.13, 1.0, 1.0),
    bbox_transform=ax_time.transAxes,
    borderpad=1.0,
)

ax_in.plot(
    df.degree, df.ckks_mae,
    marker="^", ls="-.", lw=1.4, color="tab:green"
)
ax_in.set_yscale("log")

# larger fonts for inset
ax_in.set_title("CKKS noise MAE", fontsize=12, pad=4)
ax_in.tick_params(labelsize=10)

# ax_in.set_xticks([])


# highlight chosen degree
if CHOSEN_DEG in df.degree.values:
    row  = df.loc[df.degree == CHOSEN_DEG].iloc[0]
    x_pt, y_pt = CHOSEN_DEG, row.total_ms

    ax_time.axvline(CHOSEN_DEG, color="gray", ls="--", lw=1, zorder=0)

    # box at top-center inside axes; arrow to the data point
    ax_time.annotate(
        f"deg={CHOSEN_DEG}   lat={row.total_ms:.0f} ms   poly MAE={row.poly_mae:.3g}",
        xy=(x_pt, y_pt),
        xycoords="data",
        xytext=(0.5, 0.97),
        textcoords="axes fraction",
        ha="center", va="top",
        fontsize=12, 
        # fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.9,
                        connectionstyle="angle3,angleA=0,angleB=-90"),
        clip_on=False,
        zorder=5,
    )

# legend under the axes
lines  = [lat, poly]
labels = [l.get_label() for l in lines]
ax_time.legend(
    lines, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False,
    fontsize=11,
)


ax_time.set_title("CKKS Depth / Polynomial Degree Trade‑off (exp approximation)",
                  fontsize=12)

fig.savefig("depth_tradeoff.png", dpi=300)
fig.savefig("depth_tradeoff.svg")
plt.show()
