import argparse, pathlib, numpy as np, torch
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2TokenizerFast
from torch import nn

# find the shortest interval [a,b] covering "coverage" fraction of the data (e.g. 0.95 = 95%)
def shortest_interval(x: np.ndarray, coverage: float = 0.95):
    x_sorted = np.sort(x)
    m = int(np.floor(coverage * x_sorted.size))
    spans = x_sorted[m:] - x_sorted[:-m]
    j = np.argmin(spans)
    return float(x_sorted[j]), float(x_sorted[j + m])

# plot histogram + cdf of the zoomed-in range [a,b] with a bit of padding
def nice_hist_cdf_plot(values, a, b, out_png,
                       zoom_bins=200, pad_frac=0.05,
                       title="LayerNorm input distribution"):
    # full-range cdf first so the zoom-window interp is accurate
    full_bins = np.linspace(values.min(), values.max(), 2001)
    full_counts, _ = np.histogram(values, bins=full_bins)
    full_centers = (full_bins[:-1] + full_bins[1:]) / 2
    full_cdf = np.cumsum(full_counts) / full_counts.sum()

    # zoom limits
    width = b - a
    x_lo = a - pad_frac * width
    x_hi = b + pad_frac * width

    # recompute hist inside zoom range
    z_bins = np.linspace(x_lo, x_hi, zoom_bins + 1)
    z_counts, _ = np.histogram(values, bins=z_bins)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    z_widths  = z_bins[1:] - z_bins[:-1]

    # interpolate CDF onto zoom centers
    z_cdf = np.interp(z_centers, full_centers, full_cdf)

    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=200)
    plt.style.use("default")

    ax1.bar(z_centers, z_counts, width=z_widths, alpha=0.6,
            edgecolor="none", label="Histogram")
    ax1.axvline(a, color="tab:red",   linestyle="--", lw=1.2, label=f"a = {a:.3f}")
    ax1.axvline(b, color="tab:green", linestyle="--", lw=1.2, label=f"b = {b:.3f}")
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylabel("Count")

    ax2 = ax1.twinx()
    ax2.plot(z_centers, z_cdf, lw=2, label="CDF")
    ax2.set_ylabel("CDF")

    ax1.set_title(title, pad=12)
    lines, labels = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + l2, labels + lab2, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="HF model id")
    ap.add_argument("--batches", type=int, default=50, help="random batches")
    ap.add_argument("--seq_len", type=int, default=32, help="tokens per batch")
    ap.add_argument("--coverage", type=float, default=0.95,
                    help="central mass for [a,b] via shortest interval (e.g. 0.95 = 95%)")
    ap.add_argument("--prefix", default="layernorm_inputs",
                    help="output file prefix (no extension)")
    ap.add_argument("--collect_var", action="store_true",
                    help="also collect per-token variances (σ²) used by LN")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model / tokenizer
    model = GPT2Model.from_pretrained(args.model).to(device).eval()
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    vocab_size = tok.vocab_size

    collected_x = []   # raw inputs to LN
    collected_var = [] # optional σ² values

    def ln_pre_hook(mod, inp):
        x = inp[0].detach().cpu()
        collected_x.append(x)
        if args.collect_var:
            var = x.var(dim=-1, unbiased=False)
            collected_var.append(var)

    for _, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            m.register_forward_pre_hook(ln_pre_hook)

    with torch.no_grad():
        for _ in range(args.batches):
            dummy = torch.randint(0, vocab_size, (1, args.seq_len), device=device)
            _ = model(dummy)

    out_prefix = pathlib.Path(args.prefix)

    x_vals = torch.cat(collected_x, dim=0).reshape(-1).numpy()
    np.save(out_prefix.with_suffix(".npy"), x_vals)
    print(f"Saved raw LN inputs ({x_vals.size} vals) → {out_prefix.with_suffix('.npy')}")

    if args.collect_var:
        var_vals = torch.cat(collected_var, dim=0).reshape(-1).numpy()
        np.save(out_prefix.with_name(out_prefix.stem + "_var.npy"), var_vals)
        print(f"Saved LN variances ({var_vals.size} vals) → {out_prefix.stem}_var.npy")

    # stats + shortest interval for raw x
    mean, std = x_vals.mean(), x_vals.std()
    a, b = shortest_interval(x_vals, coverage=args.coverage)

    kept = ((x_vals >= a) & (x_vals <= b)).mean() * 100
    print("Raw LN input stats:")
    print(f"  mean  = {mean:.4f}")
    print(f"  std   = {std:.4f}")
    print(f"  a,b   = ({a:.4f}, {b:.4f}) via shortest {args.coverage*100:.2f}% interval")
    print(f"  kept  = {kept:.2f}% of samples inside [a,b]")
    for p in (0.5, 0.9, 0.95, 0.99, 0.995):
        print(f"  {p*100:5.1f}%: {np.quantile(x_vals, p):.4f}")

    fig_path = out_prefix.with_suffix(".png")
    nice_hist_cdf_plot(
        values=x_vals,
        a=a, b=b,
        out_png=str(fig_path),
        zoom_bins=200,
        pad_frac=0.05,
        title="LayerNorm Input Distribution (Histogram + CDF)"
    )
    print(f"Plot saved → {fig_path}")

    # save hist artifacts for reproducibility
    full_bins = np.linspace(x_vals.min(), x_vals.max(), 2001)
    full_counts = np.histogram(x_vals, bins=full_bins)[0]
    full_cdf = np.cumsum(full_counts) / full_counts.sum()
    np.save(out_prefix.with_name(out_prefix.stem + "_bins.npy"), full_bins)
    np.save(out_prefix.with_name(out_prefix.stem + "_counts.npy"), full_counts)
    np.save(out_prefix.with_name(out_prefix.stem + "_cdf.npy"), full_cdf)

    if args.collect_var:
        v_mean, v_std = var_vals.mean(), var_vals.std()
        va, vb = shortest_interval(var_vals, coverage=args.coverage)
        kept_v = ((var_vals >= va) & (var_vals <= vb)).mean() * 100
        print("\nLN variance (σ²) stats:")
        print(f"  mean  = {v_mean:.6f}")
        print(f"  std   = {v_std:.6f}")
        print(f"  a,b   = ({va:.6f}, {vb:.6f}) via shortest {args.coverage*100:.2f}% interval")
        print(f"  kept  = {kept_v:.2f}% of samples inside [a,b]")

        v_fig = out_prefix.with_name(out_prefix.stem + "_var.png")
        nice_hist_cdf_plot(
            values=var_vals,
            a=va, b=vb,
            out_png=str(v_fig),
            zoom_bins=200,
            pad_frac=0.05,
            title="LayerNorm Variance Distribution (Histogram + CDF)"
        )
        print(f"Variance plot saved → {v_fig}")

if __name__ == "__main__":
    main()
