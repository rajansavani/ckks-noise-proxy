import argparse, pathlib, numpy as np, torch
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2TokenizerFast

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
                       title="Pre‑GeLU Activation Distribution (Histogram + CDF)"):
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

    fig, ax1 = plt.subplots(figsize=(7,4), dpi=200)
    plt.style.use("default")

    ax1.bar(z_centers, z_counts, width=z_widths, alpha=0.6,
            edgecolor="none", label="Histogram")
    # xvline(a, color="tab:red",   linestyle="--", lw=1.2, label=f"a = {a:.3f}")
    # ax1.axvline(b, color="tab:green", linestyle="--", lw=1.2, label=f"b = {b:.3f}")
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
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="gpt2", help="HF model id")
    p.add_argument("--batches", type=int, default=50, help="random batches")
    p.add_argument("--seq_len", type=int, default=32, help="tokens per batch")
    p.add_argument("--coverage", type=float, default=0.995,
                   help="shortest central mass for [a,b]; e.g. 0.95 = 95%")
    p.add_argument("--prefix",  default="gelu_inputs",
                   help="output file prefix (without extension)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model/tokenizer
    model = GPT2Model.from_pretrained(args.model).to(device).eval()
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    vocab_size = tok.vocab_size

    collected = []

    def hook_cfc(_mod, _inp, out):
        collected.append(out.detach().cpu())

    # register on all mlp.c_fc
    for block in model.h:
        block.mlp.c_fc.register_forward_hook(hook_cfc)

    # generate random batches
    with torch.no_grad():
        for _ in range(args.batches):
            dummy = torch.randint(0, vocab_size, (1, args.seq_len), device=device)
            _ = model(dummy)

    # flatten and save
    vals = torch.cat(collected, dim=0).reshape(-1).numpy()
    out_prefix = pathlib.Path(args.prefix)
    np.save(out_prefix.with_suffix(".npy"), vals)
    print(f"Saved {vals.size} samples → {out_prefix.with_suffix('.npy')}")

    # stats and shortest interval
    mean, std = vals.mean(), vals.std()
    a, b = shortest_interval(vals, coverage=args.coverage)
    kept = ((vals >= a) & (vals <= b)).mean() * 100

    print("Stats:")
    print(f"  mean  = {mean:.4f}")
    print(f"  std   = {std:.4f}")
    print(f"  a,b   = ({a:.4f}, {b:.4f}) via shortest {args.coverage*100:.2f}% interval")
    print(f"  kept  = {kept:.2f}% of samples")
    for pctl in (0.5, 0.9, 0.95, 0.99, 0.995):
        print(f"  {pctl*100:5.1f}%-tile: {np.quantile(vals, pctl):.4f}")

    # plot histogram + cdf
    fig_path = out_prefix.with_suffix(".png")
    nice_hist_cdf_plot(
        values=vals,
        a=a, b=b,
        out_png=str(fig_path),
        zoom_bins=200,
        pad_frac=0.05,
        title="Pre‑GeLU Activation Distribution (Histogram + CDF)"
    )
    print(f"Plot saved → {fig_path}")

    # save full-range hist artifacts
    full_bins = np.linspace(vals.min(), vals.max(), 2001)
    full_counts = np.histogram(vals, bins=full_bins)[0]
    full_cdf = np.cumsum(full_counts) / full_counts.sum()
    np.save(out_prefix.with_name(out_prefix.stem + "_bins.npy"), full_bins)
    np.save(out_prefix.with_name(out_prefix.stem + "_counts.npy"), full_counts)
    np.save(out_prefix.with_name(out_prefix.stem + "_cdf.npy"), full_cdf)

if __name__ == "__main__":
    main()
