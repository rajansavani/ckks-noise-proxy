#!/usr/bin/env python3
import argparse, pathlib, numpy as np, torch, math
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2TokenizerFast

# find the shortest interval [a,b] covering "coverage" fraction of the data (e.g. 0.95 = 95%)
# different for softmax scores since we want to ignore -inf values but still find a tight interval for the finite ones
def shortest_central_interval(x, coverage=0.95, nbins=5000):
    # assumes x is finite
    hist, edges = np.histogram(x, bins=nbins)
    cdf = np.cumsum(hist) / hist.sum()
    best_w = np.inf
    best_a = best_b = None
    for i in range(len(cdf)):
        target = cdf[i] + coverage
        if target > 1:
            break
        j = np.searchsorted(cdf, target)
        if j >= len(cdf):
            break
        a, b = edges[i], edges[j]
        w = b - a
        if w < best_w:
            best_w, best_a, best_b = w, a, b
    if best_a is None:
        tail = (1 - coverage) / 2
        best_a, best_b = np.quantile(x, tail), np.quantile(x, 1 - tail)
    return float(best_a), float(best_b)

def plot_hist_cdf_zoom(vals, a, b, out_png, pad_frac=0.05,
                       title="Softmax Pre-Activation Distribution (Histogram + CDF)"):
    width = b - a
    x_lo = a - pad_frac * width
    x_hi = b + pad_frac * width

    # 200 bins in zoom window
    bins = np.linspace(x_lo, x_hi, 201)
    counts, _ = np.histogram(vals, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths  = bins[1:] - bins[:-1]

    # global CDF for interpolation
    g_bins = np.linspace(vals.min(), vals.max(), 2000)
    g_counts, _ = np.histogram(vals, bins=g_bins)
    g_cdf = np.cumsum(g_counts) / g_counts.sum()
    g_centers = 0.5 * (g_bins[:-1] + g_bins[1:])
    cdf_zoom = np.interp(centers, g_centers, g_cdf)

    fig, ax1 = plt.subplots(figsize=(7,4), dpi=200)
    plt.style.use("default")

    ax1.bar(centers, counts, width=widths, alpha=0.6, edgecolor="none", label="Histogram")
    ax1.axvline(a, color="tab:red",   ls="--", lw=1.2, label=f"a ≈ {a:.3f}")
    ax1.axvline(b, color="tab:green", ls="--", lw=1.2, label=f"b ≈ {b:.3f}")
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylabel("Count")

    ax2 = ax1.twinx()
    ax2.plot(centers, cdf_zoom, lw=2, label="CDF")
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
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--coverage", type=float, default=0.95)
    ap.add_argument("--prefix", default="softmax_inputs")
    ap.add_argument("--save_probs", action="store_true")
    ap.add_argument("--keep_mask", action="store_true",
                    help="add the attention mask to scores (may introduce -inf)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model/tokenizer
    model = GPT2Model.from_pretrained(args.model, attn_implementation="eager").to(device).eval()
    model.config.use_cache = False
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    vocab = tok.vocab_size

    pre_scores = []
    post_probs = []

    def make_manual_scores(self_attn, hidden_states, attn_mask=None):
        qkv = self_attn.c_attn(hidden_states)
        q, k, v = qkv.split(self_attn.split_size, dim=2)

        def split_heads(x):
            new_shape = x.size()[:-1] + (self_attn.num_heads, self_attn.head_dim)
            x = x.view(*new_shape)
            return x.permute(0, 2, 1, 3)

        q = split_heads(q)
        k = split_heads(k)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self_attn.head_dim)

        if args.keep_mask and attn_mask is not None:
            if attn_mask.dim() == 4:
                # slice to fit
                attn_mask = attn_mask[..., :scores.size(-2), :scores.size(-1)]
            scores = scores + attn_mask
        return scores

    for block in model.h:
        attn = block.attn
        orig_fwd = attn.forward

        def hooked_forward(self_attn, hidden_states, *f_args, **f_kwargs):
            f_kwargs.setdefault("output_attentions", True)
            out = orig_fwd(hidden_states, *f_args, **f_kwargs)

            # fetch probs if provided; otherwise rebuild
            if isinstance(out, tuple) and len(out) >= 3 and out[2] is not None:
                probs = out[2]
                scores = torch.log(probs.clamp_min(1e-12))
            else:
                scores = make_manual_scores(self_attn, hidden_states, f_kwargs.get("attention_mask", None))
                probs = torch.softmax(scores, dim=-1)

            # filter finite before storing
            scores = scores.detach()
            if args.save_probs:
                post_probs.append(probs.detach().cpu())

            pre_scores.append(scores.cpu())
            return out

        attn.forward = hooked_forward.__get__(attn, attn.__class__)

    with torch.no_grad():
        for _ in range(args.batches):
            dummy = torch.randint(0, vocab, (1, args.seq_len), device=device)
            _ = model(dummy)

    out_prefix = pathlib.Path(args.prefix)

    scores = torch.cat(pre_scores, 0).view(-1)
    finite_mask = torch.isfinite(scores)
    scores = scores[finite_mask].numpy()

    np.save(out_prefix.with_suffix(".npy"), scores)
    print(f"Saved pre-softmax finite scores ({scores.size}) → {out_prefix.with_suffix('.npy')}")

    if args.save_probs:
        probs = torch.cat(post_probs, 0).view(-1).numpy()
        np.save(out_prefix.with_name(out_prefix.stem + "_probs.npy"), probs)
        print(f"Saved probs ({probs.size}) → {out_prefix.stem}_probs.npy")

    mean, std = scores.mean(), scores.std()
    a, b = shortest_central_interval(scores, coverage=args.coverage)

    print("Pre-softmax score stats (finite only):")
    print(f"  mean = {mean:.4f}  std = {std:.4f}")
    print(f"  shortest {args.coverage*100:.1f}% interval: [{a:.4f}, {b:.4f}]")
    for p in (0.5, 0.9, 0.95, 0.99):
        print(f"  {p*100:5.1f}%: {np.quantile(scores, p):.4f}")

    fig_path = out_prefix.with_suffix(".png")
    plot_hist_cdf_zoom(scores, a, b, str(fig_path))
    print(f"Plot saved → {fig_path}")

    # optional dump of full hist/CDF
    bins_full = np.linspace(scores.min(), scores.max(), 2001)
    counts_full, _ = np.histogram(scores, bins=bins_full)
    cdf_full = np.cumsum(counts_full) / counts_full.sum()
    np.save(out_prefix.with_name(out_prefix.stem + "_bins.npy"), bins_full)
    np.save(out_prefix.with_name(out_prefix.stem + "_counts.npy"), counts_full)
    np.save(out_prefix.with_name(out_prefix.stem + "_cdf.npy"), cdf_full)

if __name__ == "__main__":
    main()
