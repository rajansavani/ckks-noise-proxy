import argparse, json, pathlib, inspect, numpy as np, torch, torch.nn as nn, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAVE_SNS = True
except ImportError:
    plt.style.use("ggplot")
    HAVE_SNS = False
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict

# default benchmarks from lm-eval to run
DEFAULT_TASKS = [
    "arc_easy","hellaswag","piqa","social_iqa",
    "mnli","sst2","anli_r1","anli_r2","anli_r3","wic"
]

def ensure_dir(d: pathlib.Path):
    d.mkdir(parents=True, exist_ok=True)

# JSON fallback to convert numpy / torch types to something JSON can handle
def _json_fallback(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.dtype, torch.dtype)):
        return str(o)
    # last‑ditch: stringify
    return str(o)

# helper functions to save/load JSON, save figures, aggregate results, and pull accuracies from the lm-eval result dicts
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_fallback)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def central_agg(results_dict):
    """Aggregate anli_r* -> anli mean; leave others unchanged."""
    d = results_dict.copy()
    if all(k in d for k in ("anli_r1","anli_r2","anli_r3")):
        d["anli"] = (d["anli_r1"] + d["anli_r2"] + d["anli_r3"]) / 3
        for k in ("anli_r1","anli_r2","anli_r3"):
            d.pop(k, None)
    return d

def pull_accs(res):
    out = {}
    for t, dd in res["results"].items():
        key = "acc" if "acc" in dd else next(k for k in dd if k.startswith("acc"))
        out[t] = dd[key]
    return out

# noise pools (memory-mapped from .bin files, or None if missing)
def _pool(res_dir, tag, kind):
    fn = res_dir / f"{tag}_{kind}.bin"
    if not fn.exists():
        print(f"[WARN] missing pool: {fn}")
        return None
    return np.memmap(fn, dtype=np.float64, mode="r")

def sample_pool(pool, shape, device, dtype, mul=1.0, rng=None):
    if pool is None or pool.size == 0:
        return torch.zeros(shape, device=device, dtype=dtype)
    n = int(torch.tensor(shape).prod().item())
    idx = rng.integers(0, pool.size, size=n, endpoint=False)
    vals = torch.as_tensor(np.asarray(pool[idx], dtype=np.float32),
                           device=device, dtype=dtype).view(shape)
    return vals * mul

# we hook into the model at various points to inject noise from the pools
# there are different hooks for each activation type, and we keep track of how many times each hook fires
hook_counts = {"linear":0,"gelu":0,"ln":0,"softmax":0}

def make_hooks(POOLS, SCALE, rng):
    def hook_linear(_, __, out):
        hook_counts["linear"] += 1
        return out + sample_pool(POOLS["matmul"]["ckks"], out.shape, out.device, out.dtype, SCALE, rng)

    def hook_gelu(_, __, out):
        hook_counts["gelu"] += 1
        out = out.clone()
        out += sample_pool(POOLS["gelu"]["poly"],  out.shape, out.device, out.dtype, SCALE, rng)
        out += sample_pool(POOLS["gelu"]["ckks"], out.shape, out.device, out.dtype, SCALE, rng)
        return out

    def hook_layernorm(_, __, out):
        hook_counts["ln"] += 1
        out = out.clone()
        out += sample_pool(POOLS["layernorm"]["poly"],  out.shape, out.device, out.dtype, SCALE, rng)
        out += sample_pool(POOLS["layernorm"]["ckks"], out.shape, out.device, out.dtype, SCALE, rng)
        return out

    return hook_linear, hook_gelu, hook_layernorm

# for attention softmax, we have to re-implement the pre-softmax scores calculation to inject noise before the softmax
def patch_attention_softmax(model, POOLS, SCALE, rng):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    orig_fwd = GPT2Attention.forward
    allowed  = set(inspect.signature(orig_fwd).parameters)

    def noisy_forward(self, hidden_states, *args, **kw):
        kw["output_attentions"] = True
        kw = {k:v for k,v in kw.items() if k in allowed}
        out = orig_fwd(self, hidden_states, *args, **kw)

        if len(out)==3:
            attn_out, present, probs = out
            rest = (present,)
        else:
            attn_out, probs = out
            rest = ()

        hook_counts["softmax"] += 1
        probs = probs + sample_pool(POOLS["softmax"]["ckks"],
                                    probs.shape, probs.device, probs.dtype, SCALE, rng)
        return (attn_out, *rest, probs) if rest else (attn_out, probs)

    GPT2Attention.forward = noisy_forward

# can also inject noise into the final logits if desired (though this is a bit less principled, since the noise pools are derived from activations, not logits)
def patch_logits(model, POOLS, SCALE, rng):
    orig = model.forward
    def forward_with_noise(*a, **kw):
        out = orig(*a, **kw)
        if hasattr(out, "logits"):
            out.logits += sample_pool(POOLS["matmul"]["ckks"],
                                      out.logits.shape, out.logits.device,
                                      out.logits.dtype, SCALE, rng)
        return out
    model.forward = forward_with_noise

# plotting
def plot_bars(df, outdir):
    df_long = df.reset_index().melt(id_vars="index", var_name="Variant", value_name="Accuracy")
    if HAVE_SNS:
        fig, ax = plt.subplots(figsize=(10,5))
        import seaborn as sns
        sns.barplot(data=df_long, x="index", y="Accuracy", hue="Variant", ax=ax)
    else:
        fig, ax = plt.subplots(figsize=(10,5))
        for i, var in enumerate(df.columns):
            ax.bar(np.arange(len(df)) + i*0.35, df[var].values, width=0.35, label=var)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df.index, rotation=45, ha='right')

    ax.set_ylim(0,1)
    ax.set_title("GPT-2: baseline vs noisy")
    ax.set_xlabel("Task"); ax.set_ylabel("Accuracy")
    ax.legend()
    save_fig(fig, outdir/"bar_baseline_vs_noisy.png")

def slopegraph(data, outpath, left_col="baseline", right_col="noisy", title="Accuracy change per task"):
    labels = data.index.tolist()
    x = [0, 1]
    fig, ax = plt.subplots(figsize=(6,5))
    for i, lab in enumerate(labels):
        y0, y1 = data.iloc[i][left_col], data.iloc[i][right_col]
        ax.plot(x, [y0, y1], marker="o", linewidth=1.8)
        ax.text(-0.05, y0, f"{lab}", ha="right", va="center", fontsize=9)
        ax.text( 1.05, y1, f"{y1:.3f}", ha="left",  va="center", fontsize=9)
    ax.set_xticks([0,1])
    ax.set_xticklabels([left_col.capitalize(), right_col.capitalize()])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.2)
    save_fig(fig, outpath)

def dumbbell(data, outpath, left="baseline", right="noisy", sort_by="delta", title="Baseline vs Noisy"):
    d = data.sort_values(sort_by)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6,5))
    ax.hlines(y, d[left], d[right], linewidth=2, alpha=0.6)
    ax.plot(d[left],  y, "o", label=left)
    ax.plot(d[right], y, "o", label=right)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0,1)
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(title)
    ax.legend()
    save_fig(fig, outpath)

def delta_plot(df_plot_sorted, outpath):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(df_plot_sorted.index, df_plot_sorted["delta"])
    ax.axvline(0, color="k", linewidth=1)
    ax.set_xlabel("Δ Accuracy (noisy − baseline)")
    ax.set_title("Change in accuracy by task")
    save_fig(fig, outpath)

def hooks_plot(hook_counts, outpath):
    hk = pd.Series(hook_counts).sort_values()
    fig, ax = plt.subplots(figsize=(4.5,3.5))
    ax.barh(hk.index, hk.values)
    ax.set_xlabel("Hook calls")
    ax.set_title("Total hook firings by op type")
    for i,v in enumerate(hk.values):
        ax.text(v*1.01, i, f"{v}", va="center", fontsize=9)
    save_fig(fig, outpath)

def noise_hist_sample(path_bin, outpath, n=200000):
    try:
        mm = np.memmap(path_bin, dtype=np.float64, mode="r")
        sample = mm[:min(n, mm.size)]
        fig, ax = plt.subplots()
        ax.hist(sample, bins=100, alpha=0.7)
        ax.set_title(f"Distribution of CKKS residual samples ({path_bin.stem})")
        ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
        save_fig(fig, outpath)
    except Exception as e:
        print(f"[WARN] noise_hist_sample failed: {e}")

# evaluate
def run_eval(base_model, tasks, limit, batch_size, device):
    return evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={base_model},trust_remote_code=True",
        tasks=tasks,
        num_fewshot=None,
        limit=limit,
        batch_size=batch_size,
        device=str(device),
        verbosity="ERROR",
    )

def run_noisy(base_model, tasks, limit, batch_size, device, dtype_str,
              POOLS, SCALE, INJECT_LOGITS, rng):
    noisy_lm = HFLM(pretrained=base_model,
                    device=str(device),
                    dtype=dtype_str,
                    batch_size=batch_size,
                    trust_remote_code=True)

    # attach hooks
    from transformers.pytorch_utils import Conv1D
    hook_linear, hook_gelu, hook_layernorm = make_hooks(POOLS, SCALE, rng)

    for m in noisy_lm.model.modules():
        if isinstance(m, (nn.Linear, Conv1D)):
            m.register_forward_hook(hook_linear)
        elif isinstance(m, nn.LayerNorm):
            m.register_forward_hook(hook_layernorm)
        elif isinstance(m, nn.GELU) or "gelu" in m.__class__.__name__.lower():
            m.register_forward_hook(hook_gelu)

    patch_attention_softmax(noisy_lm.model, POOLS, SCALE, rng)
    if INJECT_LOGITS:
        patch_logits(noisy_lm.model, POOLS, SCALE, rng)

    task_dict = get_task_dict(tasks)
    return evaluator.evaluate(lm=noisy_lm, task_dict=task_dict,
                              limit=limit, bootstrap_iters=0,
                              log_samples=False, verbosity="ERROR")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--inject-logits", action="store_true")
    ap.add_argument("--residual-dir", default="residual_bins")
    ap.add_argument("--save-dir", default=None,
                    help="where to save outputs. If None, auto timestamp.")
    ap.add_argument("--load-dir", default=None,
                    help="skip eval & just load from dir (re-generate plots).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    dtype_str = str(dtype).split('.')[-1]

    if args.load_dir and args.save_dir:
        print("[WARN] --load-dir provided; ignoring --save-dir")
    out_dir = pathlib.Path(args.load_dir if args.load_dir else
                           args.save_dir if args.save_dir else
                           f"runs/run_{pd.Timestamp.now():%Y-%m-%d_%H-%M-%S}")
    ensure_dir(out_dir)

    if args.load_dir:
        print(f"[LOAD] Loading cached results from {out_dir}")
        baseline = load_json(out_dir/"baseline.json")
        noisy    = load_json(out_dir/"noisy.json")
        global hook_counts
        hook_counts = load_json(out_dir/"hook_counts.json")
    else:
        # RNG
        rng = np.random.default_rng(args.seed)

        # pools
        res_dir = pathlib.Path(args.residual_dir)
        POOLS = {
            "gelu":      {"ckks": _pool(res_dir,"gelu","ckks"),
                          "poly": _pool(res_dir,"gelu","poly")},
            "layernorm": {"ckks": _pool(res_dir,"layernorm","ckks"),
                          "poly": _pool(res_dir,"layernorm","poly")},
            "softmax":   {"ckks": _pool(res_dir,"softmax","ckks"),
                          "poly": _pool(res_dir,"softmax","poly")},  # poly not used
            "matmul":    {"ckks": _pool(res_dir,"matmul","ckks"),
                          "poly": None},
        }

        print("=== Baseline ===")
        baseline = run_eval(args.model, args.tasks, args.limit, args.batch_size, device)

        print("=== Noisy ===")
        noisy = run_noisy(args.model, args.tasks, args.limit, args.batch_size,
                          device, dtype_str, POOLS, args.scale,
                          args.inject_logits, rng)

        # save raw jsons
        save_json(baseline, out_dir/"baseline.json")
        save_json(noisy,    out_dir/"noisy.json")
        save_json(hook_counts, out_dir/"hook_counts.json")

    # make dataframe & save
    df = pd.DataFrame({
        "baseline": central_agg(pull_accs(baseline)),
        "noisy":    central_agg(pull_accs(noisy))
    })
    df.to_csv(out_dir/"accuracies.csv")
    with open(out_dir/"accuracies.txt","w") as f:
        f.write(df.to_string()+"\n")

    # plots
    plot_bars(df, out_dir)
    df_plot = df.copy()
    df_plot["delta"] = df_plot["noisy"] - df_plot["baseline"]
    df_plot_sorted = df_plot.sort_values("delta")
    slopegraph(df_plot.reset_index(), out_dir/"slopegraph.png")
    dumbbell(df_plot, out_dir/"dumbbell.png")
    delta_plot(df_plot_sorted, out_dir/"delta_barh.png")
    hooks_plot(hook_counts, out_dir/"hook_counts.png")

    # noise hist example
    noise_hist_sample(pathlib.Path(args.residual_dir)/"gelu_ckks.bin",
                      out_dir/"gelu_ckks_hist.png")

    # summary
    print(df)
    print("\nHook fire counts:", hook_counts)
    print(f"\nAll artifacts saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
