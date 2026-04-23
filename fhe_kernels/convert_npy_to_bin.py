import numpy as np, argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("npy", help="input .npy file (float64)")
p.add_argument("--out", default=None, help="output .bin (float64)")
args = p.parse_args()

arr = np.load(args.npy).astype(np.float64, copy=False)
out = pathlib.Path(args.out) if args.out else pathlib.Path(args.npy).with_suffix(".bin")
arr.tofile(out)
print(f"Wrote {arr.size} doubles to {out}")
