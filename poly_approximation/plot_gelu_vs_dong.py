import numpy as np, torch, matplotlib.pyplot as plt, pandas as pd

# coefficients for the piece-wise polynomial approximation of GeLU from Dong et al.
a3,a2,a1,a0 = (-0.011034134030615728,
               -0.11807612951181953,
               -0.42226581151983866,
               -0.5054031199708174)

b6,b4,b2,b1,b0 = (0.0018067462606141187,
                 -0.037688200365904236,
                  0.3603292692789629,
                  0.5,
                  0.008526321541038084)

def dong_gelu_pw(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    m0 = x < -4
    m1 = (-4 <= x) & (x < -1.95)
    m2 = (-1.95 <= x) & (x <= 3)
    m3 = x > 3

    # F0  (cubic)
    x1 = x[m1]
    y[m1] = ((a3*x1 + a2)*x1 + a1)*x1 + a0

    # F1  (even poly + linear term)
    x_mid = x[m2]
    u = x_mid * x_mid
    p_even = ((b6*u + b4)*u + b2)*u
    y[m2] = p_even + b1*x_mid + b0

    # identity and zero branches
    y[m3] = x[m3]
    # y[m0] already zero
    return y

x = torch.linspace(-5, 5, 1000).numpy() # generate 1000 points in range [-5, 5] as input to both GeLU and Dong's approximation
gelu_true = torch.nn.functional.gelu(torch.tensor(x)).numpy()
gelu_dong = dong_gelu_pw(x)
abs_err   = np.abs(gelu_dong - gelu_true)

# error report
idx_sort = np.argsort(-abs_err)[:11]
table = pd.DataFrame({
    "x":      x[idx_sort],
    "|err|":  abs_err[idx_sort],
    "true":   gelu_true[idx_sort],
    "approx": gelu_dong[idx_sort]
}).sort_values("x")

print("\nTop-11 absolute errors:")
print(table.to_string(index=False, float_format=lambda v:f"{v: .5f}"))

print("\nSummary over full range:")
print(f"  max |err|      : {abs_err.max():.4e}")
print(f"  mean |err|     : {abs_err.mean():.4e}")
print(f"  95-th perc |err|: {np.quantile(abs_err,0.95):.4e}")

# plot the true GeLU, Dong's piece-wise approximation, and their absolute error
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(x, gelu_true, label="True GeLU", lw=2)
plt.plot(x, gelu_dong, '--', label="Dong piece-wise", lw=2)
plt.legend(); plt.grid(True); plt.title("GeLU vs. Dong approximation")

plt.subplot(2,1,2)
plt.plot(x, abs_err, color="crimson")
plt.title("Absolute error |approx – true|"); plt.xlabel("x"); plt.grid(True)
plt.tight_layout(); plt.show()
