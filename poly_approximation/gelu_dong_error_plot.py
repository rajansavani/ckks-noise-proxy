import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# exact gelu function (like the real one)
def gelu_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

# piecewise polynomial approximation from Dong et al. (2023)
# Dong splits into different regions and fits F0 and F1 polynomials
F0 = np.array([-0.5054031199708174,
               -0.42226581151983866,
               -0.11807612951181953,
               -0.011034134030615728])  # coeffs for x^0, x^1, x^2, x^3

F1 = np.array([ 0.008526321541038084,
                0.5,
                0.3603292692789629,
                0.0,
               -0.037688200365904236,
                0.0,
                0.0018067462606141187])     # coeffs up to x^6

def gelu_dong(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    # region 1: x < -4 -> 0 (already 0)
    
    # region 2: -4 <= x < -1.95, degree-3
    mask2 = (x >= -4) & (x < -1.95)
    xx2 = x[mask2]
    y[mask2] = (F0[0] + F0[1]*xx2 + F0[2]*xx2**2 + F0[3]*xx2**3)
    
    # region 3: -1.95 <= x <= 3, degree-6
    mask3 = (x >= -1.95) & (x <= 3)
    xx3 = x[mask3]
    y[mask3] = sum(F1[i] * xx3**i for i in range(len(F1)))
    
    # region 4: x > 3 -> y = x
    mask4 = (x > 3)
    y[mask4] = x[mask4]
    
    return y

# compute absolute error across a range of inputs
x = np.linspace(-6, 6, 2000, dtype=np.float64)
error = np.abs(gelu_exact(x) - gelu_dong(x))

# plotting
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.edgecolor": "#333333",
        "axes.labelsize": 14,   # was 18
        "axes.titlesize": 16,   # was 20
        "xtick.labelsize": 12,  # was 14
        "ytick.labelsize": 12,  # was 14
        "grid.alpha": 0.3,
    }
)

# IIT colors for the poster
IIT_RED = "#CC0000"
IIT_GREYBLUE = "#dadfe1"

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, error, color=IIT_RED, linewidth=2.0, label="Absolute error")
ax.fill_between(x, error, alpha=0.15, color=IIT_RED)

ax.set_xlabel("Input $x$", fontsize=14)
ax.set_ylabel(r"$|\mathrm{GeLU}(x) - \tilde{\mathrm{GeLU}}(x)|$", fontsize=14)
ax.set_title("Absolute Error of Polynomial GeLU Approximation", pad=10, fontsize=16)
ax.set_xlim(-6, 6)
ax.set_ylim(0, error.max() * 1.05)
ax.legend(frameon=False, fontsize=14)
ax.axhline(0, color=IIT_GREYBLUE, linewidth=1.5)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig("gelu_dong_error.png", dpi=300)
plt.close()

print("Saved: gelu_dong_error.png")
