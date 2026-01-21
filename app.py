import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Function Reconstruction from Derivatives", layout="wide")

st.title("📈 Function Reconstruction from Derivatives")
st.markdown("**By: Racely Ortega**")
st.markdown(
    """
This app reconstructs an original function from its derivative(s), applies initial conditions,
and visually aligns the graphs to show relationships between **f**, **f′**, and **f″**.
"""
)

# -----------------------------
# Symbolic setup
# -----------------------------
x = sp.symbols("x")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Functions")

derivative_str = st.sidebar.text_input(
    "First derivative f'(x)",
    value="2*x"
)

second_derivative_str = st.sidebar.text_input(
    "Second derivative f''(x) (optional)",
    value=""
)

st.sidebar.header("Initial Conditions")

ic_fx = st.sidebar.text_input(
    "f(a) = b   (example: 0,1)",
    value="0,0"
)

ic_fpx = st.sidebar.text_input(
    "f'(c) = d (optional, example: 1,2)",
    value=""
)

st.sidebar.header("Critical / Inflection Points")

critical_points_str = st.sidebar.text_input(
    "Known zeros of f'(x) (comma-separated)",
    value=""
)

st.sidebar.header("Plot Range")
xmin, xmax = st.sidebar.slider("x-range", -10.0, 10.0, (-5.0, 5.0))

# -----------------------------
# Parse symbolic expressions
# -----------------------------
try:
    fprime = sp.sympify(derivative_str)
except Exception as e:
    st.error(f"Invalid f'(x): {e}")
    st.stop()

fdoubleprime = None
if second_derivative_str.strip():
    try:
        fdoubleprime = sp.sympify(second_derivative_str)
    except Exception as e:
        st.error(f"Invalid f''(x): {e}")
        st.stop()

# -----------------------------
# Reconstruct original function
# -----------------------------
C1, C2 = sp.symbols("C1 C2")

if fdoubleprime is not None:
    fprime_expr = sp.integrate(fdoubleprime, x) + C1
    f_expr = sp.integrate(fprime_expr, x) + C2
else:
    fprime_expr = fprime
    f_expr = sp.integrate(fprime, x) + C1

# -----------------------------
# Apply initial conditions
# -----------------------------
equations = []

try:
    a, b = map(float, ic_fx.split(","))
    equations.append(f_expr.subs(x, a) - b)
except Exception:
    st.error("Invalid f(a)=b input.")
    st.stop()

if ic_fpx.strip():
    try:
        c, d = map(float, ic_fpx.split(","))
        equations.append(fprime_expr.subs(x, c) - d)
    except Exception:
        st.error("Invalid f'(c)=d input.")
        st.stop()

constants = sp.solve(equations, [C1, C2], dict=True)
if constants:
    f_expr = f_expr.subs(constants[0])
    fprime_expr = fprime_expr.subs(constants[0])

# -----------------------------
# Numerical lambdas
# -----------------------------
f_num = sp.lambdify(x, f_expr, "numpy")
fp_num = sp.lambdify(x, fprime_expr, "numpy")

if fdoubleprime is not None:
    fpp_num = sp.lambdify(x, fdoubleprime, "numpy")

x_vals = np.linspace(xmin, xmax, 1000)

# -----------------------------
# Critical & inflection points
# -----------------------------
critical_points = []
if critical_points_str.strip():
    try:
        critical_points = [float(v) for v in critical_points_str.split(",")]
    except Exception:
        st.warning("Could not parse critical points.")

inflection_points = []
if fdoubleprime is not None:
    try:
        inflection_points = sp.solve(fdoubleprime, x)
        inflection_points = [float(v) for v in inflection_points if v.is_real]
    except Exception:
        pass

# -----------------------------
# Plotting
# -----------------------------
num_plots = 2 if fdoubleprime is None else 3
fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(8, 10))

if num_plots == 1:
    axes = [axes]

axes[0].plot(x_vals, f_num(x_vals))
axes[0].set_ylabel("f(x)")
axes[0].grid(True)

axes[1].plot(x_vals, fp_num(x_vals))
axes[1].set_ylabel("f'(x)")
axes[1].grid(True)

if fdoubleprime is not None:
    axes[2].plot(x_vals, fpp_num(x_vals))
    axes[2].set_ylabel("f''(x)")
    axes[2].grid(True)

for cp in critical_points:
    axes[0].axvline(cp, linestyle="--", alpha=0.6)
    axes[1].axvline(cp, linestyle="--", alpha=0.6)

for ip in inflection_points:
    axes[0].axvline(ip, linestyle="--", color="red", alpha=0.6)
    if fdoubleprime is not None:
        axes[2].axvline(ip, linestyle="--", color="red", alpha=0.6)

axes[-1].set_xlabel("x")
plt.tight_layout()
st.pyplot(fig)

# -----------------------------
# Display formulas
# -----------------------------
st.subheader("Reconstructed Functions")
st.latex(f"f(x) = {sp.latex(f_expr)}")
st.latex(f"f'(x) = {sp.latex(fprime_expr)}")
if fdoubleprime is not None:
    st.latex(f"f''(x) = {sp.latex(fdoubleprime)}")
