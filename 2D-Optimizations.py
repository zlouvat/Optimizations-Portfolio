"""
================================================================================
CLASSIC 2-D OPTIMIZATION METHODS
================================================================================
Toy function used throughout:

    f(x, y) = (x - 2)^2 - xy + (y - 3)^2

    grad f  = (2x - y - 4,  -x + 2y - 6)
    H       = [[2, -1], [-1, 2]]
    minimum at (14/3, 16/3) ≈ (4.667, 5.333)

Methods
-------
    vizualization()           3-D surface plot of f
    Bisection_Nelder_Mead()   SciPy Nelder-Mead simplex (derivative-free)
    Gradient_2D()             Fixed-step gradient descent
    Newtons_2D()              Newton's method with the analytic Hessian

NOTE: Docstrings were AI generated to provide ample context for each
optimization. Corresponding "play" was hand made and done without the
intervention of AI.
================================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def vizualization():
    """Plot the toy quadratic surface z = f(x, y) over a 2-D grid.

    Builds a mesh on [-5, 5] x [-5, 5], evaluates
    f(x, y) = (x - 2)^2 - xy + (y - 3)^2, and draws a 3-D surface
    so the bowl (and the coupling from the -xy term) can be seen
    before any optimizer is run.

    Notes
    -----
    This is visualization only: no search, no derivatives.
    The inner ``f`` accepts a 2-vector so it matches the call
    signature used later by SciPy and the hand-rolled methods.
    """
    def f(coords):
        x, y = coords
        output = (x - 2)**2 - x*y + (y - 3)**2
        return output

    # Visualizing f(x)

    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f((X, Y))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()


def Bisection_Nelder_Mead():
    """Minimize the toy surface with SciPy's Nelder-Mead simplex method.

    Starts at (0, 0) and lets Nelder-Mead grow / reflect / contract a
    simplex on the same f used in ``vizualization``. Derivative-free:
    useful as a baseline when only function values are available.

    Prints
    ------
    The reported minimizer ``res.x`` and the full ``OptimizeResult``.

    Notes
    -----
    The name includes "Bisection" from an earlier 1-D idea; this
    implementation is 2-D Nelder-Mead only. The true critical point
    of this quadratic is (14/3, 16/3) ≈ (4.667, 5.333).
    """
    # Define target surface function
    def f(point):
        x, y = point[0], point[1]
        return (x - 2)**2 - x*y + (y - 3)**2

    # Initial guess (starting simplex vertex)
    x0 = np.array([0.0, 0.0])

    # Execute Nelder-Mead optimization
    res = minimize(f, x0, method='Nelder-Mead')

    print(f"Minimum at: {res.x}")
    print(res)


def Gradient_2D():
    """Fixed-step gradient descent on the analytic gradient of f.

    Uses
        ∇f = (2x - y - 4,  -x + 2y - 6)
    which is the exact first derivative of the toy quadratic.
    Walks 20 steps from (0, 0) with learning rate α = 0.5 and
    prints the iterate after each update.

    Notes
    -----
    First-order method: no curvature. Step size is hand-chosen;
    too large and it oscillates, too small and it crawls.
    Compare the path and final point with Newton and Nelder-Mead.
    """
    # Define surface gradient
    def grad_f(v):
        x, y = v[0], v[1]
        dx = 2*x - y - 4
        dy = -x + 2*y - 6
        return np.array([dx, dy])

    # Parameters
    v = np.array([0.0, 0.0])  # Initial guess
    alpha = 0.5              # Learning rate
    epochs = 20              # Iterations

    # 2D Gradient Descent Loop
    for epoch in range(epochs):
        grad = grad_f(v)
        v = v - alpha * grad
        print(f"Step {epoch}: v = {v}")


def Newtons_2D():
    """Newton's method on f using the closed-form gradient and Hessian.

    Gradient matches ``Gradient_2D``. Hessian of this quadratic is
    constant:
        H = [[2, -1], [-1, 2]]
    so each step solves H Δ = ∇f and is theoretically exact in one
    iteration (up to floating-point). Five iterations are run from
    (0, 0) to show the iterate collapsing onto the minimizer.

    Prints
    ------
    The final (x, y) after the loop.
    """
    # Define Gradient & Hessian
    def grad(xy):
        x, y = xy[0], xy[1]
        return np.array([2*(x - 2) - y, -x + 2*(y - 3)])

    def hessian(xy):
        return np.array([[2.0, -1.0], [-1.0, 2.0]])

    # Iteration Step
    xy = np.array([0.0, 0.0])
    for _ in range(5):
        g = grad(xy)
        H = hessian(xy)
        xy = xy - np.linalg.solve(H, g)
    print(xy)


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    # vizualization()
    # Bisection_Nelder_Mead()
    # Gradient_2D()
    Newtons_2D()


# ==============================================================================
# TAKEHOME NOTES AFTER PLAY SESSION
# ==============================================================================
"""
    - Visualization: Plot contour lines with your path drawn on top to see if your algorithm is getting stuck, bouncing around, or heading the right way. This is the easiest method for understanding as it's visual and intuitive.

    - Nelder-Mead: Moves a simple area around the grid without using slopes, making it great for messy or bumpy functions.

    - Gradient Descent: Takes steps straight downhill using the slope, but can easily overshoot or get stuck moving back and forth in steep spots.

    - Newton's Method: Uses both slope and curve shape to jump straight to the bottom fast, but can fail if the curve flattens out or bends the wrong way.
"""
