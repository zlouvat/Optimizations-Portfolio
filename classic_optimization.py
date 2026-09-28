"""
================================================================================
CLASSIC 1-D OPTIMIZATION METHODS
================================================================================
Toy function used throughout (unless noted):

    f(x)  = 4 - (x - 2)^2        maximum at x = 2, f(2) = 4
    f'(x) = -2x + 4

Methods
-------
    Guess_And_Check()        Brute-force evaluation on a coarse grid
    Plotting()               Visualize f(x) on [0, 4]
    Gradient_opt()           Fixed-step gradient ascent
    Bisection()              Bisection on f'(x) to find the critical point
    newtons_optimization()   Newton's method on f'(x) for several test functions
================================================================================
"""

import math

import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# GUESS AND CHECK
# ==============================================================================

def Guess_And_Check():
    """Coarse grid search for the max of the 1-D toy parabola y = 4 - (x - 2)^2.

    Evaluates the function on the hand-picked nodes
    x = 0, 0.5, ..., 5, stores the y values, then reports
    max(y) and the x that produced it.

    Notes
    -----
    Not an optimizer: just sample-and-pick. The true vertex is at
    x = 2, y = 4, which happens to land on a grid point here.
    Contrast with ``Plotting`` (dense curve) and the derivative methods.
    """
    # Vector for the X value
    x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    y = []

    # Loop through
    for i in range(len(x)):
        value = 4 - (x[i] - 2)**2
        # print(f" X value of {x[i]}, Y value of {value}")
        y.append(value)

    # Find X and Y at Y max
    print(f"Max Y value is {max(y)}")
    print(f"Max Y at X value of {x[y.index(max(y))]}")


# ==============================================================================
# PLOTTING
# ==============================================================================

def Plotting():
    """Draw the same parabola y = 4 - (x - 2)^2 on a dense 1-D grid.

    Samples x on [0, 4] with 100 linspace points and plots the curve
    so the vertex at (2, 4) is visible. No search is performed.

    Notes
    -----
    Same surface as ``Guess_And_Check``, different resolution.
    Useful as a visual check before running ascent / bisection / Newton.
    """
    x = np.linspace(0, 4, 100)
    y = []

    # Loop through
    for i in range(len(x)):
        value = 4 - (x[i] - 2)**2
        # print(f" X value of {x[i]}, Y value of {value}")
        y.append(value)

    plt.plot(x, y)
    plt.show()


# ==============================================================================
# GRADIENT ASCENT
# ==============================================================================

def Gradient_opt():
    # Derivative
    def df(x):
        return (-2 * x) + 4

    # Parameters
    x = 0.0          # Initial guess
    alpha = 0.1      # Learning rate
    epochs = 20      # Iterations

    # Gradient Ascent Loop
    for epoch in range(epochs):
        grad = df(x)
        x = x + alpha * grad
        print(f"Step {epoch}: x = {x:.4f}")


# ==============================================================================
# BISECTION
# ==============================================================================

def Bisection():
    """Bisection on y'(x) = -2x + 4 to isolate the critical point of the parabola.

    Brackets the root of the first derivative on [0, 3], then repeatedly
    replaces the endpoint whose derivative has the same sign as the midpoint.
    Stops when |y'(c)| < 1e-5 or after 20 steps.

    Notes
    -----
    This is root-finding on the derivative, not interval-halving on y itself.
    Sign test assumes a single root in [a, b] (true here: y' is linear).
    Prints the midpoint each iteration.
    """
    # Derivative f'(x) = -2x + 4
    def df(x):
        return -2 * x + 4

    # Bracket [a, b] containing root
    a, b = 0.0, 3.0  # a and b are the surrounding points of which we're looking to find a midpoint i.e. maxima
    tol = 1e-5       # This tolerance is the direct value we are willing to be off by, essentially this tells the program when to stop, if still within the 20 steps specified

    for step in range(20):  # Step limits to 20 in case program keeps going
        c = (a + b) / 2.0
        f_c = df(c)
        if abs(f_c) < tol:
            break
        if df(a) * f_c > 0:
            a = c  # Root is in [c, b], keep adjusting c until we isolate that maxima
        else:
            b = c  # Root is in [a, c], same for here ^^^^^^
        print(f"Step {step}: c = {c:.4f}")


# ==============================================================================
# NEWTON'S METHOD FOR OPTIMIZATION - PLAY AND EXPLORE SESSION
# ==============================================================================
# Summary:
#   Newton's method finds critical points of a function f(x) by finding the
#   roots of its first derivative f'(x).
#
# Iterative Scheme:
#   x_{n+1} = x_n - [ f'(x_n) / f''(x_n) ]
#
# Properties:
#   - Quadratic functions: Converges in exactly 1 step because f'(x) is linear.
#   - General functions: Converges quadratically near local extrema (f''(x) != 0).
# ==============================================================================

def newtons_optimization():
    # Gemini Generated Functions to test each type of equation

    # --------------------------------------------------------------------------
    # 1. SIMPLE CUBIC POLYNOMIAL (Has 1 local max and 1 local min)
    #    Base function: f(x) = x^3 - 3x^2 + 2
    # --------------------------------------------------------------------------
    def df_cubic(x):
        return 3.0 * (x**2) - 6.0 * x       # f'(x) = 0 at x = 0 (max) and x = 2 (min)

    def ddf_cubic(x):
        return 6.0 * x - 6.0                # f''(x) = 0 at x = 1 (inflection point)

    # --------------------------------------------------------------------------
    # 2. QUARTIC "DOUBLE-WELL" (Has 2 local minima and 1 local max)
    #    Base function: f(x) = x^4 - 4x^2
    # --------------------------------------------------------------------------
    def df_quartic(x):
        return 4.0 * (x**3) - 8.0 * x       # f'(x) = 0 at x = 0 (max), x = ±sqrt(2) (minima)

    def ddf_quartic(x):
        return 12.0 * (x**2) - 8.0          # f''(x) = 0 at x = ±sqrt(2/3)

    # --------------------------------------------------------------------------
    # 3. EXPONENTIAL DECAY WAVE (Oscillating with dampening extrema)
    #    Base function: f(x) = sin(x) * exp(-0.1 * x)
    # --------------------------------------------------------------------------
    def df_exp_wave(x):
        return math.exp(-0.1 * x) * (math.cos(x) - 0.1 * math.sin(x))

    def ddf_exp_wave(x):
        return math.exp(-0.1 * x) * (-0.99 * math.sin(x) - 0.2 * math.cos(x))

    # --------------------------------------------------------------------------
    # 4. RATIONAL/FRACTIONAL SURFACE (Asymptote testing)
    #    Base function: f(x) = x / (x^2 + 1)
    # --------------------------------------------------------------------------
    def df_rational(x):
        return (1.0 - x**2) / ((x**2 + 1)**2)  # f'(x) = 0 at x = 1 (max) and x = -1 (min)

    def ddf_rational(x):
        return (2.0 * x * (x**2 - 3.0)) / ((x**2 + 1)**3)

    # --------------------------------------------------------------------------
    # 5. LOG-POLYNOMIAL (Defined only for x > 0)
    #    Base function: f(x) = x^2 - ln(x)
    # --------------------------------------------------------------------------
    def df_log(x):
        return 2.0 * x - (1.0 / x)          # f'(x) = 0 at x = 1/sqrt(2) ≈ 0.7071

    def ddf_log(x):
        return 2.0 + (1.0 / (x**2))

    # Initial setup
    x_n = 1.0         # Initial guess
    tol = 1e-5        # Tolerance on step magnitude
    max_steps = 20    # Iteration limit

    print("--- [Newton's Optimization Method] ---")

    for step in range(max_steps):
        f_p = df_log(x_n)
        f_pp = ddf_log(x_n)

        # Newton update step: x_{n+1} = x_n - f'(x_n) / f''(x_n)
        x_next = x_n - (f_p / f_pp)

        print(f"Step {step:2d}: x_n = {x_n:8.5f} --> x_next = {x_next:8.5f}")

        # Check convergence on step size
        if abs(x_next - x_n) < tol:
            print(f"Newton converged to optimum at x = {x_next:.6f}\n")
            return x_next

        x_n = x_next

    return x_n


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    # Guess_And_Check()
    # Plotting()
    Gradient_opt()
    # Bisection()
    # newtons_optimization()


# ==============================================================================
# TAKEHOME NOTES AFTER PLAY SESSION
# ==============================================================================

"""
    * Gradient
        - Extremely sensitive to the learning rate, this is interesting after having tried many
        different equations as I'm unsure why you wouldn't just increase the learning rate to reach
        the optimum point earlier. I was unable to find an equation that increasing the learning
        rate had falsified the outcome or reduced the meaning of the outcome.
            = After further looking into equations, this learning rate, when too large, can jump completely
            over the maximum which can impact the final product, seeming like it's unfinished at certain points.
        - I can see why the gradient method is using a lot in machine learning, it's one of the simplest and it's efficient
        at what it does. I wonder how different algorithms, like newtons, and bisection would do in a machine learning space.
        I'd imagine there are different applications for each of them, some may excel in areas better than others.
"""
