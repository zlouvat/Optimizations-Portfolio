# Optimizations Portfolio — Classic Optimization Methods

A hands-on tour of classic numerical optimization: each method is implemented from scratch (or with SciPy for comparison) and run on small toy functions so its behavior is easy to see.

Part of my DCS340 portfolio at [zlouvat.github.io](https://zlouvat.github.io/).

## Contents

| File | Dimension | Methods |
|------|-----------|---------|
| [`classic_optimization.py`](classic_optimization.py) | 1-D | Guess & check, plotting, gradient ascent, bisection, Newton's method |
| [`2D-Optimizations.py`](2D-Optimizations.py) | 2-D | Surface visualization, Nelder-Mead, gradient descent, Newton's method |

## 1-D Methods

**Toy function:** $f(x) = 4 - (x - 2)^2$, maximum at $x = 2$

| Function | Method | Idea |
|----------|--------|------|
| `Guess_And_Check()` | Brute force | Evaluate *f* on a coarse grid and pick the largest value |
| `Plotting()` | Visualization | Plot *f* on $[0, 4]$ |
| `Gradient_opt()` | Gradient ascent | $x \leftarrow x + \alpha f'(x)$ with a fixed learning rate |
| `Bisection()` | Bisection | Halve a bracket around the root of $f'(x)$ |
| `newtons_optimization()` | Newton's method | $x_{n+1} = x_n - f'(x_n) / f''(x_n)$ |

`newtons_optimization()` also includes five extra test functions (cubic, quartic double-well, damped sine wave, rational, and log-polynomial) for exploring how Newton's method behaves.

## 2-D Methods

**Toy function:** $f(x, y) = (x - 2)^2 - xy + (y - 3)^2$, minimum at $(14/3,\ 16/3) \approx (4.667,\ 5.333)$

| Function | Method | Idea |
|----------|--------|------|
| `vizualization()` | Visualization | 3-D surface plot of *f* |
| `Bisection_Nelder_Mead()` | Nelder-Mead (SciPy) | Derivative-free simplex search |
| `Gradient_2D()` | Gradient descent | $\mathbf{v} \leftarrow \mathbf{v} - \alpha \nabla f(\mathbf{v})$ |
| `Newtons_2D()` | Newton's method | Solve $H \Delta = \nabla f$ each step; exact in one step for a quadratic |

## Running

```bash
pip install numpy matplotlib scipy

python classic_optimization.py
python 2D-Optimizations.py
```

Each script runs one method by default. To try a different one, uncomment its call in the `if __name__ == "__main__":` block at the bottom of the file.
