import numpy as np
from matplotlib import pyplot as plt


def taylor_cos(x):
    return 1 - (x**2) / 2 + (x**4) / 24


def taylor_sin(x):
    return x - (x**3) / 6 + (x**5) / 120


m = 1


def first_order(x, y, lam):
    u, v = y
    first = (-2 / taylor_cos(x)) * (
        lam
        - (m**2 * taylor_cos(x) * 2) / (2 * taylor_sin(x) * 2)
        + taylor_cos(x) / taylor_sin(x)
    )
    second = -taylor_cos(2 * x) / (taylor_cos(x) * taylor_sin(x))
    matrix = np.array([[0, 1], [first, second]])
    return np.dot(matrix, np.array([u, v]))


def euler_method(f, xrange, y0, h, lambda_, tol=1e-10, max_iter=100):
    """
    Backward Euler method for solving a system of ODEs.

    Parameters:
        f: Function f(x, y, lambda_) defining the system of ODEs (dy/dx = f(x, y, lambda_)).
        y0: Initial condition (numpy array of size n).
        xrange: Tuple (x_start, x_end) defining the integration interval.
        lambda_: Parameter passed to the function f.
        steps: Number of steps for the integration.
        tol: Tolerance for Newton's method convergence.
        max_iter: Maximum number of iterations for Newton's method.

    Returns:
        x: Array of x values.
        y: Array of solution values (shape: (steps, n)).
    """
    x = np.linspace(xrange[0], xrange[1], steps + 1)
    y = np.zeros((steps, len(y0)))
    y[0] = y0

    for i in range(steps - 1):
        y_i = y[i]
        x_next = x[i + 1]
        g = lambda y_next: y_next - h * f(x_next, y_next, lambda_) - y_i
        y_next = y_i
        for _ in range(max_iter):
            residual = g(y_next)

            jacobian = np.eye(len(y0)) - h * finite_difference_jacobian(
                f, x_next, y_next, lambda_
            )

            y_next = y_next - np.linalg.solve(jacobian, residual)
            if np.linalg.norm(residual, ord=np.inf) < tol:
                break
        else:
            print(f"Warning: Newton's method did not converge at step {i + 1}")

        y[i + 1] = y_next

    return y


def finite_difference_jacobian(f, x, y, lambda_, epsilon=1e-6):
    n = len(y)
    jacobian = np.zeros((n, n))

    for j in range(n):
        y_perturbed = y.copy()
        y_perturbed[j] += epsilon
        f_perturbed = f(x, y_perturbed, lambda_)
        f_unperturbed = f(x, y, lambda_)
        jacobian[:, j] = (f_perturbed - f_unperturbed) / epsilon

    return jacobian


def fun(lam, xrange, h, steps, a1, a2, b1, b2):
    u1 = euler_method(first_order, xrange, np.array([1, 0]), h, lam, steps)
    u2 = euler_method(first_order, xrange, np.array([0, 1]), h, lam, steps)
    u1_a, v1_a = u1[0]
    u2_a, v2_a = u2[0]
    u1_b, v1_b = u1[-1]
    u2_b, v2_b = u2[-1]
    return np.linalg.det(
        np.array(
            [
                [a1 * u1_a + b1 * v1_a, a1 * u2_a + b1 * v2_a],
                [a2 * u1_b + b2 * v1_b, a2 * u2_b + b2 * v2_b],
            ]
        )
    )


def central(lam, xrange, h, steps, a1, a2, b1, b2, deltat=1e-6):
    fun1 = fun(lam + deltat, xrange, h, steps, a1, a2, b1, b2)
    fun2 = fun(lam - deltat, xrange, h, steps, a1, a2, b1, b2)
    return (fun1 - fun2) / (2 * deltat)


def newton(xrange, lam_init, a1, a2, b1, b2, h, steps, tol=1e-5, iter=100):
    x = lam_init
    for _ in range(iter):
        fx = fun(x, xrange, h, steps, a1, a2, b1, b2)
        df_dx = central(x, xrange, h, steps, a1, a2, b1, b2)

        if abs(fx) < tol:
            return x

        if df_dx == 0:
            raise ValueError("df_dx is zero in newton")

        x = x - fx / df_dx

    raise RuntimeError("Newton did not converge")


def plot(xrange, lam_values, h, steps=200):
    x_values = np.linspace(xrange[0], xrange[1], steps)
    plt.figure(figsize=(10, 8))
    for lam_value in lam_values:
        u1 = euler_method(first_order, xrange, np.array([1, 0]), h, lam_value, steps)
        u2 = euler_method(first_order, xrange, np.array([0, 1]), h, lam_value, steps)
        u1a, u1b = u1[0, 0], u1[-1, 0]
        u2a, u2b = u2[0, 0], u2[-1, 0]
        A = np.array([[u1a, u2a], [u1b, u2b]])

        c = np.array([1, 1])

        y = c[0] * u1[:, 0] + c[1] * u2[:, 0]

        if np.abs(np.linalg.det(A)) <= 1e-5:
            _, _, Vt = np.linalg.svd(A)
            c = Vt.T[:, -1]
            y = c[1] * u1[:, 0] + c[0] * u2[:, 0]

        plt.plot(x_values, y, label=f"λ = {lam_value:.4f}")

    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Eigenfunctions")
    plt.legend()
    plt.grid(True)
    plt.show()


xrange = [0, np.pi / 2]
a1, a2 = 1, 1
b1, b2 = 0, 0
steps = 100
h = (np.pi / 2) / steps
lambda_inits = np.array([1, 2, 3, 4, 5, 6, 7, 8])
lam_values = [newton(xrange, li, a1, b1, a2, b2, h, steps) for li in lambda_inits]
plot(xrange, lam_values, h, steps)
