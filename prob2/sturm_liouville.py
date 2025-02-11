import numpy as np
import matplotlib.pyplot as plt

# chebyshev's differential equation

# p = lambda x: -np.sqrt(1 - x**2)
# q = lambda _: 0
# w = lambda x: 1 / np.sqrt(1 - x**2)
# dp_dx = lambda x: x / p(x)
#
#
# def f(x, Y, lambda_):
#     u, v = Y
#     system_matrix = np.array(
#         [
#             [0, 1],
#             [(-q(x) + lambda_ * w(x)) / p(x), dp_dx(x) / p(x)],
#         ]
#     )
#     dY_dx = np.dot(system_matrix, np.array([u, v]))
#     return dY_dx


# example sturm liouville
m = 1


def cos_taylor(x):
    return 1 - x**2 / 2 + x**4 / 24


def sin_taylor(x):
    return x - x**3 / 6 + x**5 / 120


def f(x, Y, lambda_):
    u, v = Y

    a21 = (
        -2
        / cos_taylor(x) ** 4
        * (
            lambda_
            - (m**2 * cos_taylor(x) ** 2) / (2 * sin_taylor(x) ** 2)
            + cos_taylor(x) / sin_taylor(x)
        )
    )

    a22 = -cos_taylor(2 * x) / (cos_taylor(x) * sin_taylor(x))

    system_matrix = np.array([[0, 1], [a21, a22]])
    dY_dx = np.dot(system_matrix, np.array([u, v]))
    return dY_dx


def gauss_legendre_4(f, x_span, y0, h, lambda_, steps, tol=1e-6, max_iter=100):
    c = np.array([1 / 2 - np.sqrt(3) / 6, 1 / 2 + np.sqrt(3) / 6])
    a = np.array([[1 / 4, 1 / 4 - np.sqrt(3) / 6], [1 / 4 + np.sqrt(3) / 6, 1 / 4]])
    b = np.array([1 / 2, 1 / 2])

    x = np.linspace(x_span[0], x_span[-1], steps + 1)
    y = np.zeros((steps + 1, len(y0)))
    y[0] = y0

    for i in range(steps):
        x_i = x[i]
        y_i = y[i]

        K = np.zeros((len(c), len(y0)))

        def residual(K_flat):
            K = K_flat.reshape(len(c), -1)
            res = np.zeros_like(K)
            for j in range(len(c)):
                stage_sum = sum(a[j, k] * K[k] for k in range(len(c)))
                res[j] = K[j] - f(x_i + c[j] * h, y_i + h * stage_sum, lambda_)
            return res.flatten()

        def jacobian(K_flat):
            K = K_flat.reshape(len(c), -1)
            J = np.zeros((len(K_flat), len(K_flat)))

            for j in range(len(c)):
                for k in range(len(c)):
                    if j == k:
                        J_jk = np.eye(len(y0)) - h * a[j, k] * numerical_jacobian(
                            lambda Y: f(x_i + c[j] * h, Y, lambda_),
                            y_i + h * sum(a[j, l] * K[l] for l in range(len(c))),
                        )
                    else:
                        J_jk = (
                            -h
                            * a[j, k]
                            * numerical_jacobian(
                                lambda Y: f(x_i + c[j] * h, Y, lambda_),
                                y_i + h * sum(a[j, l] * K[l] for l in range(len(c))),
                            )
                        )
                    J[
                        j * len(y0) : (j + 1) * len(y0), k * len(y0) : (k + 1) * len(y0)
                    ] = J_jk

            return J

        for _ in range(max_iter):
            res = residual(K.flatten())
            if np.linalg.norm(res, ord=np.inf) < tol:
                break

            J = jacobian(K.flatten())
            delta = np.linalg.solve(J, -res)
            K = (K.flatten() + delta).reshape(len(c), -1)
        else:
            raise RuntimeError("Newton's method did not converge for implicit RK4.")

        y[i + 1] = y_i + h * sum(b[j] * K[j] for j in range(len(c)))

    return y


def gauss_legendre_2(f, x_span, y0, h, lambda_, steps, tol=1e-6, max_iter=150):
    c = np.array([0.5])
    a = np.array([[0.5]])
    b = np.array([1.0])

    x = np.linspace(x_span[0], x_span[-1], steps + 1)
    y = np.zeros((steps + 1, len(y0)))
    y[0] = y0

    for i in range(steps):
        x_i = x[i]
        y_i = y[i]

        K = np.zeros((len(c), len(y0)))

        def residual(K_flat):
            K = K_flat.reshape(len(c), -1)
            res = np.zeros_like(K)
            for j in range(len(c)):
                stage_sum = sum(a[j, k] * K[k] for k in range(len(c)))
                res[j] = K[j] - f(x_i + c[j] * h, y_i + h * stage_sum, lambda_)
            return res.flatten()

        def jacobian(K_flat):
            K = K_flat.reshape(len(c), -1)
            J = np.zeros((len(K_flat), len(K_flat)))

            for j in range(len(c)):
                for k in range(len(c)):
                    if j == k:
                        J_jk = np.eye(len(y0)) - h * a[j, k] * numerical_jacobian(
                            lambda Y: f(x_i + c[j] * h, Y, lambda_),
                            y_i + h * sum(a[j, l] * K[l] for l in range(len(c))),
                        )
                    else:
                        J_jk = (
                            -h
                            * a[j, k]
                            * numerical_jacobian(
                                lambda Y: f(x_i + c[j] * h, Y, lambda_),
                                y_i + h * sum(a[j, l] * K[l] for l in range(len(c))),
                            )
                        )
                    J[
                        j * len(y0) : (j + 1) * len(y0), k * len(y0) : (k + 1) * len(y0)
                    ] = J_jk

            return J

        for _ in range(max_iter):
            res = residual(K.flatten())
            if np.linalg.norm(res, ord=np.inf) < tol:
                break

            J = jacobian(K.flatten())
            delta = np.linalg.solve(J, -res)
            K = (K.flatten() + delta).reshape(len(c), -1)
        else:
            raise RuntimeError(
                "Newton's method did not converge for Gauss-Legendre second order."
            )

        y[i + 1] = y_i + h * sum(b[j] * K[j] for j in range(len(c)))

    return y


def numerical_jacobian(f, y, eps=1e-6):
    n = len(y)
    J = np.zeros((n, n))
    f_y = f(y)
    for i in range(n):
        y_perturbed = y.copy()
        y_perturbed[i] += eps
        f_perturbed = f(y_perturbed)
        J[:, i] = (f_perturbed - f_y) / eps
    return J


def obj_func(lambda_, x_span, h, steps, alpha1, beta1, alpha2, beta2):
    Y1 = gauss_legendre_4(f, x_span, [1, 0], h, lambda_, steps)
    Y2 = gauss_legendre_4(f, x_span, [0, 1], h, lambda_, steps)

    u1_a, v1_a = Y1[0]
    u2_a, v2_a = Y2[0]
    u1_b, v1_b = Y1[-1]
    u2_b, v2_b = Y2[-1]

    det_matrix = np.array(
        [
            [alpha1 * u1_a + beta1 * v1_a, alpha1 * u2_a + beta1 * v2_a],
            [alpha2 * u1_b + beta2 * v1_b, alpha2 * u2_b + beta2 * v2_b],
        ]
    )

    return np.linalg.det(det_matrix)


def obj_func_derivative(
    lambda_, x_span, h, steps, alpha1, beta1, alpha2, beta2, delta=1e-5
):
    return (
        obj_func(lambda_ + delta, x_span, h, steps, alpha1, beta1, alpha2, beta2)
        - obj_func(lambda_ - delta, x_span, h, steps, alpha1, beta1, alpha2, beta2)
    ) / (2 * delta)


def newton_method(
    x_span,
    lambda_initial,
    alpha1,
    beta1,
    alpha2,
    beta2,
    h,
    steps,
    tol=1e-6,
    max_iter=150,
):
    lambda_ = lambda_initial
    for _ in range(max_iter):
        f_lambda = obj_func(lambda_, x_span, h, steps, alpha1, beta1, alpha2, beta2)
        f_prime_lambda = obj_func_derivative(
            lambda_, x_span, h, steps, alpha1, beta1, alpha2, beta2
        )

        if abs(f_lambda) < tol:
            return lambda_

        if f_prime_lambda == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")

        lambda_ -= f_lambda / f_prime_lambda

    raise RuntimeError("Newton's method did not converge.")


def plot_eigenfunctions(x_span, lambda_values, h=0.01, steps=200):
    x_vals = np.linspace(x_span[0], x_span[-1], steps + 1)
    plt.figure(figsize=(10, 6))

    for lambda_val in lambda_values:
        Y1 = gauss_legendre_4(f, x_span, [1, 0], h, lambda_val, steps)
        Y2 = gauss_legendre_4(f, x_span, [0, 1], h, lambda_val, steps)

        u1a, u1b = Y1[0, 0], Y1[-1, 0]
        u2a, u2b = Y2[0, 0], Y2[-1, 0]

        A = np.array([[u1a, u2a], [u1b, u2b]])
        b = np.array([0, 0])

        c = np.array([1, 1])

        y = c[0] * Y1[:, 0] + c[1] * Y2[:, 0]

        if np.abs(np.linalg.det(A)) <= 1e-5:
            _, _, Vt = np.linalg.svd(A)

            c = Vt.T[:, -1]
            y = c[1] * Y1[:, 0] + c[0] * Y2[:, 0]

        plt.plot(x_vals, y, label=f"λ = {lambda_val:.4f}")

    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Eigenfunctions")
    plt.legend()
    plt.grid(True)
    plt.show()


# for chebyshev
# a, b = -1, 1

# for example
a, b = 0, np.pi / 2

alpha1, beta1 = 1, 0
alpha2, beta2 = 1, 0
x_span = [a, b]
steps = 300
h = (b - a) / steps

# for chebyshev
# lambda_initials = np.array([1, 5, 10, 14, 24, 35, 48, 66])


# for example
lambda_initials = np.array([1, 2, 3, 4, 5, 7, 8, 10])


roots = [
    newton_method(x_span, li, alpha1, beta1, alpha2, beta2, h, steps)
    for li in lambda_initials
]

print("Found roots:")
for i, root in enumerate(roots):
    print(f"Root {i + 1}: {root}")

plot_eigenfunctions(x_span, roots, h=h, steps=steps)
