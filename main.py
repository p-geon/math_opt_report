"""
Using a randomly generated A ∈ R^m×n,
some b ∈ R^m and some nonnegative λ ∈ R, 
create the following problem
    min f(w) := || Aw − b ||_2^2 + λ || w ||_2^2
A ∈ R^mn, where m < n.
Notice that f() is a L-smooth function and especially when λ > 0, f() is strongly convex.
Solve the problem using seepest descent with some step-size 1/L by changing lambda.
Then show plots with the iteration number k in the horizontal axis
and f(w_k) in the vertical axis to confirm the iteration complexity.
"""
from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def create_equation(
                    m: int=2, 
                    n: int=3, 
                    k: int=200, # iteration
                    lamb: float=0, # L2 regularization
                    L: float = 100, # reciprocal of step size
                    ) -> None:
    # create assertions to check the input
    assert m < n, "m must be less than n"
    assert lamb >= 0, "lamb must be nonnegative"

    step_size = 1/L
    errors = []

    # create weight vector, bias vector and data matrix
    w = np.random.rand(n) # weight: Nx1
    b = np.random.rand(m) # bias: Mx1
    A = np.random.rand(m, n) # data: MxN

    # function and derivatives
    f = lambda w: np.linalg.norm(b - A.dot(w)) + lamb * np.linalg.norm(w)
    df_dw = lambda w, b: 2 * (b - A.dot(w)).dot(-A) + 2 * lamb * w
    df_db = lambda b: 2 * b
    

    # create a loop to update w and b
    for i in range(k):
        error = f(w)
        errors.append(error)

        w = w - step_size * df_dw(w, b)
        b = b - step_size * df_db(b)
        print(f"step: {i+1}, error: {error: .6f}")

    return errors


def main():
    L = 100
    lambdas = [0, 1, 10]

    plt.figure()
    for l in lambdas:
        errors = create_equation(lamb=l, L=L)
        plt.plot(np.arange(len(errors)), errors)

    plt.legend([f"l={l}, L={L}" for l in lambdas])
    plt.xlabel("iteration k")
    plt.ylabel("f(w_k)")
    plt.savefig(f"results/error.png")
    plt.show()
    plt.close()
    

if(__name__ == '__main__'):
    main()