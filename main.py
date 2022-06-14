from re import X
import numpy as np


def create_equation(m: int, n: int, lamb: float=0.):
    """
    Using a randomly generated A ∈ R^m×n,
      some b ∈ R^m and some nonnegative λ ∈ R, 
      create the following problem
         min f(w) := || Aw − b ||_2^2 + λ || w ||_2^2
         A ∈ R^mn
         where m < n.
         Notice that f() is a L-smooth function and especially when λ > 0, f() is strongly convex.
    Solve the problem using seepest descent with some step-size 1/L by changing lambda.
    """

    # create assertions to check the input
    assert m < n, "m must be less than n"
    assert lamb >= 0, "lamb must be nonnegative"

    '''create weight matrix and bias vector
      A ∈ R^mn
      b ∈ R^m
      w ∈ R^n
    '''
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    #w = np.random.rand(n)
    f = lambda x: x*x + 2*x

    """
    # calc L2-norm of weight matrix
    l2 = np.linalg.norm(w)
    print("||w||_2^2", l2)
    """

    # create the objective function
    #f = lambda w: np.linalg.norm(A @ w - b)**2 + lamb * np.linalg.norm(w)**2

    # create the gradient of the objective function
    #grad_f = lambda w: 2 * A.T @ (A @ w - b) + 2 * lamb * w
    

    


def main():
    for l in [1]:
        create_equation(m = 2, n = 3, lamb=l)

    

if(__name__ == '__main__'):
    main()