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


def create_equation(
                    m: int, 
                    n: int, 
                    k: int=100, # iteration
                    lamb: float=0, # L2 regularization
                    L: float = 100, # reciprocal of step size
                    ) -> None:
    # create assertions to check the input
    assert m < n, "m must be less than n"
    assert lamb >= 0, "lamb must be nonnegative"

    errors = []

    '''create weight matrix and bias vector
      A ∈ R^mn
      b ∈ R^m
      w ∈ R^n
    '''
    A = np.random.rand(m, n) # data: MxN
    b = np.random.rand(m) # bias: Mx1
    w = np.random.rand(n) # weight: Nx1
    f = lambda w: np.linalg.norm(b - A.dot(w)) + lamb * np.linalg.norm(w)
    #df_dw = lambda w: 2 * A.T.dot(A.dot(w) - b) + 2 * lamb * w
    df_dw = lambda w: 2 * (b - A.dot(w)).dot(-A) + 2 * lamb * w
    df_db = lambda b: 2 * b
    
    step_size = 1/L

    # create a loop to update w and b
    for i in range(k):
        error = f(w)
        errors.append(error)

        w = w - step_size * df_dw(w)
        b = b - step_size * df_db(b)
        print(f"step: {i+1}, error: {error: .6f}")
    #print(errors)



    """
    # calc L2-norm of weight matrix
    l2 = np.linalg.norm(w)
    print("||w||_2^2", l2)
    """

    # create the objective function
    #f = lambda w: np.linalg.norm(A @ w - b)**2 + lamb * np.linalg.norm(w)**2

    

def main():
    for l in [1]:
        create_equation(m = 2, n = 3, lamb=l)
    

if(__name__ == '__main__'):
    main()


"""
def create_f():
    '''
    Create a L-smooth function.
        f(x) = x^2 + 2x + 1.
    '''
    f = lambda x: x*x + 2*x + 1
    g = lambda x: 2*x + 2

    
    return f, g

def update_weight_and_bias(
        w: np.ndarray,
        b: np.ndarray, 
        error: np.float32,
        l: float) -> List[np.ndarray, np.ndarray]:
print(f"type: ", type(error))

# calculate gradient

# update w
# update b
return w, b

"""