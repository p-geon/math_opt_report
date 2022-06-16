from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tqdm import tqdm
from easydict import EasyDict

from optimizers import SteepestDescent, Momentum


def init_variables(m: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # create weight vector, bias vector and data matrix
    w = np.random.rand(n) # weight: Nx1
    b = np.random.rand(m) # bias: Mx1
    A = np.random.rand(m, n) # data: MxN
    return w, b, A


def create_equation(
                m: int, 
                n: int, 
                lamb: float=0, # L2 regularization
                L: float = 100, # reciprocal of step size
                update_rule: str='steepest descent', # 'steepest descent' or 'nesterov'
                ) -> None:
    # create assertions to check the input
    assert m < n, "m must be less than n"
    assert lamb >= 0, "lamb must be nonnegative"

    step_size = 1/L
    errors = []

    w, b, A = init_variables(m, n)


    # function and derivatives
    f = lambda w: np.linalg.norm(b - A.dot(w)) + lamb * np.linalg.norm(w)
    df = EasyDict({
        'dw': lambda w, b: 2 * (b - A.dot(w)).dot(-A) + 2 * lamb * w,
        'db': lambda b: 2 * b,
        })


    # define the optimizer
    if(update_rule == 'steepest descent'):
        optimizer = SteepestDescent(step_size, df)
    elif(update_rule == 'momentum'):
        optimizer = Momentum(step_size, df)
    else:
        raise ValueError("update_rule must be 'steepest descent' or 'nesterov'")
    

    # create a loop to update w and b
    pbar = tqdm(range(L))
    for i in pbar:
        error = f(w)
        errors.append(error)

        w, b = optimizer.update(w, b)

        pbar.set_description(f"step: {i+1}, error: {error: .6f}")
    return errors


def show_graph(all_errors: list, 
               lambdas: list, 
               L: int, 
               _update_rule: str,
               ) -> None:
    colors = ['gray', 'salmon', 'orangered'] # with lambda = 0, 1, 10

    plt.figure()
    plt.ylim(0, 10)
    for i, l in enumerate(lambdas):
        plt.plot(np.arange(len(all_errors[i])), all_errors[i], color=colors[i])
    plt.legend([f"l={l}, L={L}" for l in lambdas])
    plt.xlabel("iteration k")
    plt.ylabel("f(w_k)")
    plt.title(f'opt: {_update_rule}')
    plt.savefig(f"results/error.png")
    plt.close()


def main():
    # Q1: create problem
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
    ===
    - 点がどれくらいのスピードで近づいていくか。古いタイプの解析
    - 強凸を仮定するとだいたい一時収束
    - 強凸を仮定せずにどのくらいの収束スピードになるのか = iteration complexity
    - convergence speed
    ===
    この問題の本質は MxN の比率かもしれない
    - n がでかいと不安定になる
    - L を大きくすると安定する
    """
    # consts
    lambdas = [0, 1, 10]
    L = 500 # n-steps
    m, n = 4, 32

    all_errors = []

    _update_rule = ['steepest descent', 'momentum'][1]
    print(f"[update rule]: {_update_rule}")

    for l in lambdas:
        errors = create_equation(m, n, lamb=l, L=L, update_rule=_update_rule)
        all_errors.append(errors)
    show_graph(all_errors, lambdas, L, _update_rule)


    # Q2 backtracking
    """
    Change the step-size for the steepest descent method from 
      the constant L1 to the one chosen by the backtracking method with 
      Armijo rule, and compare the performance between
      two step-size rules for some fixed λ.
    ===
    - ステップサイズの変更
    - backtracking method is 何
    - 2種
    """

    # Q3
    """
    Implement the Nesterov’s accelerated gradient algorithm with 
      some α ̃k and β ̃k and compare the performance to
      the steepest descent method developed in Q1.
    - Nesterov
    """

if(__name__ == '__main__'):
    main()