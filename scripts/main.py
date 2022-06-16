from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tqdm import tqdm
from easydict import EasyDict

from optimizers import SteepestDescent, Momentum, Nesterov


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
    if(update_rule == 'steepest'):
        optimizer = SteepestDescent(step_size, df)
    elif(update_rule == 'momentum'):
        optimizer = Momentum(step_size, df)
    elif(update_rule == 'nesterov'):
        optimizer = Nesterov(step_size, df)
    else:
        raise ValueError("update_rule must be 'steepest' or 'nesterov' or 'momentum'")
    

    # create a loop to update w and b
    pbar = tqdm(range(L))
    for i in pbar:
        error = f(w)
        errors.append(error)
        w, b = optimizer.update(w, b)
        #print(f"step: {i+1}, error: {error: .6f}")
        pbar.set_description(f"step: {i+1}, error: {error: .6f}")
    return errors


def show_graph(all_errors: list, 
               lambdas: list, 
               Ls: list, 
               _update_rule: str,
               fname: str,
               ) -> None:
    colors = ['gray', 'salmon', 'orangered'] # with lambda = 0, 1, 10

    plt.figure()
    plt.ylim(0, 10)
    for i, l in enumerate(lambdas):
        plt.plot(np.arange(len(all_errors[i])), all_errors[i], color=colors[i])
    plt.legend([f"l={l}, L={L}" for l, L in zip(lambdas, Ls)])
    plt.xlabel("iteration k")
    plt.ylabel("f(w_k)")
    plt.title(f'opt: {_update_rule}')
    plt.savefig(f"results/{fname}.png")
    plt.close()


def main():
    '''Q1: create problem'''
    print("[Q1]")
    lambdas = [0, 1, 10]
    L = 500 # n-steps
    m, n = 4, 32

    all_errors = []

    _update_rule = 'steepest'
    print(f"[update rule]: {_update_rule}")

    for l in lambdas:
        errors = create_equation(m, n, lamb=l, L=L, update_rule=_update_rule)
        all_errors.append(errors)
    show_graph(all_errors, lambdas, [L], _update_rule, fname=f"q1-steepest")


    '''Q2 backtracking'''
    print("[Q2]")
    lambdas = [0, 0]
    Ls = [500, 1000] # n-steps
    m, n = 4, 32

    all_errors = []
    print(f"[update rule]: {_update_rule}")

    all_errors = []
    for i, L in enumerate(Ls):
        print(L)
        errors = create_equation(m, n, lamb=lambdas[i], L=L, update_rule=_update_rule)
        all_errors.append(errors)
    show_graph(all_errors, lambdas, Ls, _update_rule, fname=f"q2-backtracking")


    '''Q3'''
    print("Q3")
    
    lambdas = [0, 1, 10]
    L = 500 # n-steps
    m, n = 4, 32

    for _update_rule in ['steepest', 'momentum', 'nesterov']:
        all_errors = []
        print(f"[update rule]: {_update_rule}")

        for l in lambdas:
            errors = create_equation(m, n, lamb=l, L=L, update_rule=_update_rule)
            all_errors.append(errors)
        show_graph(all_errors, lambdas, [L], _update_rule, fname=f"q3-{_update_rule}")


if(__name__ == '__main__'):
    main()