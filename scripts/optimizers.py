from typing import List, Callable, Tuple
import numpy as np


class SteepestDescent(object):
    def __init__(self,
            step_size: float, 
            df: Callable):
        self.step_size = step_size
        self.df = df


    def update(self, w, b) -> Tuple[np.ndarray, np.ndarray]:
        w = w - self.step_size * self.df.dw(w, b)
        b = b - self.step_size * self.df.db(b)
        return w, b


class Momentum(object):
    def __init__(self, 
            step_size: float, 
            df: Callable,
            momentum: float=0.9
            ):
        assert momentum < 1.0, "momentum value must be less than 1.0"

        self.step_size = step_size
        self.df = df

        self.momentum = 0.9


    def update(self, w, b) -> Tuple[np.ndarray, np.ndarray]:
        gw_prev = w
        gb_prev = b

        # steepest descent
        w = w - self.step_size * self.df.dw(w, b)
        b = b - self.step_size * self.df.db(b)

        # update momemutum
        w = w + self.momentum * (w - gw_prev)
        b = b + self.momentum * (b - gb_prev)

        return w, b