from typing import List, Callable, Tuple
import numpy as np


class SteepestDescent(object):
    def __init__(self,
            step_size: float, 
            df: Callable):
        self.eta = step_size # learning rate
        self.df = df # f()', not DataFrame


    def update(self, w, b) -> Tuple[np.ndarray, np.ndarray]:
        w = w - self.eta * self.df.dw(w, b)
        b = b - self.eta * self.df.db(b)
        return w, b


class Momentum(object):
    def __init__(self, 
            step_size: float, 
            df: Callable,
            momentum: float=0.9
            ):
        assert momentum < 1.0, "momentum value must be less than 1.0"
        assert momentum > 0.0, "momentum value must be greater than 1.0"

        self.alpha = step_size # learning rate
        self.beta = momentum
        self.df = df # f()', not DataFrame

        self.gw_prev = None
        self.gb_prev = None


    def update(self, w, b) -> Tuple[np.ndarray, np.ndarray]:
        gw = self.df.dw(w, b)
        gb = self.df.db(b)

        # insert zeros for the first time
        if(self.gw_prev is not None): 
            pass
        else:
            self.gw_prev = np.zeros_like(gw)
            self.gb_prev = np.zeros_like(gb)

        # gradient with momentum
        gw = gw + self.beta * self.gw_prev
        gb = gb + self.beta * self.gb_prev

        # minimization
        w = w - self.alpha * gw
        b = b - self.alpha * gb

        self.gw_prev = gw
        self.gb_prev = gb
        return w, b


class Nesterov(object):
    """[Nesterov, 83]
    """
    def __init__(self, 
            step_size: float, 
            df: Callable,
            momentum: float=0.9
            ):
        assert momentum < 1.0, "momentum value must be less than 1.0"
        assert momentum > 0.0, "momentum value must be greater than 1.0"

        self.alpha = step_size # learning rate
        self.beta = momentum
        self.df = df # f()', not DataFrame

        self.vw_prev = None
        self.vb_prev = None


    def update(self, w, b) -> Tuple[np.ndarray, np.ndarray]:
        gw = self.df.dw(w, b)
        gb = self.df.db(b)

        # insert zeros for the first time
        if(self.vw_prev is not None): 
            pass
        else:
            self.vw_prev = np.zeros_like(gw)
            self.vb_prev = np.zeros_like(gb)


        vw = self.beta * self.vw_prev + gw
        vb = self.beta * self.vb_prev + gb

        # minimization
        w = w - self.alpha * (gw + self.beta * vw)
        b = b - self.alpha * (gb + self.beta * vb)

        self.vw_prev = vw
        self.vb_prev = vb
        return w, b