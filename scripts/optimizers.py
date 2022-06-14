
def steepest_descent(w, b, df_dw, df_db, step_size):
    w = w - step_size * df_dw(w, b)
    b = b - step_size * df_db(b)
    return w, b


def nesterov(w, b, df_dw, df_db, step_size):
    # TODO
    w_prev = w
    w = w - step_size * df_dw(w, b)
    b = b - step_size * df_db(b)
    w = w + step_size * df_dw(w, b)
    w = w_prev + (1 + step_size * df_dw(w, b)) * (w - w_prev)
    return w, b