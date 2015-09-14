
def minmax(x, y):
    mins, maxs = 0.0, 0.0
    for i in range(x.shape[0]):
        a, b = x[i], y[i]
        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a
    if maxs > 0.0:
        return 1.0 - (mins / maxs)
    return 0.0