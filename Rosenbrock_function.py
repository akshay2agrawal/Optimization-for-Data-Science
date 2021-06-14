import numpy as np


def counter(fn):
    def _counted(*largs, **kargs):
        _counted.call += 1
        return fn(*largs, **kargs)

    _counted.call = 0
    return _counted


@counter
def f(x):
    x = np.array(x)
    """
    The Rosenbrock function.

    """
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


@counter
def df2dt(x):
    x = [2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0] ** 2)]
    return np.array(x)


@counter
def hessf2(x):
    x = [[-400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2, -400 * x[0]],
         [-400 * x[0], 200]]
    return np.array(x)


def generaized_descent(start_point, fx, df2dt, hessf2, direction='newton', Cd=0.05,
                       maxiter=10 ** 6, epsilon=10 ** -4, delta=10 ** -4, eta=5.0):
    iteration = 0
    x = start_point
    xtrack = []
    finished = False

    while not finished:
        gradient = df2dt(x)
        finished = np.linalg.norm(gradient) <= epsilon or iteration > maxiter

        # choosing direction
        d = -1 * gradient
        if not finished:
            if direction == 'newton':
                try:
                    d = np.linalg.solve(hessf2(x), (-1 * gradient))

                    if -1 * ((gradient @ d) / (np.linalg.norm(gradient) * np.linalg.norm(d))) < Cd:
                        d = -gradient
                    else:
                        pass


                except np.linalg.LinAlgError:
                    d = -gradient
            elif direction == 'gradient':
                d = -gradient

            # choosing stepsize
            rho = armijo(fx, df2dt, d, x, delta, eta)
            x += rho * d
        xtrack.append([x[0], x[1]])
        iteration += 1
    return iteration, xtrack


def armijo(f, dfdt, direction, startval, delta=10 ** -4, eta=2.0):
    roh = 1.0
    x = startval

    while f(x + roh * direction) <= f(x) + roh * delta * (dfdt(x) @ direction):
        roh = eta * roh

    while f(x + roh * direction) > f(x) + roh * delta * (dfdt(x) @ direction):
        roh = eta ** -1 * roh

    return roh


itr, points = generaized_descent([3, 2], f, df2dt, hessf2)


for p in points:
    print(p)

print("total iteration " + str(itr))
print("function calls " + str(f.call) + " times")
print("1st derivative calls " + str(df2dt.call) + " times")
print("2nd derivative calls " + str(hessf2.call) + " times")
