import numpy as np
import time

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
def dfdt(x):
    x = [2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0] ** 2)]
    return np.array(x)

def globalized_bfgs(start_point, fx, dfdt, Hessian, direction='newton', Cd = 0.05,
                       maxiter=10 ** 6, epsilon=10 ** -4, delta=10 ** -4, eta=3.0):
    iteration = 0
    x = start_point
    xtrack = []
    finished = False

    while not finished:
        gradient = dfdt(x)
        oldx = x.copy()
        finished = np.linalg.norm(gradient) <= epsilon or iteration > maxiter

        # choosing direction
        if not finished:
            if direction == 'newton':
                try:
                    d = np.linalg.solve(Hessian, (-1 * gradient))
                    if -1 * ((gradient @ d) / (np.linalg.norm(gradient) * np.linalg.norm(d))) < Cd:
                        d = -gradient
                    else:
                        pass

                except np.linalg.LinAlgError:
                    d = -gradient
                    # print('gradient')

            elif direction == 'gradient':
                d = -gradient

            # choosing stepsize
            rho = armijo(fx, dfdt, d, x, delta, eta)
            x += rho * d

            # update hessian
            y = dfdt(x) - gradient
            s = x - oldx


            # broyden update rank 1
            Hessian = Hessian + ((y - Hessian * s) * s.T) / (s.T * s)

            # bfgs update rack 2
            # try :
            # if np.isnan(Hessian).any() == True:
            #     Hessian = np.array([[1,0],[0,1]])
            #
            # print((Hessian * s * (Hessian * s).T),(s.T * Hessian * s))

            # Hessian = Hessian + ((y * y.T) / (y.T * s)) - np.divide((Hessian * s * (Hessian * s).T) , (s.T * Hessian * s))

            if np.isnan(Hessian).any() == True:
                Hessian = np.array([[1,0],[0,1]])



            # except  :

            # if np.isnan(Hessian).any() == True :
            #     Hessian = np.array([[2,1],[1,2]])


            # print( Hessian)
            # print (y,s)
            # time.sleep(1)

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



itr, points = globalized_bfgs([3, 2], f, dfdt, np.array([[2,1],
                                                           [1,2]]))


for p in points:
    print(p)

print("total iteration " + str(itr))
print("function calls " + str(f.call) + " times")
print("1st derivative calls " + str(dfdt.call) + " times")
# print("2nd derivative calls " + str(hessf2.call) + " times")
