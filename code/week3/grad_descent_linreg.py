import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def run():
    A = np.array([
        [1, 2],
        [1, 0],
        [1, -2],
        [1, -1],
        [1, 1]
    ])

    b = np.array([2, 0, -2, 1, -1]).transpose()

    X = A[:, 0]
    Y = b

    # Values to setup for initialization
    m = 0.6
    theta1 = 5
    theta0 = 100
    alpha = 0.001
    tol = 0.001
    max_iters = 100

    o = []
    e = (theta1 * X + theta0 - Y)
    o.append((e*e).sum())

    for i in range(1, max_iters):
        temp0 = theta0 - (1 / m) * alpha * sum(theta1 * X + theta0 - Y)
        temp1 = theta1 - (1 / m) * alpha * sum((theta1 * X + theta0 - Y) * X)
        theta0 = temp0
        theta1 = temp1
        e = (theta1 * X + theta0 - Y)
        o.append((e * e).sum())
        
        if (i > 1) and (abs(o[i] - o[i-1]) < tol):
            break;

    # print(o)

    # plt.plot(o)
    # plt.show()
    theta = np.array([1, 0]).transpose() # our guess y=x
    AT = A.transpose()
    theta = inv(AT.dot(A)).dot(AT).dot(b)
    e = A.dot(theta) - b # A x theta -b
    J = (e*e).sum() # elementwise mult of e with itself or e2 and sum

    print(f"A = {A}")
    print(f"b = {b}")
    print(f"Error = {J}")


if __name__ == '__main__':
    run()
