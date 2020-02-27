import matplotlib.pyplot as plt
import numpy as np

import six

"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""

# File: brownian.py

from math import sqrt
from scipy.stats import norm
import numpy as np


def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta * sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def run_bm(runs,T):

    for r in range(runs):
        b_t = [0.0]
        b = 0
        for t in range(1,T + 1):
            b = b + np.random.normal(0,1,1)[0]
            b_t.append(b)

        plt.plot([x for x in range(T + 1)],b_t)

    plt.show()


def run_bm_bridge(runs,T):

    for r in range(runs):

        b_t = [0.0]
        dt_list = []
        dt = 1.0 / T
        b = 0.0

        for t in range(T):

            t = t*dt

            cov = t*(1 - t - dt)

            if  1.0:

                #b = b*(1 - (dt / (1 - t))) +  dt*np.random.normal(0,t*(1 - t),1)[0]

                b = b*(1 - (dt / (1 - t))) + np.random.normal(0,cov,1)[0]
            else:

                #b = dt*np.random.normal(0,t*(1 - t),1)[0]
                b = 5.0

            b_t.append(b)

            dt_list.append(dt)

        plt.title('b(t)')
        plt.plot([x / T for x in range(T + 1)],b_t)

    plt.show()


def brownian_bridge(Z):
    '''
    Take in brownian motion process Z
    and return the brownian bridge
    :param Z:
    :return:
    '''

    N = Z.shape[0]
    dt = 1.0 / (N - 1)
    B = []
    b = 0
    for t in range(1,N):
        s = t*dt

        if s >= 1:
            print('S is done {}'.format(s))
            B.append(0)
        else:
            b = Z[t] - s * Z[1]
            B.append(b)


    return B


def sample_path_batch(M, N):
    dt = 1.0 / N
    dt_sqrt = np.sqrt(dt)
    B = np.empty((M, N), dtype=np.float32)
    B[:, 0] = 0
    for n in six.moves.range(N - 1):
        t = n * dt
        xi = np.random.randn(M) * dt_sqrt
        B[:, n + 1] = B[:, n] * (1 - dt / (1 - t)) + xi
    return B



if __name__ == '__main__':

    run_graph_bm = 0
    if run_graph_bm:

        # The Wiener process parameter.
        delta = 1
        # Total time.
        T = 1.0
        # Number of steps.
        N = 200
        # Time step size
        dt = T / N
        # Number of realizations to generate.
        m = 20
        # Create an empty array to store the realizations.
        x = np.empty((m, N + 1))
        # Initial values of x.
        x[:, 0] = 0

        brownian(x[:, 0], N, dt, delta, out=x[:, 1:])

        t = np.linspace(0.0, N * dt, N + 1)
        #for k in range(m):
        #    plt.plot(t, x[k])
        #plt.xlabel('t', fontsize=16)
        #plt.ylabel('x', fontsize=16)
        #plt.grid(True)
        #plt.show()

        for k in range(m):

            #B = brownian_bridge(np.reshape(x[k],(x[k].shape[1],)))
            B = brownian_bridge(x[k])

            plt.plot([y / N for y in range(1,N + 1)],B)

        plt.show()


    run_bm_experiment = 0
    if run_bm_experiment:
        RUNS = 50
        T = 100
        run_bm(RUNS,T)

    run_bm_bridge_experiment = 1
    if run_bm_bridge_experiment:
        RUNS = 3
        T = 100
        run_bm_bridge(RUNS,T)

    run_version_blah = 0
    if run_version_blah:

        B = sample_path_batch(1,100)
        plt.plot(B.T)
        #for i in range(B.shape[1]):
        #    plt.plot([x for x in range(B.shape[0])],B[:,i])

        plt.show()
