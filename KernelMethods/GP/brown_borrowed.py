import numpy
import six
from matplotlib import pyplot


seed = 0
N = 100
M = 10

numpy.random.seed(seed)


def sample_path_batch(M, N):

    dt = 1.0 / N

    dt_sqrt = numpy.sqrt(dt)

    B = numpy.empty((M, N), dtype=numpy.float32)

    B[:, 0] = 0

    for n in six.moves.range(N - 1):

        t = n * dt

        #xi = numpy.random.randn(M) * dt_sqrt
        xi = numpy.random.normal(0,dt,M)

        B[:, n + 1] = B[:, n] * (1 - dt / (1 - t)) + xi

    return B


B = sample_path_batch(M, N)
pyplot.plot(B.T)
pyplot.show()