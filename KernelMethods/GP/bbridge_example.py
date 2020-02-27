import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as LA;


def genBrownianBridge(n):
    tSeq = np.arange(1 / float(n), 1, 1 / float(n));
    n = len(tSeq);
    sig = np.zeros((n, n), dtype='float64');
    for i in range(n):
        sig[i, 0:i] = tSeq[0:i];
        sig[i, i:] = tSeq[i];
    sig11 = sig;
    sig21 = tSeq;
    sig12 = np.transpose(sig21);
    muCond = np.zeros(n);

    sigCond = sig11 - np.outer(sig12, sig21);
    sigCondSqrt = LA.cholesky(sigCond, lower=True);
    z = muCond + np.dot(sigCondSqrt, np.random.randn(n));
    z = np.insert(z, 0, 0);
    return z;


## Quick illustration
n = 1000;
bm = genBrownianBridge(n);
plt.plot(bm);
plt.show();