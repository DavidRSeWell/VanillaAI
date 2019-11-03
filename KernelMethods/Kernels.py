import numpy as np

def kernel_k1(x,y):
    return np.dot(x,y)

def kernel_k2(x,y,p=2):

    k1 = kernel_k1(x,y)

    k2 = (1 + k1)**p
    return k2

def kernel_k3(x,y,a=1,b=1):

    return np.tanh(a*kernel_k1(x,y) + b)


class KernelsDict(object):
    '''
    I am a simple dictionary.
    You tell me what kernel you want
    I give you that kernel
    If I dont know that kernel I blow up
    '''
    def __init__(self):
        self.kernels = [
            'k1'
        ]
        pass

    def kernel_k1(self,x, y):
        return np.inner(x, y)

    def kernel_k2(self,x, y, p=5):
        k1 = self.kernel_k1(x, y)

        k2 = (1 + k1) ** p
        return k2

    def kernel_k3(self,x, y, a=1, b=1):
        return np.tanh(a * self.kernel_k1(x, y) + b)
