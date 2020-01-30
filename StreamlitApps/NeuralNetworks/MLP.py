'''
Module for exploring MLP (Multilayer perceptrons) same as NN
For CS 545 ML PSU class
'''

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# This is necessary for importing from outisde the current directory while running streamlit
import sys
sys.path.append('.')

def sigmoid(x):

    return 1 / (1 + np.exp(-1.0*x))

def main():

    st.text('Viewing components of a NN')

    x = np.linspace(-10,10,100)
    y = sigmoid(x)
    plt.plot(x,y)
    st.pyplot()


    #st.line_chart(y)



if __name__ == '__main__':

    main()
