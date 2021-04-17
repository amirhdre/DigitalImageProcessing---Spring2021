'''
___OVERVIEW___
CODE FOR QUESTION 4 - HW 0

___SUMMARY___
Matplotlib and Numpy Excercise  

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
2/26/2021

'''

import numpy as np
import matplotlib.pyplot as plt

def myFunc(rng=[0,1], shp=[1,2]):

    arr = np.random.uniform(rng[0],rng[1],shp)
    print(arr,arr.shape)
    fig, ax = plt.subplots(1,2)
    fig.suptitle('HW0-P4-9733023')
    
    line0, = ax[0].plot(arr[:,0],label='Column 1')
    ax[0].plot(arr[:,0],'ok',alpha=0.2)
    line0.set_linestyle('dashed')
    line0.set_color('r')

    line1, = ax[1].plot(arr[:,1],label='Column 2')
    ax[1].plot(arr[:,1],'ok',alpha=0.2)
    line1.set_linestyle('dashdot')
    line1.set_color('b') 
    
    ax[0].legend()   
    ax[1].legend()   
    ax[0].set_aspect('equal', adjustable="datalim")
    ax[1].set_aspect('equal', adjustable="datalim")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('HW0-P4-9733023.png')
    plt.show()

myFunc([-20,20],[10,2])
