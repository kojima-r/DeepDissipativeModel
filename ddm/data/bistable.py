import numpy as np
import glob
import os
from numba import jit
import json

from ddm.data.util import minmax_normalize, z_normalize, save_dataset
from ddm.data.input_signal import generate_input

dh = 1e-1
T = 10

#f(x)=-1/4(x^2 -1)**2=-1/4*(x^4-2x^2+1)
@jit
def f(x,u):
    return  x*(1- x**2) + u

def compute_fx(x):
    return x*(1- x**2)

def generate(N=20000, input_type_id=5):
    np.random.seed(0)
    dh = 1e-1
    T = 10
    times = np.arange(0,T,dh)
    n_steps=times.shape[0]
    
    #  初期値は0
    """
    x01 = -np.ones(N//2)
    x02 = np.ones(N//2)
    x0 = np.concatenate([x01,x02],axis=0)
    np.random.shuffle(x0)
    """
    u_data = generate_input(N,1,T,dh,input_type_id)
    x0 = -np.ones(N)
    for i in range(N):
        r=np.random.randint(0,2)
        if r==0:
            x0[i]=1
        elif r==1:
            x0[i]=-1
    x = np.zeros((N,times.shape[0],1))

    x[:,0,0] = x0

    for k in range(times.shape[0]-1):
        x[:,k+1] = x[:,k] + dh*f(x[:,k],u_data[:,k])

    ys = np.ones(x.shape)
    ys[x<0] = -1
    return x,u_data,x,ys

def generate_dataset(args):
    N = args.num
    M = args.train_num
    name = args.prefix+args.mode
    path=args.path
    if args.input_type_id <0:
        input_type_id=5
    else:
        input_type_id = args.input_type_id

    print("> T=",T, " dh=", dh)
    print("> intput ID",input_type_id)

    x_data, u_data, y_data, ys_data = generate(N, input_type_id)
    if not args.without_normalization:
        x_data, u_data, y_data, ys_data = z_normalize(x_data, u_data, y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name, T=T, dt=dh)


