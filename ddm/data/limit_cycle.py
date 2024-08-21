import numpy as np
import glob
import os
from numba import jit
import json

from ddm.data.util import minmax_normalize, z_normalize, save_dataset
from ddm.data.input_signal import generate_input

dh = 1e-1
T = 10

@jit
def f(x,u):
    dx0 =  x[:,0] - x[:,1]  - x[:,0] * (x[:,0]**2 +x[:,1]**2) + x[:,0] * u[:,0]
    dx1 =  x[:,0] + x[:,1]  - x[:,1] * (x[:,0]**2 +x[:,1]**2) +x[:,1] * u[:,0]
    return  np.stack((dx0,dx1)).T

def compute_fx(x):
    y=f(x,0)
    return y

def generate(N, input_type_id=2):
    np.random.seed(0)
    u_sigma = 0.5

    n = 2
    times = np.arange(0,T,dh)


    x0= np.random.randn(N,n)
    u = generate_input(N,1,T,dh,input_type_id)
    x = np.zeros((N,times.shape[0],x0.shape[1]))

    x[:,0,:] = x0
    for k in range(times.shape[0]-1):
        x[:,k+1] = x[:,k] + dh*f(x[:,k],u[:,k])


    ys = np.zeros((N,times.shape[0],x0.shape[1]))
    ys[:,:,0] = x[:,:,0]  / np.sqrt(x[:,:,0]**2 + x[:,:,1]**2)
    ys[:,:,1] = x[:,:,1]  / np.sqrt(x[:,:,0]**2 + x[:,:,1]**2)

    u=np.array(u,dtype=np.float32)
    x=np.array(x,dtype=np.float32)
    y=x
    ys=np.array(ys,dtype=np.float32)
    
    return x, u, y, ys

def generate_dataset(args):
    N = args.num
    M = args.train_num
    name = args.prefix+args.mode
    path=args.path
    if args.input_type_id <0:
        input_type_id=2
    else:
        input_type_id = args.input_type_id

    print("> T=",T, " dh=", dh)
    print("> intput ID",input_type_id)

    x_data, u_data, y_data, ys_data = generate(N, input_type_id)
    if not args.without_normalization:
        x_data, u_data, y_data, ys_data = z_normalize(x_data,u_data,y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name, T=T, dt=dh)


