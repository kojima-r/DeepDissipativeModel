import numpy as np
from numba import jit
import json
import glob
import os

from ddm.data.util import minmax_normalize, z_normalize, save_dataset
from ddm.data.input_signal import generate_input

dh = 1e-2
T = 100

@jit
def f(x,u):
    a = 0.7
    b = 0.8
    c = 3
    dx0 =  c*(x[:,0] + x[:,1]  - (x[:,0] **3)/3 + u[:,0])
    dx1 =  (-x[:,0] - b*x[:,1] + a)/c
    
    return  np.stack((dx0,dx1)).T

def compute_fx(x):
    a = 0.7
    b = 0.8
    c = 3
    dx0 =  c*(x[:,0] + x[:,1]  - (x[:,0] **3)/3 )
    dx1 =  (-x[:,0] - b*x[:,1] + a)/c
    
    return  np.stack((dx0,dx1)).T


@jit
def get_stable_x():
    n = 2
    N = 2 

    times = np.arange(0,T,dh)


    x0= np.zeros((N,n))
     
    x = np.zeros((N,times.shape[0],x0.shape[1]))

    x[:,0,:] = x0

    u = np.zeros((N,times.shape[0],1))
    u[1,:,:] = - 0.5

    for k in range(times.shape[0]-1):
        x[:,k+1] = x[:,k] + dh*f(x[:,k],u[:,k])    
    return x

def generate(N, input_type_id=4):
    np.random.seed(10)
    n = 2

    times = np.arange(0,T,dh)

    Uon = -0.5
    Uoff = 0

    Toff =  0.5*T*np.random.rand(N)
    Ton  =  0.5*T*np.random.rand(N)


    # generate x0
    x=get_stable_x()
    xs =x[0,-1,:]
    x0= xs * np.ones((N,n)) 
    # generate u
    u = u=generate_input(N,1,T,dh,input_type_id)#np.zeros((N,times.shape[0],1))
    # generate x
    x = np.zeros((N,times.shape[0],x0.shape[1]))
    x[:,0,:] = x0
    for k in range(times.shape[0]-1):
        x[:,k+1] = x[:,k] + dh*f(x[:,k],u[:,k])
    # cast
    u=np.array(u,dtype=np.float32)
    x=np.array(x,dtype=np.float32)
    y=x
    #ys=np.array(ys,dtype=np.float32)
    return x, u, y, None

def generate_dataset(args):
    N = args.num
    M = args.train_num
    name = args.prefix+args.mode
    path=args.path
    if args.input_type_id <0:
        input_type_id=4
    else:
        input_type_id = args.input_type_id

    print("> T=",T, " dh=", dh)
    print("> intput ID",input_type_id)

    x_data, u_data, y_data, ys_data = generate(N, input_type_id)
    if not args.without_normalization:
        x_data, u_data, y_data, ys_data = z_normalize(x_data,u_data,y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name, T=T, dt=dh)


