import numpy as np
import glob
import os
from numba import jit
import json

from ddm.data.util import minmax_normalize, z_normalize, save_dataset
from ddm.data.input_signal import generate_input

dh = 1e-1
T = 10

m = 1
d = 1
k = 1

A = np.array([[0,1],[-k/m,-d/m]])
#B = np.array([[1.0],[0]])
B = np.array([[0.0],[1/m]])
#C = np.array([[1.0,0]])
#C = np.array([[0.0,1.0]])
C = np.eye(2)


Q = np.array([[0,0],[0,-d]])
R = np.array([[0]])
S = np.array([[0],[1/2.0]])

def f(x,u,A,B):
    return  A.dot(x.reshape(-1,1)) + B.dot(u)

def compute_fx(x):
    out_fx=[]
    for i in range(x.shape[0]):
      fx=f_(x[i],A)
      out_fx.append(fx[:,0])
    out_fx=np.array(out_fx)
    return out_fx

def generate(N,output_x0=False,output_x1=False,init_random=False,input_type_id=1):
    times = np.arange(0,T,dh)
    step = times.shape[0]
    n =  A.shape[0]

    np.random.seed(0)

    u_data = u=generate_input(N,1,T,dh,input_type_id)#np.zeros((N,times.shape[0],1))
    x_data = np.zeros((N,times.shape[0],n))
    if output_x0 or output_x1:
        y_data = np.zeros((N,times.shape[0],1))
    else:
        y_data = np.zeros((N,times.shape[0],2))

    for i_N in range(N):

        u=u_data[i_N,:,0]

        x = np.zeros((step,n))
        #  初期値はランダム
        #x0 = 0.5*np.random.randn(n)
        if init_random:
            x0 = 0.5*np.random.randn(n)
        else:
            x0 = np.zeros((n,))


        x[0] = x0
        for k in range(times.shape[0]-1):
            x[k+1] = x[k] + dh*f(x[k],u[k],A,B).reshape(-1)

        y = C.dot(x.T)

        u_data[i_N,:,0] = u
        x_data[i_N,:,:] = x
        if output_x0:
            y_data[i_N,:,0] = y[0,:] # y: dim x time  
        elif output_x1:
            y_data[i_N,:,0] = y[1,:] # y: dim x time  
        else:
            y_data[i_N,:,:] = y.T # y: dim x time  

    y=np.array(y_data,dtype=np.float32)
    u=np.array(u_data,dtype=np.float32)
    x=np.array(x_data,dtype=np.float32)
    return x,u,y,y


def generate_dataset(args):
    N = args.num
    M = args.train_num
    name = args.prefix+args.mode
    path=args.path
    if args.input_type_id <0:
        input_type_id=1
    else:
        input_type_id = args.input_type_id
    output_x0=args.linear_x0
    output_x1=args.linear_x1
    init_random=args.init_random

    print("> T=",T, " dh=", dh)
    print("> intput ID",input_type_id)
    
    x_data, u_data, y_data, ys_data = generate(N,output_x0,output_x1,init_random, input_type_id)
    if not args.without_normalization:
        x_data, u_data, y_data, ys_data = z_normalize(x_data,u_data,y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name, T=T, dt=dh)


