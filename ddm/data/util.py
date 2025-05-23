import numpy as np
import os
import json
from ddm.util import NumPyArangeEncoder

def z_normalize(x_data,u_data,y_data,ys_data, path="dataset", name="none"):
    x_m=np.mean(x_data[:,:,:],axis=(0,1))
    y_m=np.mean(y_data[:,:,:],axis=(0,1))
    u_m=np.mean(u_data[:,:,:],axis=(0,1))
    x_std=np.std(x_data[:,:,:],axis=(0,1))
    y_std=np.std(y_data[:,:,:],axis=(0,1))
    u_std=np.std(u_data[:,:,:],axis=(0,1))
    x_data=(x_data-x_m)/x_std
    y_data=(y_data-y_m)/y_std
    u_data=(u_data-u_m)/(1.0+u_std)
    if ys_data is not None:
        ys_data=(ys_data-y_m)/y_std
    minmax_data={"name":name,
            "x_m":x_m,"y_m":y_m,"u_m":u_m,
            "x_std":x_std,"y_std":y_std,"u_std":u_std,
            }
    filename=path+"/"+name+".z.json"
    os.makedirs(path,exist_ok=True)
    json.dump(minmax_data,open(filename,"w"),
            cls=NumPyArangeEncoder)
    print("[SAVE]",filename)
    return x_data, u_data, y_data, ys_data


def minmax_normalize(x_data,u_data,y_data,ys_data, path="dataset", name="none"):
    x_max=np.max(x_data[:,:,:])
    y_max=np.max(y_data[:,:,:])
    u_max=np.max(u_data[:,:,:])
    x_min=np.min(x_data[:,:,:])
    y_min=np.min(y_data[:,:,:])
    u_min=np.min(u_data[:,:,:])
    x_data=(x_data-x_min)/(x_max-x_min)
    y_data=(y_data-y_min)/(y_max-y_min)
    u_data=(u_data-u_min)/(1+(u_max-u_min))
    if ys_data is not None:
        ys_data=(ys_data-y_min)/(y_max-y_min)
    minmax_data={"name":name,
            "x_max":x_max,"y_max":y_max,"u_max":u_max,
            "x_min":x_min,"y_min":y_min,"u_min":u_min,
            }
    filename=path+"/"+name+".minmax.json"
    os.makedirs(path,exist_ok=True)
    json.dump(minmax_data,open(filename,"w"),
            cls=NumPyArangeEncoder)
    print("[SAVE]",filename)
    return x_data, u_data, y_data, ys_data


def save_dataset(x_data, u_data, y_data, ys_data, M, path="dataset", name="none", T=1.0, dt=0.01):
    ###
    os.makedirs(path,exist_ok=True)
    ###
    info_data={"name":name, "path":path, "T":T,"dt":dt}
    filename=path+"/"+name+".info.json"
    print("[SAVE]",filename)
    json.dump(info_data,open(filename,"w"))
    ###
    filename=path+"/"+name+".train.obs.npy"
    print("[SAVE]",filename)
    print(y_data[:M].shape)
    np.save(filename,y_data[:M])
    filename=path+"/"+name+".test.obs.npy"
    print("[SAVE]",filename)
    print(y_data[M:].shape)
    np.save(filename,y_data[M:])

    filename=path+"/"+name+".train.input.npy"
    print("[SAVE]",filename)
    print(u_data[:M].shape)
    np.save(filename,u_data[:M])
    filename=path+"/"+name+".test.input.npy"
    print("[SAVE]",filename)
    print(u_data[M:].shape)
    np.save(filename,u_data[M:])

    filename=path+"/"+name+".train.state.npy"
    print("[SAVE]",filename)
    print(x_data[:M].shape)
    np.save(filename,x_data[:M])
    filename=path+"/"+name+".test.state.npy"
    print("[SAVE]",filename)
    print(x_data[M:].shape)
    np.save(filename,x_data[M:])

    if ys_data is not None:
        filename=path+"/"+name+".train.stable.npy"
        print("[SAVE]",filename)
        print(ys_data[:M].shape)
        np.save(filename,ys_data[:M])
        filename=path+"/"+name+".test.stable.npy"
        print("[SAVE]",filename)
        print(ys_data[M:].shape)
        np.save(filename,ys_data[M:])


