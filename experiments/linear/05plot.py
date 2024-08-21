import matplotlib.pyplot as plt
import numpy as np
import os
from omegaconf import OmegaConf

from ddm.util_op import dissipative_w
from ddm.util_op import integral_w
from ddm.util import getQRS
import torch
dt=0.1
stat={}
print("..start")
for line in open("result_all.tsv"):
    arr=line.strip().split("\t")
    arr[0]=os.path.basename(arr[0])
    #config_study021_naive_5.retrain.yaml    0.32783845
    
    config_path=arr[0]
    print(config_path)
    config = OmegaConf.load(config_path)
    Q,R,S=getQRS(config)
    name=arr[0].split(".")[0]
    el=name.split("_")
    exp=el[1]
    method=el[2]
    trial=el[3]
    val=float(arr[1])
    npz_filename=arr[2]
    ex_name=""
    if len(arr)>=4:
        ex_name=arr[3]
    key=(exp,method,ex_name)
    if key not in stat:
        stat[key]=[]
    stat[key].append((npz_filename,Q,R,S))
dim=0
for key,el in stat.items():
    filenames=[e[0] for e in el]
    qrs=[(e[1],e[2],e[3]) for e in el]
    exp,method,ex_name=key
    objs=[np.load(filename) for filename in filenames]
    if ex_name=="":
        name=exp+"_"+method
    else:
        ex_name=os.path.basename(ex_name)
        name=exp+"_"+method+"_"+ex_name
    n=len(objs[0]["y_pred"])
    m=len(objs)
    print(key, "n=",n,"m=",m)
    y_true=objs[0]["y_true"]
    u_=objs[0]["u"]
    for idx in range(n):
        print("idx",idx)
        
        ###
        plt.plot(y_true[idx,:,dim],label="true",c="red")
        for i in range(m):
            y_pred=objs[i]["y_pred"]
            plt.plot(y_pred[idx,:,dim], label="pred", alpha=0.4)
        #plt.legend()
        save_filename="eval/{:02d}".format(idx)+name+".pred.png"
        print("[SAVE]",save_filename)
        plt.savefig(save_filename)
        plt.clf()
        plt.plot(u_[idx,:,dim])
        save_filename="eval/{:02d}".format(idx)+name+".u.png"
        print("[SAVE]",save_filename)
        plt.savefig(save_filename)
        plt.clf()
        for i in range(m):
            x_sol=objs[i]["x_sol"]
            plt.plot(x_sol[idx,:,dim])
        save_filename="eval/{:02d}".format(idx)+name+".x.png"
        print("[SAVE]",save_filename)
        plt.savefig(save_filename)
        plt.clf()
        

        ####
        with torch.no_grad():
            u =torch.tensor(u_[idx,:,:])
            y1=torch.tensor(y_true[idx,:,:])
            for i in range(m):
                Q,R,S=qrs[i]
                y_pred=objs[i]["y_pred"]
                x_sol=objs[i]["x_sol"]
                y2=torch.tensor(y_pred[idx,:,:])
                w,(yQy,uRu,ySu2)=dissipative_w(u,y2,Q,R,S)
                int_w=integral_w(w,dt=dt)
                plt.plot(w,label="w",alpha=0.5)
                #vx, _=model.v_nn(torch.tensor(x_sol[idx,:,:]))
                #vx=(torch.tensor(x_sol[idx,:,:])**2).sum(dim=(-1,))
                vx=torch.tensor(x_sol[idx,:,0])**2
                v1=vx-vx[0]
                plt.plot(int_w,label="int w")
                plt.plot(yQy,label="yQy")
                plt.plot(uRu,label="uRu")
                plt.plot(v1,label="V(x)-V(0)")#label="V(x)-V(0)")
                plt.legend()
                save_filename="eval/{:02d}".format(idx)+name+".w{}.png".format(i+1)
                print("[SAVE]",save_filename)
                plt.savefig(save_filename)
                plt.clf()

