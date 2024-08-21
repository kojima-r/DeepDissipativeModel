import pickle
import glob
import matplotlib.pyplot as plt
import os
#result_study020_l2stable_5
import numpy as np
for filename in glob.glob("result_study*/model/train_loss.pkl"):
    obj=pickle.load(open(filename,"rb"))
    loss_t=[e["total"] for e in obj.loss_dict_history]
    plt.plot(loss_t,label="train",alpha=0.5)
    path=os.path.dirname(filename)
    name=path.split("/")[0]
    valid_filename=path+"/valid_loss.pkl"
    obj=pickle.load(open(valid_filename,"rb"))
    loss_v=[e["total"] for e in obj.loss_dict_history]
    plt.plot(loss_v,label="valid",alpha=0.5)
    min_val=min([np.nanmin(loss_t),np.nanmin(loss_v)])
    plt.title("total")
    plt.ylim(0,min_val*10)
    print("eval/"+name+".loss.png")
    plt.savefig("eval/"+name+".loss.png")
    plt.clf()

