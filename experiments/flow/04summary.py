import os
stat={}
for line in open("result_all.tsv"):
    arr=line.strip().split("\t")
    #config_study021_naive_5.retrain.yaml    0.32783845
    name=arr[0].split(".")[0]
    el=name.split("_")
    exp=el[1]
    method=el[2]
    trial=el[3]
    val=float(arr[1])
    ex_name=""
    if len(arr)>=4:
        ex_name=arr[3]
        ex_name=os.path.basename(ex_name)
    key=(exp,method,ex_name)
    if key not in stat:
        stat[key]=[]
    stat[key].append(val)
import numpy as np
for key,vals in stat.items():
    exp,method,ex_name=key
    m=np.mean(vals)
    s=np.std(vals)
    print(exp,method,ex_name,m,s)

