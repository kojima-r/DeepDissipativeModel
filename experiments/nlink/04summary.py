stat={}
for line in open("result_all.tsv"):
    arr=line.strip().split("\t")
    #config_study021_naive_5.retrain.yaml    0.32783845
    name=arr[0].split(".")[0]
    el=name.split("_")
    exp=el[1]
    method=el[2]
    trial=el[3]
    if len(arr)>3:
        dataset=arr[3]
    else:
        dataset="test"
    val=float(arr[1])
    key=(exp,method,dataset)
    if key not in stat:
        stat[key]=[]
    stat[key].append(val)
import numpy as np
for key,vals in stat.items():
    exp,method,dataset=key
    m=np.mean(vals)
    s=np.std(vals)
    print(exp,method,dataset,m,s)

