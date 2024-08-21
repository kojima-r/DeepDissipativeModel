import os
stat={}

for line in open("result_all.tsv"):
    arr=line.strip().split("\t")
    #config_study021_naive_5.retrain.yaml    0.32783845
    if len(arr)>2:
        arr[0]=os.path.basename(arr[0])
        name=arr[0].split(".")[0]
        el=name.split("_")
        #print(el)
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
with open("summary_all.tsv","w") as fp:
    line = "exp\tmethod\tex_name\tmean\tstd\tmin_v\tmax_v\tdata"
    print(line)
    fp.write(line)
    fp.write("\n")
    for key,vals in stat.items():
        exp,method,ex_name=key
        m=np.mean(vals)
        s=np.std(vals)
        min_v=np.min(vals)
        max_v=np.max(vals)
        print(exp,method,ex_name,m,s,min_v,max_v)
        s=",".join(map(str,vals))
        line="\t".join(map(str,[exp,method,ex_name,m,s,min_v,max_v,s]))
        fp.write(line)
        fp.write("\n")



