#study020_dissipative_1/trial0032/log_train.txt:perf_time: 114.63 sec (0.03 h)
data={}
for line in open("time_all.txt"):
    a=line.split("/")[0]
    name=a.split("_")[1]
    time_s=line.split(" ")[1]
    if name not in data:
        data[name]=[]
    data[name].append(float(time_s))
import numpy as np
for k,v in data.items():
    print(k,np.sum(v)/(60*60))

