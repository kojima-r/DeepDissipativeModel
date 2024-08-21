#study020_dissipative_1/trial0032/log_train.txt:perf_time: 114.63 sec (0.03 h)
import glob
import os
not_end_list=[]
lines=[]
for path in glob.glob("result_study*/"):
    flag=False
    filename=path+"log_train.txt"
    if os.path.exists(filename):
        for line in open(filename):
            if "perf_time" in line:
                print(filename, line.strip())
                lines.append(filename+":"+line.strip())
                flag=True
    if not flag:
        not_end_list.append(filename)

with open("time_all.txt", "w") as fp:
    for line in lines:
        fp.write(line)
        fp.write("\n")
#study020_dissipative_1/trial0032/log_train.txt:perf_time: 114.63 sec (0.03 h)
data={}
for line in open("time_all.txt"):
    if len(line)>0:
        a=line.split("/")[0]
        names=a.split("_")
        name=names[1]+"_"+names[2]
        time_s=line.split(" ")[1]
        if name not in data:
            data[name]=[]
        data[name].append(float(time_s)/(60*60))

import numpy as np
with open("summary_all_time.tsv","w") as fp:
    line = "exp\tmethod\tmean (hour)\tstd\tmin_time\tmax_time"
    print(line)
    fp.write(line)
    fp.write("\n")
    for key,vals in data.items():
        exp,method=key.split("_")
        m=np.mean(vals)
        s=np.std(vals)
        min_v=np.min(vals)
        max_v=np.max(vals)
        print(exp,method,m,s,min_v,max_v)
        line="\t".join(map(str,[exp,method,m,s,min_v,max_v]))
        fp.write(line)
        fp.write("\n")



