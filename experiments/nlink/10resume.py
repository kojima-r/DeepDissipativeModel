import glob
import os
not_end_list=[]
data={}
for path in glob.glob("./result_study*/"):
    flag=False
    filename=path+"log_train.txt"
    if os.path.exists(filename):
        for line in open(filename):
            if "perf_time" in line:
                print(filename, line.strip())
                flag=True
    if not flag:
        not_end_list.append(filename)

def get_latest_ckpt(target_name):
    latest_nid=None
    for filename in glob.glob("result_"+target_name+"/model/model.*.checkpoint"):
        fn=os.path.basename(filename)
        nid=int(fn.split(".")[1])
        if latest_nid is None or latest_nid<nid:
            latest_nid=nid
    if latest_nid is not None:
        s=target_name+"/model/model."+str(latest_nid)+".checkpoint"
        return s
    else:
        return None

with open("resume.list1.sh","w") as ofp1:
    with open("resume.list2.sh","w") as ofp2:
        for i, name in enumerate(not_end_list):
            name=os.path.dirname(name)
            name=os.path.basename(name)
            #print(name)
            arr=name.split("_")[1:]
            target_name="_".join(arr)
            ckpt=get_latest_ckpt(target_name)
            print(target_name,ckpt)
            if ckpt is not None:
                cmd="ddm-train --config config_" \
                    + target_name+".retrain.yaml --epoch 5000 --resume ./result_" \
                    + ckpt
            else:
                cmd="ddm-train --config config_" \
                    + target_name+".retrain.yaml --epoch 5000"
            
            ofp1.write("echo \"resume{:03d}\"".format(i))
            ofp1.write("\n")
            #ofp1.write("cp ./result_{}/log_train.txt ./result_{}/log_train.first.txt  ".format(target_name,target_name))
            #ofp1.write("cp ./result_{}/log_train.txt ./result_{}/log_train.third.txt  ".format(target_name,target_name))
            ofp1.write("cp ./result_{}/log_train.txt ./result_{}/log_train.fourth.txt  ".format(target_name,target_name))
            ofp1.write("\n")
            ofp2.write(cmd)
            ofp2.write("\n")

print("[save]","resume.list1.sh")
print("[save]","resume.list2.sh")
quit()

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

