cmd_list=[]
#fp=open("run_all.list","w")
for seed in range(1,6):
    for size in range(1,4):
        #fp=open("run_all."+str(seed)+"."+str(size)+".list","w")
        for mode in ["naive","passive","l2stable","stable","dissipative"]:
            #mode="dissipative"
            cmd="sh ./run_cmd.sh {} 1{}0 {}".format(seed,size,mode)
            print(cmd)
            cmd_list.append(cmd)
            #fp.write(cmd)
            #fp.write("\n")


group_by = 4
arr=[cmd_list[i:i + group_by] for i in range(0, len(cmd_list), group_by)]
for i,cmds in enumerate(arr):
    fp=open("run_all.{:02d}.list".format(i),"w")
    for cmd in cmds:
        fp.write(cmd)
        fp.write("\n")

