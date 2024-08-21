#fp=open("run_all.list","w")
for seed in range(1,6):
    fp=open("run_all."+str(seed)+".list","w")
    for size in range(1,4):
        #size=2
        for n in range(2,4): #nlink
            for mode in ["naive","l2stable","stable","dissipative"]:
                cmd="sh ./run_cmd.sh {} 0{}{} {}".format(seed,size,n,mode)
                print(cmd)
                fp.write(cmd)
                fp.write("\n")
