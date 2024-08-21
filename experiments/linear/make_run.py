#fp=open("run_all.list","w")
for seed in range(1,6):
    fp=open("run_all."+str(seed)+".list","w")
    for size in range(1,3):
        for mode in ["naive","l2stable","stable","dissipative"]:
            #mode="dissipative"
            cmd="sh ./run_cmd.sh {} 0{}0 {}".format(seed,size,mode)
            print(cmd)
            fp.write(cmd)
            fp.write("\n")
