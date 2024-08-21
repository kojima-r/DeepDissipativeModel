import sys
import argparse
import glob
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from ddm.util_config import DDMConfig
from omegaconf import OmegaConf


def get_opt(filename):
    fp=open(filename)
    h=next(fp)
    data=[]
    for line in fp:
        arr=line.strip().split(",")
        trial=arr[1]
        f=float(arr[2])
        #print((f,trial))
        data.append((f,trial))
    data=sorted(data)
    return data[0][1]

def plot_opt(basepath="study",keyword="total", k=5, mode="train", min_mode=True, trial_names=None, target_trial_names=None):
    """
    This method shows the optimization results.
    The top-k or selected trials will be highlighted.
    Args:
      basepath (str): optimization directory
      keyword (str): total/
      k  (int): highlight top-k trials
      mode (str): train/valid/test
      min_mode (bool): this flag should be true if optimization is minimization
      trial_names (List[str]): all trial lists. If this trial_names is None, all trial is obtained from study directory.
      target_trial_names (List[str]): specifies the trials to highlight in a list
    Returns:
      out_name_list (List[str]:
    """
    ## construct filename_list
    filename_list=[]
    if trial_names is None:
        for filename in glob.glob(basepath+"/**/model/"+mode+"_loss.pkl"):
            if os.path.isfile(filename):
                filename_list.append(filename)
            else:
                print("skip:", filename)
    else:
        for name in trial_names:
            filename =basepath+"/"+name+"/model/"+mode+"_loss.pkl"
            if os.path.isfile(filename):
                filename_list.append(filename)
            else:
                print("skip:", filename)
    ## load data from filename_list
    data=[]
    for filename in filename_list:
        obj=pickle.load(open(filename,"rb"))
        val=[e[keyword] for e in obj.loss_dict_history]
        if min_mode:
            val_opt=min(val)
        else:
            val_opt=max(val)
        name=filename.split("/")[1]
        data.append((name, val,val_opt))

    if target_trial_names is not None:
        target_index_list=[]
        for i,e in enumerate(data):
            if e[0] in target_trial_names:
                target_index_list.append(i)
    elif min_mode:
        target_index_list=np.array([val_opt for _,_,val_opt in data]).argsort()[:k]
    else:
        target_index_list=np.array([val_opt for _,_,val_opt in data]).argsort()[-k:]
    ## plot
    for i in target_index_list:
        name,val,val_opt=data[i]
        plt.plot(val,label=name)
    for i,el in enumerate(data):
        name,val,val_opt=el
        if i not in target_index_list:
            plt.plot(val,":",alpha=0.2)
    best_i=target_index_list[0]
    best_val=data[best_i][2]
    print(best_i,data[best_i][0],best_val)
    plt.title(mode+" "+keyword)
    plt.legend()
    ## name list
    out_name_list=[data[i][0] for i in target_index_list]
    return out_name_list,data[best_i][0],best_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study_name", type=str, default="study", help="output csv file"
    )
    parser.add_argument(
        "--input", type=str, default="study.csv", help="output csv file"
    )
    args = parser.parse_args()
    #infile = args.input
    #opt_id=get_opt(infile)
    #print(opt_id)
    name=args.study_name
    print(name)
    out_name_list, best_name, best_val=plot_opt(basepath=name,keyword="total", k=5)
    plt.ylim(0,best_val*4)
    filename=name+".total.png"
    print("[SAVE]",filename)
    plt.savefig(filename)
    print(out_name_list)
    print(best_name, best_val)
    #################
    config_filename=name+'/'+best_name+'/config.yaml'
    config=OmegaConf.structured(DDMConfig())
    print("[LOAD]",config_filename)
    conf_ = OmegaConf.load(config_filename)
    config = OmegaConf.merge(config, conf_)

    config["init_model"]=name+'/'+best_name+'/model/best.checkpoint'
    config["result_path"]='result_'+name
    print(config)
    conf_path="config_"+name+".retrain.yaml"
    with open(conf_path,"w") as fp:
        print("[SAVE] config: ", conf_path)
        fp.write(OmegaConf.to_yaml(config))
    #
    config

if __name__ == "__main__":
    main()

