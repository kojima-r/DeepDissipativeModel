from ddm.trainer import SystemTrainer
from ddm.model import NeuralDissipativeSystem
from omegaconf import OmegaConf
import torch
import numpy as np
import logging
import argparse
import ddm.util_data
from ddm.util_data import DiosDataset
import os
import pickle
from ddm.util import getQRS
logging.basicConfig(level=logging.INFO)

external_list=[
    "dataset/ex1linear.test",
    "dataset/ex2linear.test",
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, default=None, nargs="*", help="/"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="/"
    )
    args = parser.parse_args()
    #config_path=args.config
    results=[]
    for external,ex_name in [(False,"")]+[(True,e) for e in external_list]:
        for config_path in args.config:
            print("[LOAD]",config_path)
            config = OmegaConf.load(config_path)
            if args.cpu or not torch.cuda.is_available():
                device="cpu"
            else:
                device="cuda"
            path= config["result_path"]+"/model/best.checkpoint"
            model=NeuralDissipativeSystem(config=config,device=device)
            trainer=SystemTrainer(config=config,model=model,device=device)
            trainer.load_ckpt(path)
            Q,R,S=getQRS(config)
            Q, R, S=Q.to(device), R.to(device), S.to(device)
            model.set_dissipative(Q,R,S)
            ###
            if not external:
                base_path,_=os.path.splitext(config["data_train"])
                test_name=base_path+".test"
                #continue
            else:
                test_name=ex_name
            ###
            test_data=ddm.util_data.load_all_data(test_name,{})
            print(test_data.input_dim, model.in_dim,"obs", test_data.obs_dim, model.obs_dim)
            if test_data.input_dim != model.in_dim or test_data.obs_dim != model.obs_dim:
                print("mismatch!!")
                continue
            ###
            test_loss,y_pred, x_sol, x0 =trainer.pred(test_data,batch_size=args.batch_size,device=device)
            y_true=test_data.obs
            print("y_true:",y_true.shape)
            print("y_pred:",y_pred.shape)
            
            mse=np.mean(np.sum((y_true-y_pred)**2,axis=(-2,-1)))
            ###
            idx=np.arange(len(y_true))
            np.random.seed(0)
            np.random.shuffle(idx)
            ###
            n=10
            idx_=idx[:n]
            y_true_=y_true[idx_,:,:]
            y_pred_=y_pred[idx_,:,:]
            x_sol_=x_sol[idx_,:,:]
            u=test_data.input[idx_,:,:]
            os.makedirs("eval",exist_ok=True)
            if not external:
                name,_=os.path.splitext(os.path.basename(config_path))
            else:
                name,_=os.path.splitext(os.path.basename(config_path))
                name = name+"."+os.path.basename(ex_name)
            npz_filename="eval/"+name+".npz"
            np.savez(npz_filename,y_true=y_true_, y_pred=y_pred_,x_sol=x_sol_,u=u, test_data_idx=idx_)
            #####
            results.append((config_path,mse,npz_filename,ex_name))

    with open("result_all.tsv","w") as fp:
        for res in results:
            config_path,mse,npz_filename,ex_name = res
            s=str(config_path)+"\t"+str(mse)+"\t"+npz_filename+"\t"+ex_name
            fp.write(s)
            fp.write("\n")
        

if __name__ == "__main__":
    main()
