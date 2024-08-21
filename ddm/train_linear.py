import numpy as np
import joblib
import json
import logging
import argparse
from matplotlib import pylab as plt
import os 
import ddm.model_linear as linear
import ddm.model_ar as ar
from ddm.util_config import DDMConfig, update_config_from_cli
from ddm.util_data import load_data
from ddm.util import set_file_logger

ss_type = ["ORT", "MOESP", "ORT_auto", "MOESP_auto"]
ar_type = ["ARX", "PWARX", "ARX_auto", "PWARX_auto"]

def simulate(model_name,model,u,y0):
    n_temp=y0.shape[1]
    N=u.shape[0]
    stable_y=[]
    out_y=[]
    for i in range(N):
        if model_name in ss_type:
            x0 = model.predict_initial_state(u[i,:n_temp],y0[i,:n_temp])
        elif model_name in ar_type:
            x0 = model.predict_initial_state(y0[i])
        else:
            print("unknown model:", model_name)
        y =model.predict(x0,u[i])
        stable_y.append(y[-1,:])
        out_y.append(y)
    stable_y=np.array(stable_y)
    out_y=np.array(out_y)
    return out_y, stable_y



def run_pred_mode(config, logger):
    # ... loading data
    logger.info("... loading data")
    all_data = load_data(mode="test", config=config, logger=None)
    # ... confirmation of input data and dimensions
    model_name=config["method"] 
    print("method:", model_name)
    print("data_size:", all_data.num)
    print("observation dimension:", all_data.obs_dim)
    print("input dimension:", all_data.input_dim)
    print("state dimension:", all_data.state_dim)
    obs_dim = all_data.obs_dim
    input_dim = all_data.input_dim
    state_dim = all_data.state_dim
    # ... defining data
    y_test =all_data.obs
    u_test =all_data.input
    x_test =all_data.state
    ##
    # ... loading
    model_path=config["result_path"]+"/linear"
    filename = model_path+"/"+model_name+".pkl"
    print("[LOAD]", filename)
    model = joblib.load(filename)

    ####
    mean_error = model.score(u_test,y_test)
    logger.info("mean error: {}".format(mean_error))
    # チェックに使うデータを決める
    # 初期値の推定(最初の2点のデータのみ)
    #obs_gen=[]
    #n=len(u_test)
    #n_temp=20
    #obs_gen, _ = simulate(model_name,model,u_test,y_test[:,:n_temp,:])
    ####
    #print(obs_gen)

def run_train_mode(config, logger):
    # ... loading data
    logger.info("... loading data")
    train_data = load_data(mode="train", config=config, logger=None)
    # ... confirmation of input data and dimensions
    model_name=config["method"] 
    print("method:", model_name)
    print("data_size:", train_data.num)
    print("observation dimension:", train_data.obs_dim)
    print("input dimension:", train_data.input_dim)
    print("state dimension:", train_data.state_dim)
    obs_dim = train_data.obs_dim
    # ... defining data
    y_train=train_data.obs
    u_train=train_data.input
    x_train=train_data.state
    # ... training
    n = config["state_dim"]
    k = 20
    if config["method"] == "ORT":
        model=linear.ORT(n=n, k =k,initial_state =  'estimate')
        model.fit(u_train,y_train)
    elif config["method"] == "MOESP":
        model=linear.MOESP(n=n, k =k,initial_state =  'estimate')
        model.fit(u_train,y_train)
    elif config["method"] == "ARX":
        #nty=2,ntu=2,N_max = 1000,N_alpha = 10,max_class = 25
        model=ar.ARX()
        model.fit(u_train,y_train)
    elif config["method"] == "PWARX":
        model=ar.PWARX()
        model.fit(u_train,y_train)
    elif config["method"] == "ORT_auto":
        model=linear.ORT(n=n, k =k,initial_state =  'estimate')
        n,k = model.autofit(u_train,y_train,n_max = 10)
    elif config["method"] == "MOESP_auto":
        model=linear.ORT(n=n, k =k,initial_state =  'estimate')
        n,k = model.autofit(u_train,y_train,n_max = 10)
    elif config["method"] == "ARX_auto":
        model=ar.ARX()
        nty,ntu=model.autofit(u_train,y_train)
    elif config["method"] == "PWARX_auto":
        model=ar.PWARX()
        nty,ntu=model.autofit(u_train,y_train)
    else:
        print("unknown method:",config["method"])
        exit()
    ##
    mean_error = model.score(u_train,y_train)
    logger.info("mean error: {}".format(mean_error))
    # ...saving
    model_path=config["result_path"]+"/linear"
    os.makedirs(model_path,exist_ok=True)
    filename = model_path+"/"+model_name+".pkl"
    print("[SAVE]", filename)
    joblib.dump(model, filename)
    ##
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train/infer")
    parser.add_argument(
        "--config", type=str, default=None, nargs="+", help="config json file"
    )
    m=["ORT","MOESP","ARX","PWARX","ORT_auto","MOESP_auto","ARX_auto","PWARX_auto"]
    parser.add_argument(
            "--method", type=str, default="MOESP", nargs="?", help="method:"+str(m)
    )
     
    parser.add_argument("--model", type=str, default=None, help="model")
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode (calcuration only with cpu)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="constraint gpus (default: all) (e.g. --gpu 0,2)",
    )
    parser.add_argument("--profile", default=None, action="store_true", help="")
    #####
    reserved_args=["profile","gpu","cpu","config","model", "mode", "method"]
    config, args = update_config_from_cli(parser, reserved_args)
    #####
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")
    config["method"]=args.method
    os.makedirs(config["result_path"],exist_ok=True)
    # setup
    mode_list = args.mode.split(",")
    for mode in mode_list:
        # mode
        if mode == "train":
            set_file_logger(logger,config,"log_linear_train."+config["method"]+".txt")
            run_train_mode(config, logger)
        elif mode == "test":
            set_file_logger(logger,config,"log_linear_test."+config["method"]+".txt")
            run_pred_mode(config, logger)

if __name__ == "__main__":
    main()
