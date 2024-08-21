import optuna
import argparse
import os
import copy
import subprocess, shlex
import json
from ddm.util_config import DDMConfig, update_config_from_cli
from omegaconf import OmegaConf

def objective(trial,src_config,args):
    #x = trial.suggest_uniform('x', 0, 10)
    name="trial%04d"%(trial.number,)
    path=args.study_name+"/"+name
    config=copy.deepcopy(src_config)
    config["result_path"]=path
    ##
    #config["alpha_recons"] = trial.suggest_uniform("alpha_recons", 0, 1.0)
    #config["alpha_HJ"]     = 1.0- config["alpha_recons"]
    #config["alpha_gamma"] = trial.suggest_uniform("alpha_gamma", 0.0, 1.0)
    config["lr"]= trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["weight_decay"]= trial.suggest_float("weight_decay", 1e-10, 1e-6, log=True)
    config["eps_P"]= 1.0#trial.suggest_float("eps_P", 1e-8, 10, log=True)
    config["eps_f"]= 0.01#trial.suggest_float("eps_f", 1e-8, 10, log=True)
    config["eps_g"]= 0.01#trial.suggest_float("eps_g", 1e-8, 10, log=True)
    
    config["scale_f"]= trial.suggest_float("scale_f", 1.0e-5, 1.0, log=True)
    config["alpha_h"]= trial.suggest_float("alpha_h", 1.0e-10, 1.0, log=True)
    #config["residual_f"] = trial.suggest_categorical('residual_f', [True,False])
    #config["residual_h"] = trial.suggest_categorical('residual_h', [True,False])
    """
    config["scale_g"]= trial.suggest_float("scale_g", 1.0e-5, 1.0, log=True)
    config["scale_h"]= trial.suggest_float("scale_h", 1.0e-5, 1.0, log=True)
    config["scale_j"]= trial.suggest_float("scale_j", 1.0e-5, 1.0, log=True)
    config["scale_L"]= trial.suggest_float("scale_L", 1.0e-5, 1.0, log=True)

    config["with_bn_f"] = trial.suggest_categorical('with_bn_f', [True,False])
    config["with_bn_g"] = trial.suggest_categorical('with_bn_g', [True,False])
    config["with_bn_h"] = trial.suggest_categorical('with_bn_h', [True,False])
    config["with_bn_j"] = trial.suggest_categorical('with_bn_j', [True,False])
    config["with_bn_L"] = trial.suggest_categorical('with_bn_L', [True,False])
    """

    #config["v_type"] = trial.suggest_categorical('v_type', ['single','double','many'])
    config["activation"] = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'sigmoid'])
    config["optimizer"] = trial.suggest_categorical('optimizer', ['adamw', 'adam', 'rmsprop'])
    
    """
    config["detach_f"] = trial.suggest_categorical('detach_f', [True,False])
    config["detach_g"] = trial.suggest_categorical('detach_g', [True,False])
    config["detach_h"] = trial.suggest_categorical('detach_h', [True,False])
    config["detach_j"] = trial.suggest_categorical('detach_j', [True,False])
    config["detach_diff_f"] = trial.suggest_categorical('detach_diff_f', [True,False])
    config["detach_diff_g"] = trial.suggest_categorical('detach_diff_g', [True,False])
    """
    #config["diag_j"] = trial.suggest_categorical('diag_j', [True,False])
    #config["diag_g"] = trial.suggest_categorical('diag_g', [True,False])

    n_layer_f = trial.suggest_int('n_layer_f', 0, 3)
    n_layer_g = trial.suggest_int('n_layer_g', 0, 3)
    #n_layer_h = trial.suggest_int('n_layer_h', 0, 3)
    n_layer_h = 0
    hidden_layer_f=[]
    for i in range(n_layer_f):
        ii = trial.suggest_int("hidden_layer_f_{:02d}".format(i), 8, 32)
        hidden_layer_f.append(ii)
    config["hidden_layer_f"]=hidden_layer_f
    hidden_layer_g=[]
    for i in range(n_layer_g):
        ii = trial.suggest_int("hidden_layer_g_{:02d}".format(i), 8, 64)
        hidden_layer_g.append(ii)
    config["hidden_layer_g"]=hidden_layer_g
    hidden_layer_h=[]
    for i in range(n_layer_h):
        ii = trial.suggest_int("hidden_layer_h_{:02d}".format(i), 8, 64)
        hidden_layer_h.append(ii)
    config["hidden_layer_h"]=hidden_layer_h
    #config["state_dim"]= trial.suggest_int("state_dim", 2, 16)
    config["batch_size"]= trial.suggest_int("batch_size", 16, 128)
    ##########

    if args.dissipative_mode is not None:
        config["dissipative_mode"]=args.dissipative_mode
    if args.data_train is not None:
        config["data_train"]=args.data_train
    if args.data_test is not None:
        config["data_test"]=args.data_test


    ##########
    os.makedirs(path,exist_ok=True)
    conf_path=args.study_name+"/"+name+"/config.yaml"
    with open(conf_path,"w") as fp:
        print("[SAVE] config: ", conf_path)
        fp.write(OmegaConf.to_yaml(config))
    ## running command
    cmd=["ddm-train","--config",conf_path,"--seed",str(args.seed)]
    if args.gpu:
        cmd+=["--gpu",args.gpu]
    print("[EXEC]",cmd)
    #subprocess.Popen(shlex.split(cmd))
    subprocess.run(cmd)
    ##
    score=0
    result_path=path+"/model/best.result.json"
    print("[check] ",result_path)
    try:
        with open(result_path, "r") as fp:
            result=json.load(fp)
        score=result["valid-mse-loss"]
        #score=result["valid-loss"]
    except:
        score=1.0e10
    print("score:",score)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, nargs="+", help="config json file"
    )
    parser.add_argument(
        "--study_name", type=str, default="study", help="config json file"
    )
    parser.add_argument(
        "--db", type=str, default="./study.db", help="config json file"
    )
    parser.add_argument(
        "--n_trials", type=int, default=100, help="config json file"
    )
    parser.add_argument(
        "--output", type=str, default="study.csv", help="output csv file"
    )
    parser.add_argument(
            "--gpu", type=str, default=None, help="gpu"
    )
    parser.add_argument(
        "--dissipative_mode", type=str, default=None, help="opt"
    )
    parser.add_argument(
        "--data_train", type=str, default=None, help="opt"
    )
    parser.add_argument(
        "--data_test", type=str, default=None, help="opt"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="opt"
    )
    args = parser.parse_args()
    ## load config
    config=OmegaConf.structured(DDMConfig())
    for config_filename in args.config:
        print("[LOAD]",config_filename)
        conf_ = OmegaConf.load(config_filename)
        config = OmegaConf.merge(config, conf_)
    # start
    if os.path.exists(args.db):
        print("[REUSE]",args.db)
    else:
        print("[CREATE]",args.db)
    study = optuna.create_study(
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage='sqlite:///'+args.db,
        direction="minimize",
        load_if_exists=True)
    study.optimize(lambda trial: objective(trial,config,args), n_trials=args.n_trials)
    #study.optimize(objective, timeout=120)
    
    outfile = args.output
    study.trials_dataframe().to_csv(outfile)

if __name__ == "__main__":
    main()

