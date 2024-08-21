import logging
import json
import numpy as np
import torch
import os
import argparse

def set_file_logger(logger,config,filename):
    filename=config["result_path"]+"/"+filename
    h = logging.FileHandler(filename=filename, mode="w")
    h.setLevel(logging.INFO)
    logger.addHandler(h)
 
class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

def init_gpu(args):
    # gpu/cpu
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # cuda
    if torch.cuda.is_available():
        device = 'cuda'
        print("device: cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        print("device: cpu")
    return device

def init_log(config, filename):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")
    set_file_logger(logger, config, filename)
    return logger

def init_result(config):
    os.makedirs(config["result_path"],exist_ok=True)
    os.makedirs(config["result_path"]+"/model",exist_ok=True)

def init_argparse(parser):
    parser.add_argument(
        "--config", type=str, default=None, nargs="+", help="config json file"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode (calcuration only with cpu)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="constraint gpus (default: all) (e.g. --gpu 0,2)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="resume training using checkpoint (default: all) (e.g. --resume epoch111.ckpt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed",
    )
    parser.add_argument("--profile",default=None, action="store_true", help="")
    reserved_args=["profile","gpu","cpu","config", "resume","seed"]
    return reserved_args

def init_dim_from_data(config,train_data):
    print("observation dimension:", train_data.obs_dim)
    print("input dimension:", train_data.input_dim)
    print("state dimension(data):", train_data.state_dim)
    state_dim = config["state_dim"]
    print("state dimension:", state_dim)
    in_dim = train_data.input_dim if train_data.input_dim is not None else 1
    config["in_dim"]=in_dim
    obs_dim = train_data.obs_dim
    config["obs_dim"]=obs_dim
    return in_dim, obs_dim, state_dim

def getQRS(config):
    in_dim=config["in_dim"]
    obs_dim=config["obs_dim"]
    if config["Q"] is None:
        Q=torch.tensor(-np.eye(obs_dim),dtype=torch.float32)
    else:
        Q_np=np.array(config["Q"]).reshape((obs_dim,obs_dim))
        Q=torch.tensor(Q_np,dtype=torch.float32)
    if config["R"] is None:
        R=torch.tensor(np.eye(in_dim)*config["gamma"],dtype=torch.float32)
    else:
        R_np=np.array(config["R"]).reshape((in_dim,in_dim))
        R=torch.tensor(R_np,dtype=torch.float32)
    if config["S"] is None:
        S=torch.tensor(np.zeros((obs_dim,in_dim)),dtype=torch.float32)
    else:
        S_np=np.array(config["S"]).reshape((obs_dim,in_dim))
        S=torch.tensor(S_np,dtype=torch.float32)
    print("Q:",Q)
    print("R:",R)
    print("S:",S)
    return Q,R,S
