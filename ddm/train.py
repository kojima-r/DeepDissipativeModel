import os
import logging

import numpy as np
import joblib
import argparse
from omegaconf import OmegaConf

from ddm.trainer import SystemTrainer
from ddm.model import NeuralDissipativeSystem
from ddm.util_config import DDMConfig, update_config_from_cli
from ddm.util_data import load_data
from ddm.util import init_gpu, init_log, init_argparse, init_result, init_dim_from_data, getQRS

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(config, logger, device, resume_ckpt=None):
    logger.info("... loading data")
    all_data = load_data(mode="train", config=config, logger=logger)
    train_data, valid_data = all_data.split(1.0 - config["train_valid_ratio"])
    print("train_data_size:", train_data.num)
    in_dim, obs_dim, state_dim=init_dim_from_data(config,train_data)

    # defining dimensions from given data
    # defining system
    flag=False
    max_retry=10
    retry_cnt=0
    while not flag and retry_cnt < max_retry:
        logger.info("... trial: {}".format(retry_cnt))
        # setting up Q, R, S
        Q,R,S=getQRS(config)
        Q, R, S=Q.to(device), R.to(device), S.to(device)
        # model construction
        model = NeuralDissipativeSystem(
                config=config,
                device=device
                )
        model.set_dissipative(Q,R,S)
        # training NN from data
        trainer = SystemTrainer(config, model, device=device)
        if config["init_model"]!="":
            trainer.load_ckpt(config["init_model"])
        train_loss,valid_loss,flag=trainer.fit(train_data, valid_data, resume_ckpt=resume_ckpt)
        retry_cnt+=1

    joblib.dump(train_loss, config["result_path"]+"/model/train_loss.pkl")
    joblib.dump(valid_loss, config["result_path"]+"/model/valid_loss.pkl")
    trainer.load_ckpt(config["result_path"]+"/model/best.checkpoint")
    #plot_loss(train_loss,valid_loss,config["result_path"]+"/loss.png")
    #plot_loss_detail(train_loss,valid_loss,config["result_path"]+"/loss_detail.png")

    #loss, states, obs_gen, _ = model.simulate_with_data(valid_data)
    #save_simulation(config,valid_data,states,obs_gen)
    #joblib.dump(loss, config["result_path"]+"/model/last_loss.pkl")

def main():
    parser = argparse.ArgumentParser()
    reserved_args=init_argparse(parser)
    #####
    config, args = update_config_from_cli(parser, reserved_args)
    # initialization
    init_result(config)
    device = init_gpu(args)
    logger = init_log(config, "log_train.txt")
    print(config)
    # init seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ## save config
    save_config_path=config["result_path"]+"/config.yaml"
    with open(save_config_path,"w") as fp:
        print("[SAVE] config: ", save_config_path)
        fp.write(OmegaConf.to_yaml(config))
    # setup
    resume_ckpt=args.resume
    train(config, logger, device,resume_ckpt)
    """
    ## save config
    save_config_path=config["result_path"]+"/config.yaml"
    with open(save_config_path,"w") as fp:
        print("[SAVE] config: ", save_config_path)
        fp.write(OmegaConf.to_yaml(config))
    """ 

if __name__ == "__main__":
    main()
