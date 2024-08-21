import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import json
import logging
import numpy as np

from ddm.util_data import DiosDataset
from ddm.util_trainer import LossLogger
import time



class SystemTrainer:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger("logger")
    
    def batch_to_yux(self,batch):
        y_true, u, x = None, None, None
        y_true=batch["obs"]
        y_true=torch.transpose(y_true,0,1)
        if "input" in batch:
            u = batch["input"]
            u=torch.transpose(u,0,1)
        if "state" in batch:
            x = batch["state"]
            x=torch.transpose(x,0,1)
        return y_true, u, x
    def compute_loss(self, batch, dt_x, dt_u, T, epoch,  num_sample=100, return_sol=False):
        y=None
        y_true, u, x = self.batch_to_yux(batch)
        alpha_f=self.config["alpha_f"]
        alpha_g=self.config["alpha_g"]
        alpha_h=self.config["alpha_h"]
        state_dim=self.model.state_dim
        if x is not None:
            x0_ = x[0,:,:]
            x0=torch.zeros((x0_.shape[0],state_dim),device=self.device)
            d=min([x.shape[-1],state_dim])
            x0[:,:d] = x0_[:,:d]
            #self.logger.info("initial state x0 is set as true state")
        else:
            x0=torch.zeros((y_true.shape[1],state_dim),device=self.device)
            #self.logger.info("initial state x0 is set as zero")
        ##
        #x0: batch_size x state_dim
        t_ode=torch.arange(0, T-dt_x, dt_x).to(self.device)
        if epoch < self.config["pre_fit_epoch"]:
            y,x_sol=self.model(x0, u, dt_u, t_ode, enforce_naive=True)
        else:
            y,x_sol=self.model(x0, u, dt_u, t_ode)
        if self.config["consistency_h"]:
            e=(self.model.h_inv_nn(y)-x_sol)
            if self.config["one_step_loss"]: # time x batch x dim
                c_loss=torch.mean(torch.sum(e**2,dim=-1))
            else:
                c_loss=torch.mean(torch.sum(e**2,dim=(0,-1)))
        # y: time x batch x #state
        ## computing error loss
        T_x=min(len(y),len(y_true))
        if self.config["one_step_loss"]:
            loss_error=torch.mean(torch.sum((y[:T_x]-y_true[:T_x])**2,dim=-1))
        else:
            loss_error=torch.mean(torch.sum((y[:T_x]-y_true[:T_x])**2,dim=(0,-1)))
        ## computing projection loss
        x=torch.normal(0,1,size=(num_sample,state_dim),device=self.device)
        fx_new, gx_new, jx_mat, hx, fx_diff, gx_diff, Lx =self.model.forward_proj(x)
        loss_proj_f=torch.mean(torch.sum(fx_diff**2,dim=-1))
        loss_proj_g=torch.mean(torch.sum(gx_diff**2,dim=(-2,-1)))
        ## for logging
        norm2_fx=torch.mean(torch.sum(fx_new**2,dim=-1)).item()
        norm2_gx=torch.mean(torch.sum(gx_new**2,dim=(-2,-1))).item()
        norm2_hx=torch.mean(torch.sum(hx**2,dim=-1)).item()
        norm2_jx=torch.mean(torch.sum(jx_mat**2,dim=(-2,-1))).item()
        if Lx is not None:
            norm2_Lx=torch.mean(torch.sum(Lx**2,dim=-1)).item()
        else:
            norm2_Lx=0
        ## computing total loss
        if epoch < self.config["pre_fit_epoch"]:
            target_loss=loss_error
        else:
            target_loss=loss_error+alpha_f*loss_proj_f+alpha_g*loss_proj_g
        total_loss=loss_error+alpha_f*loss_proj_f+alpha_g*loss_proj_g
        if self.config["consistency_h"]:
            total_loss+=alpha_h*c_loss
            target_loss+=alpha_h*c_loss
        loss_dict={
                "target": total_loss.item(),
                "total": total_loss.item(),
                "mse":loss_error.item(),
                "proj_f":(alpha_f*loss_proj_f).item(),
                "proj_g":(alpha_g*loss_proj_g).item(),
                "*norm2-fx": norm2_fx,
                "*norm2-gx": norm2_gx,
                "*norm2-hx": norm2_hx,
                "*norm2-jx": norm2_jx,
                "*norm2-Lx": norm2_Lx,
                "consistency_loss":c_loss.item()
                }
        if return_sol:
            return target_loss,loss_dict, y,x_sol,x0
        else:
            return target_loss,loss_dict

    def save(self,path):
        self.logger.info("[save model]"+path)
        torch.save(self.model.state_dict(), path)

    def load(self,path):
        self.logger.info("[load model]"+path)
        state_dict=torch.load(path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)

    def save_ckpt(self, epoch, loss, optimizer, path):
        self.logger.info("[save ckeck point]"+path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

    def load_ckpt(self, path):
        self.logger.info("[load ckeck point]"+path)
        ckpt=torch.load(path, map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt["model_state_dict"])
        return ckpt

    def get_optimizer(self,config):
        if config["optimizer"]=="adam":
            optimizer = optim.Adam(
                self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
            )
        elif config["optimizer"]=="adamw":
            optimizer = optim.AdamW(
                self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
            )
        else:
            optimizer = optim.RMSprop(
                self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
            )
        return optimizer

    def get_default_input(self,batch_size,T,dt_x,dt_u=None,device=None, x0_std=0.5):
        if device is None:
            device=self.model.device
        if dt_u is None:
            dt_u=dt_x
        dt_u=torch.tensor(dt_u)
        dt_x=torch.tensor(dt_x)
        T=torch.tensor(T)
        ###
        t_u=torch.arange(0,T,dt_u)
        t=torch.arange(0,T,dt_x)
        ###
        u_zero=torch.zeros((len(t_u),batch_size,self.model.in_dim))
        x0=torch.normal(torch.zeros((batch_size,self.model.state_dim)),x0_std)
        x0=x0.to(device)
        u_zero=u_zero.to(device)
        t=t.to(device)
        dt_u=dt_u.to(device)
        return x0,u_zero,dt_u,t

    def pred(self, test_data, batch_size=None,device=None):
        config = self.config
        if batch_size is None:
            batch_size = config["batch_size"]
        if device is None:
            device=self.device
        testset = DiosDataset(test_data, train=False)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=4, timeout=20,
            pin_memory=True,
            )
        ## 
        T, dt_x, dt_u = test_data.T, test_data.dt_x, test_data.dt_u
        dt_x=torch.tensor(dt_x).to(device)
        dt_u=torch.tensor(dt_u).to(device)
        T=torch.tensor(T).to(device)
        ##
        test_loss_logger=LossLogger()
        test_loss_logger.start_epoch()
        y_list=[]
        x_sol_list=[]
        x0_list=[]
        for i, batch in enumerate(testloader,0):
            batch={k:el.to(device) for k,el in batch.items()}
            loss, loss_dict, y, x_sol, x0 = self.compute_loss(batch,dt_x,dt_u,T,0,return_sol=True)
            test_loss_logger.update(loss.item(), loss_dict)
            y_list.append(y.to("cpu").detach().numpy())
            x_sol_list.append(x_sol.to("cpu").detach().numpy())
            x0_list.append(x0.to("cpu").detach().numpy())
            
        test_loss_logger.end_epoch()
        ## print message
        msg=test_loss_logger.get_msg("test")
        self.logger.info(msg)
        ### post processing
        y=np.concatenate(y_list,axis=1)
        x_sol=np.concatenate(x_sol_list,axis=1)
        x0=np.concatenate(x0_list,axis=0)
        y=np.transpose(y,(1,0,2))
        x_sol=np.transpose(x_sol,(1,0,2))

        return test_loss_logger, y, x_sol, x0

    def fit(self, train_data, valid_data, resume_ckpt=None):
        config = self.config
        batch_size = config["batch_size"]
        trainset = DiosDataset(train_data, train=True)
        validset = DiosDataset(valid_data, train=False)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, timeout=20,
            pin_memory=True,
        )
        validloader = DataLoader(
            validset, batch_size=batch_size, shuffle=False, num_workers=4, timeout=20,
            pin_memory=True,
        )
        T, dt_x, dt_u = train_data.T, train_data.dt_x, train_data.dt_u
        dt_x=torch.tensor(dt_x).to(self.device)
        dt_u=torch.tensor(dt_u).to(self.device)
        T=torch.tensor(T).to(self.device)

        optimizer = self.get_optimizer(config)
        train_loss_logger = LossLogger()
        valid_loss_logger = LossLogger()
        prev_valid_loss=None
        best_valid_loss=None
        patient_count=0
        start_epoch=0
        start_time = time.perf_counter()
        if resume_ckpt is not None:
            ckpt=self.load_ckpt(resume_ckpt)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch=ckpt["epoch"]
            print("start epoch:",start_epoch)
        for epoch in range(start_epoch,config["epoch"]):
            train_loss_logger.start_epoch()
            valid_loss_logger.start_epoch()
            for i, batch in enumerate(trainloader, 0):
                optimizer.zero_grad()
                batch={k:el.to(self.device) for k,el in batch.items()}
                loss, loss_dict = self.compute_loss(batch,dt_x,dt_u,T,epoch)
                train_loss_logger.update(loss.item(), loss_dict)
                loss.backward()
                # grad clipping by norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1, norm_type=2)
                # grad clipping by value
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0e-1)
                optimizer.step()
                ##print(loss.item())
                del loss
                del loss_dict

            for i, batch in enumerate(validloader, 0):
                batch={k:el.to(self.device) for k,el in batch.items()}
                loss, loss_dict = self.compute_loss(batch,dt_x,dt_u,T,epoch)
                valid_loss_logger.update(loss.item(), loss_dict)
            train_loss_logger.end_epoch()
            valid_loss_logger.end_epoch()
            ## Early stopping
            l=valid_loss_logger.get_loss()
            if np.isnan(l):
                self.logger.info("... nan is detected in training")
                msg="\t".join(["[{:4d}] ".format(epoch + 1),
                    train_loss_logger.get_msg("train"),
                    valid_loss_logger.get_msg("valid"),
                    "({:2d})".format(patient_count),])
                self.logger.info(msg)
                return train_loss_logger, valid_loss_logger, False
            elif epoch >20 and l>1.0e15:
                self.logger.info("... loss is too learge")
                msg="\t".join(["[{:4d}] ".format(epoch + 1),
                    train_loss_logger.get_msg("train"),
                    valid_loss_logger.get_msg("valid"),
                    "({:2d})".format(patient_count),])
                self.logger.info(msg)
                return train_loss_logger, valid_loss_logger, False
            if prev_valid_loss is None or l < prev_valid_loss:
                patient_count=0
            else:
                patient_count+=1
            prev_valid_loss=l
            ## check point
            check_point_flag=False
            if best_valid_loss is None or l < best_valid_loss:
                path = config["result_path"]+f"/model/model.{epoch}.checkpoint"
                self.save_ckpt(epoch, l, optimizer, path)
                path = config["result_path"]+f"/model/best.checkpoint"
                self.save_ckpt(epoch, l, optimizer, path)
                path = config["result_path"]+f"/model/best.result.json"
                with open(path, "w") as fp:
                    res=train_loss_logger.get_dict("train")
                    res.update(valid_loss_logger.get_dict("valid"))
                    res["best_epoch"]=epoch
                    json.dump(
                        res,
                        fp,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    )
                check_point_flag=True
                best_valid_loss=l

            ## print message
            ckpt_msg = "*" if check_point_flag else ""
            msg="\t".join(["[{:4d}] ".format(epoch + 1),
                train_loss_logger.get_msg("train"),
                valid_loss_logger.get_msg("valid"),
                "({:2d})".format(patient_count),
                ckpt_msg,])
            self.logger.info(msg)
        ### save last ckpt
        path = config["result_path"]+f"/model/last.checkpoint"
        self.save_ckpt(epoch, l, optimizer, path)
        end_time = time.perf_counter()
        dtime_sec=end_time-start_time
        self.logger.info('perf_time: {:.2f} sec ({:.2f} h)'.format(dtime_sec,dtime_sec/(60*60)))
        return train_loss_logger, valid_loss_logger, True

if __name__ == '__main__':
    batch_size=5
    state_dim=2
    obs_dim=1
    in_dim=1
    device="cpu"
       
    logging.basicConfig(level=logging.INFO)
    #logger = logging.getLogger("logger")
    from util_nn import SimpleV3
    from model import NeuralDissipativeSystem
    v_nn=SimpleV3(state_dim,device=device)
    model=NeuralDissipativeSystem(
            batch_size=batch_size,
            state_dim=state_dim,
            obs_dim=obs_dim,
            in_dim=in_dim,
            v_nn=v_nn,device=device)

    ###
    import data.linear
    import util_data
    from util_data import DiosDataset
    from torch.utils.data import DataLoader
    data.linear.generate_dataset(N = 100, M = 90)
    train_data=util_data.load_all_data("dataset/linear.train",{})
    valid_data=util_data.load_all_data("dataset/linear.test",{})
    ###
    config={"batch_size":batch_size, "optimizer":"adam", "lr":0.1, "weight_decay":0.01, "epoch":10,
            "alpha_f":1.0,"alpha_g":1.0,
            "save_model_path":"model_out/"}
    trainer=SystemTrainer(config,model,"cpu")
    trainer.fit(train_data,valid_data)
