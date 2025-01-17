import os
import numpy as np
import logging
import torch


class DiosDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, train=True):
        super(DiosDataset, self).__init__()
        self.transform = transform
        self.data = data

    def __len__(self):
        return self.data.num

    def __getitem__(self, idx):
        out={}
        if self.data.input is not None:
            out_input = self.data.input[idx]
            out["input"]=out_input
        if self.data.state is not None:
            out_state = self.data.state[idx]
            out["state"]=out_state

        out_obs = self.data.obs[idx]
        if self.transform:
            out_obs = self.transform(out_obs)
        out["obs"]=out_obs

        return out #out_obs, out_input, out_state


class DiosData:
    def __init__(self):
        self.obs = None
        self.obs_mask = None
        self.obs_dim = None
        self.input = None
        self.input_mask = None
        self.input_dim = None
        self.state = None
        self.state_mask = None
        self.state_dim = None
        self.stable = None
        self.stable_mask = None
        self.step = None
        self.idx = None
        self.dt_x = None
        self.dt_u = None
        self.T = None

    def split(self, rate):
        idx = list(range(self.num))
        np.random.shuffle(idx)
        m = int(self.num * rate)
        idx1 = idx[:m]
        idx2 = idx[m:]
        data1 = DiosData()
        data2 = DiosData()
        copy_attrs = ["obs_dim", "input_dim", "state_dim", "dt_x", "dt_u", "T"]
        split_attrs = [
            "obs",
            "obs_mask",
            "input",
            "input_mask",
            "state",
            "state_mask",
            "stable",
            "stable_mask",
            "idx",
        ]
        ## split
        for attr in split_attrs:
            val = getattr(self, attr)
            if val is not None:
                setattr(data1, attr, val[idx1])
                setattr(data2, attr, val[idx2])
        data1.num = m
        data2.num = self.num - m
        ## copy
        for attr in copy_attrs:
            val = getattr(self, attr)
            if val is not None:
                setattr(data1, attr, val)
                setattr(data2, attr, val)
        return data1, data2

    def set_dim_from_data(self):
        attrs = ["obs", "input", "state"]
        for attr in attrs:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr + "_dim", val.shape[2])


def np_load(filename, logger=None, dtype=None, shape_num=None):
    if logger is None: logger=logging.getLogger(__name__)
    logger.info("[LOAD] " + filename)
    obj=np.load(filename)
    if dtype is not None:
        if obj.dtype != dtype:
            obj=obj.astype(dtype)
    if shape_num is not None:
        n=len(obj.shape)
        for _ in range(shape_num-n):
            obj=np.expand_dims(obj, -1)
    return obj


def load_data(mode, config, logger=None):
    if logger is None: logger=logging.getLogger(__name__)
    name = config["data_" + mode]
    if os.path.exists(name):
        return load_simple_data(name, config, logger)
    else:
        return load_all_data(name, config, logger)


def load_all_data(name, config, logger=None):
    if logger is None: logger=logging.getLogger(__name__)
    data = DiosData()
    data.obs = np_load(name + ".obs.npy", logger, dtype=np.float32, shape_num=3)
    data.num = data.obs.shape[0]
    data.idx = np.array(list(range(data.num)))
    ###
    filename = name + ".info.json"
    if os.path.exists(filename):
        info=json.load(open(filename, 'r'))
        data.T=info["T"]
        data.dt_x=info["dt"]
        data.dt_u=info["dt"]
    else:
        data.T=1.0
        data.dt_x=1.0/data.obs.shape[1]
        data.dt_u=1.0/data.obs.shape[1]
    ###
    filename = name + ".obs_mask.npy"
    if os.path.exists(filename):
        data.obs_mask = np_load(filename, logger, dtype=np.float32, shape_num=3)
    else:
        data.obs_mask = np.ones_like(data.obs)
    ###
    filename = name + ".step.npy"
    if os.path.exists(filename):
        data.step = np_load(filename, logger)
    else:
        s = data.obs.shape[1]
        data.step = np.array([s] * data.num)
    ###
    keys = ["input", "state", "stable"]
    for key in keys:
        filename = name + "." + key + ".npy"
        val = None
        if os.path.exists(filename):
            val = np_load(filename, logger, dtype=np.float32, shape_num=3)
            setattr(data, key, val)
            valid_flag = True
        else:
            setattr(data, key, None)
        ###
        filename = name + "." + key + "_mask.npy"
        if os.path.exists(filename):
            setattr(data, key + "_mask", np_load(filename, logger, dtype=np.float32, shape_num=3))
        if val is not None:
            setattr(data, key + "_mask", np.ones_like(val))
        else:
            setattr(data, key + "_mask", None)
        ###
    data.set_dim_from_data()
    return data


def load_simple_data(filename, config, logger=None):
    if logger is None: logger=logging.getLogger(__name__)
    data = DiosData()
    data.obs = np_load(filename, logger, dtype=np.float32, shape_num=3)
    data.obs_mask = np.ones_like(data.obs)
    data.num = data.obs.shape[0]
    data.idx = np.array(list(range(data.num)))
    s = data.obs.shape[1]
    data.step = np.array([s] * data.num)
    data.set_dim_from_data()
    data.T=1.0
    data.dt_x=1.0/obs.shape[1]
    data.dt_u=1.0/obs.shape[1]
    return data
