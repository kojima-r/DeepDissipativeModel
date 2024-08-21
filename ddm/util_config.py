from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from omegaconf import OmegaConf
from typing import Optional

@dataclass
class DDMConfig:
    method:str = "ddm"
    state_dim: int = 2
    obs_dim: int = 1
    in_dim: int = 1

    # training
    epoch: int = 10
    pre_fit_epoch: int = -1
    patience: int = 5
    batch_size: int = 10
    activation:str = "leaky_relu"
    optimizer :str = "adam"
    lr: float = 1.0e-2

    # system model
    # single, double, many, single_cycle
    v_type: str = "single"
    scale_f: float = 0.1
    scale_g: float = 1.0
    scale_h: float = 1.0
    scale_j: float = 1.0
    scale_L: float = 1.0
    
    # dataset
    train_valid_ratio: float = 0.2
    data_train: str = ""
    data_test: str = ""
    
    # save/load model
    init_model:str = ""
    result_path:str = ""
    
    # sys
    dissipative_mode: str = "naive" # "dissipative", "f", "fg", "fgh"
    gamma: float = 1.0
    eps_P: float = 0.01
    eps_f: float = 0.01
    eps_g: float = 0.01
    Q: Optional[List[float]] = None #field(default_factory=lambda: [32])
    R: Optional[List[float]] = None #field(default_factory=lambda: [32])
    S: Optional[List[float]] = None #field(default_factory=lambda: [32])
    alpha_f: float=1.0
    alpha_g: float=1.0
    alpha_h: float=1.0
    one_step_loss: bool=False
    diag_g: bool=True
    diag_j: bool=True
    #detach_proj: bool = True
    detach_f: bool=False
    detach_g: bool=False
    detach_h: bool=False
    detach_j: bool=False
    detach_diff_f: bool=False
    detach_diff_g: bool=False

    fix_f: bool=False
    fix_g: bool=False
    fix_h: bool=False
    fix_j: bool=False
    fix_L: bool=False

    with_bn_f: bool = False
    with_bn_g: bool = False
    with_bn_h: bool = False
    with_bn_j: bool = False
    with_bn_L: bool = False

    without_j: bool=True
    identity_h: bool=False
    consistency_h: bool=False
    
    weight_decay: float = 0.0001
    hidden_layer_f: List[int] = field(default_factory=lambda: [32])
    hidden_layer_g: List[int] = field(default_factory=lambda: [32])
    hidden_layer_h: List[int] = field(default_factory=lambda: [32]) 
    hidden_layer_j: List[int] = field(default_factory=lambda: [32])
    hidden_layer_L: List[int] = field(default_factory=lambda: [32])
    residual_f: bool = False
    residual_h: bool = False
    residual_coeff_f: float=-1.0

    profile: bool=False

def update_config_from_cli(parser, reserved_args=[]):
    ## config
    conf=OmegaConf.structured(DDMConfig())
    for key, val in conf.items():
        if key in reserved_args:
            pass
        elif type(val) is int:
            parser.add_argument("--"+key, type=int, default=None, help="[config integer]")
        elif type(val) is float:
            parser.add_argument("--"+key, type=float, default=None, help="[config float]")
        elif type(val) is bool:
            parser.add_argument("--"+key, type=bool, default=None, help="[config float]")
            #parser.add_argument("--"+key, action="store_true", help="[config bool]")
        else:
            parser.add_argument("--"+key, type=str, default=None, help="[config string]")
    
    args = parser.parse_args()
    # config
    conf_args = {}
    for key, val in conf.items():
        val_new=getattr(args,key)
        if val_new != "" and val_new is not None:
            conf_args[key]=val_new
    # 
    if args.config is None or len(args.config) ==0:
        if not args.no_config:
            parser.print_help()
            quit()
    else:
        for config_filename in args.config:
            print("[LOAD]",config_filename)
            conf_ = OmegaConf.load(config_filename)
            conf = OmegaConf.merge(conf, conf_)
    config = OmegaConf.merge(conf, conf_args)
    return config, args
 
if __name__ == '__main__':
    conf=OmegaConf.structured(DDMConfig())
    print(conf["result_path"])
    
    conf_ = OmegaConf.load("default_config.yaml")
    #with raises(ValidationError):
    conf = OmegaConf.merge(conf, conf_)
    print(conf)
    """
    with open("default_config.yaml","w") as fp:
        fp.write(OmegaConf.to_yaml(conf))
    """
