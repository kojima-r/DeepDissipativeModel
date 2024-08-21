import torch
from torchdiffeq import odeint
import torch.nn.functional as F
import scipy
import numpy as np

from ddm.util_op import xAy, bmatvecmul,bvecmatmul, bvecvecmul_xyt
from ddm.util_op import reshape_last_dim, get_gradv, get_P_mat
from ddm.util_nn import SimpleMLP, get_v_nn ,IdentityNet

class NeuralDissipativeSystem(torch.nn.Module):
  def __init__(self,
               ode_method='euler',
               batch_size=5,
               config={},
               device='cpu'):
    """
    Args:
      dissipative_mode (str): dissipative/l2stable/passive
      ode_method (str): ODE methods (method for odeint): euler/dopri5/...
      batch_size (int): batch size
      device (str): cpu/cuda
    """
    super(NeuralDissipativeSystem, self).__init__()
    state_dim = config["state_dim"]
    obs_dim = config["obs_dim"]
    in_dim = config["in_dim"]


    hidden_layer_f=config["hidden_layer_f"]
    hidden_layer_g=config["hidden_layer_g"]
    hidden_layer_h=config["hidden_layer_h"]
    hidden_layer_j=config["hidden_layer_j"]
    hidden_layer_L=config["hidden_layer_L"]
    diag_g=config["diag_g"]
    diag_j=config["diag_j"]
    
    
    self.detach_f=config["detach_f"]
    self.detach_g=config["detach_g"]
    self.detach_h=config["detach_h"]
    self.detach_j=config["detach_j"]
    self.detach_diff_f=config["detach_diff_f"]
    self.detach_diff_g=config["detach_diff_g"]

    self.id_h=config["identity_h"]
    self.without_j=config["without_j"]
    
    #v_nn (torch.nn.Module): neural network or function for V(x): state space -> R
    v_type=config["v_type"]
    self.v_nn=get_v_nn(v_type,state_dim,device)

    self.batch_size=batch_size
    self.state_dim=state_dim # n
    self.obs_dim=obs_dim     # l
    self.in_dim=in_dim       # m
    self.diag_g=diag_g
    self.diag_j=diag_j
    self.f_nn=SimpleMLP(state_dim,hidden_layer_f,state_dim,scale=config["scale_f"],
            residual=config["residual_f"], residual_coeff=config["residual_coeff_f"],
            with_bn=config["with_bn_f"])
    self.h_inv_nn=None
    if self.id_h:
        self.h_nn=IdentityNet(state_dim,obs_dim)
    else:
        self.h_nn=SimpleMLP(state_dim,hidden_layer_h,obs_dim,scale=config["scale_h"],
                residual=config["residual_h"], with_bn=config["with_bn_h"])
        if config["consistency_h"]:
            self.h_inv_nn=SimpleMLP(obs_dim,hidden_layer_h[::-1],state_dim,scale=config["scale_h"],
                residual=config["residual_h"], with_bn=config["with_bn_h"])

    if self.diag_g:
      dim=min(state_dim,in_dim)
      self.g_nn=SimpleMLP(state_dim,hidden_layer_g,dim,scale=config["scale_g"],
            with_bn=config["with_bn_g"])
    else:
      self.g_nn=SimpleMLP(state_dim,hidden_layer_g,state_dim*in_dim,scale=config["scale_g"],
            with_bn=config["with_bn_g"])
    if self.diag_j:
      dim=min(obs_dim,in_dim)
      self.j_nn=SimpleMLP(state_dim,hidden_layer_j,dim,scale=config["scale_j"],
            with_bn=config["with_bn_j"])
    else:
      self.j_nn=SimpleMLP(state_dim,hidden_layer_j,obs_dim*in_dim,scale=config["scale_j"],
            with_bn=config["with_bn_j"])
    self.ode_method=ode_method
    
    self.dissipative_mode=config["dissipative_mode"]
    #logger.info("... model mode: ", self.dissopative_mode)
    if self.dissipative_mode=="dissipative":
      if self.without_j:
        self.L_nn=SimpleMLP(state_dim,hidden_layer_L,in_dim,scale=config["scale_L"],
              with_bn=config["with_bn_L"])
      else:
        self.L_nn=SimpleMLP(state_dim,hidden_layer_L,obs_dim+in_dim,scale=config["scale_L"],
              with_bn=config["with_bn_L"])
    elif self.dissipative_mode=="l2stable":
      self.L_nn=SimpleMLP(state_dim,hidden_layer_L,in_dim,scale=config["scale_L"],
            with_bn=config["with_bn_L"])
    else:
      self.L_nn=None
    if self.dissipative_mode=="passive":
      assert obs_dim==in_dim
    ###
    if config["fix_f"]:
        for param in self.f_nn.parameters():
            param.requires_grad = False
    if config["fix_g"]:
        for param in self.g_nn.parameters():
            param.requires_grad = False
    if config["fix_h"]:
        for param in self.h_nn.parameters():
            param.requires_grad = False
    if config["fix_j"]:
        for param in self.j_nn.parameters():
            param.requires_grad = False
    if config["fix_L"] and self.L_nn is not None:
        for param in self.L_nn.parameters():
            param.requires_grad = False
    ###
    self.l2stable_gamma=config["gamma"]
    self.eps_P=config["eps_P"]
    self.eps_f=config["eps_f"]
    self.eps_g=config["eps_g"]
    ###
    self.device=device

  def set_dissipative(self,Q,R,S):
    """
    Args:
      Q (tensor): l x l (obs_dim x obs_dim)
      R (tensor): m x m (in_dim x in_dim)
      S (tensor): l x m (obs_dim x in_dim)
    """
    self.Q=Q
    self.R=R
    self.S=S
    #
    a=torch.cat([-Q,S],dim=1)
    b=torch.cat([S.T,R],dim=1)
    A=torch.cat([a,b],dim=0)
    rtA_=scipy.linalg.sqrtm(A.detach().cpu().numpy())
    self.rootQRS=torch.real(torch.tensor(rtA_)) # complex => float
    self.rootQRS=self.rootQRS.to(self.device)
    #
    rtR_=scipy.linalg.sqrtm(R.detach().cpu().numpy())
    self.rootR=torch.real(torch.tensor(rtR_)) # complex => float
    self.rootR=self.rootR.to(self.device)
    #
  def set_qrs(self,q=1.0,r=1.0,s=0.0):
    """
    Args:
      Q (tensor): l x l (obs_dim x obs_dim)
      R (tensor): m x m (in_dim x in_dim)
      S (tensor): l x m (obs_dim x in_dim)
    """
    Q=torch.tensor(- q*np.eye(self.obs_dim),dtype=torch.float32)
    R=torch.tensor(r*np.eye(self.in_dim),dtype=torch.float32)
    S=torch.tensor(s*np.eye(self.obs_dim,self.in_dim),dtype=torch.float32)
    Q, R, S=Q.to(self.device), R.to(self.device), S.to(self.device)
    self.set_dissipative(Q,R,S)

 
  def f_conservation(self, x, fx,gradv,gradv_norm2,hx,Lx_norm2,P_mat):
    """
    compute f(x) in dissipative models

    Args:
      x (tensor): ... x state_dim
      fx (tensor): ... x state_dim : naive f(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
      hx (tensor): ... x obs_dim: naive h(x)
      Lx_norm2 (tensor): ... : ||L(x)||^2
      P_mat (tensor): ... x state_dim x state_dim : P
    Returns:
      f_new (tensor):  ... x state_dim: new f(x)
      fx_diff (tensor): ... x state_dim: differemce between new f(x) and naive f(x)
    """
    hQh=xAy(hx,self.Q,hx)
    #
    eps=self.eps_f
    scale_f_modify=(hQh/(gradv_norm2+eps))
    scale_f_modify_vec=torch.unsqueeze(scale_f_modify,-1)
    #
    f_modify=scale_f_modify_vec*gradv
    #
    #fx_proj=fx-bmatvecmul(P_mat,fx)
    #f_new=fx_proj+f_modify
    fx_diff=-bmatvecmul(P_mat,fx)+f_modify
    f_new=fx+fx_diff
    if self.detach_diff_f:
        f_new=fx+fx_diff.detach()
    else:
        f_new=fx+fx_diff
    return f_new,fx_diff

  def g_conservation(self, x, gx_mat,gradv,gradv_norm2, Lx,hx,P_mat):
    """
    compute g(x) in dissipative models

    Args:
      x (tensor): ... x state_dim
      gx_mat (tensor): ... x state_dim x in_dim : naive g(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
      Lx (tensor): ... x obs_dim: L(x)
      Lx_norm2 (tensor): ... : ||L(x)||^2
      hx (tensor): ... x obs_dim : naive h(x)
      P_mat (tensor): ... x state_dim x state_dim : P
    Returns:
      g_new (tensor): ... x state_dim x in_dim: new g(x)
      gx_diff (tensor): ... x state_dim x in_dim: differemce between new g(x) and naive g(x)
    """
    if self.detach_g:
        gx_mat_=gx_mat.detach()
    else:
        gx_mat_=gx_mat
    if self.detach_h:
        hx=hx.detach()
    #
    hx_mat=torch.unsqueeze(hx,-1)
    hx_mat_t=torch.transpose(hx_mat,-1,-2)
    hx_S=torch.matmul(hx_mat_t,self.S)
    #
    #
    eps=self.eps_g
    gradv_norm2_mat=torch.unsqueeze(torch.unsqueeze(gradv_norm2,-1),-1)
    gradv_mat=torch.unsqueeze(gradv,-1)
    #print("grad_v",gradv_mat.shape)
    #print("hx_S",hx_S.shape)   # (1 x output). (output x input)
    #print("Lx",Lx_mat_t.shape) # 1 x input
    #print("RootR",self.rootR.shape) # input x input
    G_modify=2/(gradv_norm2_mat+eps)*gradv_mat@(hx_S)
    #
    #gx_proj=gx_mat-torch.matmul(P_mat,gx_mat)
    #gx_new=gx_proj+G_modify
    gx_diff=-torch.matmul(P_mat,gx_mat_)+G_modify
    if self.detach_diff_g:
        gx_new=gx_mat+gx_diff.detach()
    else:
        gx_new=gx_mat+gx_diff
    return gx_new,gx_diff



   
  def f_dissipative(self, x, fx,gradv,gradv_norm2,hx,Lx_norm2,P_mat):
    """
    compute f(x) in dissipative models

    Args:
      x (tensor): ... x state_dim
      fx (tensor): ... x state_dim : naive f(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
      hx (tensor): ... x obs_dim: naive h(x)
      Lx_norm2 (tensor): ... : ||L(x)||^2
      P_mat (tensor): ... x state_dim x state_dim : P
    Returns:
      f_new (tensor):  ... x state_dim: new f(x)
      fx_diff (tensor): ... x state_dim: differemce between new f(x) and naive f(x)
    """
    hQh=xAy(hx,self.Q,hx)
    #
    eps=self.eps_f
    scale_f_modify=((hQh-Lx_norm2)/(gradv_norm2+eps))
    scale_f_modify_vec=torch.unsqueeze(scale_f_modify,-1)
    #
    f_modify=scale_f_modify_vec*gradv
    #
    #fx_proj=fx-bmatvecmul(P_mat,fx)
    #f_new=fx_proj+f_modify
    fx_diff=-bmatvecmul(P_mat,fx)+f_modify
    f_new=fx+fx_diff
    if self.detach_diff_f:
        f_new=fx+fx_diff.detach()
    else:
        f_new=fx+fx_diff
    return f_new,fx_diff

  def g_dissipative(self, x, gx_mat,gradv,gradv_norm2, jx_mat,Lx,hx,P_mat):
    """
    compute g(x) in dissipative models

    Args:
      x (tensor): ... x state_dim
      gx_mat (tensor): ... x state_dim x in_dim : naive g(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
      jx_mat (tensor): ... x obs_dim x in_dim: naive j(x)
      Lx (tensor): ... x obs_dim: L(x)
      Lx_norm2 (tensor): ... : ||L(x)||^2
      hx (tensor): ... x obs_dim : naive h(x)
      P_mat (tensor): ... x state_dim x state_dim : P
    Returns:
      g_new (tensor): ... x state_dim x in_dim: new g(x)
      gx_diff (tensor): ... x state_dim x in_dim: differemce between new g(x) and naive g(x)
    """
    if self.detach_g:
        gx_mat_=gx_mat.detach()
    else:
        gx_mat_=gx_mat
    if self.detach_h:
        hx=hx.detach()
    if self.detach_j:
        jx_mat=jx_mat.detach()
    Qjx=torch.matmul(self.Q,jx_mat)
    Qjx_S=Qjx+self.S
    #
    hx_mat=torch.unsqueeze(hx,-1)
    hx_mat_t=torch.transpose(hx_mat,-1,-2)
    hxQj_S=torch.matmul(hx_mat_t,Qjx_S)
    #
    Lx_mat=torch.unsqueeze(Lx,-1)
    Lx_mat_t=torch.transpose(Lx_mat,-1,-2)
    #
    I=torch.eye(self.in_dim, device=self.device)
    eye_batch=torch.tile(I,jx_mat.shape[:-2]+(1,1))
    jxI=torch.cat([jx_mat,eye_batch],dim=-2)
    #
    eps=self.eps_g
    gradv_norm2_mat=torch.unsqueeze(torch.unsqueeze(gradv_norm2,-1),-1)
    gradv_mat=torch.unsqueeze(gradv,-1)
    G_modify=2/(gradv_norm2_mat+eps)*(gradv_mat@(hxQj_S - Lx_mat_t@self.rootQRS@jxI))
    #
    #gx_proj=gx_mat-torch.matmul(P_mat,gx_mat)
    #gx_new=gx_proj+G_modify
    gx_diff=-torch.matmul(P_mat,gx_mat_)+G_modify
    if self.detach_diff_g:
        gx_new=gx_mat+gx_diff.detach()
    else:
        gx_new=gx_mat+gx_diff
    return gx_new,gx_diff

  def g_dissipative_nonj(self, x, gx_mat,gradv,gradv_norm2, Lx,hx,P_mat):
    """
    compute g(x) in dissipative models

    Args:
      x (tensor): ... x state_dim
      gx_mat (tensor): ... x state_dim x in_dim : naive g(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
      Lx (tensor): ... x obs_dim: L(x)
      Lx_norm2 (tensor): ... : ||L(x)||^2
      hx (tensor): ... x obs_dim : naive h(x)
      P_mat (tensor): ... x state_dim x state_dim : P
    Returns:
      g_new (tensor): ... x state_dim x in_dim: new g(x)
      gx_diff (tensor): ... x state_dim x in_dim: differemce between new g(x) and naive g(x)
    """
    if self.detach_g:
        gx_mat_=gx_mat.detach()
    else:
        gx_mat_=gx_mat
    if self.detach_h:
        hx=hx.detach()
    #
    hx_mat=torch.unsqueeze(hx,-1)
    hx_mat_t=torch.transpose(hx_mat,-1,-2)
    hx_S=torch.matmul(hx_mat_t,self.S)
    #
    Lx_mat=torch.unsqueeze(Lx,-1)
    Lx_mat_t=torch.transpose(Lx_mat,-1,-2)
    #
    eps=self.eps_g
    gradv_norm2_mat=torch.unsqueeze(torch.unsqueeze(gradv_norm2,-1),-1)
    gradv_mat=torch.unsqueeze(gradv,-1)
    #print("grad_v",gradv_mat.shape)
    #print("hx_S",hx_S.shape)   # (1 x output). (output x input)
    #print("Lx",Lx_mat_t.shape) # 1 x input
    #print("RootR",self.rootR.shape) # input x input
    G_modify=2/(gradv_norm2_mat+eps)*gradv_mat@(hx_S-(Lx_mat_t@self.rootR))
    #
    #gx_proj=gx_mat-torch.matmul(P_mat,gx_mat)
    #gx_new=gx_proj+G_modify
    gx_diff=-torch.matmul(P_mat,gx_mat_)+G_modify
    if self.detach_diff_g:
        gx_new=gx_mat+gx_diff.detach()
    else:
        gx_new=gx_mat+gx_diff
    return gx_new,gx_diff


  def f_stable(self, x, fx,gradv,gradv_norm2):
    """
    compute f(x) in internal stable models

    Args:
      x (tensor): ... x state_dim
      fx (tensor): ... x state_dim : naive f(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
    Returns:
      f_new (tensor):  ... x state_dim: new f(x)
      fx_diff (tensor): ... x state_dim: differemce between new f(x) and naive f(x)
    """
    if self.detach_f:
        fx_=fx.detach()
    else:
        fx_=fx
    eps=self.eps_f
    scale_f=torch.sum(F.relu(gradv*fx_),dim=-1)/(gradv_norm2+eps)
    scale_f_=torch.unsqueeze(scale_f,-1)
    #
    #f_modify=scale_f_*gradv
    #f_new=fx-f_modify
    fx_diff=-scale_f_*gradv
    if self.detach_diff_f:
        f_new=fx+fx_diff.detach()
    else:
        f_new=fx+fx_diff
    return f_new,fx_diff

  def f_l2stable(self, x, fx, gradv,gradv_norm2,hx_norm2,Lx_norm2,P_mat):
    """
    compute f(x) in L2 stable models

    Args:
      x (tensor): ... x state_dim
      fx (tensor): ... x state_dim : naive f(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
      hx_norm2 (tensor): ... : ||h(x)||^2
      Lx_norm2 (tensor): ... : ||L(x)||^2
      P_mat (tensor): ... x state_dim x state_dim : P
    Returns:
      f_new (tensor):  ... x state_dim: new f(x)
      fx_diff (tensor): ... x state_dim: differemce between new f(x) and naive f(x)
    """
    eps=self.eps_f
    if self.detach_h:
        hx_norm2=hx_norm2.detach()
    if self.detach_f:
        fx_=fx.detach()
    else:
        fx_=fx
    scale_f=(hx_norm2+Lx_norm2)/(gradv_norm2+eps)
    scale_f_=torch.unsqueeze(scale_f,-1)
    f_modify=scale_f_*gradv
    #
    #fx_proj=fx-bmatvecmul(P_mat,fx)
    #f_new=fx_proj-f_modify
    fx_diff=-bmatvecmul(P_mat,fx_)-f_modify
    if self.detach_diff_f:
        f_new=fx+fx_diff.detach()
    else:
        f_new=fx+fx_diff
    return f_new,fx_diff

  def g_l2stable(self, x, gx_mat,gradv,gradv_norm2,Lx,P_mat):
    """
    compute g(x) in L2 stable models

    Args:
      x (tensor): ... x state_dim
      gx_mat (tensor): ... x state_dim x in_dim : naive g(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
      Lx (tensor): ... x obs_dim: L(x)
      P_mat (tensor): ... x state_dim x state_dim : P
    Returns:
      g_new (tensor): ... x state_dim x in_dim: new g(x)
      gx_diff (tensor): ... x state_dim x in_dim: differemce between new g(x) and naive g(x)
    """
    gamma=self.l2stable_gamma
    if self.detach_g:
        gx_mat_=gx_mat.detach()
    else:
        gx_mat_=gx_mat
    #
    eps=self.eps_f
    scale=2*gamma/(gradv_norm2+eps)
    scale_mat=torch.unsqueeze(torch.unsqueeze(scale,-1),-1)
    #
    gradvLx=bvecvecmul_xyt(gradv, Lx)
    gx_modify=scale_mat*gradvLx
    #
    #gx_proj=gx_mat-torch.matmul(P_mat,gx_mat)
    #gx_new=gx_proj-gx_modify
    gx_diff=-torch.matmul(P_mat,gx_mat_)-gx_modify
    if self.detach_diff_g:
        gx_new=gx_mat+gx_diff.detach()
    else:
        gx_new=gx_mat+gx_diff
    return gx_new,gx_diff

  def f_passive(self,x,fx,gradv,gradv_norm2):
    """
    compute f(x) in passive models

    Args:
      x (tensor): ... x state_dim
      fx (tensor): ... x state_dim : naive f(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
    Returns:
      f_new (tensor):  ... x state_dim: new f(x)
      fx_diff (tensor): ... x state_dim: differemce between new f(x) and naive f(x)
    """
    return self.f_stable(x,fx,gradv,gradv_norm2)

  def g_passive(self,x,gx_mat,gradv,gradv_norm2,hx):
    """
    compute g(x) in passive models

    Args:
      x (tensor): ... x state_dim
      gx_mat (tensor): ... x state_dim x in_dim : naive g(x)
      gradv (tensor): ... x state_dim : grad V(x)
      gradv_norm2 (tensor): ...  : ||grad V(x)||^2
      hx (tensor): ... x obs_dim: h(x)
    Returns:
      g_new (tensor): ... x state_dim x in_dim: new g(x)
      gx_diff (tensor): ... x state_dim x in_dim: differemce between new g(x) and naive g(x)
    """

    if self.detach_h:
        hx=hx.detach()
    LgV=bvecmatmul(gradv,gx_mat)
    LgV_2h=LgV - 2*hx
    #
    eps=self.eps_g
    scale_g_=torch.unsqueeze(torch.unsqueeze(gradv_norm2,-1),-1)
    gx_modify=bvecvecmul_xyt(gradv,LgV_2h)/(scale_g_+eps)
    #
    gx_diff=-gx_modify
    if self.detach_diff_g:
        g_new=gx_mat+gx_diff.detach()
    else:
        g_new=gx_mat+gx_diff
    return g_new,gx_diff

  def get_jx_mat(self, x):
    """
    compute naive j(x)

    Args:
      x (tensor): ... x state_dim
    Returns:
      jx (tensor): ... x obs_dim x in_dim: j(x)
    """
    if self.without_j:
        jx_out=torch.zeros(x.shape[:-1]+(self.obs_dim, self.in_dim),device=self.device)
        return jx_out
    jx=self.j_nn(x)
    if self.diag_j and self.obs_dim<=self.in_dim:
      #jx_mat=torch.diag(jx)@torch.eye(self.obs_dim, self.in_dim,device=self.device)
      I=torch.eye(self.obs_dim, self.in_dim,device=self.device)
      eye_batch=torch.tile(I,jx.shape[:-1]+(1,1))
      jx_temp=torch.unsqueeze(jx,-1)
      jx_mat=eye_batch*jx_temp # (... x obs_dim x 1) * (... x obs_dim x in_dim)
    elif self.diag_j and self.obs_dim>self.in_dim:
      #jx_mat=torch.eye(self.obs_dim, self.in_dim,device=self.device)@torch.diag(jx)
      I=torch.eye(self.obs_dim, self.in_dim,device=self.device)
      eye_batch=torch.tile(I,jx.shape[:-1]+(1,1))
      jx_temp=torch.unsqueeze(jx,-2)
      jx_mat=eye_batch*jx_temp # (... x 1 x in_dim) * (... x obs_dim x in_dim)
    else:
      jx_mat=reshape_last_dim(jx,(self.obs_dim,self.in_dim))
    return jx_mat

  def get_gx_mat(self,x):
    """
    compute naive g(x)

    Args:
      x (tensor): ... x state_dim
    Returns:
      gx (tensor): ... x state_dim x in_dim: g(x)
    """
    gx=self.g_nn(x)
    if self.diag_g and self.state_dim <= self.in_dim:
      I=torch.eye(self.state_dim, self.in_dim,device=self.device)
      eye_batch=torch.tile(I,gx.shape[:-1]+(1,1))
      gx_temp=torch.unsqueeze(gx,-1)
      gx_mat=gx_temp*eye_batch # (... x state_dim x 1) * (... x state_dim x in_dim)
    elif self.diag_g and self.state_dim > self.in_dim:
      I=torch.eye(self.state_dim, self.in_dim,device=self.device)
      eye_batch=torch.tile(I,gx.shape[:-1]+(1,1))
      gx_temp=torch.unsqueeze(gx,-2)
      gx_mat=eye_batch*gx_temp # (... x 1 x in_dim) * (... x state_dim x in_dim)
    else:
      gx_mat=reshape_last_dim(gx,(self.state_dim, self.in_dim))
    return gx_mat

  def forward_proj(self,x):
    """
    Args:
      x (tensor): ... x #state
    Returns:
      fx_new (tensor): ... x #state
      gx_new (tensor): ... x #state x in_dim
      jx_mat (tensor):  ... x obs_dim x in_dim
      hx (tensor):  ... x obs_dim
      fx_diff (tensor): ... x #state
      gx_diff (tensor): ... x #state x in_dim

    """
    # g_mat: batch_size x state_dim x state_dim
    # u_vec: batch_size x state_dim
    # gu_vec:batch_size x state_dim
    gradv=get_gradv(x, self.v_nn)
    gradv_norm2=(torch.linalg.norm(gradv,dim=-1)**2)
    #
    jx_mat=self.get_jx_mat(x)
    gx_mat=self.get_gx_mat(x)
    Lx=None
    if self.L_nn is not None:
      Lx=self.L_nn(x)
      Lx_norm2=torch.linalg.norm(Lx,dim=-1)**2

    hx=self.h_nn(x)
    hx_norm2=(torch.linalg.norm(hx,dim=-1)**2)

    fx=self.f_nn(x)
    #
    fx_diff, gx_diff=torch.zeros((1,1)),torch.zeros((1,1,1))
    if self.dissipative_mode=="dissipative":
      P_mat = get_P_mat(gradv,self.eps_P)
      fx_new,fx_diff=self.f_dissipative(x, fx,gradv,gradv_norm2,hx,Lx_norm2,P_mat)
      if self.without_j:
        gx_new,gx_diff=self.g_dissipative_nonj(x, gx_mat,gradv,gradv_norm2, Lx,hx,P_mat)
      else:
        gx_new,gx_diff=self.g_dissipative(x, gx_mat,gradv,gradv_norm2, jx_mat,Lx,hx,P_mat)
    elif self.dissipative_mode=="stable":
      fx_new,fx_diff=self.f_stable(x, fx,gradv,gradv_norm2)
      gx_new=gx_mat
    elif self.dissipative_mode=="l2stable":
      P_mat = get_P_mat(gradv,self.eps_P)
      fx_new,fx_diff=self.f_l2stable(x, fx, gradv,gradv_norm2,hx_norm2,Lx_norm2,P_mat)
      gx_new,gx_diff=self.g_l2stable(x, gx_mat,gradv,gradv_norm2,Lx,P_mat)
    elif self.dissipative_mode=="passive":
      fx_new,fx_diff=self.f_passive(x,fx,gradv,gradv_norm2)
      gx_new,gx_diff=self.g_passive(x,gx_mat,gradv,gradv_norm2,hx)
    elif self.dissipative_mode=="conservation":
      P_mat = get_P_mat(gradv,self.eps_P)
      fx_new,fx_diff=self.f_conservation(x, fx,gradv,gradv_norm2,hx,None,P_mat)
      gx_new,gx_diff=self.g_conservation(x, gx_mat,gradv,gradv_norm2, None,hx,P_mat)
    else:
      fx_new=fx
      gx_new=gx_mat
    return fx_new, gx_new, jx_mat, hx, fx_diff, gx_diff, Lx

  def forward(self,x0, u, dt_u, t_ode, enforce_naive=False):
    """
    Args:
      x0 (tensor): batch x #state
      u  (tensor): time  x batch x in_dim
      dt_u (float): delta time for input signal u
      t_ode (tensor): output time points
    Returns:
      y (tensor):  time x batch x #state
      x_sol (tensor): time x batch x #state
    """
    def get_u_t(t):
      """
      compute u_t from u: t x batch x in_dim
      Args:
        t (tensor): float scalar value
      Returns:
        u_t: batch x in_dim
      """
      i=torch.floor_divide(t,dt_u).to(torch.int64)
      if i<len(u):
        u_out=u[i,:,:]
      else:
        u_out=u[-1,:,:]
      return u_out

    def func(t,x):
      if self.dissipative_mode=="naive" or enforce_naive:
        fx_new=self.f_nn(x)
        gx_new=self.get_gx_mat(x)
      else:
        fx_new, gx_new, _, _, _, _, _ = self.forward_proj(x)

      u_t=get_u_t(t)
      gu_vec=bmatvecmul(gx_new,u_t)
      x_out=fx_new+gu_vec
      return x_out


    x_sol=odeint(func, x0, t_ode,method=self.ode_method)
    # x_sol: time x batch x state_dim
    # j_mat: time  x batch x obs_dim x in_dim
    # hx   : time  x batch x obs_dim
    # ju_vec: time  x batch x obs_dim

    hx=self.h_nn(x_sol)
    j_mat=self.get_jx_mat(x_sol)
    ## resampling u for t_ode
    u_idx=(t_ode//dt_u).to(torch.int64)
    u_idx[u_idx>=u.shape[0]]=u.shape[0]-1
    u_=u[u_idx,:,:]
    ju_vec=bmatvecmul(j_mat,u_)
    y=hx+ju_vec
    return y,x_sol

