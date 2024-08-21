import torch


#input : ... x a
#return: ... x b(a<b)
def expand_last_dim(x, dim_a):
  """
  Args:
    x (tensor): ... x a

  Returns:
    bvec (tensor): ... x b
  """
  in_dim=x.shape[-1]
  if in_dim>dim_a:
      # x[:,...,:out_dim]
      new_shape=x.shape[:-1]+(dim_a,)
      x=x.view([-1,in_dim])
      x=x[:,:dim_a].view(new_shape)
  else:
      padding_out=torch.nn.ZeroPad1d((0,dim_a-in_dim))
      x=padding_out(x)
  return x


####
def xAy(x,A,y):
  """
  Args:
    x (tensor): ... x a
    A (tensor): ... x a x b
    y (tensor): ... x b

  Returns:
    result (tensor): ...
  """
  x_mat   =torch.unsqueeze(x,-1)
  y_mat   =torch.unsqueeze(y,-1)
  x_mat_t =torch.transpose(x_mat,-1,-2)
  xAy  = torch.matmul(torch.matmul(x_mat_t,A),y_mat)
  xAy_ = torch.squeeze(torch.squeeze(xAy,-1),-1)
  return xAy_

def bmatvecmul(bmat,bvec):
  """
  Args:
    bmat (tensor): ... x a x b
    bvec (tensor): ... x b

  Returns:
    result (tensor): ... x a
  """
  bvec_mat=torch.unsqueeze(bvec,-1)
  res=torch.matmul(bmat,bvec_mat)
  return torch.squeeze(res,-1)

def bvecmatmul(bvec,bmat):
  """
  Args:
    bvec (tensor): ... x a
    bmat (tensor): ... x a x b

  Returns:
    result (tensor): ... x b
  """
  bvec_mat=torch.unsqueeze(bvec,-1)
  bvec_mat_t=torch.transpose(bvec_mat,-1,-2)
  res=torch.matmul(bvec_mat_t,bmat)
  return torch.squeeze(res,-2)


#input : ... x (a x b)
#return: ... x  a x b
def reshape_last_dim(x, shape):
  """
  Args:
    x (tensor): ... x (a x b)

  Returns:
    bvec (tensor): ... x a x b
  """
  x_mat=torch.reshape(x,x.shape[:-1]+shape)
  return x_mat

def bvecvecmul_xyt(x,y):
  """
  Args:
    x (tensor): ... x a
    y (tensor): ... x b

  Returns:
    bmat (tensor): ... x a x b
  """
  x_mat=torch.unsqueeze(x,-1)
  y_mat=torch.unsqueeze(y,-1)
  y_mat_t=torch.transpose(y_mat,-1,-2)
  return torch.matmul(x_mat, y_mat_t)

def get_gradv(x,v_nn):
  """
  Args:
    x (tensor): ... x #state
    v_nn (torch.nn.Module): funciton of V(x)
  Returns:
    gradv (tensor): ... x #state
  """
  x_=x.clone().detach().requires_grad_(True)
  v_val,v_idx=v_nn(x_)
  # v is independet w.r.t. batch and time
  gradv = torch.autograd.grad(v_val.sum(), x_, create_graph=True)[0]
  return gradv

def get_P_mat(gradv,eps=1.0e-5):
  """
  Args:
    gradv (tensor): ... x #state
  Returns:
    gvgv (tensor): ... x #state x #state
  """
  gradv_mat=torch.unsqueeze(gradv,-1)
  gradv_mat_t=torch.transpose(gradv_mat,-1,-2)
  gvgv=torch.matmul(gradv_mat,gradv_mat_t)

  scale_P=(torch.linalg.norm(gradv,dim=-1)**2)
  scale_P_mat=torch.unsqueeze(scale_P,-1)
  scale_P_mat=torch.unsqueeze(scale_P_mat,-1)
  scale_P_mat.shape
  # scale_P_mat : ... x 1 x 1
  P_mat=gvgv/(scale_P_mat+eps)
  # P_mat : ... x #state x #state
  return P_mat
#####


def dissipative_w(u,y,Q,R,S):
  yQy=xAy(y,Q,y)
  uRu=xAy(u,R,u)
  ySu=xAy(y,S,u)

  out_w=yQy+uRu+2*ySu

  return out_w, (yQy,uRu,2*ySu)

def integral_w(W,dt=0.01,scale=1):
    int_w=[]
    cum_e=0
    for e in W:
      cum_e += e*dt*scale
      int_w.append(float(cum_e))
    return int_w

