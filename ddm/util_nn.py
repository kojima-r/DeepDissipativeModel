import torch
import torch.nn as nn
import torch.nn.functional as F
def get_v_nn(v_type,state_dim,device):
    if v_type=="single":
        v_nn= SimpleV1(state_dim,device=device)
    elif v_type=="double":
        v_nn = SimpleV2(state_dim,device=device)
    elif v_type=="many":
        v_nn = SimpleV3(state_dim,device=device)
    elif v_type=="single_cycle":
        v_nn = SimpleV4(state_dim,device=device)
    else:
        v_nn=None
        print("[ERROR] unknown:",v_type)
    return v_nn

class IdentityNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, scale=1.0, offset=0.0, positive=False):
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.scale=scale
        self.offset=offset
        self.positive=positive
    def forward(self, x):
        """
        Args:
            x (tensor): ... x  in_dim
        Returns:
            out (tensor): ... x out_dim
        """
        # x[:,...,:out_dim]
        new_shape=x.shape[:-1]+(self.out_dim,)
        x=x.view([-1,self.in_dim])
        x=x[:,:self.out_dim].view(new_shape)
        # computing output
        out = self.scale*x+self.offset
        if self.positive:
            return 1.0e-4+F.relu(out)
        else:
            return out
        return out



class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, activation=F.leaky_relu, scale=0.1, residual=False, residual_coeff=-1.0, offset=0.0, positive=False, with_bn=False,with_dropout=False):
        """
        Simple malti-layer perceptron neural network 
        Output = NN(x) * scale + residual_coeff*x + offset
        x:input

        Args:
          in_dim (int): #dimensions of input 
          h_dim  (List[int]): list of #dimensions of hidden layers
          out_dim (int): #dimensions of input
          activation (function): activation function (default: F.leaky_relu)
          scale (float): initial scaling factor of output (default: 0.1)
          residual (bool): residual connection (default: False)
          residual_coeff (float): (default: -1.0)
          offset (float):  (default: 0.0)
          positive (bool): enforcing positive output (default: False)
          with_bn (bool): enabling batch normalization (default: False)
          with_dropout (bool): enabling dropout (default: False)
        """
        super(SimpleMLP, self).__init__()
        linears=[]
        bns=[]
        dos=[]
        self.with_bn=with_bn
        self.with_dropout=with_dropout
        prev_d=in_dim
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self._get_layer(prev_d,d))
            if with_bn:
                bns.append(nn.BatchNorm1d(d))
            if with_dropout:
                dos.append(nn.Dropout(p=0.3))
            prev_d=d
        linears.append(self._get_layer(prev_d,out_dim))
        self.linears = nn.ModuleList(linears)
        self.bns = nn.ModuleList(bns)
        self.dos = nn.ModuleList(dos)
        self.activation = activation
        self.scale = scale
        self.residual = residual
        self.residual_coeff = residual_coeff
        self.offset = offset
        self.positive = positive
        self.out_dim = out_dim
        self.in_dim = in_dim
        if self.out_dim>self.in_dim:
            self.padding_out=torch.nn.ZeroPad1d((0,self.out_dim-self.in_dim))

    def _get_layer(self,in_d,out_d):
        """
        get initialized linear layer
        """
        l=nn.Linear(in_d, out_d)
        #nn.init.kaiming_uniform_(l.weight)
        nn.init.kaiming_normal_(l.weight,nonlinearity="leaky_relu")
        return l

    def forward(self, x):
        """
        Args:
            x (tensor): ... x  in_dim
        Returns:
            out (tensor): ... x out_dim
        """
        res_x=x
        for i in range(len(self.linears)-1):
            x = self.linears[i](x)
            if self.with_bn:
                x = self.bns[i](x)
            if self.with_dropout:
                x = self.dos[i](x)
            x = self.activation(x)
        x = self.linears[len(self.linears)-1](x)
        if self.residual:
            # res_x[:,...,:out_dim]
            if self.out_dim<=self.in_dim:
                new_shape=res_x.shape[:-1]+(self.out_dim,)
                res_x=res_x.view([-1,self.in_dim])
                res_x=res_x[:,:self.out_dim].view(new_shape)
            else:
                new_shape=res_x.shape[:-1]+(self.out_dim,)
                res_x=self.padding_out(res_x)
            # computing output
            out = x*self.scale+self.residual_coeff*res_x+self.offset
        else:
            out = x*self.scale+self.offset
        if self.positive:
            return 1.0e-4+F.relu(out)
        else:
            return out


class SimpleV1(torch.nn.Module):
    """
    simple V function without parameters
    V(x) = ||x||^2
    stable points: x=0
    """
    def __init__(self, in_dim, device=None):
        super(SimpleV1, self).__init__()
        self.in_dim = in_dim
        self.pt1 = torch.zeros((self.in_dim,))
        self.pt1=self.pt1.to(device)

    def get_stable_points(self):
        """
        list of stable points
        """
        return [self.pt1]
    
    def get_limit_cycles(self):
        """
        list of center points of limit cycles
        """
        return []
    
    def get_stable(self):
        """
        list of pairs of (type,  point)
        type (str): limit_cycle/point
        point (tensor): tensor with (in_dim,) size
        """
        return [("point",self.pt1)]

    def forward(self, x):
        """
        Args:
            x (tensor): batch (x time) x state dimension
        Returns:
            min_vv (tensor): batch (x time)
            min_index (tensor): batch (x time)
        """
        v1 = x - self.pt1
        v1 = (v1 ** 2).sum(dim=(-1,))
        vv = torch.stack([v1])
        min_vv, min_index = torch.min(vv, dim=0)
        return min_vv,min_index

class SimpleV2(torch.nn.Module):
    """
    simple V function without parameters
    V(x) = min(||x-u1||^2,||x-u2||^2)
    stable points: x=u1, u2
    - u1=[ 1,0,0,...]
    - u2=[-1,0,0,...]
    """
    def __init__(self, in_dim, device=None):
        super(SimpleV2, self).__init__()
        self.in_dim = in_dim
        self.pt1 = torch.zeros((self.in_dim,))
        self.pt2 = torch.zeros((self.in_dim,))
        self.pt1[0] = 1.0
        self.pt2[0] = -1.0
        self.pt1=self.pt1.to(device)
        self.pt2=self.pt2.to(device)

    def get_stable_points(self):
        """
        list of stable points
        """
        return [self.pt1,self.pt2]
    
    def get_limit_cycles(self):
        """
        list of center points of limit cycles
        """
        return []
    
    def get_stable(self):
        """
        list of pairs of (type,  point)
        type (str): limit_cycle/point
        point (tensor): tensor with (in_dim,) size
        """
        return [("point",self.pt1),("point",self.pt2)]

    def forward(self, x):
        """
        Args:
            x (tensor): batch (x time) x state dimension
        Returns:
            min_vv (tensor): batch (x time)
            min_index (tensor): batch (x time)
        """
        v1 = x - self.pt1
        v2 = x - self.pt2
        v1 = (v1 ** 2).sum(dim=(-1,))
        v2 = (v2 ** 2).sum(dim=(-1,))
        vv = torch.stack([v1, v2])
        min_vv, min_index = torch.min(vv, dim=0)
        return min_vv,min_index

class SimpleV3(torch.nn.Module):
    """
    simple V function without parameters
    V(x) = min(||x-u1||^2,||x-u2||^2, ||x-u3||^2, ..., ||x-u_2d||^2,)
    stable points: x=u1, u2, u3,..., u_2d
    - u1=[ 1, 0, 0,...]
    - u2=[-1, 0, 0,...]
    - u3=[ 0, 1, 0,...]
    - u4=[-1,-1, 0,...]
    ...
    """
    def __init__(self, in_dim, device=None):
        super(SimpleV3, self).__init__()
        self.in_dim = in_dim
        self.pts=[]
        for i in range(in_dim):
            pt1 = torch.zeros((self.in_dim,))
            pt2 = torch.zeros((self.in_dim,))
            pt1[i] = 1.0
            pt2[i] = -1.0
            pt1=pt1.to(device)
            pt2=pt2.to(device)
            self.pts.append(pt1)
            self.pts.append(pt2)

    def get_stable_points(self):
        """
        list of stable points
        """
        return self.pts

    def get_limit_cycles(self):
        """
        list of center points of limit cycles
        """
        return []
    
    def get_stable(self):
        """
        list of pairs of (type,  point)
        type (str): limit_cycle/point
        point (tensor): tensor with (in_dim,) size
        """
        return [("point",pt) for pt in self.pts]

    def forward(self, x):
        """
        Args:
            x (tensor): batch (x time) x state dimension
        Returns:
            min_vv (tensor): batch (x time)
            min_index (tensor): batch (x time)
        """
        vv=[]
        for pt in self.pts:
            v1 = x - pt
            v1 = (v1 ** 2).sum(dim=(-1,))
            vv.append(v1)
        vv = torch.stack(vv)
        min_vv, min_index = torch.min(vv, dim=0)
        return min_vv,min_index

## limit cycle
class SimpleV4(torch.nn.Module):
    """
    simple V function without parameters
    V(x) = ||x-n||^2
    n=x/||x||
    stable points: x on a unit circle
    """
    def __init__(self, in_dim, device=None):
        super(SimpleV4, self).__init__()
        self.in_dim = in_dim
        self.device = device
        self.pt1 = torch.zeros((self.in_dim,),device=device)

    def get_stable_points(self):
        """
        list of stable points
        This function returns an empty list since there are infinite numbers.
        """
        return []

    def get_limit_cycles(self):
        """
        list of center points of limit cycles
        """
        return [self.pt1]

    def get_stable(self):
        """
        list of pairs of (type,  point)
        type (str): limit_cycle/point
        point (tensor): tensor with (in_dim,) size
        """
        return [("limit_cycle",self.pt1)]

    def forward(self, x):
        """
        Args:
            x (tensor): batch (x time) x state dimension
        Returns:
            min_vv (tensor): batch (x time)
            min_index (tensor): batch (x time)
        """
        eps=1.0e-4
        nx=x/(torch.sqrt(torch.sum(x**2,dim=-1,keepdim=True)+eps)+eps)
        d=torch.sum((x-nx)**2,dim=-1)
        min_i=torch.zeros(d.size(),device=self.device)
        return d,min_i

