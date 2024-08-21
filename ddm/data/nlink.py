import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from scipy.integrate import odeint
import numpy as np
me.mechanics_printing(pretty_print = True)

from ddm.data.util import minmax_normalize, z_normalize, save_dataset
from ddm.data.input_signal import generate_input
Q = np.array([[-1,0],[0,0]]) ## [u0,q0]
R = np.array([[0.0]])  ##
S = np.array([[1/2.0]])  ##
#C = np.array([[1.0,0]])
#C = np.array([[0.0,1.0]])



class NLinkPendulum:
    def __init__(self,n,lengths=None, masses=None, dampers =None, simplify= False):
        self.n = n
        self.simplify = simplify
        self.g_val = 9.81
        if  lengths is None:
            self.lengths = np.ones(n)/n
        else:
            self.lengths = lengths
        if  masses is None:
            self.masses = np.ones(n) /np.sum(np.arange(n+1))
        else:
            self.masses = masses
        if dampers is None:
            self.dampers = np.ones(n)
        else:
            self.dampers = dampers
            
        self.mk_dynamics()
        self.parameter_vals = [9.81] + list(self.lengths) + list(self.masses) + list(self.dampers)
        self.tau_func = lambda x : 0
        return

    def mk_dynamics(self):

        print('Make Lagrangian')
        # Define symbols
        g = sm.symbols('g')        # gravity
        m = sm.symbols('m:{0}'.format(self.n))
        l = sm.symbols('l:{0}'.format(self.n))
        c = sm.symbols('c:{0}'.format(self.n))
        t =  me.dynamicsymbols._t            # Time 

        tau = sm.symbols(r'\tau')

        q =  me.dynamicsymbols('q:{0}'.format(self.n))
        qd = [tmp.diff(t,1) for tmp in q]
        # position

        for i in range(self.n):
            if i == 0:
                rx = [l[i] * sm.sin(q[i])]
                ry = [l[i] * sm.cos(q[i])]
                y0 = [l[i]]
            else:
                rx.append(rx[i-1] + l[i] * sm.sin(q[i]) )
                ry.append(ry[i-1] + l[i] * sm.cos(q[i]) )
                y0.append(y0[i-1] + l[i]) 

        rxd  =  [x.diff(t) for x in rx]
        ryd  =  [y.diff(t) for y in ry]

        #  energies
        D=0
        P=0
        P0 = 0
        K=0

        for i in range(self.n):
            K = K+ 0.5* m[i] * (rxd[i]**2 + ryd[i]**2)
            P = P -  m[i] * g* (ry[i]  - y0[i])
            P0 = P0  -  m[i] * g *l[i]
            if i == 0:
                D = D+ 0.5 * c[i] *qd[i]**2
            else:
                D = D+ 0.5 * c[i] *(qd[i] - qd[i-1])**2
        if self.simplify:
            K = K.simplify()
            P = P.simplify()



        L = K -P
        print('Make Lagranges Equation')
        if self.simplify:
            damper_forces = sm.Matrix([ - D.diff(qd[i]).simplify() for i in range(self.n)])
            external_force = sm.Matrix([tau] + [0]*(self.n-1))
            deltaD  = (- damper_forces.T * sm.Matrix(qd))[0].simplify()
            deltaW = (external_force.T * sm.Matrix(qd))[0].simplify()
            LM =me.LagrangesMethod(L,q)
            LM.form_lagranges_equations().simplify()
        else:
            damper_forces = sm.Matrix([ - D.diff(qd[i]) for i in range(self.n)])
            external_force = sm.Matrix([tau] + [0]*(self.n-1))
            deltaD  = (- damper_forces.T * sm.Matrix(qd))[0]
            deltaW = (external_force.T * sm.Matrix(qd))[0]
            LM =me.LagrangesMethod(L,q)
            LM.form_lagranges_equations()

        print('Make function of dynamics')

        force = LM.forcing + damper_forces + external_force  
        mass = LM.mass_matrix

        #  dynamicsymbols -> symbols

        r = sm.symbols('x_:{0}'.format(self.n))
        rd =  sm.symbols('x\'_:{0}'.format(self.n))
        state  =  list(r+ rd ) + [tau]
        replace_state = [tmp for  tmp in zip(qd + q,rd + r)]

        force = force.subs(replace_state)
        mass = mass.subs(replace_state)
        P_func =P.subs(replace_state)
        K_func =K.subs(replace_state)
        deltaD_func = deltaD.subs(replace_state)
        deltaW_func = deltaW.subs(replace_state)

        #  make function
        parameters = [g] +list(m) + list(l) + list(c)
        parameters_val  =  [self.g_val] +list(self.masses) +list(self.lengths)+list(self.dampers)
        set_parameter =  [tmp for tmp in zip(parameters,parameters_val)]

        force = force.subs(set_parameter)
        mass = mass.subs(set_parameter)
        P_func = P_func.subs(set_parameter)
        K_func = K_func.subs(set_parameter)
        deltaD_func = deltaD_func.subs(set_parameter)
        deltaW_func = deltaW_func.subs(set_parameter)

        self.force = sm.lambdify(state,force)
        self.mass = sm.lambdify(state,mass)
        self.P =  sm.lambdify(state,P_func)
        self.K =  sm.lambdify(state,K_func)
        self.deltaD = sm.lambdify(state,deltaD_func)
        self.deltaW = sm.lambdify(state,deltaW_func)
        print('End : Make Dynamics')
        return 

    def gradient(self,x,t):
        tau_tmp = self.tau_func(t)
        state =  np.r_[x,tau_tmp]
        return np.r_[x[self.n:],np.linalg.solve(self.mass(*state),self.force(*state)).T[0]]
        
    def set_tau(self,func): 
        self.tau_func = func
        return
    
    def simulation(self,times,x0 = None):
        if x0 is None:
             x0 = np.zeros(2*self.n)
        self.times = times
        self.tau =  np.array([self.tau_func(t) for t in times]) 
        self.qu = odeint(self.gradient,x0,times,hmax=0.1)
        self.q = self.qu[:,:self.n]
        self.u = self.qu[:,self.n:]
        zeros = np.zeros(times.shape[0])[:, None]
        self.x = np.cumsum(np.hstack([zeros, self.lengths * np.sin(self.q)]),1)
        self.y = np.cumsum(np.hstack([zeros, -self.lengths * np.cos(self.q)]),1)        
        return 
    
    def energies(self):
        state = np.c_[self.qu,self.tau.reshape(-1,1)]
        dt =  (self.times[1:] - self.times[:-1]).mean()
        P_vals =  np.array([self.P(*tmp) for tmp in state])
        K_vals =  np.array([self.K(*tmp) for tmp in state])
        deltaD_vals =  np.array([self.deltaD(*tmp) for tmp in state])
        deltaW_vals =  np.array([self.deltaW(*tmp) for tmp in state])
        D_vals =  np.cumsum(deltaD_vals*dt)
        W_vals =  np.cumsum(deltaW_vals*dt)
        return {'Potential':P_vals,'Kinetics': K_vals,'Damper':D_vals,'Work': W_vals}
        
    



T=1
dh=0.01

#N:  numner of data 
#n:  numner of pendulum 
def generate_old(N, input_type_id=1, n=1, output_q0=False, output_qs=False):
    np.random.seed(0)

    #  numner of time points
    t_N = int(T/dh)

    t = np.linspace(0, T, t_N)
    x_data =[]
    u_data = []
    y_data = []
    # モデル初期化
    model = NLinkPendulum(n)
    for omega in np.linspace(0,T,N): #omega=T*i*1.0/N
        # 入力関数の設定
        func = lambda t: 5*np.sin(omega * t)
       
        model.set_tau(func)
        # シミュレーション
        model.simulation(t)
        # データ保存
        x_data.append(np.c_[model.q,model.u])
        if output_q0:
            y_data.append(np.c_[model.q[:,0]])
        elif output_qs:
            y_data.append(model.q[:,:])
        else:
            y_data.append(np.c_[model.x[:,-1],model.y[:,-1] + 1])
        u_data.append(model.tau)
    x_data  = np.array(x_data,np.float32)
    y_data  = np.array(y_data,np.float32)
    u_data  = np.array(u_data,np.float32)
    u_data = np.expand_dims(u_data, 2)
    return x_data,u_data,y_data,y_data


#N:  numner of data 
#n:  numner of pendulum 
def generate(N, input_type_id=1, n=1,output_q0u0=False, output_q0=False, output_qs=False,output_u0=False, input_scale=1):
    np.random.seed(0)

    #  numner of time points
    t_N = int(T/dh)

    t = np.linspace(0, T, t_N)
    x_data =[]
    u_data =generate_input(N,1,T,dh,input_type_id)*input_scale
    y_data = []
    # モデル初期化
    model = NLinkPendulum(n)
    #for omega in np.linspace(0,T,N): #omega=T*i*1.0/N
    for i in range(N):
        # 入力関数の設定
        def func(t_):
          u_out=[]
          if type(t_) is float or type(t_) is np.float64:
            j=int(t_/dh)
            if j<len(u_data[i]):
                u_out=u_data[i,j,0]
            else:
                u_out=u_data[i,-1,0]
            return u_out
          else:
              for t in t_:
                  j=int(t/dh)
                  if j<len(u_data[i]):
                    u_out.append(u_data[i,j,0])
                  else:
                    u_out.append(u_data[i,-1,0])
              return np.array(u_out)
        
        model.set_tau(func)
        # シミュレーション
        model.simulation(t)
        # データ保存
        x_data.append(np.c_[model.q,model.u])
        if output_q0:
            y_data.append(np.c_[model.q[:,0]])
        elif output_qs:
            y_data.append(model.q[:,:])
        elif output_q0u0:
            y_data.append(np.c_[model.u[:,0], model.q[:,0]])
        elif output_u0:
            y_data.append(np.c_[model.u[:,0]])
        else:
            y_data.append(np.c_[model.x[:,-1],model.y[:,-1] + 1])
        #u_data.append(model.tau)
    x_data  = np.array(x_data,np.float32)
    y_data  = np.array(y_data,np.float32)
    u_data  = np.array(u_data,np.float32)
    #u_data = np.expand_dims(u_data, 2)
    return x_data,u_data,y_data,y_data


def generate_dataset(args):
    global T
    global dh
    N = args.num
    M = args.train_num
    name = args.prefix+args.mode
    path=args.path
    if args.input_type_id <0:
        input_type_id=1
    else:
        input_type_id = args.input_type_id
    n=args.nlink_n
    output_q0=args.nlink_q0
    output_qs=args.nlink_qs
    output_q0u0=args.nlink_q0u0
    output_u0=args.nlink_u0
    input_scale=args.input_scale

    if args.T >=0:
        T=args.T
    if args.dh >=0:
        dh=args.dh
    print("> T=",T, " dh=", dh)
    print("> intput ID:",input_type_id)
    print("> nlink:",n)
    print("> input_scale:",input_scale)

    if args.nlink_old_input:
        x_data, u_data, y_data, ys_data = generate_old(N,input_type_id, n, output_q0u0, output_q0, output_qs)
    else:
        x_data, u_data, y_data, ys_data = generate(N,input_type_id, n, output_q0u0, output_q0, output_qs, output_u0, input_scale)
    if not args.without_normalization:
        x_data, u_data, y_data, ys_data = z_normalize(x_data,u_data,y_data, ys_data, path=path, name=name)
    save_dataset(x_data, u_data, y_data, ys_data, M=M, path=path, name=name, T=T, dt=dh)


