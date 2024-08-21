import numpy as np

def generate_input(N, dim, T, dh, input_type_id):
    times = np.arange(0,T,dh)
    n_steps=len(times)
    out_u = np.zeros((N,n_steps,dim))
    # ステップ入力の設定
    if   input_type_id==0:
        # zero input
        u = np.zeros((N,n_steps,dim))
    elif input_type_id==1: # linear
        # random binary input
        u=2 * np.random.randint(0,2,[N,n_steps,dim]) -1
    elif input_type_id==2: # limit cycle
        # ranodom normal input
        u_sigma = 0.5
        u = u_sigma * np.random.randn(N,n_steps,dim)
        u=cumsum(u,axis=1)*0.01
        
    elif input_type_id==3: # bistable 1
        # rectangle inputs
        # positive input at T=0~5
        # negative input at T=5~10 
        # outputs 1-dim
        u1 = np.zeros((N//2,n_steps,1))
        u_step = 4*np.random.rand(N//2)
        tmp =np.tile(u_step.reshape(-1,1), (1,n_steps))
        u1 [times<tmp,:]= 1

        u_step = 4*np.random.rand(N//2)
        tmp =np.tile(u_step.reshape(-1,1), (1,n_steps))
        u1 [(times<tmp+5)&(5<=times),:]= -1

        u2 = np.zeros((N//2,n_steps,1))
        u_step = 4*np.random.rand(N//2)
        tmp =np.tile(u_step.reshape(-1,1), (1,n_steps))
        u2 [times<tmp,:]= -1

        u_step = 4*np.random.rand(N//2)
        tmp =np.tile(u_step.reshape(-1,1), (1,n_steps))
        u2 [(times<tmp+5)&(5<=times),:]= 1
        u=np.concatenate([u1,u2],axis=0)
    elif input_type_id==4: #nagumo
        # negative (Uon) rectangle PWM inputs with a period (T_on + T_off) 
        # outputs 1-dim
        Uon = -0.5
        Uoff = 0
        Toff =  np.random.uniform(0,0.5*T,N)
        Ton  =  np.random.uniform(0,0.5*T,N)
        u =  Uoff *np.ones((N,n_steps,1))
        for i in range(N):
            j=0
            Tperiod=Ton[i]+Toff[i]
            while(j*Tperiod<=T):
                flags=(j*Tperiod + Toff[i]<times)&(times <= (j+1)*Tperiod)
                u[i, flags , 0] =  Uon
                j = j+1
    elif input_type_id==5: # bistable 2
        # one or two random rectangle inputs (U_on=1/-1)
        # outputs 1-dim
        u = np.zeros((N,n_steps,1))
        for i in range(N):
            r=np.random.randint(0,4)
            if r==0:
                s1=np.random.randint(1,n_steps)
                s2=np.random.randint(1,n_steps)
                ss=sorted([s1,s2])
                u[i,ss[0]:ss[1],0]=1
            elif r==1:
                s1=np.random.randint(1,n_steps)
                s2=np.random.randint(1,n_steps)
                ss=sorted([s1,s2])
                u[i,ss[0]:ss[1],0]=-1
            elif r==2:
                s1=np.random.randint(1,n_steps)
                s2=np.random.randint(1,n_steps)
                s3=np.random.randint(1,n_steps)
                s4=np.random.randint(1,n_steps)
                ss=sorted([s1,s2,s3,s4])
                u[i,ss[0]:ss[1],0]=1
                u[i,ss[2]:ss[3],0]=-1
            elif r==3:
                s1=np.random.randint(1,n_steps)
                s2=np.random.randint(1,n_steps)
                s3=np.random.randint(1,n_steps)
                s4=np.random.randint(1,n_steps)
                ss=sorted([s1,s2,s3,s4])
                u[i,ss[0]:ss[1],0]=-1
                u[i,ss[2]:ss[3],0]=1
    elif input_type_id==6: # glucose
        # three rectangle inputs (U_on=random positive value with the same are)
        # outputs 1-dim
        BW = 78
        dt_u = 30
        k=3
        D0 = BW * 1.0 * 1000 # area
        u_data = np.zeros((N,times.shape[0],1))
        for i_sample in range(N):
          # u0: mean=1, sigma=0.1
          u0 = 1  + 0.1 *np.random.randn(k)# D0 + 0.1 D0 sigma [mg]
          u0[u0<0] = 0
          # start time of on
          u_times = times[np.random.randint(0,n_steps,k)]

          u = np.zeros((n_steps,1))
          for i in range(k):
              u[(u_times[i]<=times) & (times<u_times[i] + dt_u)] = D0 * u0[i]/dt_u
          u_data[i_sample,:,:]=u
          u=u_data
    elif input_type_id==7: # glucose
        u = np.zeros((N,n_steps,dim))
        u[:,10:,:]=1.0
    out_u[:,:,:u.shape[2]] =u
    return out_u

