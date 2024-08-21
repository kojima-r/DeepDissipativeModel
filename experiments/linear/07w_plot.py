from ddm.util_op import dissipative_w
from ddm.util_op import integral_w
y1=torch.tensor(y_true[idx,:,:])
y2=torch.tensor(y_pred[idx,:,:])
y1.shape

u =torch.tensor(test_data.input[idx,:,:])
w,(yQy,uRu,ySu2)=dissipative_w(u,y1,Q,R,S)
int_w=integral_w(w,dt=0.01)
plt.plot(w,label="w")
plt.legend()
