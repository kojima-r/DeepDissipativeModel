import numpy as np
import json
import os
obj=np.load("flow_around_a_cylinder_triangle_wave.npz")
all_idx=np.arange(len(obj["right_data"]))
np.random.shuffle(all_idx)

u=obj["left_data"]
y=obj["right_data"]
N=u.shape[0]
os.makedirs("dataset",exist_ok=True)
for name,m in zip(["010","020","030"],[50,100,N]):
    idx=all_idx[:m]
    k=int(m*0.9)
    train_u=u[idx[:k]]
    train_y=y[idx[:k]]
    test_u=u[idx[k:]]
    test_y=y[idx[k:]]
    steps=u.shape[1]
    print("train u:",train_u.shape)
    print("train y:",train_y.shape)

    np.save("dataset/"+name+"flow.train.input.npy",train_u)
    np.save("dataset/"+name+"flow.train.obs.npy",train_y)
    np.save("dataset/"+name+"flow.test.input.npy",test_u)
    np.save("dataset/"+name+"flow.test.obs.npy",test_y)

    dt=0.01
    T=steps*dt

    info={"name": name+"flow", "path": "dataset", "T": T, "dt": dt}
    with open("dataset/"+name+"flow.info.json","w") as fp:
        json.dump(info,fp)

