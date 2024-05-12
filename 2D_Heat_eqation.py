import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pandas as pdp
import csv


a=-1.0
b=-1.0*a/3.0
t1 = 0
t2 = 1
end_time = 1

def pde(X,T):
    dT_xx = dde.grad.hessian(T, X ,j=0)
    dT_yy = dde.grad.hessian(T, X, i=1, j=1)
    dT_t = dde.grad.jacobian(T, X, j=2)
    return (dT_t - dT_xx - dT_yy)


def r_boundary(X,on_boundary):
    x,y,t = X
    return on_boundary and np.isclose(x,1)
def l_boundary(X,on_boundary):
    x,y,t = X
    return on_boundary and np.isclose(x,0)
def up_boundary(X,on_boundary):
    x,y,t = X
    return on_boundary and np.isclose(y,1)
def down_boundary(X,on_boundary):
    x,y,t = X
    return on_boundary and np.isclose(y,0)

def boundary_initial(X, on_initial):
    x,y,t = X
    return on_initial and np.isclose(t, 0)

def init_func(X):
    t = np.zeros((len(X),1))
    return t
    
def neu_func(X):
    return b*np.ones((len(X),1))

def neu_func_low(X):
    return a*np.ones((len(X),1))
def zero_func(X):
    return np.zeros((len(X),1))

num_domain = 9600
num_boundary = 400
num_initial = 2
layer_size = [3] + [60] * 5 + [1]
activation_func = "tanh"
initializer = "Glorot uniform"
lr = 1e-3

loss_weights = [10, 1, 1, 1, 1, 10]
epochs = 9000
optimizer = "adam"
batch_size_ = 256

geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])
timedomain = dde.geometry.TimeDomain(0, end_time)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_l = dde.NeumannBC(geomtime, neu_func, l_boundary)
bc_r = dde.NeumannBC(geomtime, neu_func, r_boundary)
bc_up = dde.NeumannBC(geomtime, neu_func , up_boundary) 
bc_low = dde.NeumannBC(geomtime, neu_func_low, down_boundary)
ic = dde.IC(geomtime, init_func, boundary_initial)


data = dde.data.TimePDE(geomtime,
                        pde,
                        [bc_l, bc_r, bc_up, bc_low,ic],
                        num_domain=num_domain,
                        num_boundary=num_boundary,
                        num_initial=num_initial,
                        train_distribution="uniform",)



net = dde.maps.FNN(layer_size, activation_func, initializer)
net.apply_output_transform(lambda x, y: abs(y))

model = dde.Model(data, net)

model.compile(optimizer, lr=lr)

checker = dde.callbacks.ModelCheckpoint("model/model1.ckpt", save_better_only=True, period=1000)
losshistory, trainstate = model.train(epochs=epochs,batch_size = batch_size_,callbacks = [checker])
model.compile("L-BFGS-B")
dde.optimizers.set_LBFGS_options( maxcor=50,)
losshistory, train_state = model.train(epochs = epochs, batch_size = batch_size_)
dde.saveplot(losshistory, trainstate, issave=True, isplot=True)


nelx = 100
nely = 100
timesteps = 67
x = np.linspace(0,1,nelx+1)
y = np.linspace(0,1,nely+1)
t = np.linspace(0,1,timesteps)
delta_t = t[1]-t[0]
xx,yy = np.meshgrid(x,y)

x_ = np.zeros(shape = ((nelx+1) * (nely+1),)) 
y_ = np.zeros(shape = ((nelx+1) * (nely+1),))
for c1,ycor in enumerate(y):
    for c2,xcor in enumerate(x):
        x_[c1*(nelx+1) + c2] = xcor
        y_[c1*(nelx+1) + c2] = ycor
Ts = []
for time in t:
    t_ = np.ones((nelx+1) * (nely+1),) * (time)
    X = np.column_stack((x_,y_))
    X = np.column_stack((X,t_))
    T = model.predict(X)
    T=T
    T = T.reshape(T.shape[0],)
    T = T.reshape(nelx+1,nely+1)
    Ts.append(T)
    with open('Ts.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(Ts)

    
def plotheatmap(T,time):
    plt.clf()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolor(xx, yy, T,cmap = 'RdBu_r',vmin=0,vmax=1)
    plt.colorbar()
   
    return plt

def animate(k):
    plotheatmap(Ts[k], k)
   

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=len(t), repeat=False)
anim.save("proba.gif")  
