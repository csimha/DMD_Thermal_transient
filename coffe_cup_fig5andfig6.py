# -*- coding: utf-8 -*-
"""
@author: dhruvin dhankara 
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import data_prep_utils

def sum_rel_error(x1, x2):
    return np.sum(np.abs(np.divide(x1 - x2, x1 )))

"""
#Get temperatures from the images and save in numpy array
temps = data_prep_utils.get_images('output', 100, 10, [40,240,0,250])

with open('cup_data.pickle', 'wb') as f:
    pickle.dump(temps, f)
"""
      
data = []
file = open('cup_data.pickle', 'rb')    

data = pickle.load(file)

data_r = data.reshape(  ( data.shape[0], data.shape[1]*data.shape[2]))  #---reshaped data

Xi_smooth = np.array([np.mean(data_r[i:i+30,:], axis=0) for i in range (data_r.shape[0] - 30)])
Xi_smooth = data_r.T


skip = 30; start_frame = 1000; end_frame = 11000

Xi = Xi_smooth[:, start_frame: end_frame:skip]   #---this is our sampled DMD snapshot matrix
Xi =Xi[:,:300]


X1 = Xi[:, : -1]
X2 = Xi[:, 1 : ]
## Perform singular value decomposition on X1
u, s, v = np.linalg.svd(X1, full_matrices = False)

def reconstruct(r, u, s, v):
    ur, sr, vr = u[:, : r], s[: r], v[: r, :]
    ## Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    ## Perform eigenvalue decomposition on A_tilde
    eig_val_, eig_vec_ = np.linalg.eig(A_tilde)
    eig_val = eig_val_.real
    eig_vec = eig_vec_.real
    ## Compute the dmd_modes
    Phi = X2 @ vr.conj().T @ np.diag(np.reciprocal(sr)) @ eig_vec
    
    b = np.linalg.lstsq(Phi, X1[:,0], rcond=-1)[0]
    omega = np.log(eig_val)/1.0
    
    
    #Time dynamics
    mm1 = Xi.shape[1]
    time_dynamics = np.zeros((r, mm1))
    t = np.array(list(range(0,mm1)))
    for i in range(0,mm1):
        time_dynamics[:,i] += (b * np.exp(omega*t[i]))
    
    #Reconstruction
    sol = Phi @ time_dynamics
    return sol

#Plotting relation between rank and error
err = []

for r in range (2, 21):
    sol = reconstruct(r, u, s, v)
    error = sum_rel_error(sol[:,-1],Xi[:,-1])
    err.append(error)
err = np.array(err)

fig, ax = plt.subplots(figsize = (5,4))
ax.plot(np.arange(2,21,1), err[:]/50000*100)
ax.grid()
ax.set_xticks(np.arange(2,21,2))
ax.set_xlabel('Number of retained Modes')
ax.set_ylabel('Average relative error (%)')
#fig.savefig('plots/coffee_modes_error.eps', dpi=600)

#More detailed plots for one selected ranks
r =3
sol = reconstruct(r, u, s, v)
error = np.divide(np.abs(sol - Xi),Xi)

#Plotting error snaps at regular intervals
image = []
for i in range (0,5):
    image.append(error[:,i*60]*100)
image.append(error[:,-1]*100)

cmap = plt.cm.binary
norm = colors.BoundaryNorm(np.linspace(-0.0, 15.0,11), cmap.N)
fig, axes = plt.subplots(nrows=1, ncols=6)
fig.set_figheight(3)
fig.set_figwidth(10)
for i,ax in enumerate(axes.flat):
    im = ax.imshow(image[i].reshape((250,200)), cmap=cmap, norm=norm)
    ax.plot(100,150,'ob', markersize=4)
    ax.plot(83,12,'xr', markersize=4)
    ax.plot(15,200,'dc', markersize=4)
    ax.set_title(f'{i*60} s')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.patch.set_linewidth(3)
    
plt.tight_layout(w_pad=0.25, h_pad=0.5)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbr = fig.colorbar(im, cax=cbar_ax)
cbr.ax.set_ylabel('Relative Error (%)')
#plt.savefig(f'plots/coffee_error_images_r{r}.jpg', dpi=600)

#Error on 3 pixels for all times
sol_image = sol.T
sol_image = sol_image.reshape((300,250,200))

fig = plt.figure(figsize=(10,4.5))
ax = fig.add_subplot()
y = np.linspace(0,sol_image.shape[0],sol_image.shape[0])

plt.plot(y, sol_image[:,150,100], 'b--', label = 'Point 1 DMD')
plt.plot(y, sol_image[:,12,83], 'r--', label = 'Point 2 DMD')
plt.plot(y, sol_image[:,200,15], 'c--', label = 'Point 3 DMD')

Xi_image = Xi.T
Xi_image = Xi_image.reshape((300,250,200))

plt.plot(y, Xi_image[:,150,100], marker = 'o', color = 'b', markersize=4, markevery=5, label = 'Point 1 Actual', alpha=0.3)
plt.plot(y, Xi_image[:,12,83], marker = 'x', color = 'r', markersize=4, markevery=5, label = 'Point 2 Actual', alpha=0.3)
plt.plot(y, Xi_image[:,200,15], marker = 'd', color = 'c', markersize=4, markevery=5, label = 'Point 3 Actual', alpha=0.3)

plt.legend(loc="upper right")
plt.xlabel('Time (s)')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
#plt.savefig(f'plots/coffee_temp_Comparision_r{r}.jpg', dpi=600)

