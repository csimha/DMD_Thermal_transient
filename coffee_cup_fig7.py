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

Xi_smooth = np.array([np.mean(data_r[i:i+60,:], axis=0) for i in range (data_r.shape[0] - 60)])
Xi_smooth = data_r.T


skip = 30; start_frame = 1000; end_frame = 11000

Xi = Xi_smooth[:, start_frame: end_frame:skip]   #---this is our sampled DMD snapshot matrix
Xi =Xi[:,:300]
Xi_image = Xi.T
Xi_image = Xi_image.reshape((Xi_image.shape[0],250,200))

"""

Xi = Xi_smooth[:,1000:10000]
"""
"""
Using complete DMD method

"""

window = 200

X1 = Xi[:, :window]
X2 = Xi[:, 1 : window+1]
## Perform singular value decomposition on X1
u, s, v = np.linalg.svd(X1, full_matrices = False)

r = 4

ur, sr, vr = u[:, : r], s[: r], v[: r, :]
## Compute the Koopman matrix
A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
## Perform eigenvalue decomposition on A_tilde
eig_val_, eig_vec_ = np.linalg.eig(A_tilde)
eig_val = eig_val_.real
eig_vec = eig_vec_.real
## Compute the dmd_modes
Phi = X2 @ vr.conj().T @ np.diag(np.reciprocal(sr)) @ eig_vec

b = np.linalg.lstsq(Phi, Xi[:,0], rcond=-1)[0]
omega = np.log(eig_val)


#Time dynamics
mm1 = Xi.shape[1]
time_dynamics = np.zeros((r, mm1))
t = np.array(list(range(0,mm1)))
for i in range(0,mm1):
    time_dynamics[:,i] += (b * np.exp(omega*t[i]))

#Reconstruction
sol = Phi @ time_dynamics


#More detailed plots for one selected ranks
error = np.divide(np.abs(sol - Xi), Xi)*100
error_avg = np.mean(error, 0)
x_axis = np.linspace(1,300,300)

fig, ax = plt.subplots(figsize = (5,4))
ax.plot(x_axis[window:], error_avg[window:], marker = 'v', markersize=4, markevery=10, label=f'{window} s')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Average relative error (%)')
ax.legend(title='Training window', loc = 'upper left')
#fig.savefig('plots/coffee_prediction.eps', dpi=600)
