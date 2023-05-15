from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
from ssim import calculate_ssim

# In[]: load model
with open('dgmm.pkl','rb') as f:  # Python 3: open(..., 'rb')
    numTest, img_chns, img_rows, img_cols,H_mu,D2,sigma_h,gamma_mu,C,S,B_mu,rho,K,Y_test,Z_mu,L,resolution,X_test = pickle.load(f)
imagereconstruct = keras.models.load_model('dgmmmodel.h5')


denoising = keras.models.load_model('mnistaedenoising.h5')

print("load model selesai")

# In[]: reconstruct X (image) from Y (fmri)
X_reconstructed_mu = np.zeros((numTest, img_chns, img_rows, img_cols))
HHT = H_mu * H_mu.T + D2 * sigma_h
Temp = gamma_mu * np.mat(np.eye(D2)) - (gamma_mu**2) * (H_mu.T * (np.mat(np.eye(C)) + gamma_mu * HHT).I * H_mu)
for i in range(numTest):
    s=S[:,i]
    z_sigma_test = (B_mu * Temp * B_mu.T + (1 + rho * s.sum(axis=0)[0,0]) * np.mat(np.eye(K)) ).I
    z_mu_test = (z_sigma_test * (B_mu * Temp * (np.mat(Y_test)[i,:]).T + rho * np.mat(Z_mu).T * s )).T
    temp_mu = np.zeros((1,img_chns, img_rows, img_cols))#1,1,28,28
    epsilon_std = 1
    for l in range(L):
        epsilon=np.random.normal(0,epsilon_std,1)
        z_test = z_mu_test + np.sqrt(np.diag(z_sigma_test))*epsilon
        x_reconstructed_mu = imagereconstruct.predict(z_test, batch_size=1)#1,28,28,1
        #edit rolly move axis
        x_reconstructed_mu=np.moveaxis(x_reconstructed_mu,-1,1)
        temp_mu = temp_mu + x_reconstructed_mu # ati2 nih disini main tambahin aja
    x_reconstructed_mu = temp_mu / L
    X_reconstructed_mu[i,:,:,:] = x_reconstructed_mu

# In[]:# visualization the reconstructed images, output in var X_reconstructed_mu
n = 10
mse1=0
mse2=0
psnr1=0
psnr2=0
ssim1=0
ssim2=0
#inputcmap = 'hot'
inputcmap=plt.cm.binary
for j in range(1):
    plt.figure(figsize=(12, 3))    
    for i in range(n):
        # display original images
        ax = plt.subplot(3, n, i +j*n*2 + 1)
        stim=np.rot90(np.fliplr(X_test[i+j*n].reshape(resolution ,resolution )))
        plt.imshow(stim,cmap=inputcmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstructed images
        ax = plt.subplot(3, n, i + n + j*n*2 + 1)
        recons=np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution )))#28x28
        plt.imshow(recons,cmap=inputcmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display denoising images
        ax = plt.subplot(3, n, i + n +n+ j*n*2 + 1)
        inputres=recons.reshape(1,28,28,1)
        res=denoising.predict(inputres)
        fig=res.reshape(1,28,28)[0]
        plt.imshow(fig,cmap=inputcmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Hitung MSE
        msedgmm=np.square(np.subtract(stim,recons)).mean()
        psnrdgmm=20 * math.log10(1.0 / math.sqrt(msedgmm))
        ssimdgmm=calculate_ssim(stim, recons)
        msedae=np.square(np.subtract(stim,fig)).mean()
        psnrdae=20 * math.log10(1.0 / math.sqrt(msedae))
        ssimdae=calculate_ssim(stim, fig)
        mse1=mse1+msedgmm
        mse2=mse2+msedae
        psnr1=psnr1+psnrdgmm
        psnr2=psnr2+psnrdae
        ssim1=ssim1+ssimdgmm
        ssim2=ssim2+ssimdae
        print("mse",i,msedgmm,msedae, psnrdgmm,psnrdae,ssimdgmm,ssimdae)
    print("mse akhir : ",mse1/10,mse2/10,psnr1/10,psnr2/10,ssim1/10,ssim2/10)
    plt.show()
