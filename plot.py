import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

# rcParams["font.family"] = "serif"
# rcParams["font.serif"] = ["Times New Roman"]
root_dir = '/data1/wenjie/github/pytorch-CycleGAN-and-pix2pix/results/original2USBplug_mfcc_zscore_scaled_batch16/test_latest/images/results'
realA = np.load(os.path.join(root_dir,'1902_real_A.npy')).T
fakeA = np.load(os.path.join(root_dir,'1902_fake_A.npy')).T
realB = np.load(os.path.join(root_dir,'1902_real_B.npy')).T
fig, axs = plt.subplots(1,3,figsize=(8,2))
x = [0,20,40,60,80,100]
xtick = [0,0.2,0.4,0.6,0.8,1]
y = [0,10,20,30,40]
ytick = [0,1,2,3,4]
axs[0].pcolormesh(realA,cmap='Spectral')
axs[0].set_title('Original')
axs[0].set_xticks(x)
axs[0].set_xticklabels(xtick)
axs[0].set_xlabel('Time (s)')
axs[0].set_yticks(y)
axs[0].set_yticklabels(ytick)
axs[0].set_ylabel('Frequency (kHz)')
axs[1].pcolormesh(fakeA,cmap='Spectral')
axs[1].set_title('M5 to Original')
axs[1].set_xticks(x)
axs[1].set_xticklabels(xtick)
axs[1].set_xlabel('Time (s)')
axs[1].set_yticks(y)
axs[1].set_yticklabels(ytick)
axs[2].pcolormesh(realB,cmap='Spectral')
axs[2].set_title('M5')
axs[2].set_xticks(x)
axs[2].set_xticklabels(xtick)
axs[2].set_xlabel('Time (s)')
axs[2].set_yticks(y)
axs[2].set_yticklabels(ytick)
plt.show()
