import numpy as np
import h5py
import matplotlib.pyplot as plt
import mne  # 用于绘制 EEG 头皮图

# 读取 EEG 数据
file_path = "../4_processed_data/KUL_2D.mat"  # 2D 数据文件路径
with h5py.File(file_path, 'r') as f:
    EEG = np.array(f['EEG'])  # 形状: (16, 8, 时间步, 10, 11)

# 设定受试者编号
sbnum = EEG.shape[0]  # 16 名受试者

# 创建一个文件夹保存图片
import os
save_dir = "EEG_Visualization"
os.makedirs(save_dir, exist_ok=True)

# 可视化每位受试者的数据
for sb in range(sbnum):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. EEG 头皮分布图 (Topomap)
    eeg_data = np.mean(EEG[sb, :, :, :, :], axis=(0, 1))  # 平均不同 trial
    im = axes[0].imshow(np.mean(eeg_data, axis=0), cmap='RdBu_r', aspect='auto')
    axes[0].set_title(f"Subject {sb+1} - EEG Topomap")
    plt.colorbar(im, ax=axes[0])

    # 2. EEG 时序波形图 (Time Series)
    time_series = EEG[sb, 0, :500, :, :].reshape(500, -1)[:, :5]  # 取前 500 时间步，展示 5 个通道
    axes[1].plot(time_series)
    axes[1].set_title(f"Subject {sb+1} - EEG Time Series")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("EEG Amplitude")

    # 3. EEG 频谱图 (Spectrogram)
    from scipy.signal import spectrogram
    f, t, Sxx = spectrogram(time_series[:, 0], fs=128)  # 选取第一个通道
    im2 = axes[2].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet')
    axes[2].set_title(f"Subject {sb+1} - EEG Spectrogram")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    plt.colorbar(im2, ax=axes[2])

    # 保存为 PNG
    save_path = os.path.join(save_dir, f"Subject_{sb+1}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

print(f"可视化完成，图片已保存到 '{save_dir}' 文件夹！")
