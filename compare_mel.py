from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    mel0 = np.load('input/001_000_raw.npy',allow_pickle=True)[1]
    mel_rmvpe = np.load('rmvpe_mel.npy',allow_pickle=True)
    mel_fc = np.load('fc_mel.npy',allow_pickle=True)

    x_pm = np.arange(len(mel0))
    x_fc = np.arange(len(mel_fc))
    x_rm = np.arange(len(mel_rmvpe))
    # x_xiaoma = np.arange(len(f0_xiaoma_non_zero))

    # f0_pm 和 f0_rm 是两个数组，可以直接使用 matplotlib 的 plot 函数绘制折线图
    plt.plot(x_pm, mel0, label='Raw')
    plt.plot(x_fc, mel_fc, label='FC')
    plt.plot(x_rm, mel_rmvpe, label='RMVPE')
    # plt.plot(x_xiaoma, f0_xiaoma_non_zero, label='xiaoma')

    # 设置图表标题和坐标轴标签
    plt.title('Mel Comparison')
    plt.xlabel('Time')
    plt.ylabel('Mel')

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()
    plt.savefig('Mel_compare.png',dpi=300, bbox_inches='tight')
