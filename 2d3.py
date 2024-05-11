import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve

# モデルパラメータ
beta = 0.5  # 感染率
sigma = 3  # 回復率
gamma = 120  # 免疫喪失率
D = 1  # 拡散係数

# シミュレーション規模・精度
Nx, Ny = 50, 50  # 空間グリッド数
dx = 1  # 空間ステップサイズ
dt = 0.1  # 時間ステップサイズ
T = 100  # 総シミュレーション時間
Nt = int(T / dt)  # 時間ステップ数
N = 10000  # 人口
I0 = 1  # 初期感染者数

# 初期条件
S = np.full((Nx, Ny), (N - I0) / (Nx * Ny))  # 感受性保持者、一様に
I = np.zeros((Nx, Ny))  # 感染者、全体0
R = np.zeros((Nx, Ny))  # 回復者0
I[Nx // 2, Ny // 2] = I0  # 中央に感染者配置


# ルンゲ・クッタ法の実装
def runge_kutta_step(u, f, dt):
    k1 = f(u)
    k2 = f(u + dt / 2 * k1)
    k3 = f(u + dt / 2 * k2)
    k4 = f(u + dt * k3)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# 4次精度ラプラシアン
def laplacian_4th(u):
    kernel = np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 16, 0, 0],
        [-1, 16, -30 - 30, 16, -1],
        [0, 0, 16, 0, 0],
        [0, 0, -1, 0, 0]
    ])
    return convolve(u, kernel, mode='mirror', cval=0) / 12 / dx ** 2  # miror:反射する境界条件


# 2次精度ラプラシアン
def laplacian_2th(u):
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    return convolve(u, kernel, mode='mirror', cval=0) / dx ** 2


# SIRモデル
def sir_model(u):
    S, I, R = u
    dSdt = -beta * S * I + D * laplacian_2th(S)
    dIdt = beta * S * I - 1 / sigma * I + D * laplacian_2th(I)
    dRdt = 1 / sigma * I + D * laplacian_2th(R)
    return np.array([dSdt, dIdt, dRdt])


# SIRSモデル
def sirs_model(u):
    S, I, R = u
    dSdt = -beta * S * I + 1 / gamma * R + D * laplacian_2th(S)
    dIdt = beta * S * I - 1 / sigma * I + D * laplacian_2th(I)
    dRdt = 1 / sigma * I - 1 / gamma * R + D * laplacian_2th(R)
    return np.array([dSdt, dIdt, dRdt])


# シミュレーション実行
S_list = [S.copy()]  # 計算結果を格納する配列
I_list = [I.copy()]
R_list = [R.copy()]
for _ in range(Nt):
    S, I, R = runge_kutta_step(np.array([S, I, R]), sirs_model, dt)
    S_list.append(S.copy())
    I_list.append(I.copy())
    R_list.append(R.copy())


# グラフ生成
def generateScalarGraph(x: int, y: int):
    plt.figure(figsize=(12, 8))
    times = np.arange(0, T, dt)
    plt.plot(times, S_list[:, x, y], label='Susceptible (S)')
    plt.plot(times, I_list[:, x, y], label='Infectious (I)')
    plt.plot(times, R_list[:, x, y], label='Recovered (R)')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIRS Model Simulation using RK4')
    plt.legend()
    plt.grid(True)
    plt.show()


# 静止画生成
def generateImage(step: int):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(S_list[step], cmap='viridis')
    plt.title('Susceptible')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(I_list[step], cmap='viridis')
    plt.title('Infected')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(R_list[step], cmap='viridis')
    plt.title('Recovered')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# 動画生成
def generateAnimation():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    flag = True

    # 動画生成でフレームごとに呼び出される関数
    def update(frame):
        global flag

        for ax in axes:
            ax.clear()
        # 各パネルに対応するデータをプロット
        susceptible = axes[0].imshow(S_list[frame], cmap='viridis')
        infectious = axes[1].imshow(I_list[frame], cmap='hot')
        recovered = axes[2].imshow(R_list[frame], cmap='winter')

        # タイトルの設定
        axes[0].set_title("Susceptible")
        axes[1].set_title("Infectious")
        axes[2].set_title("Recovered")

        # カラーバーを追加
        if flag:
            plt.colorbar(susceptible, ax=axes[0])
            plt.colorbar(infectious, ax=axes[1])
            plt.colorbar(recovered, ax=axes[2])
        flag = False

        return axes

    ani = FuncAnimation(fig, update, frames=Nt, blit=False)
    # MP4ファイルとして再保存
    mp4_path = 'simulation.mp4'
    ani.save(mp4_path, writer='ffmpeg', fps=Nt // 10)


generateAnimation()
