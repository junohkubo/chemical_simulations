# 40秒くらいかかる
import numpy as np
import matplotlib.pyplot as plt
import sys

#-- 各種設定
# 定数の設定
Omega = 100.0
c = [0.8, 0.02, 0.8]

# 状態に依存する関数(propensity functioin)の設定
propensities = [lambda x: c[0]*x[0],
                lambda x: c[1]*x[0]*x[1],
                lambda x: c[2]*x[1]]
# 各イベントでの状態遷移の設定
transitions = np.array([[+1, 0],
                        [-1,+1],
                        [ 0,-1]])

# 乱数の種の設定
rng = np.random.default_rng(41736)

# シミュレーションの設定
x_ini = [40, 20] # 振動を見る設定

# Make sample paths and data sets
Dt = 1.0e-3          # Delta t for Euler-Maruyama method
Dt_obs = 1.0e-1      # Time-interval for snapshot pairs
T = 100.0             # Final time for a sample path

# Set parameters
dim = 2

#-- シミュレーション（データ作成）
num_event = len(transitions)

# Perform Monte Carlo simulations
time_array = np.arange(0, T+Dt, Dt)
obs_time_array = np.arange(0, T+2*Dt_obs, Dt_obs)

x = x_ini
dW = rng.standard_normal(size=(len(time_array)-1, num_event))
obs_t = 0.0
result = []
result.append(x)
for i, t in enumerate(time_array[:-1]):
    drift = 0.0
    diff = 0.0
    for e, propensity in enumerate(propensities):
        comp = propensity(x)
        drift = drift + transitions[e] * comp * Dt
        diff = diff + transitions[e] * np.sqrt(comp*Dt/Omega) * dW[i][e]
    x = x + drift + diff
    if x[0] < 0.0:
        x[0] = 0.0
    if x[1] < 0.0:
        x[1] = 0.0
    if t+Dt >= (obs_t-1.0e-6):
        obs_t = obs_t + Dt_obs
        result.append(x)

time_series_data = np.array(result).T

#-- プロット
# 今は状態変数が2つの場合のみに対応
plt.plot(obs_time_array, time_series_data[0,:], color='r', linewidth=0.5)
plt.plot(obs_time_array, time_series_data[1,:], color='b', linewidth=0.5)
plt.xlabel('time')
plt.ylabel('$x_1(t), x_2(t)$')
plt.show()

# 40秒くらいかかる

from scipy.stats import multivariate_normal

# 粒子フィルタ: c[2] が未知で、推定したいとする
rng = np.random.default_rng(25172)
Omega = 100.0
c = [0.8, 0.02]

# 推定したいパラメータ
num_params = 1
z = np.zeros(num_params)
sigma_z = np.array([[0.5/Dt_obs]])
R = np.array([[1.0,0.0],[0.0,1.0]])
H = np.array([[1,0,0],[0,1,0]])

# 初期値の設定
z[0] = 1.2

# 状態に依存する関数(propensity functioin)の設定（推定するパラメータの部分を変更）
propensities = [lambda x, z: c[0]*x[0],
                lambda x, z: c[1]*x[0]*x[1],
                lambda x, z: z[0]*x[1]]
# 各イベントでの状態遷移の設定
transitions = np.array([[+1, 0],
                        [-1,+1],
                        [ 0,-1]])

x = time_series_data[:,0]

results = []

num_particles = 200
p = np.zeros([num_particles, dim+num_params])
p[:] = np.concatenate([x,z])

num_data = 40
obs_t = 0.0
results = np.copy(p)
results = results[:,:,np.newaxis]
time_array = np.arange(0, Dt_obs, Dt)
for obs_i, obs_t in enumerate(obs_time_array[0:num_data]):
    start_t = obs_time_array[obs_i]
    end_t = obs_time_array[obs_i+1]
    time_array = np.arange(start_t, end_t, Dt)
    # simulation
    result = []
    lambda_t = np.zeros(num_particles)
    beta_t = np.zeros(num_particles)
    for ip in range(num_particles):
        dW = rng.standard_normal(size=(len(time_array), num_event))
        dWz = rng.standard_normal(size=(len(time_array), num_params))
        x = p[ip,0:dim]
        z = p[ip,dim:]
        for i, t in enumerate(time_array):
            drift = 0.0
            diff = 0.0
            for e, propensity in enumerate(propensities):
                comp = propensity(x, z)
                drift = drift + transitions[e] * comp * Dt
                diff = diff + transitions[e] * np.sqrt(comp*Dt/Omega) * dW[i][e]
            x = x + drift + diff
            z = z + sigma_z@dWz[i]*Dt
            if x[0] < 0.0:
                x[0] = 0.0
            if x[1] < 0.0:
                x[1] = 0.0
            if z[0] < 0.0:
                z[0] = 0.0

        xz = np.concatenate([x,z])
        p[ip] = xz
        obs_noise = rng.standard_normal(size=(dim))
        y = H@xz + R@obs_noise
        # 尤度の計算
        y_true = time_series_data[:,obs_i+1]
        lambda_t[ip] = multivariate_normal.pdf(y, mean=y_true, cov=R)

    beta_t = lambda_t / np.sum(lambda_t)
    # リサンプリング
    p_index = rng.choice(num_particles, num_particles, p=beta_t)
    p = np.copy(p[p_index])
    results = np.concatenate([results, p[:,:,np.newaxis]], axis=2)

#-- プロット
average_z = np.mean(results[:,2,:], axis=0)
for i in range(num_particles):
    plt.plot(obs_time_array[0:num_data+1], results[i,2,:], color='r', linewidth=0.5)
plt.plot(obs_time_array[0:num_data+1], average_z[:], color='b', linewidth=1.0)
plt.xlabel('time')
plt.ylabel('$z_1(t)$')
plt.show()
