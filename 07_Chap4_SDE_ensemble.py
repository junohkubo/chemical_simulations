import numpy as np
import matplotlib.pyplot as plt

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
Dt_obs = 5.0e-2      # Time-interval for snapshot pairs
T = 10.0             # Final time for a sample path
num_samples = 10

# Set parameters
dim = 2

#-- シミュレーション
num_event = len(transitions)

# Perform Monte Carlo simulations
time_array = np.arange(0, T+Dt, Dt)
obs_time_array = np.arange(0, T+2*Dt_obs, Dt_obs)
results = []
for s in range(num_samples):
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
    result = np.array(result).T
    results.append(result)

results = np.array(results)

av = np.mean(results, axis=0)
std = np.std(results, axis=0)

#-- プロット
# 今は状態変数が2つの場合のみに対応
plt.errorbar(obs_time_array, av[0], yerr=std[0], color='r', elinewidth=0.5)
plt.errorbar(obs_time_array, av[1], yerr=std[1], color='b', elinewidth=0.5)
plt.xlabel('time')
plt.ylabel('$x_1(t), x_2(t)$')
plt.show()


