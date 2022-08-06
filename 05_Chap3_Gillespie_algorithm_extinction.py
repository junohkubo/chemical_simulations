# 1分ちょっとかかる
import numpy as np
import matplotlib.pyplot as plt

import sys
#-- 各種設定
# 定数の設定
c = [0.5, 1.0, 1.0]

# 状態に依存する関数(propensity functioin)の設定
propensities = [lambda n: c[0]*n[0],
                lambda n: c[1]*n[0]*n[1],
                lambda n: c[2]*n[1]]
# 各イベントでの状態遷移の設定
transitions = np.array([[+1, 0],
                        [-1,+1],
                        [ 0,-1]])

# 乱数の種の設定
rng = np.random.default_rng(41736)

# シミュレーションの設定
n_ini = [4, 2]
t_end = 2
sampling_dt = 0.2
num_ensembles = 20

#-- シミュレーション
# 定数 c_0 を計算する関数
def calc_c0(n):
    c0 = 0.0
    for e, prop in enumerate(propensities):
        c0 = c0 + prop(n)
    return c0

# 結果を格納するための変数
sampling_times = np.arange(0.0, t_end+1.0e-10, sampling_dt)
sampling_size = [10, 40, 160, 640, 2560, 5120]

extinct_probs = []
for num_samples in sampling_size:
    results_extinct = []
    for num_ensemble in range(num_ensembles):
        result_extinct = np.zeros(len(sampling_times))
        for sample in range(num_samples):
            t = 0.0
            n = n_ini
            # 時間発展（アルゴリズムの解説参照）
            c0 = calc_c0(n)
            while t <= t_end:
                tau = rng.exponential(1.0/c0)
                if t+tau > t_end:
                    break
                t = t + tau
                r2 = rng.uniform(0.0, 1.0)
                c_event = 0
                for e, prop in enumerate(propensities):
                    c_event = c_event + prop(n)/c0
                    if r2 < c_event:
                        event = e
                        break
                n = n + transitions[event]
                c0 = calc_c0(n)
                # 絶滅したかどうかの確認
                if np.all(n == 0):
                    t_arr = sampling_times > t
                    result_extinct[t_arr] = result_extinct[t_arr] + 1
                    break
                
                # 種2がなくなると種1が増え続けるだけなので終了させる
                if n[1] == 0:
                    break
        # 絶滅確率の計算
        result_extinct = result_extinct / num_samples
        results_extinct.append(result_extinct[-1])
    extinct_probs.append(results_extinct)

extinct_probs = np.array(extinct_probs)

print(extinct_probs)
true_extinct_prob = 0.19933537885735433

avs = []
stds = []
line = []
for i, num_samples in enumerate(sampling_size):
    dist = np.abs(extinct_probs[i] - true_extinct_prob)
    av = np.mean(dist)
    std = np.std(dist)
    avs.append(av)
    stds.append(std)
    comp = 8.0e-2*1.0/np.power(num_samples, 0.5)
    line.append(comp)
    print("{0} {1} {2}".format(num_samples, av, std))

avs = np.array(avs)
stds = np.array(stds)
line = np.array(line)
#-- プロット
plt.xscale("log")
plt.yscale("log")
plt.errorbar(sampling_size, avs, stds, color='r', linewidth=2, elinewidth=0.1)
plt.plot(sampling_size, line, color="b")
plt.xlabel('sample size')
plt.ylabel('error')
plt.show()

