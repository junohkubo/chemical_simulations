# 30秒くらいかかる
import numpy as np
import matplotlib.pyplot as plt

#-- 各種設定
# 定数の設定
Omega = 100.0
c = [0.8, 0.02, 0.8]

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
n_ini = [40, 20]

t_end = 10
num_samples = 10
sampling_dt = 0.05
#-- シミュレーション

# 定数 c_0 を計算する関数
def calc_c0(n):
    c0 = 0.0
    for e, prop in enumerate(propensities):
        c0 = c0 + prop(n)
    return c0*Omega

# 結果を格納するための変数
sampling_times = np.arange(0.0, t_end+1.0e-10, sampling_dt)
results = []
for sample in range(num_samples):
    print("sample {0}".format(sample))
    t = 0.0
    n = n_ini
    # 時刻 0 の状態の格納
    result = np.zeros((len(sampling_times),len(n_ini)))

    # 時間発展（アルゴリズムの解説参照）
    c0 = calc_c0(n)
    prev_t = -1.0
    while t <= t_end:
        tau = rng.exponential(1.0/c0)
        # サンプリング時刻での状態の保存
        t_arr = np.logical_and((prev_t < sampling_times), (sampling_times <= t+tau))
        result[t_arr] = n
        if t+tau > t_end:
            break
        prev_t = t
        t = t + tau
        r2 = rng.uniform(0.0, 1.0)
        c_event = 0
        for e, prop in enumerate(propensities):
            c_event = c_event + (prop(n)*Omega)/c0
            if r2 < c_event:
                event = e
                break
        n = n + transitions[event]/Omega
        c0 = calc_c0(n)
        # 種2がなくなると種1が増え続けるだけなので終了させる
        if n[1] == 0:
            break

    # 最後の時刻まで状態保存
    t = t_end+1.0e-10
    t_arr = np.logical_and((prev_t < sampling_times), (sampling_times <= t+tau))
    result[t_arr] = n
    results.append(result)

results = np.array(results)

av = np.mean(results, axis=0).T
std = np.std(results, axis=0).T
maximum = np.max(results, axis=0).T
minimum = np.min(results, axis=0).T
#-- プロット
# 今は状態変数が2つの場合のみに対応
#plt.plot(sampling_times, av[0], 'r', linewidth=2)
#plt.plot(sampling_times, av[1], 'b', linewidth=2)

plt.errorbar(sampling_times, av[0], yerr=std[0], color='r', elinewidth=0.5)
plt.errorbar(sampling_times, av[1], yerr=std[1], color='b', elinewidth=0.5)

"""
plt.plot(sampling_times, maximum[0], 'r', linewidth=0.2)
plt.plot(sampling_times, minimum[0], 'r', linewidth=0.2)
plt.plot(sampling_times, maximum[1], 'b', linewidth=0.2)
plt.plot(sampling_times, minimum[1], 'b', linewidth=0.2)
"""
plt.xlabel('time')
plt.ylabel('$n_1(t), n_2(t)$')
plt.show()


