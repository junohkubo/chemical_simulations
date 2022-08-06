import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys

#-- 各種設定
# 定数の設定
c = [0.5, 0.1, 0.4]

# 状態に依存する関数(propensity functioin)の設定
propensities = [lambda n: c[0]*n[0],
                lambda n: c[1]*n[0]*n[1],
                lambda n: c[2]*n[1]]
# 各イベントでの状態遷移の設定
transitions = np.array([[+1, 0],
                        [-1,+1],
                        [ 0,-1]])

# 乱数の種の設定
rng = np.random.default_rng(641736)

# シミュレーションの設定
n_ini = [4, 2]
t_end = 2
num_samples = 100
sampling_dt = 0.1

#-- シミュレーション
# 定数 c_0 を計算する関数
def calc_c0(n):
    c0 = 0.0
    for e, prop in enumerate(propensities):
        c0 = c0 + prop(n)
    return c0

# 結果を格納するための変数
sampling_times = np.arange(0.0, t_end+1.0e-10, sampling_dt)
results = []

# サンプリング
result_extinct = np.zeros(len(sampling_times))
for sample in range(num_samples):
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
            c_event = c_event + prop(n)/(c0)
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
            t_arr = sampling_times > t
            result_extinct[t_arr] = result_extinct[t_arr] + 1
            break

    # 最後の時刻まで状態保存
    t = t_end+1.0e-10
    t_arr = np.logical_and((prev_t < sampling_times), (sampling_times <= t+tau))
    result[t_arr] = n
    results.append(result)

# 結果出力の準備
results = np.array(results)
result_extinct = result_extinct / num_samples
# プロットのために最大値と最小値を求める
maximum = np.max(results, axis=0).T
minimum = np.min(results, axis=0).T

av = np.mean(results, axis=0).T
std = np.std(results, axis=0).T # 標準偏差（今回は利用しない）


#-- プロット
# 今は状態変数が2つの場合のみに対応

plt.plot(sampling_times, av[0], 'r', linewidth=2)
plt.plot(sampling_times, av[1], 'b', linewidth=2)
plt.plot(sampling_times, maximum[0], 'r', linewidth=0.2)
plt.plot(sampling_times, minimum[0], 'r', linewidth=0.2)
plt.plot(sampling_times, maximum[1], 'b', linewidth=0.2)
plt.plot(sampling_times, minimum[1], 'b', linewidth=0.2)


# もし標準偏差をプロットする場合には上の部分をコメントアウトして下の2行を利用
#plt.errorbar(sampling_times, av[0], yerr=std[0], color='r', elinewidth=0.5)
#plt.errorbar(sampling_times, av[1], yerr=std[1], color='b', elinewidth=0.5)

plt.xlabel('time')
plt.ylabel('$n_1(t), n_2(t)$')
plt.show()



# 四分位数(25%および75%）のプロット
median = np.percentile(results, 50, axis=0).T
yerr_bottom = median - np.percentile(results, 25, axis=0).T
yerr_upper = np.percentile(results, 75, axis=0).T - median
yerr_percentile = np.dstack([yerr_bottom,yerr_upper])
plt.errorbar(sampling_times, median[0], yerr=yerr_percentile[0].T, capsize=3, color='r', elinewidth=0.5)
plt.errorbar(sampling_times, median[1], yerr=yerr_percentile[1].T, capsize=3, color='b', elinewidth=0.5)
plt.xlabel('time')
plt.ylabel('$n_1(t), n_2(t)$')
plt.show()


# seabornを使った箱ひげ図
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
data4boxplot = results.T
dfs = pd.DataFrame()
for s, s_data in enumerate(data4boxplot):
    df = pd.DataFrame()
    for i, data_line in enumerate(s_data):
        df[i] = data_line
    df_melt = pd.melt(df)
    df_melt = df_melt.assign(species=s)
    dfs = pd.concat([dfs,df_melt])
dfs = dfs.assign(time=0)
for t_index, t in enumerate(sampling_times):
    dfs.loc[dfs['variable'] == t_index, 'time'] = '{0:.1f}'.format(t)
sns.boxplot(x='time', y='value', data=dfs, ax=ax, hue="species")
ax.set_xlabel('time')
ax.set_ylabel('values')
plt.show()


print("平均")
for i, time in enumerate(sampling_times):
    print("t = {0:e}: {1} {2}".format(time, av[0][i], av[1][i]))

print("絶滅の確率")
for i, time in enumerate(sampling_times):
    print("t = {0:e}: {1}".format(time, result_extinct[i]))

