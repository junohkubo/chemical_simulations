import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

import sys

#-- 各種設定
# 定数の設定
c = [0.5, 0.1, 0.4]

# 状態に依存する関数(propensity functioin)の設定
propensities = [lambda n: c[0]*n[0],
                lambda n: c[1]*n[0]*n[1],
                lambda n: c[2]*n[1]]
hazards = [lambda n: n[0],
          lambda n: n[0]*n[1],
          lambda n: n[1]]
# 各イベントでの状態遷移の設定
transitions = np.array([[+1, 0],
                        [-1,+1],
                        [ 0,-1]])

# 乱数の種の設定
rng = np.random.default_rng(641736)

# シミュレーションの設定
sampling_times = [0, 100, 200, 400]
num_event = len(transitions)
n_ini_low = 3   # 初期化の際の最小個体数
n_ini_high = 10 # 初期化の際の最大個体数

#-- シミュレーション
# 定数 c_0 を計算する関数
def calc_c0(n):
    c0 = 0.0
    for e, prop in enumerate(propensities):
        c0 = c0 + prop(n)
    return c0

# 結果を格納するための変数
chis = np.zeros(num_event)
integral_gs = np.zeros(num_event)
result_chis = []
result_integral_gs = []

# サンプリング
t = 0.0
n = rng.integers(low=n_ini_low,high=n_ini_high,size=2)
sampling_count = 1
c0 = calc_c0(n)
result_chis.append(chis.copy())
result_integral_gs.append(integral_gs.copy())

while True:
    # 次のイベントの時刻
    tau = rng.exponential(1.0/c0)

    """
    # 終了時刻を超したら終了
    if t+tau >= sampling_times[-1]:
        dt = sampling_times[-1]-t
        for r, hazard in enumerate(hazards):
            integral_gs[r] = integral_gs[r] + hazard(n)
        break
    """
    # サンプリングの時刻での処理
    if t+tau >= sampling_times[sampling_count]:
        dt = sampling_times[sampling_count]-t
        tmp_integral_gs = integral_gs.copy()
        for r, hazard in enumerate(hazards):
            tmp_integral_gs[r] = integral_gs[r] + hazard(n)*dt
        # 保存
        result_chis.append(chis.copy())
        result_integral_gs.append(tmp_integral_gs.copy())
        sampling_count = sampling_count + 1
        if sampling_count == (len(sampling_times)):
            break
            
    t = t + tau
    r2 = rng.uniform(0.0, 1.0)
    c_event = 0
    for e, prop in enumerate(propensities):
        c_event = c_event + prop(n)/(c0)
        if r2 < c_event:
            event = e
            break
    for r, hazard in enumerate(hazards):
        integral_gs[r] = integral_gs[r] + hazard(n)*tau
    chis[event] = chis[event]+1
    
    n = n + transitions[event]

    # 絶滅したかどうかの確認: もし絶滅していたら状態を初期化し直す
    if np.all(n == 0):
        n = rng.integers(low=n_ini_low,high=n_ini_high,size=2)
        #print("extinction: new state {0}".format(n))
        
    # 種2がなくなると種1が増え続けるだけなので状態を初期化し直す
    if n[1] == 0:
        n = rng.integers(low=n_ini_low,high=n_ini_high,size=2)
        #print("extinction: new state {0}".format(n))
    #print("t:{0}, event: {1}, state {2}".format(t, event, n))
    c0 = calc_c0(n)

result_chis = np.array(result_chis)
result_integral_gs = np.array(result_integral_gs)

alphas = [1.0, 1.0, 1.0]
betas = [1.0, 1.0, 1.0]

result_alphas = []
result_betas = []

for i, chi in enumerate(result_chis):
    tmp = alphas + chi
    result_alphas.append(tmp.copy())
for i, integral_gs in enumerate(result_integral_gs):
    tmp = betas + integral_gs
    result_betas.append(tmp.copy())

x = np.linspace(0.0, 1.0, 100)

plot_r = 0
transp = np.linspace(0.2, 1.0, len(sampling_times))
    
for i, (t, alphas, betas) in enumerate(zip(sampling_times, result_alphas, result_betas)):
    comp = alphas[plot_r]/betas[plot_r]
    print("time: {0}, estimate: {1}".format(t, comp))
    plt.plot(x, gamma.pdf(x, a=alphas[plot_r], scale=1.0/betas[plot_r]), 'r-',
             lw=1, alpha=transp[i], label='beta pdf')

plt.show()
