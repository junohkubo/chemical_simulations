import numpy as np
import matplotlib.pyplot as plt

#-- Settings
# Set parameters
c = [0.3, 0.01, 0.2]
# 状態に依存する関数(propensity function)の設定
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
n_ini = [20, 10]
t_ini = 0
t_end = 20

#-- シミュレーション
t = t_ini
n = n_ini

# 定数 c_0 を計算する関数
def calc_c0(n):
    c0 = 0.0
    for e, prop in enumerate(propensities):
        c0 = c0 + prop(n)
    return c0

# 結果を格納するための変数
result_time = []
result_n = []
result_time.append(t)
result_n.append(n)

# 時間発展（アルゴリズムの解説参照）
c0 = calc_c0(n)
while t <= t_end:
    tau = rng.exponential(1.0/c0)
    if t+tau > t_end:
        break
    t = t + tau
    result_time.append(t)
    result_n.append(n)
    r2 = rng.uniform(0.0, 1.0)
    c_event = 0
    for e, prop in enumerate(propensities):
        c_event = c_event + prop(n)/c0
        if r2 < c_event:
            event = e
            break
    n = n + transitions[event]
    result_time.append(t)
    result_n.append(n)
    c0 = calc_c0(n)
    # 種2がなくなると種1が増え続けるだけなので終了させる
    if n[1] == 0:
        break


#-- プロット
# 今は状態変数が2つの場合のみに対応
result_time = np.array(result_time)
result_n = np.array(result_n).T
plt.plot(result_time, result_n[0], 'r', linewidth=2)
plt.plot(result_time, result_n[1], 'b', linewidth=2)
plt.xlabel('time')
plt.ylabel('$n_1(t), n_2(t)$')
plt.show()

