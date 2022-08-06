# 1分ちょっとかかる
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

#-- 各種設定
# 定数の設定
c = [0.5, 0.1, 0.4]

# 状態に依存する関数(propensity functioin)の設定
propensities = [lambda n : c[0]*n[0],
                lambda n : c[1]*n[0]*n[1],
                lambda n : c[2]*n[1]]
# 各イベントでの状態遷移の設定
transitions = np.array([[+1, 0],
                        [-1,+1],
                        [ 0,-1]])

# シミュレーションの設定
n_ini = [4, 2]
t_end = 2
num_plot_points = 20 # プロットする点の数
dim = len(n_ini) # 種の数の設定
max_states = [20, 20] # マスター方程式の上限

# --連立方程式のための設定
# 0から数えるので各次元1つずつ増やしておく
max_states_arr = (max_states + np.ones(len(max_states),dtype="int8")).tolist()
# 初期状態の設定
prob_ini = np.zeros(tuple(max_states_arr)) # 初期状態のための配列確保
prob_ini[tuple(n_ini)] = 1.0 # 初期状態だけ 1 に設定
prob_ini = prob_ini.flatten() # 初期状態を1次元配列に変換
# 1次元配列を作るための準備（ここは少しややこしい）
rev_max_states = list(max_states_arr)
rev_max_states.reverse()
rev_max_states_cumprod = np.cumprod(rev_max_states)
total_1dim_states = rev_max_states_cumprod[-1]
rev_max_states_cumprod = rev_max_states_cumprod[:-1]
def calc_index2state(i):
    # 1次元のindexを元の状態（多次元）に変換する関数
    # 少しややこしいので簡単な例を作って確認を・・
    n = []
    val = i
    for cumprod in reversed(rev_max_states_cumprod):
        q, mod = divmod(val, cumprod)
        n.append(q)
        val = mod
    n.append(val)
    return(n)
# 1次元配列のindexを元の状態（多次元）に変換する配列
index2state = []
for i in range(total_1dim_states):
    index2state.append(calc_index2state(i))
# 元の状態（多次元）を1次元のindexに変換するための準備
arr_for_convert = [1]
arr_for_convert.extend(list(rev_max_states_cumprod))
arr_for_convert.reverse()
arr_for_convert = np.array(arr_for_convert)
def state2index(n):
    # 元の状態（多次元）を1次元のindexに変換する関数
    comp = n * arr_for_convert
    return(np.sum(comp))

#-- シミュレーション
# 時間発展演算子
def evol_op(t, y):
    derivs = np.zeros(total_1dim_states)
    for i in range(total_1dim_states):
        state = np.array(index2state[i])
        deriv = 0.0
        for e, prop in enumerate(propensities):
            deriv = deriv - prop(state)*y[i]
            next_state = state - transitions[e]
            if any(next_state > max_states) or any(next_state < np.zeros(state.shape)):
                continue
            j = state2index(next_state)
            deriv = deriv + prop(next_state)*y[j]
        derivs[i] = deriv
    return derivs

# solve.inpでは状態変数は y で記述されることに注意
t_span = [0, t_end]
t_list = np.linspace(0, t_end, num_plot_points)
ansivp = scipy.integrate.solve_ivp(evol_op, t_span, prob_ini, t_eval = t_list, 
                                   args=(), rtol = 1.e-12, atol = 1.e-14)

probs = ansivp.y.T

# 2次元用: 平均を計算
av = []
prob_extinction = [] # 絶滅する確率の保存用
check_prob_conservation = [] # 確率保存の確認用
for prob in probs:
    prob = prob.reshape(max_states_arr)
    p1 = np.sum(prob, axis=1)
    av1 = np.sum(p1*np.arange(max_states_arr[0]))
    check_prob = np.sum(p1)
    p2 = np.sum(prob, axis=0)
    av2 = np.sum(p2*np.arange(max_states_arr[1]))
    av.append([av1,av2])
    prob_extinction.append(prob[0,0])
    check_prob_conservation.append(check_prob)

# プロット
av = np.array(av).T
plt.plot(t_list, av[0], 'r', linewidth=2)
plt.plot(t_list, av[1], 'b', linewidth=2)
plt.xlabel('time')
plt.ylabel('$n_1(t), n_2(t)$')
plt.show()

print("平均")
for i, time in enumerate(t_list):
    print("t = {0:e}: {1} {2}".format(time, av[0][i], av[1][i]))

print("絶滅の確率")
for i, time in enumerate(t_list):
    print("t = {0:e}: {1}".format(time, prob_extinction[i]))

    
print("確率保存の確認（すべての値を足し算）")
for i, time in enumerate(t_list):
    print("t = {0:f}: {1}".format(time, check_prob_conservation[i]))
