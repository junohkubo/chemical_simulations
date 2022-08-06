import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

#-- 各種設定
# 定数の設定
c = [1.5, 1.0, 1.0]
# 状態に依存する関数(propensity functioin)の設定
propensities = [lambda x: c[0]*x[0],
                lambda x: c[1]*x[0]*x[1],
                lambda x: c[2]*x[1]]
# 各イベントでの状態遷移の設定
transitions = np.array([[+1, 0],
                        [-1,+1],
                        [ 0,-1]])

# シミュレーションの設定
y_ini = [1.5, 1]
t_end = 10
num_plot_points = 400 # プロットする点の数

#-- シミュレーション
# 時間発展演算子
def evol_op(t, y):
    deriv = np.zeros(transitions.shape[-1])
    for e, propensity in enumerate(propensities):
        comp = propensity(y)
        deriv = deriv + transitions[e] * comp
    return deriv

# solve.inpでは状態変数は y で記述されることに注意
t_span = [0, t_end]
t_list = np.linspace(0, t_end, num_plot_points)

ansivp = scipy.integrate.solve_ivp(evol_op, t_span, y_ini, t_eval = t_list, 
                                   args=(), rtol = 1.e-12, atol = 1.e-14)

#-- プロット
# 今は状態変数が2つの場合のみに対応
plt.plot(t_list, ansivp.y[0], 'r', linewidth=2)
plt.plot(t_list, ansivp.y[1], 'b', linewidth=2)
plt.xlabel('time')
plt.ylabel('$x_1(t), x_2(t)$')
plt.show()
