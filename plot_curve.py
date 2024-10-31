#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

## plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 22})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42



# 定数
L = 1.0

# 媒介変数 t の範囲
t_main = np.linspace(0.1, np.pi, 1000)

# x, y の定義
x_main = L * (1 - np.cos(t_main)) / t_main
y_main = L * np.sin(t_main) / t_main

# プロット
plt.figure(figsize=(8, 6))
## plt.plot(x_main, y_main, label=r'$\frac{L (1 - \cos(t))}{t}, \frac{L \sin(t)}{t}$')
plt.plot(x_main, y_main, linestyle='--', color='k', label=r'$\frac{L (1 - \cos(\theta))}{\theta}, \frac{L \sin(\theta)}{\theta}$')


# x軸とy軸の範囲を設定
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)

# 軸の比率を1:1に設定してグリッドを均等にする
plt.gca().set_aspect('equal', adjustable='box')

# t の異なる値でプロット
t_values = [0.01, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6, np.pi]
colors = ['r', 'g', 'b', 'c', 'orange', 'm', 'k']  # 各プロットの色

for i, t in enumerate(t_values):
    a = np.linspace(0.001, t, 100)
    x = L * (1 - np.cos(a)) / t
    y = L * np.sin(a) / t
    plt.plot(x, y, color=colors[i], label=f'θ = {t:.2f}')


# ラベルとタイトル
plt.xlabel('x')
plt.ylabel('y')
## plt.title('Parametric Plot with Different t Values')

# グリッド表示
plt.grid(True)

# 凡例の表示
plt.legend()
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# グラフを表示
plt.show()

plt.savefig("plot.svg", format="svg")
plt.savefig("parametric_curve.pdf", format="pdf")