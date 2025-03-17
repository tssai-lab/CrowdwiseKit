# coding=utf-8
import numpy as np
import sympy
import random
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42

# 绘制图表代码

#random.seed(3)
#np.random.seed(3)
random.seed(300) # Ts
np.random.seed(300) #Ts
alpha = 300
T_min = 5
T_max = 30

def r_min(gamma_list):  # r最小值
    return (max(gamma_list)+1)

def r_max(gamma_list):  # r MAX
    return (max(gamma_list)+8)

def r_random(gamma_list):  # r Random
    return (max(gamma_list)+1+random.random()*7)

def T_valid(T_list):  # 指示器函数
    T_list_valid = []
    for i in range(len(T_list)):
        if T_list[i] >= T_min and T_list[i] <= T_max:
            T_list_valid.append(1)
        else:
            T_list_valid.append(0)
    # print('T_list--T_list_valid')
    # print(T_list)
    # print(T_list_valid)
    return T_list_valid

def r_cal(gamma_list, T_list):
    r_cal_list = []
    T_list_valid = T_valid(T_list)
    for i in range(len(gamma_list)):
        r_cal_list.append(gamma_list[i]*T_list_valid[i])
    return (1.0/sum(T_list_valid))*sympy.sqrt(alpha*sum(r_cal_list))

def a_cal(gamma, r):
    return (1-gamma/r)

def utility(gamma_list, T_list, num):
    a_list = []
    a_list_valid = []
    T_list_valid_list = []
    T_list_valid = T_valid(T_list)
    m = sum(T_list_valid)
    r = r_cal(gamma_list, T_list)
    for i in range(num):
        a_list.append(a_cal(gamma_list[i], r))
        T_list_valid_list.append(T_list_valid[i]*T_list[i])

    for i in range(num):
            a_list_valid.append(a_list[i]*T_list_valid[i])

    sum_cost = 0
    for i in range(num):
        sum_cost += r * a_list_valid[i]

    client_utility_list = []
    for i in range(num):
        client_utility_list.append((r * a_list[i] + gamma_list[i] * sympy.log(1 - a_list[i])) * T_list_valid[i])

    return (1/m) * (alpha * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost


def utility_max(gamma_list, T_list, num):
    a_list = []
    a_list_valid = []
    T_list_valid_list = []
    T_list_valid = T_valid(T_list)
    m = sum(T_list_valid)
    r = r_max(gamma_list)
    for i in range(num):
        a_list.append(a_cal(gamma_list[i], r))
        T_list_valid_list.append(T_list_valid[i] * T_list[i])

    for i in range(num):
        a_list_valid.append(a_list[i] * T_list_valid[i])

    sum_cost = 0
    for i in range(num):
        sum_cost += r * a_list_valid[i]

    client_utility_list = []
    for i in range(num):
        client_utility_list.append((r * a_list[i] + gamma_list[i] * sympy.log(1 - a_list[i])) * T_list_valid[i])

    return (1 / m) * (alpha * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost

def utility_min(gamma_list, T_list, num):
    a_list = []
    a_list_valid = []
    T_list_valid_list = []
    T_list_valid = T_valid(T_list)
    m = sum(T_list_valid)
    r = r_min(gamma_list)
    for i in range(num):
        a_list.append(a_cal(gamma_list[i], r))
        T_list_valid_list.append(T_list_valid[i] * T_list[i])

    for i in range(num):
        a_list_valid.append(a_list[i] * T_list_valid[i])

    sum_cost = 0
    for i in range(num):
        sum_cost += r * a_list_valid[i]

    client_utility_list = []
    for i in range(num):
        client_utility_list.append((r * a_list[i] + gamma_list[i] * sympy.log(1 - a_list[i])) * T_list_valid[i])

    return (1 / m) * (alpha * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost


def utility_random(gamma_list, T_list, num):
    a_list = []
    a_list_valid = []
    T_list_valid_list = []
    T_list_valid = T_valid(T_list)
    m = sum(T_list_valid)
    r = r_random(gamma_list)
    for i in range(num):
        a_list.append(a_cal(gamma_list[i], r))
        T_list_valid_list.append(T_list_valid[i] * T_list[i])

    for i in range(num):
        a_list_valid.append(a_list[i] * T_list_valid[i])

    sum_cost = 0
    for i in range(num):
        sum_cost += r * a_list_valid[i]

    client_utility_list = []
    for i in range(num):
        client_utility_list.append((r * a_list[i] + gamma_list[i] * sympy.log(1 - a_list[i])) * T_list_valid[i])

    return (1 / m) * (alpha * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost


num = 30
T_list = np.random.randint(0, 35, num)
gamma_list_list = []
for j in range(6):
    gamma_list_list.append([np.random.uniform(j/2.0+1.0, j/2.0+1.5) for i in range(num)])

# gamma_list_list = [np.random.uniform(1.0, 1.5) for i in range(num)]


### iFedCrowd
Utility_list_gamma = []
client_utility_list_list = []
r_list = []
m_list = []
sum_cost = []
for j in range(6):
    res = utility(gamma_list_list[j], T_list, num)
    Utility_list_gamma.append(res[0])
    client_utility_list_list.append(res[1])
    m_list.append(res[2])
    r_list.append(r_cal(gamma_list_list[j], T_list))
    sum_cost.append(res[3])

client_utility_avg_gamma = []

for i in range(len(client_utility_list_list)):
    client_utility_avg_gamma.append(sum(client_utility_list_list[i]) / m_list[i])

# print('client_utility_avg_gamma')
# print(client_utility_avg_gamma)
# print('T_list')
# print(T_list)
# print('m_list')
# print(m_list)
# print('gamma_list_list')
# print(gamma_list_list)
# print('r_list')
# print(r_list)
# print('Utility_list_gamma')
# print(Utility_list_gamma)
# print('client_utility_list_list')
# print(client_utility_list_list)
##### max
Utility_max_list_gamma = []
client_utility_list_list = []
r_max_list = []
m_list = []
sum_cost_max = []
for j in range(6):
    res = utility_max(gamma_list_list[j], T_list, num)
    Utility_max_list_gamma.append(res[0])
    client_utility_list_list.append(res[1])
    m_list.append(res[2])
    r_max_list.append(r_max(gamma_list_list[j]))
    sum_cost_max.append(res[3])

client_utility_max_avg_gamma = []

for i in range(len(client_utility_list_list)):
    client_utility_max_avg_gamma.append(sum(client_utility_list_list[i]) / m_list[i])
# #
#
# ##### random
# R = []
# UC = []
# UC1 = []
# US =[]
# TC = []
Utility_random_list_gamma = []
client_utility_list_list = []
r_random_list = []
m_list = []
sum_cost_random = []
# for j in range(20):
#     res = utility_random(gamma_list_list, T_list, num)
#     US.append(res[0])
#     UC.append(sum(res[1])/16)
#     R.append(r_random(gamma_list_list))
#     TC.append(res[3])


for j in range(6):
    res = utility_random(gamma_list_list[j], T_list, num)
    Utility_random_list_gamma.append(res[0])
    client_utility_list_list.append(res[1])
    m_list.append(res[2])
    r_random_list.append(r_random(gamma_list_list[j]))
    sum_cost_random.append(res[3])

client_utility_random_avg_gamma = []
for i in range(len(client_utility_list_list)):
   client_utility_random_avg_gamma.append(sum(client_utility_list_list[i]) / m_list[i])





##### min
Utility_min_list_gamma = []
client_utility_list_list = []
r_min_list = []
m_list = []
sum_cost_min = []
for j in range(6):
    res = utility_min(gamma_list_list[j], T_list, num)
    Utility_min_list_gamma.append(res[0])
    client_utility_list_list.append(res[1])
    m_list.append(res[2])
    r_min_list.append(r_min(gamma_list_list[j]))
    sum_cost_min.append(res[3])

client_utility_min_avg_gamma = []

for i in range(len(client_utility_list_list)):
    client_utility_min_avg_gamma.append(sum(client_utility_list_list[i]) / m_list[i])

plt.rcParams['font.sans-serif'] = ['STSong']  # 设置中文
plt.figure(figsize=(33, 33), dpi=35)
plt.subplot(221)
a = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0']

bar_width = 0.15
x = np.arange(len(a))
x_1 = x - 1.5*bar_width
x_2 = x - 0.5*bar_width
x_3 = x + 0.5*bar_width
x_4 = x + 1.5*bar_width

#plt.figure(figsize=(4, 4), dpi=150)
plt.bar(x_1, r_list, width=bar_width, color='chocolate', label='TsIFedCrowd')
plt.bar(x_2, r_max_list, width=bar_width, color='olive', label='RWD_MAX')
plt.bar(x_3, r_random_list, width=bar_width, color='pink', label='RWD_RAND')
plt.bar(x_4, r_min_list, width=bar_width, color='c', label='RWD_MIN')
params = {'legend.fontsize': 50,
          'legend.handlelength': 3}
plt.rcParams.update(params)
plt.tick_params(labelsize=50)
plt.legend(loc='upper left')
plt.xlabel('∆', fontsize=80)
plt.ylabel('Reward rate(r)',  fontsize=80)
plt.tick_params(labelsize=50)
plt.xticks(x, a)
# plt.show()

plt.subplot(222)
b = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0']

bar_width = 0.15
x = np.arange(len(b))
x_5 = x - 1.5*bar_width
x_6 = x - 0.5*bar_width
x_7 = x + 0.5*bar_width
x_8 = x + 1.5*bar_width


#plt.figure(figsize=(4, 4), dpi=150)
plt.bar(x_5, client_utility_avg_gamma, width=bar_width, color='chocolate', label='TsIFedCrowd')
plt.bar(x_6, client_utility_max_avg_gamma, width=bar_width, color='olive', label='RWD_MAX')
plt.bar(x_7, client_utility_random_avg_gamma, width=bar_width, color='pink', label='RWD_RAND')
plt.bar(x_8, client_utility_min_avg_gamma, width=bar_width, color='c', label='RWD_MIN')
params = {'legend.fontsize': 50,
          'legend.handlelength': 3}
plt.rcParams.update(params)
plt.legend(loc='upper right')
plt.xlabel('∆', fontsize=80)
plt.ylabel('Average utility of clients',  fontsize=80)
plt.tick_params(labelsize=50)
plt.xticks(x, b)

# plt.show()

plt.subplot(223)
c = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0']

bar_width = 0.15
x = np.arange(len(c))
x_9 = x - 1.5*bar_width
x_10 = x - 0.5*bar_width
x_11 = x + 0.5*bar_width
x_12 = x + 1.5*bar_width


#plt.figure(figsize=(4, 4), dpi=150)
plt.bar(x_9, Utility_list_gamma, width=bar_width, color='chocolate', label='TsIFedCrowd')
plt.bar(x_10, Utility_max_list_gamma, width=bar_width, color='olive', label='RWD_MAX')
plt.bar(x_11, Utility_random_list_gamma, width=bar_width, color='pink', label='RWD_RAND')
plt.bar(x_12, Utility_min_list_gamma, width=bar_width, color='c', label='RWD_MIN')
params = {'legend.fontsize': 50,
          'legend.handlelength': 3}
plt.rcParams.update(params)
plt.legend(loc='upper right')
plt.xlabel('∆', fontsize=80)
plt.ylabel('Utility of the server',  fontsize=80)
plt.tick_params(labelsize=50)
plt.xticks(x, c)

plt.subplot(224)
d = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0']

bar_width = 0.15
x = np.arange(len(c))
x_13 = x - 1.5*bar_width
x_14 = x - 0.5*bar_width
x_15 = x + 0.5*bar_width
x_16 = x + 1.5*bar_width

plt.bar(x_13, sum_cost, width=bar_width, color='chocolate', label='TsIFedCrowd')
plt.bar(x_14, sum_cost_max, width=bar_width, color='olive', label='RWD_MAX')
plt.bar(x_15, sum_cost_random, width=bar_width, color='pink', label='RWD_RAND')
plt.bar(x_16, sum_cost_min, width=bar_width, color='c', label='RWD_MIN')
params = {'legend.fontsize': 50,
          'legend.handlelength': 3}
plt.rcParams.update(params)
plt.legend(loc='upper right')
plt.xlabel('∆', fontsize=80)
plt.ylabel('Total incentive cost',  fontsize=80)
plt.tick_params(labelsize=50)
plt.xticks(x, d)


plt.savefig('fig22.pdf', bbox_inches='tight')
plt.show()

# print(r_list[0])
# print(client_utility_avg_gamma[0])
# print(Utility_list_gamma[0])
# print(sum_cost[0])
# print(r_max_list[0])
# print(client_utility_max_avg_gamma[0])
# print(Utility_max_list_gamma[0])
# print(sum_cost_max[0])
# print(r_min_list[0])
# print(client_utility_min_avg_gamma[0])
# print(Utility_min_list_gamma[0])
# print(sum_cost_min[0])
# print(r_random_list[0])
# print(client_utility_random_avg_gamma[0])
# print(Utility_random_list_gamma[0])
# print(sum_cost_random[0])

# print(sum(R)/20)
# print(sum(UC)/20)
# print(sum(US)/20)
# print(sum(TC)/20)

