import numpy as np
import sympy
import random
import matplotlib.pyplot as plt


random.seed(0)
np.random.seed(0)
alpha = 300
T_min = 5
T_max = 30

def r_min(gamma_list):
    return max(gamma_list)

def T_valid(T_list):
    T_list_valid = []
    for i in range(len(T_list)):
        if T_list[i] >= T_min and T_list[i] <= T_max:
            T_list_valid.append(1)
        else:
            T_list_valid.append(0)
    return T_list_valid

def r_cal(gamma_list, T_list):
    r_cal_list = []
    T_list_valid = T_valid(T_list)
    for i in range(len(gamma_list)):
        r_cal_list.append(gamma_list[i]*T_list_valid[i])
    return (1.0/sum(T_list_valid))*sympy.sqrt(alpha*sum(r_cal_list))

def r_cal_without(gamma_list):
    r_cal_list = []
    for i in range(len(gamma_list)):
        r_cal_list.append(gamma_list[i])
    return (1.0/num)*sympy.sqrt(alpha*sum(r_cal_list))

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

def utility_without(gamma_list,T_list, num):
    a_list = []
    r = r_cal_without(gamma_list)
    T_valid_list = T_valid(T_list)
    for i in range(num):
        if T_valid_list[i] == 1:
            a_list.append(a_cal(gamma_list[i], r))
        else:
            a_list.append(np.random.uniform(0.1, 0.5))

    sum_cost = 0
    for i in range(num):
        sum_cost += r * a_list[i]

    client_utility_list = []
    for i in range(num):
        client_utility_list.append((r * a_list[i] + gamma_list[i] * sympy.log(1 - a_list[i])))
    return (1/num) * (alpha * sum(a_list)) - sum_cost, client_utility_list, sum_cost

def ranrandom (num, rate, x1, x2):
    ran1 = np.random.randint(x1, x2, int(num*rate))
    ran2 = np.random.randint(1, x1, num-int(num*rate))
    ran3 = np.append(ran1, ran2)
    index = np.random.permutation(ran3.size)
    ran4 = ran3[index]
    return ran4

num = 30
T_list = []
T_valid_list = []
gamma_list_list = []
for j in range(6):
    gamma_list_list.append([np.random.uniform(1.0, 1.5) for i in range(num)])
    T_list.append(ranrandom(num, 1.0-j/10.0, 5, T_max))


# TiFedCrowd
Utility_list_gamma = []
client_utility_list_list = []
r_list = []
m_list = []
sum_cost = []
for j in range(6):
    res = utility(gamma_list_list[j], T_list[j], num)
    Utility_list_gamma.append(res[0])
    client_utility_list_list.append(res[1])
    m_list.append(res[2])
    r_list.append(r_cal(gamma_list_list[j], T_list[j]))
    sum_cost.append(res[3])

client_utility_avg_gamma = []

for i in range(len(client_utility_list_list)):
    client_utility_avg_gamma.append(sum(client_utility_list_list[i]) / m_list[i])


# TiFedCrowd_without time control
Utility_without_list_gamma = []
client_utility_without_list_list = []
r_without_list = []
sum_cost_withut = []

for j in range(6):
    res = utility_without(gamma_list_list[j], T_list[j], num)
    Utility_without_list_gamma.append(res[0])
    client_utility_without_list_list.append(res[1])
    r_without_list.append(r_cal_without(gamma_list_list[j]))
    sum_cost_withut.append(res[2])
client_utility_without_avg_gamma = []

for i in range(len(client_utility_without_list_list)):
    client_utility_without_avg_gamma.append(sum(client_utility_without_list_list[i]) / num)




