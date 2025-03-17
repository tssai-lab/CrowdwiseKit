import numpy as np
import sympy
import random
import matplotlib.pyplot as plt


random.seed(3)
np.random.seed(3)
alpha = 300
T_min = 5
T_max = 30

def r_min(gamma_list):
    return (max(gamma_list)+1)

def r_max(gamma_list):  # r MAX
    return (max(gamma_list)+8)

def r_random(gamma_list):  # r Random
    return (max(gamma_list)+1+random.random()*7)

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


# TiFedCrowd
Server_utility_list = []

for j in range(6):
    res = utility(gamma_list_list[j], T_list, num)
    Server_utility_list.append(res[0])


# RWD_MAX
Server_utility_max_list = []

for j in range(6):
    res = utility_max(gamma_list_list[j], T_list, num)
    Server_utility_max_list.append(res[0])


# RWD_RAND
Server_utility_random_list = []

for j in range(6):
    res = utility_random(gamma_list_list[j], T_list, num)
    Server_utility_random_list.append(res[0])


# RWD_MIN
Server_utility_min_list = []

for j in range(6):
    res = utility_min(gamma_list_list[j], T_list, num)
    Server_utility_min_list.append(res[0])
