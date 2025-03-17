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
    print('T_list--T_list_valid')
    print(T_list)
    print(T_list_valid)
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

    return (1/m) * (alpha * sum(a_list_valid)) - sum_cost, client_utility_list, m, r, sum_cost


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

    return (1/m) * (alpha * sum(a_list_valid)) - sum_cost, client_utility_list, m, r, sum_cost


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

    return (1/m) * (alpha * sum(a_list_valid)) - sum_cost, client_utility_list, m, r, sum_cost



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

    return (1/m) * (alpha * sum(a_list_valid)) - sum_cost, client_utility_list, m, r, sum_cost


num_list = [10, 15, 20, 25, 30, 35]
gamma_range = [1.0, 1.5]



# TiFedCrowd
Utility_list_TiFed = []
r_list_TiFed = []
client_utility_list_list_TiFed = []
sum_cost = []
for j in range(6):
    worker_number = num_list[j]
    gamma_list = []
    T_list = []
    for i in range(worker_number):
        gamma_list.append(random.uniform(gamma_range[0], gamma_range[1]))
        T_list.append(random.uniform(T_min, T_max))
    res = utility(gamma_list, T_list, worker_number)
    Utility_list_TiFed.append(res[0])
    client_utility_list_list_TiFed.append(res[1])
    r_list_TiFed.append(res[3])
    sum_cost.append(res[4])


client_utility_avg_TiFed = []
for utility_list in client_utility_list_list_TiFed:
    client_utility_avg_TiFed.append(sum(utility_list)/len(utility_list))


# RWD_MAX
Utility_list_MAX = []
r_list_MAX = []
client_utility_list_list_MAX = []
sum_cost_max = []
for j in range(6):
    worker_number = num_list[j]
    gamma_list = []
    delta_list = []
    T_list = []
    for i in range(worker_number):
        gamma_list.append(random.uniform(gamma_range[0], gamma_range[1]))
        T_list.append(random.uniform(T_min, T_max))
    res = utility_max(gamma_list, T_list, worker_number)
    Utility_list_MAX.append(res[0])
    client_utility_list_list_MAX.append(res[1])
    r_list_MAX.append(res[3])
    sum_cost_max.append(res[4])

client_utility_avg_MAX = []
for utility_list in client_utility_list_list_MAX:
    client_utility_avg_MAX.append(sum(utility_list)/len(utility_list))


# RWD_RAND
Utility_list_random = []
r_list_random = []
client_utility_list_list_random = []
sum_cost_random = []
for j in range(6):
    worker_number = num_list[j]
    gamma_list = []
    delta_list = []
    T_list = []
    for i in range(worker_number):
        gamma_list.append(random.uniform(gamma_range[0], gamma_range[1]))
        T_list.append(random.uniform(T_min, T_max))
    res = utility_random(gamma_list, T_list, worker_number)
    Utility_list_random.append(res[0])
    client_utility_list_list_random.append(res[1])
    r_list_random.append(res[3])
    sum_cost_random.append(res[4])

client_utility_avg_random = []
for utility_list in client_utility_list_list_random:
    client_utility_avg_random.append(sum(utility_list)/len(utility_list))


# RWD_MIN
Utility_list_min = []
r_list_min = []
client_utility_list_list_min = []
sum_cost_min = []
for j in range(6):
    worker_number = num_list[j]
    gamma_list = []
    delta_list = []
    T_list = []
    for i in range(worker_number):
        gamma_list.append(random.uniform(gamma_range[0], gamma_range[1]))
        T_list.append(random.uniform(T_min, T_max))
    res = utility_min(gamma_list, T_list, worker_number)
    Utility_list_min.append(res[0])
    client_utility_list_list_min.append(res[1])
    r_list_min.append(res[3])
    sum_cost_min.append(res[4])

client_utility_avg_min = []
for utility_list in client_utility_list_list_min:
    client_utility_avg_min.append(sum(utility_list)/len(utility_list))
