import numpy as np
import sympy
import random
import matplotlib.pyplot as plt


random.seed(3)
np.random.seed(3)
alpha1 = 200
alpha2 = 250
alpha3 = 300
T_min = 5
T_max = 30

def r_min(gamma_list):
    return (max(gamma_list)+1)

def r_max(gamma_list):
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

def r_cal1(gamma_list, T_list):
    r_cal_list = []
    T_list_valid = T_valid(T_list)
    for i in range(len(gamma_list)):
        r_cal_list.append(gamma_list[i]*T_list_valid[i])
    return (1.0/sum(T_list_valid))*sympy.sqrt(alpha1*sum(r_cal_list))

def r_cal2(gamma_list, T_list):
    r_cal_list = []
    T_list_valid = T_valid(T_list)
    for i in range(len(gamma_list)):
        r_cal_list.append(gamma_list[i]*T_list_valid[i])
    return (1.0/sum(T_list_valid))*sympy.sqrt(alpha2*sum(r_cal_list))
def a_cal(gamma, r):

    return (1-gamma/r)
def r_cal3(gamma_list, T_list):
    r_cal_list = []
    T_list_valid = T_valid(T_list)
    for i in range(len(gamma_list)):
        r_cal_list.append(gamma_list[i]*T_list_valid[i])
    return (1.0/sum(T_list_valid))*sympy.sqrt(alpha3*sum(r_cal_list))

def utility1(gamma_list, T_list, num):
    a_list = []
    a_list_valid = []
    T_list_valid_list = []
    T_list_valid = T_valid(T_list)
    m = sum(T_list_valid)
    r = r_cal1(gamma_list, T_list)
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

    return (1/m) * (alpha1 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost

def utility2(gamma_list, T_list, num):
    a_list = []
    a_list_valid = []
    T_list_valid_list = []
    T_list_valid = T_valid(T_list)
    m = sum(T_list_valid)
    r = r_cal2(gamma_list, T_list)
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

    return (1/m) * (alpha2 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost

def utility3(gamma_list, T_list, num):
    a_list = []
    a_list_valid = []
    T_list_valid_list = []
    T_list_valid = T_valid(T_list)
    m = sum(T_list_valid)
    r = r_cal3(gamma_list, T_list)
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

    return (1/m) * (alpha3 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost


def utility_max1(gamma_list, T_list, num):
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

    return (1 / m) * (alpha1 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost

def utility_max2(gamma_list, T_list, num):
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

    return (1 / m) * (alpha2 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost

def utility_max3(gamma_list, T_list, num):
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

    return (1 / m) * (alpha3 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost

def utility_min1(gamma_list, T_list, num):
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

    return (1 / m) * (alpha1 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost


def utility_min2(gamma_list, T_list, num):
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

    return (1 / m) * (alpha2 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost

def utility_min3(gamma_list, T_list, num):
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

    return (1 / m) * (alpha3 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost



def utility_random1(gamma_list, T_list, num):
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

    return (1 / m) * (alpha1 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost

def utility_random2(gamma_list, T_list, num):
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

    return (1 / m) * (alpha2 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost


def utility_random3(gamma_list, T_list, num):
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

    return (1 / m) * (alpha3 * sum(a_list_valid)) - sum_cost, client_utility_list, m, sum_cost


num1 = 15
num2 = 20
num3 = 25
T_list1 = np.random.randint(5, 30, num1)
T_list2 = np.random.randint(5, 30, num2)
T_list3 = np.random.randint(5, 30, num3)
gamma_list_list1 = []
gamma_list_list2 = []
gamma_list_list3 = []

for j in range(6):
    gamma_list_list1.append([np.random.uniform(j/2.0+1.0, j/2.0+1.5) for i in range(num1)])

for j in range(6):
    gamma_list_list2.append([np.random.uniform(j/2.0+1.0, j/2.0+1.5) for i in range(num2)])

for j in range(6):
    gamma_list_list3.append([np.random.uniform(j/2.0+1.0, j/2.0+1.5) for i in range(num3)])


# TiFedCrowd
Utility_list_gamma1 = []
Utility_list_gamma2 = []
Utility_list_gamma3 = []
client_utility_list_list1 = []
client_utility_list_list2 = []
client_utility_list_list3 = []
r_list1 = []
r_list2 = []
r_list3 = []
m_list1 = []
m_list2 = []
m_list3 = []
sum_cost1 = []
sum_cost2 = []
sum_cost3 = []
for j in range(6):
    res1 = utility1(gamma_list_list1[j], T_list1, num1)
    Utility_list_gamma1.append(res1[0])
    client_utility_list_list1.append(res1[1])
    m_list1.append(res1[2])
    r_list1.append(r_cal1(gamma_list_list1[j], T_list1))
    sum_cost1.append(res1[3])
    res2 = utility2(gamma_list_list2[j], T_list2, num2)
    Utility_list_gamma2.append(res2[0])
    client_utility_list_list2.append(res2[1])
    m_list2.append(res2[2])
    r_list2.append(r_cal2(gamma_list_list2[j], T_list2))
    sum_cost2.append(res2[3])
    res3 = utility3(gamma_list_list3[j], T_list3, num3)
    Utility_list_gamma3.append(res3[0])
    client_utility_list_list3.append(res3[1])
    m_list3.append(res3[2])
    r_list3.append(r_cal3(gamma_list_list3[j], T_list3))
    sum_cost3.append(res3[3])

client_utility_avg_gamma = []
Utility_list_gamma = []
sum_cost = []
for i in range(6):
    client_utility_avg_gamma.append((sum(client_utility_list_list1[i])+sum(client_utility_list_list2[i])+sum(client_utility_list_list3[i])) / (m_list1[i]+m_list2[i]+m_list3[i]))
    Utility_list_gamma.append(Utility_list_gamma1[i]+Utility_list_gamma2[i]+Utility_list_gamma3[i])
    sum_cost.append(sum_cost1[i]+sum_cost2[i]+sum_cost3[i])


# RWD_MAX
Utility_max_list_gamma1 = []
Utility_max_list_gamma2 = []
Utility_max_list_gamma3 = []
client_utility_list_list1 = []
client_utility_list_list2 = []
client_utility_list_list3 = []
r_max_list1 = []
r_max_list2 = []
r_max_list3 = []
m_list1 = []
m_list2 = []
m_list3 = []
sum_cost_max1 = []
sum_cost_max2 = []
sum_cost_max3 = []
for j in range(6):
    res1 = utility_max1(gamma_list_list1[j], T_list1, num1)
    Utility_max_list_gamma1.append(res1[0])
    client_utility_list_list1.append(res1[1])
    m_list1.append(res1[2])
    r_max_list1.append(r_max(gamma_list_list1[j]))
    sum_cost_max1.append(res1[3])
    res2 = utility_max2(gamma_list_list2[j], T_list2, num2)
    Utility_max_list_gamma2.append(res2[0])
    client_utility_list_list2.append(res2[1])
    m_list2.append(res2[2])
    r_max_list2.append(r_max(gamma_list_list2[j]))
    sum_cost_max2.append(res2[3])
    res3 = utility_max3(gamma_list_list3[j], T_list3, num3)
    Utility_max_list_gamma3.append(res3[0])
    client_utility_list_list3.append(res3[1])
    m_list3.append(res3[2])
    r_max_list3.append(r_max(gamma_list_list3[j]))
    sum_cost_max3.append(res3[3])

client_utility_max_avg_gamma = []
Utility_max_list_gamma = []
sum_cost_max = []
for i in range(6):
    client_utility_max_avg_gamma.append((sum(client_utility_list_list1[i])+sum(client_utility_list_list2[i])+sum(client_utility_list_list3[i])) / (m_list1[i]+m_list2[i]+m_list3[i]))
    Utility_max_list_gamma.append(Utility_max_list_gamma1[i]+Utility_max_list_gamma2[i]+Utility_max_list_gamma3[i])
    sum_cost_max.append(sum_cost_max1[i]+sum_cost_max2[i]+sum_cost_max3[i])



# RWD_RAND
Utility_random_list_gamma1 = []
Utility_random_list_gamma2 = []
Utility_random_list_gamma3 = []
client_utility_list_list1 = []
client_utility_list_list2 = []
client_utility_list_list3 = []
r_random_list1 = []
r_random_list2 = []
r_random_list3 = []
m_list1 = []
m_list2 = []
m_list3 = []
sum_cost_random1 = []
sum_cost_random2 = []
sum_cost_random3 = []

for j in range(6):
    res1 = utility_random1(gamma_list_list1[j], T_list1, num1)
    Utility_random_list_gamma1.append(res1[0])
    client_utility_list_list1.append(res1[1])
    m_list1.append(res1[2])
    r_random_list1.append(r_random(gamma_list_list1[j]))
    sum_cost_random1.append(res1[3])
    res2 = utility_random2(gamma_list_list2[j], T_list2, num2)
    Utility_random_list_gamma2.append(res2[0])
    client_utility_list_list2.append(res2[1])
    m_list2.append(res2[2])
    r_random_list2.append(r_random(gamma_list_list2[j]))
    sum_cost_random2.append(res2[3])
    res3 = utility_random3(gamma_list_list3[j], T_list3, num3)
    Utility_random_list_gamma3.append(res3[0])
    client_utility_list_list3.append(res3[1])
    m_list3.append(res3[2])
    r_random_list3.append(r_random(gamma_list_list3[j]))
    sum_cost_random3.append(res3[3])

client_utility_random_avg_gamma = []
Utility_random_list_gamma = []
sum_cost_random = []
for i in range(6):
    client_utility_random_avg_gamma.append((sum(client_utility_list_list1[i])+sum(client_utility_list_list2[i])+sum(client_utility_list_list3[i])) / (m_list1[i]+m_list2[i]+m_list3[i]))
    Utility_random_list_gamma.append(Utility_random_list_gamma1[i]+Utility_random_list_gamma2[i]+Utility_random_list_gamma3[i])
    sum_cost_random.append(sum_cost_random1[i]+sum_cost_random2[i]+sum_cost_random3[i])


# RWD_MIN
Utility_min_list_gamma1 = []
Utility_min_list_gamma2 = []
Utility_min_list_gamma3 = []
client_utility_list_list1 = []
client_utility_list_list2 = []
client_utility_list_list3 = []
r_min_list1 = []
r_min_list2 = []
r_min_list3 = []
m_list1 = []
m_list2 = []
m_list3 = []
sum_cost_min1 = []
sum_cost_min2 = []
sum_cost_min3 = []

for j in range(6):
    res1 = utility_min1(gamma_list_list1[j], T_list1, num1)
    Utility_min_list_gamma1.append(res1[0])
    client_utility_list_list1.append(res1[1])
    m_list1.append(res1[2])
    r_min_list1.append(r_min(gamma_list_list1[j]))
    sum_cost_min1.append(res1[3])
    res2 = utility_min2(gamma_list_list2[j], T_list2, num2)
    Utility_min_list_gamma2.append(res2[0])
    client_utility_list_list2.append(res2[1])
    m_list2.append(res2[2])
    r_min_list2.append(r_min(gamma_list_list2[j]))
    sum_cost_min2.append(res2[3])
    res3 = utility_min3(gamma_list_list3[j], T_list3, num3)
    Utility_min_list_gamma3.append(res3[0])
    client_utility_list_list3.append(res3[1])
    m_list3.append(res3[2])
    r_min_list3.append(r_min(gamma_list_list3[j]))
    sum_cost_min3.append(res3[3])

client_utility_min_avg_gamma = []
Utility_min_list_gamma = []
sum_cost_min = []
for i in range(6):
    client_utility_min_avg_gamma.append((sum(client_utility_list_list1[i])+sum(client_utility_list_list2[i])+sum(client_utility_list_list3[i])) / (m_list1[i]+m_list2[i]+m_list3[i]))
    Utility_min_list_gamma.append(Utility_min_list_gamma1[i]+Utility_min_list_gamma2[i]+Utility_min_list_gamma3[i])
    sum_cost_min.append(sum_cost_min1[i]+sum_cost_min2[i]+sum_cost_min3[i])