# -*- coding: utf-8 -*-
import numpy

def gen_rand_sum_one(size):
    """
    generate size random numbers that sum up to one
    :param size: number of random numbers
    :return: list of numbers
    """
    lst = []
    sum = 0
    for i in range(0, size):
        r = numpy.random.random()
        sum += r
        lst.append(r)
    lst2 = [x/sum for x in lst]
    return lst2


def gen_uniform_sum_one(size):
    """
    generate size uniform numbers that sum up to one
    :param size: number of uniform numbers
    :return: list of numbers
    """
    lst = []
    sum = 0
    val = 1.0/size
    for i in range(0, size-1):
        lst.append(val)
        sum += val
    lst.append(1.0-sum)
    return lst


def exclusive_rand(exval, valset, size = 1):
    vallist = list()
    ret_list = list()
    for e in valset:
        if e != exval:
            vallist.append(e)
    list_len = len(vallist)
    positions = rand_int_no_repeat(0, list_len - 1, size)
    for pos in positions:
        ret_list.append(vallist[pos])
    return ret_list


def rand_int_no_repeat(low, high, size):
    lst = list()
    if size == (high - low + 1):
        for i in range(low, high + 1):
            lst.append(i)
            return lst
    elif (size > int((high-low) / 2)):
        exclusive_list = list()
        exclusiv_num = (high - low + 1) - size
        while (len(exclusive_list) < exclusiv_num):
            e = numpy.random.randint(low, high + 1, 1)
            if e[0] in exclusive_list:
                continue
            else:
                exclusive_list.append(e[0])
        for x in range(low, high + 1):
            if x not in exclusive_list:
                lst.append(x)
        return  lst
    else:
        while(len(lst) < size):
            x = numpy.random.randint(low, high + 1, 1)
            if x[0] in lst:
                continue
            else:
                lst.append(x[0])
        return  lst


def get_max_index(vals, flag=False):
    maxindex = 1
    maxval = vals[1]
    size = len(vals)
    for i in range(2, size):
        if vals[i] > maxval:
            maxval = vals[i]
            maxindex = i
    # process multiple maximum
    if (flag == True):
        pos = []
        for i in range(1, size):
            if vals[i] == maxval:
                pos.append(i)
        size2 = len(pos)
        if size2 == 1:
            return maxindex
        else:
            x = numpy.random.randint(0, size2)
            maxindex = pos[x]
    return maxindex


def get_full_combination(M, K):
    current = 0
    comb = []
    s = [None]
    for m in range(1, M+1):
        s.append(0)
    pos = 1
    full_combination(M, K, pos, comb, s)
    return comb


def full_combination(m, K, pos, comb, s):
    if (m != 0):
        for i in range(1, K+1):
            s[pos] = i
            full_combination(m - 1, K, pos + 1, comb, s)
    else:
        comb.append(s.copy())


def split_val_rand(val, n):
    """
    randomly split a val into n pieces
    :param val:
    :param n:
    :return:
    """
    parts = numpy.random.uniform(0, 1, n)
    s = sum(parts)
    for i in range (0, n):
        parts[i] = parts[i]*val/s
    return parts


def gen_rand_sum_one_in_range(size, low, high, try_count=100000):
    """
    generate size random numbers that sum up to one
    :param size: number of random numbers
    :param low: lower bound
    :param high: upper bound
    :try_count: the number of trying
    :return: list of numbers
    """
    count = 0
    while (count < try_count):
        lst = []
        sum = 0.0
        for i in range(0, size-1):
            r = numpy.random.uniform(low, high)
            sum += r
            lst.append(r)
        remainder = 1 - sum
        if (remainder >= low) and (remainder <= high):
            lst.append(remainder)
            return lst
        count += 1
    return None
