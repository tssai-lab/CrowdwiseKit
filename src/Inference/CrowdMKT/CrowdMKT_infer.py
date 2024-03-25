
import numpy as np
import datetime
from scipy.optimize import minimize
import sys
from scipy import io

np.set_printoptions(formatter={'float': lambda x: '%.2f' % x})

epsilon = 1e-5


def function(var, xs, ls, zs):

    '''
    global c
    c += 1
    if c % 1000 == 0:
        print("调用第{}轮".format(c))
    '''

    Me = np.zeros((ls.shape[1], xs.shape[1]))
    for i in range(ls.shape[1]):
        Me[i] = var[xs.shape[1] * i : xs.shape[1] * (i + 1)]

    num = 0
    for i in range(1,xs.shape[0]+1):
        for j in range(ls.shape[1]):

            num += (np.log((1 + np.exp(-np.dot(Me[j], xs[i-1]))) ** -1 + epsilon)
                    # + np.log((1 + np.exp(-np.dot(gama, xs[i]))) ** -1 + epsilon)
                    + np.log(show(ls[i][j], zs[i], (1 + np.exp(-np.dot(Me[j], xs[i-1]))) ** -1) + epsilon))

    return -num  


def show(x1, x2, p):

    if x1 == x2:
        return p
    else:
        return (1 - p) / (label_num-1)


def get_truth(var, xs, ls):
    print(ls.shape[0],ls.shape[1],xs.shape[0],xs.shape[1])
    Me = np.zeros((ls.shape[1], xs.shape[1]))
    for i in range(ls.shape[1]):
        Me[i] = var[xs.shape[1] * i: xs.shape[1] * (i + 1)]

    ans = np.zeros(xs.shape[0]+1, dtype=int)

    for i in range(1,xs.shape[0]+1):
        p = []
        for n in range(0,label_num):
            p.append(0.0)

        for j in range(0,ls.shape[1]):
            ability = np.dot(Me[j], xs[i-1])
            prob = (1 + np.exp(-ability)) ** -1
            for k in range(0, label_num):
                p[k] += np.log(prob + epsilon) + np.log(show(ls[i][j], k+1, prob) + epsilon)

        ans[i] = np.argmax(p) + 1

    return ans


def EM(xs, ls, truth, max_iter=50, tol=1e-1):


    var_old = np.ones(ls.shape[1] * xs.shape[1])
    zs = np.zeros(xs.shape[0]+1, dtype=int)

    bound = np.array([0, 1])
    for i in range(ls.shape[1]  * xs.shape[1] - 1):
        bound = np.vstack((bound, [0, 1]))

    ops = {'maxiter': 10, 'disp': True}

    for i in range(max_iter):
        zs = get_truth(var_old, xs, ls)
        print("zs", zs)

        res = minimize(function, var_old, args=(xs, ls, zs), bounds=bound, options=ops)
        print("epoch{}, ".format(i + 1), "The reason of stopping:", res.message)

        var_new = res.x
        if np.linalg.norm(var_new - var_old) < tol:
            break
        var_old = var_new

    k = 0
    for i in range(lss.shape[0]):
        if truth[i] == zs[i]:
            k += 1
    acc = k / ls.shape[0]

    return  zs, var_old, acc


def getdata(datafile, truthfile, sparsefile):
    a = np.loadtxt(open(datafile))
    label = a.astype(int)
    row = int(max(label[:,0]))
    line = int(max(label[:,1]))

    label_set = []
    label_num = 0

    lss = np.zeros((line+1,row+1))
    for lines in label:
        w_id, e_id, value =lines
        lss[e_id][w_id] = value
        if value not in label_set:
            label_set.append(value)
            label_num += 1

    truth = []
    truth.append(0)
    t = np.loadtxt(open(truthfile))
    lt= t.astype(int)
    for lines in lt:
        example, gold = lines
        truth.append(gold)

    vec = []
    with open(sparsefile) as f:
        for line in f:
            temp = line.strip('\n').split(' ')
            vec.append(list(map(float, temp)))

    vec = np.array(vec)
    vec = vec[:,1:line+1].T

    return lss, truth, vec, label_set, label_num

if __name__ == '__main__':
    datafile = sys.argv[1]
    truthfile = sys.argv[2]
    sparsefile = sys.argv[3]

    lss, truth, vec ,label_set,label_num = getdata(datafile, truthfile, sparsefile)

    a, b, c = EM(vec, lss, truth)

    print("acc:",c)

    # 如果文件存在，则先删除
    filename = './CrowdMKT.txt'
    if os.path.exists(filename):
        os.remove(filename)

    inference_label = open(filename, "a+")
    for i in range(1,len(a)):
        inference_label.write(str(i) + ' ' + str(a[i]) + '\n')

