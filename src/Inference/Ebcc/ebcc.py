import random
import math
from data_pipeline import gete2wlandw2el,getaccuracy
from collections import defaultdict
from copy import deepcopy
import time
from results_summary import OfflineResultsSummary
import sys
random.seed(1)


def from_z_to_truths_multi(z_i, label_set):
    truths = {}
    for item in z_i.keys():
        votes = z_i.get(item)
        candidate = []
        max_ = -1
        for class_ in label_set:
            if votes.get(class_) is None:
                continue
            if votes.get(class_) > max_:
                candidate.clear()
                candidate.append(class_)
                max_ = votes.get(class_)
            elif votes.get(class_) == max_:
                candidate.append(class_)
        truths[item] = random.choice(candidate)
    return truths

def list_min_thresh(list_,thresh):
    #return true if the smallest element in list_ is smaller than the given thresh
    for ele in list_:
        if ele <= thresh:
            return True
    return False

def list_max_thresh(list_,thresh):
    # return true if the biggest element in list_ is bigger than the given thresh
    for ele in list_:
        if ele >= thresh:
            return True
    return False

def list_add_scalar(list_,scalar):
    for ind,ele in enumerate(list_):
        list_[ind] += scalar
    return list_




def stablize_exp_helper(list_):
    while list_min_thresh(list_, -300):
        list_add_scalar(list_,300)
        if list_max_thresh(list_,300):
            break
    return list_


def stablize_exp_zg_ikm(zg_ikm):
    for item in zg_ikm.keys():
        for class_ in zg_ikm[item].keys():
            values = zg_ikm[item][class_]
            zg_ikm[item][class_] = stablize_exp_helper(values)
    return zg_ikm



def entr(x):
    try:
        sum_ = 0
        for ele in x:
            if ele != 0:
                sum_ -= ele * math.log(ele)
    except ValueError as e:
        raise e
    return sum_

def list_sum(list_):
    sum_ = 0
    for ele in list_:
        sum_ += ele
    return sum_

def list_mean(list_):
    sum_ = list_sum(list_)
    return sum_ * 1.0 / len(list_)

def dict_values_allclose(dict1, dict2, rtol = 1e-3, atol=1e-03):
    for item in dict1.keys():
        for class_ in dict1[item].keys():
            if math.fabs(dict1[item][class_] - dict2[item][class_]) > atol + rtol * math.fabs(dict2[item][class_]):
                return False
    return True

def log_multi_beta_f(x):
    sum1, sum2 = 0,0
    for ele in x:
        sum1 += math.lgamma(ele)
        sum2 += ele
    return sum1 - math.lgamma(sum2)

def dirichlet_entropy(x):
    val = log_multi_beta_f(x)
    K = len(x)

    sum_ = list_sum(x)
    val += (sum_ - K) * digamma(sum_)

    for ind, ele in enumerate(x):
            val -= (ele - 1) * digamma(ele)
    return val

def digamma(x):
    r = 0
    while (x<=5):
        r -= 1/x
        x += 1
    f = 1/(x*x)
    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
        + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
    return r + math.log(x) - 0.5/x + t

def mv(e2wl, label_set):
    items = list(e2wl.keys())
    votes = {}
    truths={}


    for item in items:
        #aggregate truth
        item_votes = {}
        for class_ in label_set:
            item_votes[class_] = 0
        for worker, worker_label in e2wl[item]:
            item_votes[worker_label]  = item_votes[worker_label] + 1
        candidate = []
        max_ = -1
        for class_ in label_set:
            if item_votes.get(class_) is None:
                continue
            if item_votes.get(class_) > max_:
                candidate.clear()
                candidate.append(class_)
                max_ = item_votes.get(class_)
            elif item_votes.get(class_) == max_:
                candidate.append(class_)
        truths[item] = random.choice(candidate)
        for class_ in label_set:
            item_votes[class_] = item_votes[class_] / len(e2wl[item])
        votes[item] = item_votes

    return truths, votes

def dirichlet_samples(seq_size):
    #assume parameters are all ones
    # params = [1, 1, 1, 1, 1]
    params = [1 for x in range(seq_size)]
    sample = [random.gammavariate(a, 1) for a in params]
    sample = [v / sum(sample) for v in sample]
    return sample

def init_zg_ikm(num_groups,z_ik, label_set):
    zg_ikm = {} #shape: (num_items, num_classes, num_groups)
    for item in z_ik.keys():
        temp = {}
        for class_ in label_set:
            weighted_samples = dirichlet_samples(num_groups)
            for g in range(num_groups):
                weighted_samples[g] *= z_ik[item][class_]
            temp[class_] = weighted_samples
        zg_ikm[item] = temp

    return zg_ikm



def compute_Eq_log_pi_km(eta_km):
    digamma_eta_km_sum_per_class = {}
    for class_, values_ in eta_km.items():
        digamma_eta_km_sum_per_class[class_] = digamma(list_sum(values_))

    Eq_log_pi_km = {}
    for class_, values_ in eta_km.items():
        Eq_log_pi_km[class_] = [digamma(value) - digamma_eta_km_sum_per_class[class_] for value in values_]
    return Eq_log_pi_km

def compute_Eq_log_tau_k(nu_k):
    nu_k_sum = list_sum(list(nu_k.values()))
    Eq_log_tau_k = {}
    for class_ in nu_k.keys():
        Eq_log_tau_k[class_] = digamma(nu_k[class_]) - digamma(nu_k_sum)
    return Eq_log_tau_k

def compute_Eq_log_v_jkml(mu_jkml):
    Eq_log_v_jkml = {}
    for worker,dict_worker in mu_jkml.items():
        dict_worker_log = {}
        Eq_log_v_jkml[worker] = dict_worker_log
        for class_1, dict_class_1 in dict_worker.items():
            dict_class_1_log = {}
            dict_worker_log[class_1] = dict_class_1_log
            for group, dict_group in dict_class_1.items():
                dict_group_log = {}
                dict_class_1_log[group] = dict_group_log
                sum_ = 0
                for _, val in dict_group.items():
                    sum_ += val
                digamma_sum_ = digamma(sum_)
                for class_2, val in dict_group.items():
                    dict_group_log[class_2] = digamma(val) - digamma_sum_
    return Eq_log_v_jkml

def get_max_accuracy(list_):
    #list contains (accuracy, ELBO) pairs
    max_acc, max_ELBO = -sys.float_info.max, -sys.float_info.max
    for (acc, elbo) in list_:
        if elbo > max_ELBO:
            max_acc = acc
            max_ELBO = elbo
    return max_acc,max_ELBO

def ebcc_vb(e2wl, w2el, label_set, num_groups=10, a_pi=0.1, alpha=1,
            a_v=4, b_v=1, seed=1234, max_iter=500, empirical_prior=True):
    num_items = len(e2wl)
    num_workers = len(w2el)
    num_classes = len(label_set)

    # create beta_kl
    beta_kl = [[b_v for x in range(num_classes)] for y in range(num_classes)]
    for x in range(num_classes):
        beta_kl[x][x] += (a_v - b_v)

    _, z_ik = mv(e2wl, label_set)

    # initialize alpha
    if empirical_prior:
        alpha = {}
        for item in e2wl.keys():
            for class_ in label_set:
                if alpha.get(class_) is None:
                    alpha[class_] = 0
                alpha[class_] = alpha[class_] + z_ik[item][class_]
    else:
        alpha_copy = alpha
        alpha = {}
        for class_ in label_set:
            alpha[class_] = alpha_copy

    zg_ikm = init_zg_ikm(num_groups, z_ik, label_set)


    for it in range(max_iter):
        eta_km = {}
        eta_km = defaultdict(lambda: [a_pi / num_groups for x in range(num_groups)], eta_km)
        for item, dict_ in zg_ikm.items():
            for class_ in label_set:
                for pos in range(num_groups):
                    eta_km[class_][pos] += dict_[class_][pos]

        nu_k = alpha.copy()
        for item in z_ik.keys():
            for class_ in label_set:
                nu_k[class_] += z_ik[item][class_]

        mu_jkml = {}
        for worker in w2el.keys():
            dict_worker = {}
            mu_jkml[worker] = dict_worker
            for class_ in label_set:
                dict_class = {}
                dict_worker[class_] = dict_class
                for group in range(num_groups):
                    dict_group = {}
                    dict_class[group] = dict_group
                    for class_2 in label_set:
                        dict_group[class_2] = beta_kl[int(class_)][int(class_2)]

        for l in label_set:
            for k in label_set:

                for worker in w2el.keys():
                    for group in range(num_groups):

                        for worker_item, worker_label in w2el[worker]:
                            if worker_label == l:
                                mu_jkml[worker][k][group][l] += zg_ikm[worker_item][k][group]

        Eq_log_pi_km = compute_Eq_log_pi_km(eta_km)
        Eq_log_tau_k = compute_Eq_log_tau_k(nu_k)
        Eq_log_v_jkml = compute_Eq_log_v_jkml(mu_jkml)

        zg_ikm.clear()
        temp = {}
        for class_ in label_set:
            temp[class_] = [val + Eq_log_tau_k[class_] for val in Eq_log_pi_km[class_]]
        for item in e2wl.keys():
            zg_ikm[item] = deepcopy(temp)

        for l in label_set:
            for k in label_set:

                for item in e2wl.keys():
                    for group in range(num_groups):

                        for item_worker, item_label in e2wl[item]:
                            if item_label == l:
                                zg_ikm[item][k][group] += Eq_log_v_jkml[item_worker][k][group][l]

        zg_ikm = stablize_exp_zg_ikm(zg_ikm)

        for item in zg_ikm.keys():
            for class_ in label_set:
                for group in range(num_groups):
                    val = zg_ikm[item][class_][group]
                    zg_ikm[item][class_][group] = math.exp(val)

        for item, dict_ in zg_ikm.items():
            sum_ = 0
            for class_ in label_set:
                sum_ += list_sum(dict_[class_])
            for class_ in label_set:
                for group in range(num_groups):
                    zg_ikm[item][class_][group] /= sum_

        last_z_ik = deepcopy(z_ik)

        z_ik.clear()
        for item, dict_ in zg_ikm.items():
            dict_2 = {}
            z_ik[item] = dict_2
            for class_ in label_set:
                sum_ = list_sum(zg_ikm[item][class_])
                dict_2[class_] = sum_

        if dict_values_allclose(last_z_ik, z_ik):
            break
    ELBO = compute_ELBO(eta_km,Eq_log_pi_km,nu_k,Eq_log_tau_k,mu_jkml,Eq_log_v_jkml, zg_ikm, label_set, num_groups)
    truths = from_z_to_truths_multi(z_ik, label_set)
    return truths, ELBO, it + 1

def compute_ELBO(eta_km,Eq_log_pi_km,nu_k,Eq_log_tau_k,mu_jkml,Eq_log_v_jkml, zg_ikm, label_set, num_groups):
    ELBO = 0
    for class_ in label_set:
        for eta_km_i, Eq_log_pi_km_i in zip(eta_km[class_],Eq_log_pi_km[class_]):
            ELBO += (eta_km_i-1) * Eq_log_pi_km_i

    for class_ in label_set:
        ELBO += (nu_k[class_] - 1) * Eq_log_tau_k[class_]

    for worker in mu_jkml.keys():
        for class_ in label_set:
            for group in range(num_groups):
                for class_2 in label_set:
                    ELBO += (mu_jkml[worker][class_][group][class_2]-1) * \
                            Eq_log_v_jkml[worker][class_][group][class_2]


    ELBO += dirichlet_entropy(list(nu_k.values()))
    for class_ in label_set:
        ELBO += dirichlet_entropy(eta_km[class_])

    alpha0_jkm = {}
    for worker in mu_jkml.keys():
        alpha0_jkm[worker] = {}
        for class_ in label_set:
            alpha0_jkm[worker][class_] = {}
            for group in range(num_groups):
                alpha0_jkm[worker][class_][group] = 0
                for class_2 in label_set:
                    val = mu_jkml[worker][class_][group][class_2]
                    ELBO += math.lgamma(val) - (val - 1) * digamma(val)

                    alpha0_jkm[worker][class_][group] += val

    for worker in mu_jkml.keys():
        for class_ in label_set:
            for group in range(num_groups):
                val = alpha0_jkm[worker][class_][group]
                ELBO += (val - len(label_set)) * digamma(val) - math.lgamma(val)

    for item in zg_ikm.keys():
        temp = []
        for class_ in label_set:
            temp = temp + zg_ikm[item][class_]
        ELBO += entr(temp)
    return ELBO


