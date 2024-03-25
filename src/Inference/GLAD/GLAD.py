import csv
import math
import random
import sys
import numpy as np
from scipy.optimize import minimize

class GLAD:
    def __init__(self,e2wl,w2el,label_set,e2t):
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.examples = self.e2wl.keys()
        self.label_set = label_set
        self.e2t = e2t


    def sigmoid(self,x):
        if (-x)>math.log(sys.float_info.max):
            return 0
        if (-x)<math.log(sys.float_info.min):
            return 1

        return 1/(1+math.exp(-x))

    def logsigmoid(self,x):
        # For large negative x, -log(1 + exp(-x)) = x
        if (-x)>math.log(sys.float_info.max):
            return x
        # For large positive x, -log(1 + exp(-x)) = 0
        if (-x)<math.log(sys.float_info.min):
            return 0

        value = -math.log(1+math.exp(-x))
        #if (math.isinf(value)):
        #    return x

        return value

    def logoneminussigmoid(self,x):
        # For large positive x, -log(1 + exp(x)) = -x
        if (x)>math.log(sys.float_info.max):
            return -x
        # For large negative x, -log(1 + exp(x)) = 0
        if (x)<math.log(sys.float_info.min):
            return 0

        value = -math.log(1+math.exp(x))
        #if (math.isinf(value)):
        #    return -x

        return value

    def kronecker_delta(self,answer,label):
        if answer==label:
            return 1
        else:
            return 0

    def expbeta(self,beta):
        if beta>=math.log(sys.float_info.max):
            return sys.float_info.max
        else:
            return math.exp(beta)


#E step
    def Update_e2lpd(self):
        self.e2lpd = {}
        for example, worker_label_set in self.e2wl.items():
            lpd = {}

            if example in self.e2t:
                truth = e2t[example]
                for label in self.label_set:
                    if label == truth:
                        lpd[label] = 1.0
                    else:
                        lpd[label] = 0.0
                self.e2lpd[example] = lpd
                continue

            total_weight = 0

            for tlabel, prob in self.prior.items():
                weight = math.log(prob)
                for (worker, label) in worker_label_set:
                    logsigma = self.logsigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    logoneminussigma = self.logoneminussigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label,tlabel)
                    weight = weight + delta*logsigma + (1-delta)*(logoneminussigma-math.log(len(label_set)-1))

                if weight<math.log(sys.float_info.min):
                     lpd[tlabel] = 0
                else:
                    lpd[tlabel] = math.exp(weight)
                total_weight = total_weight + lpd[tlabel]

            for tlabel in lpd:
                if total_weight == 0:
                    lpd[tlabel] = 1.0/len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel]*1.0/total_weight

            self.e2lpd[example] = lpd

#M_step


    def gradientQ(self):

        self.dQalpha={}
        self.dQbeta={}

        for example, worker_label_set in self.e2wl.items():
            dQb = 0
            for (worker, label) in worker_label_set:
                for tlabel in self.prior.keys():
                    sigma = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label,tlabel)
                    dQb = dQb + self.e2lpd[example][tlabel]*(delta-sigma)*self.alpha[worker]*self.expbeta(self.beta[example])
            self.dQbeta[example] = dQb - (self.beta[example] - self.priorbeta[example])

        for worker, example_label_set in self.w2el.items():
            dQa = 0
            for (example, label) in example_label_set:
                for tlabel in self.prior.keys():
                    sigma = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label,tlabel)
                    dQa = dQa + self.e2lpd[example][tlabel]*(delta-sigma)*self.expbeta(self.beta[example])
            self.dQalpha[worker] = dQa - (self.alpha[worker] - self.prioralpha[worker])

    def computeQ(self):

        Q = 0
        # the expectation of examples given priors, alpha and beta
        for worker, example_label_set in self.w2el.items():
            for (example, label) in example_label_set:
                logsigma = self.logsigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                logoneminussigma = self.logoneminussigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                for tlabel in self.prior.keys():
                    delta = self.kronecker_delta(label,tlabel)
                    Q = Q + self.e2lpd[example][tlabel]*(delta*logsigma+(1-delta)*(logoneminussigma-math.log(len(label_set)-1)))

        # the expectation of the sum of priors over all examples
        for example in self.e2wl.keys():
            for tlabel, prob in self.prior.items():
                Q = Q + self.e2lpd[example][tlabel] * math.log(prob)
        # Gaussian (standard normal) prior for alpha
        for worker in self.w2el.keys():
             Q = Q + math.log((pow(2*math.pi,-0.5)) * math.exp(-pow((self.alpha[worker]-self.prioralpha[worker]),2)/2))
        # Gaussian (standard normal) prior for beta
        for example in self.e2wl.keys():
            Q = Q + math.log((pow(2*math.pi,-0.5)) * math.exp(-pow((self.beta[example]-self.priorbeta[example]),2)/2))
        return Q

    def optimize_f(self,x):
        # unpack x
        i=0
        for worker in self.workers:
            self.alpha[worker] = x[i]
            i = i + 1
        for example in self.examples:
            self.beta[example] = x[i]
            i = i + 1

        return -self.computeQ() #Flip the sign since we want to minimize

    def optimize_df(self,x):
        # unpack x
        i=0
        for worker in self.workers:
            self.alpha[worker] = x[i]
            i = i + 1
        for example in self.examples:
            self.beta[example] = x[i]
            i = i + 1

        self.gradientQ()

        # pack x
        der = np.zeros_like(x)
        i = 0
        for worker in self.workers:
            der[i] = -self.dQalpha[worker] #Flip the sign since we want to minimize
            i = i + 1
        for example in self.examples:
            der[i] = -self.dQbeta[example] #Flip the sign since we want to minimize
            i = i + 1

        return der

    def Update_alpha_beta(self):

        x0=[]
        for worker in self.workers:
            x0.append(self.alpha[worker])
        for example in self.examples:
            x0.append(self.beta[example])

#        res = minimize(self.optimize_f, x0, method='BFGS', jac=self.optimize_df,tol=0.01,
#               options={'disp': True,'maxiter':100})

        res = minimize(self.optimize_f, x0, method='CG', jac=self.optimize_df,tol=0.01,
               options={'disp': False,'maxiter':25})

        self.optimize_f(res.x)

#likelihood
    def computelikelihood(self):
        L = 0

        for example, worker_label_set in self.e2wl.items():
            L_example= 0;
            for tlabel, prob in self.prior.items():
                L_label = prob
                for (worker, label) in worker_label_set:
                    sigma = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label, tlabel)
                    L_label = L_label * pow(sigma, delta)*pow((1-sigma)/(len(label_set)-1),1-delta)
                L_example = L_example +L_label
            L = L + math.log(L_example)

        for worker in self.w2el.keys():
             L = L + math.log((1/pow(2*math.pi,1/2)) * math.exp(-pow((self.alpha[worker]-self.prioralpha[worker]),2)/2))


        for example in self.e2wl.keys():
            L = L + math.log((1/pow(2*math.pi,1/2)) * math.exp(-pow((self.beta[example]-self.priorbeta[example]),2)/2))


#initialization
    def Init_prior(self):
        #uniform probability distribution
        prior = {}
        for label in self.label_set:
            prior[label] = 1.0/len(self.label_set)
        return prior

    def Init_alpha_beta(self):
        prioralpha={}
        priorbeta={}
        for worker in self.w2el.keys():
            prioralpha[worker]=1
        for example in self.e2wl.keys():
            priorbeta[example]=1
        return prioralpha,priorbeta


    def get_workerquality(self):
        sum_worker = sum(self.alpha.values())
        norm_worker_weight = dict()
        for worker in self.alpha.keys():
            norm_worker_weight[worker] = self.alpha[worker] / sum_worker
        return norm_worker_weight


    def Run(self, threshold = 1e-5):

        self.prior = self.Init_prior()
        self.prioralpha, self.priorbeta = self.Init_alpha_beta()

        self.alpha=self.prioralpha
        self.beta=self.priorbeta

        Q = 0
        self.Update_e2lpd()
        Q = self.computeQ()

        while True:
            lastQ = Q

            # E-step
            self.Update_e2lpd()
            Q = self.computeQ()
            # print( Q )

            # M-step
            self.Update_alpha_beta()
            Q = self.computeQ()
            # print(Q)

            # compute the likelihood
            #print self.computelikelihood()

            if (math.fabs((Q-lastQ)/lastQ))<threshold:
                break

        return self.e2lpd, self.alpha

###################################
# The above is the EM method (a class)
# The following are several external functions
###################################

def getaccuracy(truthfile, e2lpd, label_set):
    e2truth = {}
    f = np.loadtxt(open(truthfile))
    reader = f.astype(int)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    tcount = 0
    count = 0

    for e in e2lpd:

        if e not in e2truth:
            continue

        temp = 0
        for label in e2lpd[e]:
            if temp < e2lpd[e][label]:
                temp = e2lpd[e][label]

        candidate = []

        for label in e2lpd[e]:
            if temp == e2lpd[e][label]:
                candidate.append(label)

        truth = random.choice(candidate)

        count += 1

        if truth == e2truth[e]:
            tcount += 1

    return tcount*1.0/count


def gete2wlandw2el(datafile):
    e2wl = {}
    w2el = {}
    label_set = []

    f = np.loadtxt(open(datafile))
    reader = f.astype(int)

    for line in reader:

        worker, example, label = line
        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker, label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([example, label])

        if label not in label_set:
            label_set.append(label)

    return e2wl, w2el, label_set

def gete2t(known_true):

    e2truth = {}
    f = np.loadtxt(open(known_true))
    reader = f.astype(int)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    return e2truth

if __name__ == "__main__":
    datafile = "MAML/data/label2.resp"
    # datafile = sys.argv[1]
    e2wl,w2el,label_set = gete2wlandw2el(datafile)
    #truthfile = sys.argv[2]
    truthfile = "MAML/data/truth2.gold"
    e2t = gete2t(truthfile)

    e2lpd, weight = GLAD(e2wl,w2el,label_set,e2t).Run(1e-4)

    # print( weight)
    # print (e2lpd)

    accuracy = getaccuracy(truthfile, e2lpd, label_set)
    print(accuracy)
