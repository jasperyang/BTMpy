# -*- coding: utf-8 -*-
from pvec import *
import numpy as np
from doc import Doc
from sampler import *

class Model():
    bs = []
    W = 0   # vocabulary size
    K = 0   # number of topics
    n_iter = 0  # maximum number of iteration of Gibbs Sampling
    save_step = 0
    alpha = 0   # hyperparameters of p(z)
    beta = 0    # hyperparameters of p(w|z)
    nb_z = Pvec()   # n(b|z), size K*1
    nwz = np.zeros((1,1)) # n(w,z), size K*W
    pw_b = Pvec()   # the background word distribution


    '''
        If true, the topic 0 is set to a background topic that 
        equals to the empirical word distribution. It can filter
        out common words
    '''
    has_background = False

    def __init__(self,K,W,a,b,n_iter,save_step,has_b=False):
        self.K = K
        self.W = W
        self.alpha = a
        self.beta = b
        self.n_iter = n_iter
        self.save_step = save_step
        self.has_background = has_b
        self.pw_b.resize(W)
        self.nwz.resize((K,W))
        self.nb_z.resize(K)

    def run(self,doc_pt,res_dir):
        self.load_docs(doc_pt)
        self.model_init()

        print("Begin iteration")
        out_dir = res_dir + "k" + str(self.K) + "."
        for i in range(1,self.n_iter+1):
            print("\riter "+str(i)+"/"+str(self.n_iter))
            for b in range(len(self.bs)):
                self.update_biterm(self.bs[b])

            if i%self.save_step == 0:
                self.save_res(out_dir)

        self.save_res(out_dir)

    def model_init(self):
        for biterm in self.bs:
            k = uni_sample(self.K)
            self.assign_biterm_topic(biterm,k)

    def load_docs(self,docs_pt):
        print("load docs: " + docs_pt)
        rf = open(docs_pt)
        if not rf:
            print("file not found: " + docs_pt)

        for line in rf.readlines():
            d = Doc(line)
            biterms = []
            d.gen_biterms(biterms)
            # statistic the empirical word distribution
            for i in range(d.size()):
                w = d.get_w(i)
                self.pw_b[w] += 1
            for b in biterms:
                self.bs.append(b)

        self.pw_b.normalize()

    def update_biterm(self,bi):
        self.reset_biterm_topic(bi)

        # comput p(z|b)
        pz = Pvec()
        self.comput_pz_b(bi,pz)

        # sample topic for biterm b
        k = mul_sample(pz.to_vector())
        self.assign_biterm_topic(bi,k)

    def reset_biterm_topic(self,bi):
        k = bi.get_z()
        w1 = bi.get_wi()
        w2 = bi.get_wj()

        self.nb_z[k] -= 1
        self.nwz[k][w1] -= 1
        self.nwz[k][w2] -= 1
        assert(self.nb_z[k] > -10e-7 and self.nwz[k][w1] > -10e-7 and self.nwz[k][w2] > -10e-7)
        bi.reset_z()

    def assign_biterm_topic(self,bi,k):
        bi.set_z(k)
        w1 = bi.get_wi()
        w2 = bi.get_wj()
        self.nb_z[k] += 1
        self.nwz[k][w1] += 1
        self.nwz[k][w2] += 1

    def comput_pz_b(self,bi,pz):
        pz.resize(self.K)
        w1 = bi.get_wi()
        w2 = bi.get_wj()

        for k in range(self.K):
            if (self.has_background and k == 0) :
                pw1k = self.pw_b[w1];
                pw2k = self.pw_b[w2];
            else:
                pw1k = (self.nwz[k][w1] + self.beta) / (2 * self.nb_z[k] + self.W * self.beta);
                pw2k = (self.nwz[k][w2] + self.beta) / (2 * self.nb_z[k] + 1 + self.W * self.beta);

            pk = (self.nb_z[k] + self.alpha) / (len(self.bs) + self.K * self.alpha);
            pz[k] = pk * pw1k * pw2k;

    def save_res(self,res_dir):
        pt = res_dir + "pz"
        print("\nwrite p(z): "+pt)
        self.save_pz(pt)

        pt2 = res_dir + "pw_z"
        print("write p(w|z): "+pt2)
        self.save_pw_z(pt2)

    # p(z) is determinated by the overall proportions of biterms in it
    def save_pz(self,pt):
        pz = Pvec(pvec_v=self.nb_z)
        pz.normalize(self.alpha)
        pz.write(pt)

    def save_pw_z(self,pt):
        pw_z = np.ones((self.K,self.W))
        wf = open(pt,'w')
        for k in range(self.K):
            for w in range(self.W):
                pw_z[k][w] = (self.nwz[k][w] + self.beta) / (self.nb_z[k] * 2 + self.W * self.beta)
                wf.write(str(pw_z[k][w]) + ' ')
            wf.write("\n")


