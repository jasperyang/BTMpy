# -*- coding: utf-8 -*-
import time
from Model import *
import sys
import indexDocs
import topicDisplay
import os

def usage() :
    print("Training Usage: \
    btm est <K> <W> <alpha> <beta> <n_iter> <save_step> <docs_pt> <model_dir>\n\
    \tK  int, number of topics, like 20\n \
    \tW  int, size of vocabulary\n \
    \talpha   double, Pymmetric Dirichlet prior of P(z), like 1.0\n \
    \tbeta    double, Pymmetric Dirichlet prior of P(w|z), like 0.01\n \
    \tn_iter  int, number of iterations of Gibbs sampling\n \
    \tsave_step   int, steps to save the results\n \
    \tdocs_pt     string, path of training docs\n \
    \tmodel_dir   string, output directory")


def BTM(argvs):
    if(len(argvs)<4):
        usage()
    else:
        if (argvs[0] == "est"):
            K = argvs[1]
            W = argvs[2]
            alpha = argvs[3]
            beta = argvs[4]
            n_iter = argvs[5]
            save_step = argvs[6]
            docs_pt = argvs[7]
            dir = argvs[8]
            print("Run BTM, K="+str(K)+", W="+str(W)+", alpha="+str(alpha)+", beta="+str(beta)+", n_iter="+str(n_iter)+", save_step="+str(save_step)+"=====")
            clock_start = time.clock()
            model = Model(K, W, alpha, beta, n_iter, save_step)
            model.run(docs_pt,dir)
            clock_end = time.clock()
            print("procedure time : "+str(clock_end-clock_start))
        else:
            usage()

if __name__ ==  "__main__":
    mode = "est"
    K = 15
    W = None
    alpha = 0.5
    beta = 0.5
    n_iter = 1000
    save_step = 100
    dir = "../output/"
    input_dir = "../sample-data/"
    model_dir = dir + "model/"
    voca_pt = dir + "voca.txt"
    dwid_pt = dir + "doc_wids.txt"
    doc_pt = input_dir + "trndocs.dat"

    print("=============== Index Docs =============")
    W = indexDocs.run_indexDocs(['indexDocs',doc_pt,dwid_pt,voca_pt])


    print("W : "+str(W))

    argvs = []
    argvs.append(mode)
    argvs.append(K)
    argvs.append(W)
    argvs.append(alpha)
    argvs.append(beta)
    argvs.append(n_iter)
    argvs.append(save_step)
    argvs.append(dwid_pt)
    argvs.append(model_dir)


    print("=============== Topic Learning =============")
    BTM(argvs)

    print("================ Topic Display =============")
    topicDisplay.run_topicDicplay(['topicDisplay',model_dir,K,voca_pt])