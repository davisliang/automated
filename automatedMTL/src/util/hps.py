from os.path import expanduser
import sys
import numpy
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","model")))
from mcrnn_model_gen2 import model
from train_gen import trainModel as TM
#does hyperparameter search over some set of hyperparams.


LR = [0.01,0.001,0.0001]
LR_MOD = [1.0] #4
N_EPOCHS = [50] # 30
N_EXPERIMENTS = [10] # 5

#3*4*30*5/60=30 hrs.


def runExperiment(lr, lr_mod,n_epoch,n_experiments,f1):
    M= model()

    print M.is_multi_task
    if lr_mod == 0.0:
        M.is_multi_task = False
    else:
        M.is_multi_task = True
    print M.is_multi_task

    print M.lr
    M.lr = lr
    print M.lr

    print M.lr_mod
    M.lr_mod = lr_mod
    print M.lr_mod

    print M.n_epoch
    M.n_epoch = n_epoch
    print M.n_epoch

    maxAccList = [];
    for i in range(n_experiments):
        accuracyVec = TM(M)#INSERT CODE TO run for n epochs
        maxAcc = numpy.max(accuracyVec)
        maxAccList.append(maxAcc)
    expVal = numpy.mean(maxAccList)
    string_result = "lr = " + str(lr) + " lr_mod = "+ "self-annealing" + " avg_acc = " + str(expVal)+'\n'
    f1.write(string_result)
    f1.flush()
    print string_result



f1 = open(expanduser('~/tweetnet/logs/hps_log_mrnn_bidir.log'),'w+') 
for lr in LR:
    for lr_mod in LR_MOD:
        for n_epoch in N_EPOCHS:
            for n_experiments in N_EXPERIMENTS:
                runExperiment(lr,lr_mod,n_epoch,n_experiments,f1)      
f1.close()
