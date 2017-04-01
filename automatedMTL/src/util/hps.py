from os.path import expanduser
import sys
import numpy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","model")))
from mcrnn_model_gen2 import model

#does hyperparameter search over some set of hyperparams.


LR = [0.001, 0.0005, 0.0001] #3
LR_MOD = [0.0, 0.1, 0.5, 1.0] #4
N_EPOCHS = 30 # 30
N_EXPERIMENTS = 5 # 5

#3*4*30*5/60=30 hrs.

for lr in LR:
	for lr_mod in LR_MOD:
		runExperiment(lr,lr_mod,N_EPOCHS,N_EXPERIMENTS)


def runExperiment(lr, lr_mod,n_epochs,n_experiments):
	M= model()

	print M.lr
	M.lr = lr
	print M.lr

    print M.lr_mod
	M.lr_mod = lr_mod
	print M.lr_mod

	maxAccList = [];
	for i in range(n_experiments):
		accuracyVec = run_code()#INSERT CODE TO run for n epochs
		maxAcc = numpy.max(accuracyVec)
		maxAccList.append(maxAcc)
	expVal = numpy.mean(average)
	print "lr = " + str(lr) + " lr_mod = " + str(lr_mod) + " avg_acc = " + str(expVal)



