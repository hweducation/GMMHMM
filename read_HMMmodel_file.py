import pickle
from hmmlearn.hmm import *
rq =GMMHMM()
with open("1.pkl", 'rb') as file:
    rq = pickle.loads(file.read())
