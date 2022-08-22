import os.path
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from tensorflow.keras import backend as K
from scipy.special import log_softmax
from argparse import ArgumentParser
from joblib import Parallel, delayed
import load_data
from numba import jit, njit

np.set_printoptions(threshold=9999999999)

argparser = ArgumentParser()
argparser.add_argument('model', type=str, default=None, help='Model used to estimate the right key')
argparser.add_argument('-a', '--average', type=int, default=1, help='The number of trials for key guessing')
argparser.add_argument('-n', '--num-traces', type=int, default=2000, help='The number of traces')
argparser.add_argument('-d', '--dataset', type=str, default="AES_TI", help='Dataset')
args = argparser.parse_args()

def model_fit(model, data):
    predictions = log_softmax(model.predict(data['traces'], batch_size=4096), axis=1) #shape=(20000,9)
    return predictions


inv = np.load('./datasets/AES_TI/inv_LUT.npy')

#@njit('u4[:](f4[:,:], u1[:], u1[:], u4)')
def perm(preds, pt, rr, key):
    keys = np.arange(256).astype(np.uint8)
    tmp = np.zeros((256)).astype(np.float32)
    rank = np.zeros((len(preds))).astype(np.uint32)
    for i in range(len(pt)):
        indices = inv[keys^pt[i]]^rr[i]
        tmp += -preds[i][indices]
        rank[i] = np.sum(tmp <= tmp[key])
    return rank

def loop(i, preds, r, plaintext, key):
    # bootstrapping
    indices = np.random.choice(np.arange(preds.shape[0]), size=args.num_traces)
    X, pt, rr = preds[indices].astype(np.float32), plaintext[indices].astype(np.uint8), r[indices].astype(np.uint8)

    return perm(X, pt, rr, key)

def add_prefix(name, pre):
    n = name.split('.')[:-1]
    p = name.split('.')[-1]
    return '.'.join(n) + "_" + pre + "." + p

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dataset = args.dataset
    model_file = args.model
    num_traces = args.num_traces

    test_data = load_data.test_data(dataset)
    plaintext = test_data["pt"]
    key = test_data["key"]
    r = test_data["5byte"]

    model = load_model(model_file, compile=False)
    preds = model_fit(model, test_data)

    ge = np.zeros(num_traces) # guessing entropy
    sr = np.zeros(num_traces) # success rate

    res = Parallel(n_jobs=1, verbose=3)([delayed(loop)(i, preds, r, plaintext, key) for i in range(args.average)])

    outs = np.array(res)
    for i in range(args.average):
        ge += outs[i]
        sr += (outs[i] == 1)
    ge /= args.average
    sr /= args.average


    # plot
    plt.style.use('ggplot')
    plt.rcParams["font.size"] = 12
    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.plot(ge+1, label="Actual value")
    plt.ylim([0, 255])
    plt.xlabel("Number of traces")
    plt.ylabel("Guessing entropy")

    plt.subplot(1, 2, 2)
    plt.plot(sr, label="Actual value")
    plt.ylim([-0.1, 1.1])
    plt.xlabel("Number of traces")
    plt.ylabel("Success rate")
    plt.tight_layout()
    plt.show()

