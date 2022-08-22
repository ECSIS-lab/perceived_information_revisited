import os
import numpy as np
from sklearn import preprocessing

def normalize_mat(data):
    for i in range(len(data)):
        data[i] -= data[i].mean()
        data[i] /= data[i].std()
    return data

#def train_data(dataset):
#    data_len = 0
#    train = {}
#    v = np.load(f'./datasets/{dataset}/train_traces.npy').astype(np.float32)
#    train['traces'] = v.reshape(v.shape[0], v.shape[1], 1)
#    data_len = train['traces'].shape[0]
#    train['labels'] = np.load(f'./datasets/{dataset}/train_labels.npy')[:data_len]
#    return train

def test_data(dataset):
    data_len = 0
    test = {}
    v = np.load(f'./datasets/{dataset}/test_traces.npy').astype(np.float32)
    test['traces'] = v.reshape(v.shape[0], v.shape[1], 1)
    data_len = test['traces'].shape[0]

    test['key'] = np.load(f'./datasets/{dataset}/test_key.npy')[:data_len]
    test['pt'] = np.load(f'./datasets/{dataset}/test_pt.npy')[:data_len]
    test['5byte'] = np.load(f'./datasets/AES_TI/test_5byte.npy')[:data_len]
    test['key'] = test['key'][0]
    test['key1'] = np.load(f'./datasets/{dataset}/test_key_byte1.npy')[0]
    test['key5'] = np.load(f'./datasets/{dataset}/test_key_byte5.npy')[0]
    test['pt1'] = np.load(f'./datasets/{dataset}/test_pt_byte1.npy')[:data_len]
    test['pt5'] = np.load(f'./datasets/{dataset}/test_pt_byte5.npy')[:data_len]
    return test
