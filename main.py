#!/usr/bin/env python

from __future__ import division
import numpy as np
from ggnn import GGNN
from preprocess import Preprocess
from utils import Data
import pickle
import datetime


#####preprocess data 

pre_process = Preprocess()
pre_process.start()

dataset = 'sample'
method = 'ggnn'
epoch = 30
batchSize = 50
hiddenSize = 100
l2=1e-5
lr = 0.001
step = 1
nonhybrid = 'store_true'
lr_dc = 0.1
lr_dc_step = 3
n_node = 310
best_result = [0, 0]
best_epoch = [0, 0]

train_data = pickle.load(open(dataset + '/train.txt', 'rb'))
test_data = pickle.load(open(dataset + '/test.txt', 'rb'))

train_data = Data(train_data, sub_graph=True, method=method, shuffle=True)
test_data = Data(test_data, sub_graph=True, method=method, shuffle=False)
model = GGNN(hidden_size=hiddenSize, out_size=hiddenSize, batch_size=batchSize, n_node=n_node, lr=lr, l2=l2, 
                step=step, decay=lr_dc_step * len(train_data.inputs) / batchSize, lr_dc=lr_dc, nonhybrid=nonhybrid)

for epoch in range(epoch):
    print('epoch: ', epoch, '===========================================')
    slices = train_data.generate_batch(model.batch_size)
    fetches = [model.opt, model.loss_train, model.global_step]
    print('start training: ', datetime.datetime.now())
    loss_ = []
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
        _, loss, _ = model.run(fetches, targets, item, adj_in, adj_out, alias,  mask)
        loss_.append(loss)
    loss = np.mean(loss_)
    slices = test_data.generate_batch(model.batch_size)
    print('start predicting: ', datetime.datetime.now())
    hit, mrr, test_loss_ = [], [],[]
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)
        scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, adj_in, adj_out, alias,  mask)
        test_loss_.append(test_loss)
        index = np.argsort(scores, 1)[:, -20:]
        for score, target in zip(index, targets):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (20-np.where(score == target - 1)[0][0]))
    hit = np.mean(hit)*100
    mrr = np.mean(mrr)*100
    test_loss = np.mean(test_loss_)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1]=epoch
    
    print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'%
          (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
