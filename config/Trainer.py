# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
import logging


class Trainer(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 valid_data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None):

        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))

        self.lib = ctypes.cdll.LoadLibrary(base_file)

        self.lib.validHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.validTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.valid_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getValidLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getValidLinkMR.argtypes = [ctypes.c_int64]

        self.lib.getValidLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getValidLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getValidLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getValidLinkMRR.restype = ctypes.c_float
        self.lib.getValidLinkMR.restype = ctypes.c_float

        self.lib.getValidLinkHit10.restype = ctypes.c_float
        self.lib.getValidLinkHit3.restype = ctypes.c_float
        self.lib.getValidLinkHit1.restype = ctypes.c_float

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

    def set_logger(self, data_name, save_path):
        directory = save_path + '/' + data_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        log_file = os.path.join(directory, 'train.log')
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def valid_one_step(self, data):
        return self.model.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def run_valid_link_prediction(self, type_constrain=True):
        self.lib.initValid()
        self.valid_data_loader.set_sampling_mode('link')

        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0

        validation_range = tqdm(self.valid_data_loader)

        for index, [data_head, data_tail] in enumerate(validation_range):
            score = self.valid_one_step(data_head)
            self.lib.validHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.valid_one_step(data_tail)
            self.lib.validTail(score.__array_interface__["data"][0], index, type_constrain)

        self.lib.valid_link_prediction(type_constrain)

        mrr = self.lib.getValidLinkMRR(type_constrain)
        mr = self.lib.getValidLinkMR(type_constrain)

        hit10 = self.lib.getValidLinkHit10(type_constrain)
        hit3 = self.lib.getValidLinkHit3(type_constrain)
        hit1 = self.lib.getValidLinkHit1(type_constrain)
        print(hit10)
        return mrr, mr, hit10, hit3, hit1

    def run(self):
        # Check use GPU or CPU
        if self.use_gpu:
            self.model.cuda()

        # optimizer method choose
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")

        #training_range = tqdm(range(self.train_times))
        for epoch in range(self.train_times):
            res = 0.0
            for data in self.data_loader:
                loss = self.train_one_step(data)
                res += loss
            #training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
            logging.info("Epoch %d | loss: %f" % (epoch, res))

            if epoch % 10 == 0:
                self.model.eval()
                valid_mrr, valid_mr, valid_hit10, valid_hit3, valid_hit1 = self.run_valid_link_prediction(self.valid_data_loader)
                logging.info("Valid Epoch %d | MRR: %f Hits@10: %f Hits@3: %f Hits@1: %f" % (epoch, valid_mrr, valid_hit10, valid_hit3, valid_hit1))

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                #print("Epoch %d has finished, saving..." % (epoch))
                logging.info("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
