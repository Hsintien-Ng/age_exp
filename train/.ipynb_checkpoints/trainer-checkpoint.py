from config import Config
from utils.logger import Logger
from loss.loss import Loss
from predictor.predictor import Predictor
from acc_calc.acc_calculator import AccuracyCalculator
import torch as t
import torch.nn as nn
from torch.utils import data
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import time
import os


class Trainer:
    def __init__(self, model_cls, loss, predictor, acc_calc, train_dataset, val_dataset,
                 config, logger):
        """

        :param model_cls: an class whose super class is torch.nn.Module
        :param loss: an instance of train.loss.Loss
        :param predictor: an instance of train.predictor.Predictor
        :param acc_calc: an instance of train.acc_calc.AccuracyCalculator
        :param train_dataset: dataset for training
        :param val_dataset: dataset for validation
        :param config: instance of Config
        :param logger: instance of utils.Logger
        """
        self.Model = model_cls
        self.loss = loss
        self.predictor = predictor
        self.acc_calc = acc_calc
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.logger = logger
        assert issubclass(model_cls, nn.Module)
        assert isinstance(logger, Logger)
        assert isinstance(config, Config)
        assert isinstance(loss, Loss)
        assert isinstance(predictor, Predictor)
        assert isinstance(acc_calc, AccuracyCalculator)

    def train(self):
        """
        training according to self.config
        Note that it will do validation and save model every epoch.
        :return:
        """
        net = self.Model().cuda()
        net = nn.DataParallel(net, self.config.gpu_id)
        if self.config.pretrain:
            net = self.config.load_function(self.config.pretrained_model_dir,
                                            net)
        print 'loading data...'
        train_loader = data.DataLoader(self.train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=True)
        val_loader = data.DataLoader(self.val_dataset,
                                     batch_size=1,
                                     shuffle=False)
        print 'Data loaded!'

        stage = 0
        global_step = 0
        max_validation_acc = 0
        best_epoch = -1
        tb_writer = SummaryWriter(log_dir=self.config.tb_log_path)
        for e in range(self.config.epoch_num):
            if e in self.config.decay_points:
                stage += 1
            lr = self.config.learning_rates[stage]
            optimizer = optim.SGD(params=self.config.parameters_func(net, lr),
                                  lr=lr, momentum=self.config.momentum,
                                  weight_decay=self.config.weight_decay)

            # train for an epoch
            net.train()
            info = '------EPOCH %i START------' % e
            self.logger.log_to(info, self.config.logger_alias)
            global_step = self.train_an_epoch(net, optimizer, train_loader, tb_writer, e, global_step)
            info = '------EPOCH %i COMPLETE------' % e
            self.logger.log_to(info, self.config.logger_alias)
            self.logger.flush(self.config.logger_alias)

            # do validation
            net.eval()
            val_acc = self.validate(net, val_loader, e, tb_writer, global_step)
            if max_validation_acc < val_acc:
                max_validation_acc = val_acc
                best_epoch = e

            #save model
            self.save_model(net, e)
        # training complete
        info = '==========Training Complete=========='
        self.logger.log_to(info, self.config.logger_alias)
        info = 'Best accuracy is %.3f, at epoch %i' % (max_validation_acc, best_epoch)
        self.logger.log_to(info, self.config.logger_alias)

    def train_an_epoch(self, net, optimizer, train_loader, tb_writer, epoch_num, global_step):
        """
        train for an epoch.
        :param net:
        :param optimizer:
        :param train_loader:
        :param tb_writer: writer that record info for tensorboard
        :param epoch_num:
        :param global_step:
        :return:
        """
        total_acc = 0
        total_loss = 0
        time_start = time.time()
        for i, (img, label) in enumerate(train_loader):
            # make data gpu variable
            img = img.cuda()
            img = Variable(img)
            label = label.cuda()
            label = Variable(label)

            # clear grad each iterator
            optimizer.zero_grad()

            # forward->calculate loss->backward calculate grad->apply grad
            output = net(img)
            loss = self.loss.calculate_loss(output, label)
            loss.backward()
            optimizer.step()

            # accuracy calculating
            prediction = self.predictor.predict(output)
            batch_acc = self.acc_calc.record(prediction, label)

            # for logging
            total_loss += loss.item()
            total_acc += batch_acc

            # tensorboard
            tb_writer.add_scalar('accuracy', batch_acc, global_step)
            tb_writer.add_scalar('loss', loss.item(), global_step)

            # log
            if i % 100 == 0 and i > 0:
                time_end = time.time()
                print 'time cost for 100 step: {}'.format(time_end - time_start)
                time_start = time_end
                ave_acc = total_acc/100
                ave_loss = total_loss/100
                info = '[epoch %i, step %i]: accuracy=%.3f, loss=%.3f' % (epoch_num, i, ave_acc, ave_loss)
                self.logger.log_to(info, self.config.logger_alias)
                total_loss = 0
                total_acc = 0
                # flush tb_writer every 100 step
                tb_writer.export_scalars_to_json("./temp.json")

            # count steps number
            global_step += 1
        return global_step

    def validate(self, net, val_loader, epoch_num, tb_writer, global_step):
        """
        do validation
        :param net:
        :param val_loader:
        :param epoch_num:
        :param tb_writer:
        :param global_step:
        :return: final accuracy of this validation
        """
        total_acc = 0
        count = 0
        for i, (img, label) in enumerate(val_loader):
            # make data gpu variable
            img = img.cuda()
            img = Variable(img)
            label = label.cuda()
            label = Variable(label)

            # forward->calculate loss->backward calculate grad->apply grad
            output = net(img)

            # accuracy calculating
            prediction = self.predictor.predict(output)
            total_acc += self.acc_calc.record(prediction, label)

            count += 1

        # log
        ave_acc = total_acc/count
        info = '[epoch %i\'s validation result]: accuracy=%.3f' % (epoch_num, ave_acc)
        self.logger.log_to(info, self.config.logger_alias)

        # tensorboard
        tb_writer.add_scalar('val_accuracy', ave_acc, global_step)
        tb_writer.export_scalars_to_json("./temp.json")

        return ave_acc

    def save_model(self, net, epoch_num):
        model_path = os.path.join(self.config.save_dir, 'epoch_%i.pkl' % epoch_num)
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        t.save(net.state_dict(), model_path)
