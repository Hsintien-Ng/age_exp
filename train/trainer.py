from train.config import Config
from utils.logger import Logger
from utils.figure import scatter_plot
from train.loss.loss import Loss
from train.predictor.predictor import Predictor
from train.calculator.calculator import Calculator
import torch as t
import torch.nn as nn
from torch.utils import data
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import time
import os


class Trainer:
    def __init__(self, model_cls, loss, predictor, calculator, train_dataset, val_dataset,
                 config, logger):
        """

        :param model_cls: an class whose super class is torch.nn.Module
        :param loss: an instance of train.loss.Loss
        :param predictor: an instance of train.predictor.Predictor
        :param calculator: an instance of train.calculator.Calculator
        :param train_dataset: dataset for training
        :param val_dataset: dataset for validation
        :param config: instance of Config
        :param logger: instance of utils.Logger
        """
        self.Model = model_cls
        self.loss = loss
        self.predictor = predictor
        self.calculator = calculator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.logger = logger
        assert issubclass(model_cls, nn.Module)
        assert isinstance(logger, Logger)
        assert isinstance(config, Config)
        assert isinstance(loss, Loss)
        assert isinstance(predictor, Predictor)
        assert isinstance(calculator, Calculator)
        assert calculator.get_alias() == 'Exp' or calculator.get_alias() == 'Age'

    def train(self):
        """
        training according to self.config
        Note that it will do validation and save model every epoch.
        :return:
        """
        net = self.Model().cuda()
        net = nn.DataParallel(net, self.config.gpu_id)
        if self.config.pretrain:
            print('loading pretrained model...')
            net = self.config.load_function(self.config.pretrained_model_dir,
                                            net)
        print('loading data...')
        train_loader = data.DataLoader(self.train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=True)
        val_loader = data.DataLoader(self.val_dataset,
                                     batch_size=1,
                                     shuffle=False)
        print('Data loaded!')

        stage = 0
        global_step = 0
        min_validation_mae = 1e6
        max_validation_acc = 0
        best_epoch = -1
        tb_writer = SummaryWriter(log_dir=self.config.tb_log_path)
        # tb_writer = FileWriter(logdir=self.config.tb_log_path)
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
            val_cal = self.validate(net, val_loader, e, tb_writer, global_step)
            if self.calculator.get_alias() == 'Exp':
                if max_validation_acc < val_cal:
                    max_validation_acc = val_cal
                    best_epoch = e
            elif self.calculator.get_alias() == 'Age':
                if min_validation_mae > val_cal:
                    min_validation_mae = val_cal
                    best_epoch = e


            #save model
            self.save_model(net, e)
        # training complete
        info = '==========Training Complete=========='
        self.logger.log_to(info, self.config.logger_alias)
        if self.calculator.get_alias() == 'Exp':
            info = 'Best accuracy is %.3f, at epoch %i' % (max_validation_acc, best_epoch)
            self.logger.log_to(info, self.config.logger_alias)
        elif self.calculator.get_alias() == 'Age':
            info = 'Best mean absolute error is %.5f, at epoch %i' % (min_validation_mae, best_epoch)
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
        total_cal = 0
        total_loss = 0
        time_start = time.time()
        for i, (img, label, norm_dtb_label) in enumerate(train_loader):
            # make data gpu variable
            img = img.cuda()
            img = Variable(img)
            img = img.float()
            label = label.cuda()
            label = Variable(label)
            label = label.float()
            norm_dtb_label = norm_dtb_label.cuda()
            norm_dtb_label = Variable(norm_dtb_label)
            norm_dtb_label = norm_dtb_label.float()

            # clear grad each iterator
            optimizer.zero_grad()

            # forward->calculate loss->backward calculate grad->apply grad
            output = net(img)
            if isinstance(output, list) or isinstance(output, tuple):
                loss = 0
                for o in output:
                    # loss += self.loss.calculate_loss(o, label)
                    loss += self.loss.calculate_loss(o, norm_dtb_label)
                output = output[-1]
            else:
                # loss = self.loss.calculate_loss(output, label)
                loss = self.loss.calculate_loss(output, norm_dtb_label)
            loss.backward()
            optimizer.step()

            # accuracy calculating
            prediction = self.predictor.predict(output)
            batch_cal = self.calculator.record(prediction, label)
            # print('image_shape:', img.shape)
            # print('prediction:', prediction)
            # print('prediction_shape:', prediction.shape)
            # print('label:', label)
            # print('label_shape:', label.shape)

            # for logging
            total_loss += loss.item()
            total_cal += batch_cal

            # tensorboard
            tb_writer.add_image('image', img.cpu(), global_step)
            tb_writer.add_figure('predict',
                                 scatter_plot(prediction.detach().cpu(), self.calculator.get_alias()), global_step)
            tb_writer.add_figure('groudTruth',
                                 scatter_plot(label.cpu(), self.calculator.get_alias()), global_step)
            if self.calculator.get_alias() == 'Exp':
                tb_writer.add_scalar('accuracy', batch_cal.cpu(), global_step)
            elif self.calculator.get_alias() == 'Age':
                tb_writer.add_scalar('MAE', batch_cal.cpu(), global_step)
            tb_writer.add_scalar('loss', loss.item(), global_step)

            # log
            num = 50
            if i % num == 0 and i > 0:
                time_end = time.time()
                print('time cost for 100 step: {}'.format(time_end - time_start))
                time_start = time_end
                ave_cal = total_cal / num
                ave_loss = total_loss / num
                if self.calculator.get_alias() == 'Exp':
                    info = '[epoch %i, step %i]: accuracy=%.3f, loss=%.3f' % (epoch_num, i, ave_cal, ave_loss)
                else:
                    info = '[epoch %i, step %i]: MAE=%.3f, loss=%.3f' % (epoch_num, i, ave_cal, ave_loss)
                self.logger.log_to(info, self.config.logger_alias)
                total_loss = 0
                total_cal = 0
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
        total_cal = 0
        count = 0
        for i, (img, label, _) in enumerate(val_loader):
            # make data gpu variable
            img = img.cuda()
            label = label.cuda()
            with t.no_grad():
                img = Variable(img)
                label = Variable(label)

                # forward->calculate loss->backward calculate grad->apply grad
                output = net(img)
                if isinstance(output, list) or isinstance(output, tuple):
                    output = output[-1]

                # accuracy calculating
                prediction = self.predictor.predict(output)
                total_cal += self.calculator.record(prediction, label)

                count += 1
        total_cal = total_cal.cpu()

        # log
        ave_cal = total_cal / count
        if self.calculator.get_alias() == 'Exp':
            info = '[epoch %i\'s validation result]: accuracy=%.3f' % (epoch_num, ave_cal)
        else:
            info = '[epoch %i\'s validation result]: MAE=%.3f' % (epoch_num, ave_cal)
        self.logger.log_to(info, self.config.logger_alias)

        # tensorboard
        if self.calculator.get_alias() == 'Exp':
            tb_writer.add_scalar('val_accuracy', ave_cal, global_step)
        else:
            tb_writer.add_scalar('val_mae', ave_cal, global_step)
        tb_writer.export_scalars_to_json("./temp.json")

        return ave_cal

    def save_model(self, net, epoch_num):
        model_path = os.path.join(self.config.save_dir, 'epoch_%i.pkl' % epoch_num)
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        t.save(net.state_dict(), model_path)