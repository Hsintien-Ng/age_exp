import os
from dataset.MORPH import MORPH
from model.plain_cnn_asymmetry import PlainCNN_Asym
from model.plain_cnn_atrous import PlainCNN_Atrous
from model.plain_cnn_asymmetry import plain_parameters_func
from train.calculator.age_mae_calculator import AgeMAECalculator
from train.loss.KL_Divergence import KLDivLoss
from train.loss.focal import FocalLoss
from train.predictor.expection_predictor import ExpectionPredictor
from train.predictor.max_predictor import MaxPredictor
from train.config import Config
from train.trainer import Trainer
from utils.logger import Logger
from utils.model import load_pretrained_func
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_alias(model_cls, task):
    assert task == 'Age' or task == 'Exp'
    model = model_cls.__name__
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    alias = '{}_{}_{}'.format(task, model, current_time)
    return alias


def generate_file_msg(sps, loss, predictor, acc_calc):
    sp_descriptor = ''
    for sp in sps:
        sp_descriptor += '{}:{}\n'.format(sp, sps[sp])
    loss_info = '{}:{}\n'.format('loss', loss.get_alias())
    predictor_info = '{}:{}\n'.format('predictor', predictor.get_alias())
    acc_calc_info = '{}:{}\n'.format('acc_calc', acc_calc.get_alias())

    msg = '{}{}{}{}'.format(sp_descriptor, loss_info, predictor_info, acc_calc_info)

    return msg


# dir define

index_dir = os.path.join('/', 'home', 'xintian', 'projects', 'age_exp', 'MORPH_Split')
pretrained_model_dir = os.path.join('/', 'home', 'xintian', 'projects', 'age_exp',
                                    'models', 'Age_PlainCNN_Asym_Oct15_23-21-48', 'epoch_28.pkl')

# super param define
sps = {'epoch_num': 20, 'momentum': 0.9, 'weight_decay': 0.0002,
               # 'learning_rates': [1e-2, 5e-3, 1e-3, 1e-4, 1e-5],
               'learning_rates': [1e-3, 1e-4],
               # 'decay_points': [10, 20, 30, 40],
               'decay_points': [10],
               'batch_size': 64, 'pretrain': True,
               'pretrained_model_dir': pretrained_model_dir,
               'load_function': load_pretrained_func, 'balance': False}
parameters_func = plain_parameters_func

gpu_id = [0]

# trainer component
#model_cls = PlainCNN_Atrous
model_cls = PlainCNN_Asym
loss = KLDivLoss()
predictor = ExpectionPredictor()
calculator = AgeMAECalculator()
print('dataset')
train_dataset = MORPH(data_dir=index_dir, mode='train', balance=sps['balance'])
print('train complete')
valid_dataset = MORPH(data_dir=index_dir, mode='valid', balance=False)
print('valid complete')

def run(model_cls, loss, predictor, acc_calc, train_dataset, valid_dataset, sps):

    # log setting
    alias = generate_alias(model_cls, task='Age')
    msg = generate_file_msg(sps, loss, predictor, acc_calc)
    tb_log_path = os.path.join('runs', alias)
    save_dir = os.path.join('models', alias)
    logger_alias = alias

    config = Config(epoch_num=sps['epoch_num'], momentum=sps['momentum'], weight_decay=sps['weight_decay'],
                    learning_rates=sps['learning_rates'], decay_points=sps['decay_points'],
                    batch_size=sps['batch_size'], parameters_func=parameters_func,
                    tb_log_path=tb_log_path, save_dir=save_dir, pretrain=sps['pretrain'],
                    pretrained_model_dir=sps['pretrained_model_dir'],
                    load_function=sps['load_function'], logger_alias=logger_alias, gpu_id=gpu_id)

    logger = Logger()
    logger.open_file(os.path.join('log'), alias=alias, file_name=alias+'.txt', file_msg=msg)

    trainer = Trainer(model_cls=model_cls, loss=loss, predictor=predictor,  calculator=calculator,
                      train_dataset=train_dataset, val_dataset=valid_dataset, config=config, logger=logger)

    trainer.train()
    logger.close_file(alias)


if __name__ == '__main__':
    run(model_cls, loss, predictor, calculator, train_dataset, valid_dataset, sps)
# loss = FocalLoss(2)
# run(model_cls, loss, predictor, calculator, train_dataset, val_dataset, sps)
# loss = FocalLoss(5)
# run(model_cls, loss, predictor, calculator, train_dataset, val_dataset, sps)
