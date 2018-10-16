class Config:
    def __init__(self, epoch_num, momentum, weight_decay, learning_rates,
                 decay_points, batch_size, parameters_func, tb_log_path,
                 save_dir, pretrain, pretrained_model_dir=None,
                 load_function=None, logger_alias=None, gpu_id=None):
        """
        :param epoch_num: stop training when trained epoch_num epochs
        :param momentum:
        :param weight_decay:
        :param learning_rates: a list of lrs. training lr will change to the
                next one in this list when meet the occasion implied in
                decay_points
        :param decay_points: a list of int, indicates the time(epoch number
                the model has been trained) that lr may need to decay.
        :param batch_size:
        :param parameters_func: function provied parameters of model that need to be
                trained. Note That if you want to apply different lr to different layers,
                you may remark the detail in this param.
                this function take model object and learning rate as input.
        :param tb_log_path: path to log tensorbord event file
        :param save_dir: dir to save trained model
        :param pretrain: whether use pretrained model or not
        :param pretrained_model_dir: where to load the pretrained model
        :param load_function: a function that take pretrained model's path
                and an instance of model as input, load the pretrained param
                into that model.
        :param logger_alias: alias of logger
        :param gpu_id: a list of gpu ids. if None, use all gpu, else use the
                given ones in the list.
        """
        self.epoch_num = epoch_num
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.learning_rates = learning_rates
        self.decay_points = decay_points
        self.batch_size = batch_size
        self.parameters_func = parameters_func
        self.tb_log_path = tb_log_path
        self.save_dir = save_dir
        self.pretrain = pretrain
        self.pretrained_model_dir = pretrained_model_dir
        self.load_function = load_function
        self.logger_alias = logger_alias
        self.gpu_id = gpu_id
