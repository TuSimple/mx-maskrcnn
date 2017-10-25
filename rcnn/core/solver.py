import mxnet as mx
import logging


class Solver(object):
    def __init__(self, symbol, data_names, label_names,
                 logger=logging, context=mx.cpu(), work_load_list=None,
                 max_data_shapes=None, max_label_shapes=None, fixed_param_prefix=None):
        self._symbol = symbol
        self._data_names = data_names
        self._label_names = label_names
        self._logger = logger
        self._context = context
        self._work_load_list = work_load_list

        self._curr_module = None
        self._max_data_shapes = [] if max_data_shapes is None else max_data_shapes
        self._max_label_shapes = [] if max_label_shapes is None else max_label_shapes
        self._mutable_data_shape = len(self._max_data_shapes + self._max_label_shapes) > 0
        self._fixed_param_prefix = [] if fixed_param_prefix is None else fixed_param_prefix

        fixed_param_names = list()
        for name in self._symbol.list_arguments():
            for prefix in self._fixed_param_prefix:
                if prefix in name:
                    fixed_param_names.append(name)
        self._fixed_param_names = fixed_param_names

    def check_params(self, arg_params, aux_params):
        arg_names = self._symbol.list_arguments()
        arg_params = {k: v for k, v in arg_params.items() if k in arg_names}
        aux_names = self._symbol.list_auxiliary_states()
        aux_params = {k: v for k, v in aux_params.items() if k in aux_names}
        return arg_names, aux_names, arg_params, aux_params

    def fit(self, train_data, eval_metric=None,
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=None,
            arg_params=None, aux_params=None,
            begin_epoch=None, num_epoch=None):
        arg_names, aux_names, arg_params, aux_params = self.check_params(arg_params, aux_params)
        data_names = dict(train_data.provide_data + train_data.provide_label).keys()
        param_names = [name for name in arg_names if name not in self._fixed_param_names + data_names]
        optimizer = mx.optimizer.create(optimizer, **optimizer_params)
        (kvstore, update_on_kvstore) = mx.model._create_kvstore(kvstore, len(self._context), arg_params)

        mx.model._train_multi_device(self._symbol, self._context, arg_names, param_names, aux_names,
                                     arg_params, aux_params, begin_epoch, num_epoch,
                                     epoch_size=None, optimizer=optimizer,
                                     kvstore=kvstore, update_on_kvstore=update_on_kvstore,
                                     train_data=train_data, eval_data=None, eval_metric=eval_metric,
                                     epoch_end_callback=epoch_end_callback, batch_end_callback=batch_end_callback,
                                     logger=self._logger, work_load_list=self._work_load_list, monitor=None,
                                     mutable_data_shape=self._mutable_data_shape, max_data_shape=self._max_data_shapes,
                                     max_label_shape=self._max_label_shapes)
