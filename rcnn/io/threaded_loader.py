import mxnet as mx
import numpy as np
from Queue import Queue
from threading import Thread

from ..config import config
from ..io.rpn import get_rpn_batch, assign_anchor_fpn
from ..io.rcnn import get_fpn_maskrcnn_batch


class ThreadedMaskROIIter(mx.io.DataIter):
    def __init__(self, roidb, batch_size=2, shuffle=False, short=None, long=None, num_thread=8):
        """
        This iter will provide roi and mask data to Mask-RCNN with FPN backbone
        :param roidb: list of dict, must be preprocessed
        :param batch_size: int
        :param shuffle: bool
        :param short: int, shorter edge in pixel
        :param long: int, longer edge in pixel
        :param num_thread: int, number of workers for pre-processing
        :return: ROIIter
        """
        super(ThreadedMaskROIIter, self).__init__()
        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # image_pad size
        self.short = short
        self.long = long

        # decide data and label names (only for training)
        self.data_name = ['data']
        for s in config.RCNN_FEAT_STRIDE:
            self.data_name.append('rois_stride%s' % s)
            self.data_name.append('rois_mask_stride%s' % s)
        self.label_name = ['label', 'bbox_target', 'bbox_weight', 'mask_target', 'mask_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # multi-thread primitive
        self.index_queue = Queue()
        self.result_queue = Queue(maxsize=4*num_thread)
        self.workers = None

        self._thread_start(num_thread)
        # get first batch to fill in provide_data and provide_label
        self.get_batch()
        self.reset()

    def _insert_queue(self):
        for i in range(0, len(self.index), self.batch_size)[:-1]:
            self.index_queue.put(self.index[i:i+self.batch_size])

    def _thread_start(self, num_thread):
        self.workers = [Thread(target=self.worker)
                        for _ in range(num_thread)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)
        self._insert_queue()

    def iter_next(self):
        return self.cur + self.batch_size < self.size

    def next(self):
        TEST_NETWORK_SPEED = False

        if TEST_NETWORK_SPEED and self.data is not None and self.label is not None:
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)

        if self.iter_next():
            self.cur += self.batch_size
            data_dict, label_dict = self.result_queue.get()
            self.data = [data_dict[name] for name in self.data_name]
            self.label = [label_dict[name] for name in self.label_name]

            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        data_dict, label_dict = get_fpn_maskrcnn_batch(roidb, (len(roidb), 3, self.short, self.long))

        self.data = [mx.nd.array(data_dict[name]) for name in self.data_name]
        self.label = [mx.nd.array(label_dict[name]) for name in self.label_name]

    def worker(self):
        while True:
            indexes = self.index_queue.get()
            if indexes is None:
                return

            roidb = [self.roidb[idx] for idx in indexes]
            data_dict, label_dict = get_fpn_maskrcnn_batch(roidb, (len(roidb), 3, self.short, self.long))

            for k, v in data_dict.items():
                data_dict[k] = mx.nd.array(v, ctx=mx.cpu())
            for k, v in label_dict.items():
                label_dict[k] = mx.nd.array(v, ctx=mx.cpu())

            result = [data_dict, label_dict]

            self.result_queue.put(result)


    def __init__(self, roidb, batch_size=2, shuffle=False, short=None, long=None, num_thread=8):
        """
        This iter will provide roi and mask data to Mask-RCNN with FPN backbone
        :param roidb: list of dict, must be preprocessed
        :param batch_size: int
        :param shuffle: bool
        :param short: int, shorter edge in pixel
        :param long: int, longer edge in pixel
        :param num_thread: int, number of workers for pre-processing
        :return: ROIIter
        """
        super(ThreadedAspectMaskROIIter, self).__init__()
        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # image_pad size
        self.short = short
        self.long = long

        # decide data and label names (only for training)
        self.data_name = ['data']
        for s in config.RCNN_FEAT_STRIDE:
            self.data_name.append('rois_stride%s' % s)
            self.data_name.append('rois_mask_stride%s' % s)
        self.label_name = ['label', 'bbox_target', 'bbox_weight', 'mask_target', 'mask_weight']

        # status variable for synchronization between get_data and get_label
        self.horizontal_cur = 0
        self.vertical_cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # multi-thread primitive
        self.horizontal_indexes = list()
        self.horizontal_index_queue = Queue()
        self.horizontal_result_queue = Queue(maxsize=num_thread)
        self.horizontal_works = None
        self.vertical_indexes = list()
        self.vertical_index_queue = Queue()
        self.vertical_result_queue = Queue(maxsize=num_thread)
        self.vertical_works = None
        self._group_indexes()

        self._thread_start(num_thread)
        # get first batch to fill in provide_data and provide_label
        self.get_batch()
        self.reset()

    def _group_indexes(self):
        for i, roirec in enumerate(self.roidb):
            w = roirec["width"]
            h = roirec["height"]
            if w > h:
                self.horizontal_indexes.append(i)
            else:
                self.vertical_indexes.append(i)

    def _insert_queue(self):
        for i in range(0, len(self.horizontal_indexes), self.batch_size)[:-1]:
            self.horizontal_index_queue.put(self.horizontal_indexes[i:i + self.batch_size])
        for i in range(0, len(self.vertical_indexes), self.batch_size)[:-1]:
            self.vertical_index_queue.put(self.vertical_indexes[i:i + self.batch_size])

    def _thread_start(self, num_thread):
        self.vertical_workers = [Thread(target=self.worker,
                                        args=[self.vertical_index_queue,
                                              self.vertical_result_queue,
                                              (self.batch_size, 3, self.long, self.short)])
                                 for _ in range(num_thread)]
        for worker in self.vertical_workers:
            worker.daemon = True
            worker.start()

        self.horizontal_workers = [Thread(target=self.worker,
                                          args=[self.horizontal_index_queue,
                                                self.horizontal_result_queue,
                                                (self.batch_size, 3, self.short, self.long)])
                                   for _ in range(num_thread)]
        for worker in self.horizontal_workers:
            worker.daemon = True
            worker.start()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.vertical_cur = 0
        self.horizontal_cur = 0
        if self.shuffle:
            np.random.shuffle(self.vertical_indexes)
            np.random.shuffle(self.horizontal_indexes)
        self._insert_queue()

    def iter_next(self):
        return self.horizontal_cur + self.batch_size < len(self.horizontal_indexes) or \
               self.vertical_cur + self.batch_size < len(self.vertical_indexes)

    def check_aspect_queues(self):
        return self.horizontal_cur + self.batch_size < len(self.horizontal_indexes), \
               self.vertical_cur + self.batch_size < len(self.vertical_indexes)

    def load_vertical_batch(self):
        self.vertical_cur += self.batch_size
        result = self.vertical_result_queue.get()
        return result

    def load_horizontal_batch(self):
        self.horizontal_cur += self.batch_size
        result = self.horizontal_result_queue.get()
        return result

    def next(self):
        TEST_NETWORK_SPEED = False

        if TEST_NETWORK_SPEED and self.data is not None and self.label is not None:
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   provide_data=self.provide_data, provide_label=self.provide_label)

        if self.iter_next():
            h, v = self.check_aspect_queues()
            if h:
                data_dict, label_dict = self.load_horizontal_batch()
            else:
                data_dict, label_dict = self.load_vertical_batch()

            self.data = [data_dict[name] for name in self.data_name]
            self.label = [label_dict[name] for name in self.label_name]
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def get_batch(self):
        # slice roidb
        cur_from = self.horizontal_cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.horizontal_indexes[i]] for i in range(cur_from, cur_to)]

        data_dict, label_dict = get_fpn_maskrcnn_batch(roidb, (len(roidb), 3, self.short, self.long))

        self.data = [mx.nd.array(data_dict[name]) for name in self.data_name]
        self.label = [mx.nd.array(label_dict[name]) for name in self.label_name]

    def worker(self, index_queue, result_queue, input_shape):
        while True:
            indexes = index_queue.get()
            if indexes is None:
                return

            roidb = [self.roidb[idx] for idx in indexes]
            data_dict, label_dict = get_fpn_maskrcnn_batch(roidb, input_shape)

            for k, v in data_dict.items():
                data_dict[k] = mx.nd.array(v, ctx=mx.cpu())
            for k, v in label_dict.items():
                label_dict[k] = mx.nd.array(v, ctx=mx.cpu())

            result = [data_dict, label_dict]

            result_queue.put(result)


class ThreadedAnchorLoaderFPN(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, batch_size=1, shuffle=False, feat_stride=(32, 16, 8, 4),
                 anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), short=600, long=1000, allowed_border=0,
                 num_thread=4):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :return: AnchorLoader
        """
        super(ThreadedAnchorLoaderFPN, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border

        # decide data and label names
        self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']
        self.data_shape = [("data", (batch_size, 3, short, long))]
        self.feat_shape_list = ThreadedAnchorLoaderFPN.get_feat_shape_list(self.feat_sym, self.data_shape)
        self.label_shape = self.infer_label_shape(dict(self.data_shape), self.feat_shape_list)

        # infer properties from roidb
        self.size = len(roidb)
        self.index = range(self.size)

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # multi-process primitives
        self.index_queue = Queue()
        self.result_queue = Queue(maxsize=num_thread)
        self.workers = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self._thread_start(num_thread)

    @property
    def provide_data(self):
        if self.data is None:
            return self.data_shape
        else:
            return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        if self.label is None:
            return self.label_shape
        else:
            return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def _insert_queue(self):
        for i in range(0, len(self.index), self.batch_size)[:-1]:
            self.index_queue.put(self.index[i:i+self.batch_size])

    def _thread_start(self, num_thread):
        self.workers = [Thread(target=ThreadedAnchorLoaderFPN._worker,
                               args=[self.roidb, self.index_queue, self.result_queue,
                                     self.feat_stride, self.anchor_scales,
                                     self.anchor_ratios, self.allowed_border,
                                     self.data_shape, self.feat_shape_list])
                        for _ in range(num_thread)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)
        self._insert_queue()

    def iter_next(self):
        return self.cur + self.batch_size < self.size

    def next(self):
        TEST_NETWORK_SPEED = False

        if TEST_NETWORK_SPEED and self.data is not None and self.label is not None:
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   provide_data=self.provide_data, provide_label=self.provide_label)

        if self.iter_next():
            self.cur += self.batch_size
            result = self.result_queue.get()
            self.data = [result[name] for name in self.data_name]
            self.label = [result[name] for name in self.label_name]
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_label_shape(self, data_shape, feat_shape_list):
        """
        calculate label shape according to data shape
        :param data_shape: dict of shape
        {"data": (1, 3, 600, 1000)}
        :param feat_shape_list: list of shape
        [(1, 2048, 16, 32), (1, 1024, 32, 64)]
        :return: label_shape: dict of label shape
        {"label": (1, 10000), "bbox_target": (1, 12, 2500), "bbox_weight": (1, 12, 2500)}
        """
        batch_size = data_shape["data"][0]
        dummy_boxes = np.zeros((0, 5))
        dummy_info = [data_shape["data"][2], data_shape["data"][3], 1.0]

        # assign_anchor_fpn only support batch_size=1, make it happy
        feat_shape_list = [(1, ) + feat_shape[1:] for feat_shape in feat_shape_list]
        label_dict = assign_anchor_fpn(feat_shape_list, dummy_boxes, dummy_info, self.feat_stride,
                                       self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label_list = [label_dict['label'], label_dict['bbox_target'], label_dict['bbox_weight']]
        label_shape = [(k, (batch_size, ) + v.shape[1:])
                       for k, v in zip(self.label_name, label_list)]
        return label_shape

    @staticmethod
    def get_feat_shape_list(feat_sym, data_shape):
        _, feat_shape_list, _ = feat_sym.infer_shape(**dict(data_shape))
        return feat_shape_list

    @staticmethod
    def _worker(all_roidb, index_queue, result_queue, feat_stride, anchor_scales,
                anchor_ratios, allowed_border, data_shape, feat_shape_list):
        batch_size = dict(data_shape)["data"][0]
        spacial_dim = sum([shape[2] * shape[3] for shape in feat_shape_list])
        num_anchor = len(anchor_scales) * len(anchor_ratios)

        label_buffer       = np.empty(shape=(batch_size, spacial_dim * num_anchor), dtype="float32")
        bbox_target_buffer = np.empty(shape=(batch_size, 4 * num_anchor, spacial_dim), dtype="float32")
        bbox_weight_buffer = np.empty(shape=(batch_size, 4 * num_anchor, spacial_dim), dtype="float32")

        while True:
            indexes = index_queue.get()
            if indexes is None:
                return

            roidb = [all_roidb[idx] for idx in indexes]
            data, label = get_rpn_batch(roidb, max_shape=(batch_size,) + dict(data_shape)["data"][1:])

            for i, idx in enumerate(indexes):
                im_info = data['im_info'][i]
                gt_boxes = label['gt_boxes'][i][0]  # remove unused singleton dimension
                # assign_anchor_fpn only support batch_size=1, make it happy
                feat_shape_list = [(1,) + feat_shape[1:] for feat_shape in feat_shape_list]
                label_dict = assign_anchor_fpn(feat_shape=feat_shape_list,
                                               gt_boxes=gt_boxes,
                                               im_info=im_info,
                                               feat_strides=feat_stride,
                                               scales=anchor_scales,
                                               ratios=anchor_ratios,
                                               allowed_border=allowed_border)
                label_buffer[i] = label_dict["label"].squeeze()
                bbox_target_buffer[i] = label_dict["bbox_target"].squeeze()
                bbox_weight_buffer[i] = label_dict["bbox_weight"].squeeze()

            result_dict = dict()
            result_dict["data"] = mx.nd.array(data["data"], ctx=mx.cpu())
            result_dict["label"] = mx.nd.array(label_buffer, ctx=mx.cpu())
            result_dict["bbox_target"] = mx.nd.array(bbox_target_buffer, ctx=mx.cpu())
            result_dict["bbox_weight"] = mx.nd.array(bbox_weight_buffer, ctx=mx.cpu())
            result_queue.put(result_dict)