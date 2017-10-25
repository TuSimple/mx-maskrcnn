import cPickle
import os
import time
import mxnet as mx
import numpy as np

from module import MutableModule
from rcnn.config import config
from rcnn.io import image
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.nms import py_nms_wrapper
bbox_pred = nonlinear_pred


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs()))


def im_proposal(predictor, data_batch, data_names, scale):
    data_dict = dict(zip(data_names, data_batch.data))
    output = predictor.predict(data_batch)

    # drop the batch index
    boxes = output['rois_output'].asnumpy()[:, 1:]
    scores = output['rois_score'].asnumpy()

    # transform to original scale
    boxes = boxes / scale

    return scores, boxes, data_dict


def generate_proposals(predictor, test_data, imdb, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    i = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scale = im_info[0, 2]
        scores, boxes, data_dict = im_proposal(predictor, data_batch, data_names, scale)
        t2 = time.time() - t
        t = time.time()

        # assemble proposals
        dets = np.hstack((boxes, scores))
        original_boxes.append(dets)

        # filter proposals
        keep = np.where(dets[:, 4:] > thresh)[0]
        dets = dets[keep, :]
        imdb_boxes.append(dets)

        if vis:
            vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale)

        print 'generating %d/%d' % (i + 1, imdb.num_images), 'proposal %d' % (dets.shape[0]), \
            'data %.4fs net %.4fs' % (t1, t2)
        i += 1

    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.root_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_boxes

def im_detect_mask(predictor, data_batch, data_names, scale=1):
    output = predictor.predict(data_batch)
    data_dict = dict(zip(data_names, data_batch.data))
    if config.TEST.HAS_RPN:
        rois = output['rois_output'].asnumpy()[:, 1:]
    else:
        raise NotImplementedError

    im_shape = data_dict['data'].shape

    if config.TEST.HAS_RPN:
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        mask_output = output['mask_prob_output'].asnumpy()
    else:
        raise NotImplementedError
    # post processing
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

    # we used scaled image & roi to train, so it is necessary to transform them back
    pred_boxes = pred_boxes / scale

    return scores, pred_boxes, data_dict, mask_output

def pred_eval_mask(predictor, test_data, imdb, roidb, result_path, vis=False, thresh=1e-1):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    num_images = imdb.num_images

    i = 0
    t = time.time()
    results_list = []
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_masks = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    for im_info, data_batch in test_data:
        roi_rec = roidb[i]
        t1 = time.time() - t
        t = time.time()

        scores, boxes, data_dict, mask_output = im_detect_mask(predictor, data_batch, data_names)

        t2 = time.time() - t
        t = time.time()

        CLASSES = imdb.classes

        label = np.argmax(scores, axis=1)
        label = label[:, np.newaxis]

        for cls in CLASSES:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_masks = mask_output[:, cls_ind, :, :]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
            cls_masks = cls_masks[keep, :, :]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            all_boxes[cls_ind][i] = dets[keep, :]
            all_masks[cls_ind][i] = cls_masks[keep, :]

        boxes_this_image = [[]] + [all_boxes[cls_ind][i] for cls_ind in range(1, imdb.num_classes)]
        masks_this_image = [[]] + [all_masks[cls_ind][i] for cls_ind in range(1, imdb.num_classes)]

        results_list.append({'image': roi_rec['image'],
                           'im_info': im_info,
                           'boxes': boxes_this_image,
                           'masks': masks_this_image})
        t3 = time.time() - t
        t = time.time()
        print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(i, imdb.num_images, t1, t2, t3)
        i += 1
    results_pack = {'all_boxes': all_boxes,
                    'all_masks': all_masks,
                    'results_list': results_list}
    imdb.evaluate_mask(results_pack)

def pred_demo_mask(predictor, test_data, imdb, roidb, result_path, vis=False, thresh=1e-1):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = -1

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    i = 0
    for im_info, data_batch in test_data:
        roi_rec = roidb[i]
        scale = im_info[0, 2]
        scores, boxes, data_dict, mask_output = im_detect_mask(predictor, data_batch, data_names)

        CLASSES = imdb.classes

        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        all_masks = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        label = np.argmax(scores, axis=1)
        label = label[:, np.newaxis]

        for cls in CLASSES:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_masks = mask_output[:, cls_ind, :, :]
            cls_scores = scores[:, cls_ind, np.newaxis]
            #print cls_scores.shape, label.shape
            keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
            cls_masks = cls_masks[keep, :, :]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            #print dets.shape, cls_masks.shape
            all_boxes[cls_ind] = dets[keep, :]
            all_masks[cls_ind] = cls_masks[keep, :, :]

        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(CLASSES))]
        masks_this_image = [[]] + [all_masks[j] for j in range(1, len(CLASSES))]
        filename = roi_rec['image'].split("/")[-1]
        filename = result_path + '/' + filename.replace('.png', '') + '.jpg'
        data_dict = dict(zip(data_names, data_batch.data))
        draw_detection_mask(data_dict['data'], boxes_this_image, masks_this_image, scale, filename)
        i += 1


def vis_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()

def draw_detection_mask(im_array, boxes_this_image, masks_this_image, scale, filename):
    import cv2
    import random
    class_names = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'mcycle', 'bicycle')
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array.asnumpy(), config.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.cv.CV_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = boxes_this_image[j]
        masks = masks_this_image[j]
        for i in range(len(dets)):
            bbox = dets[i, :4] * scale
            score = dets[i, -1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            mask = masks[i, :, :]
            mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            mask_color = random.randint(0, 255)
            c = random.randint(0, 2)
            target = im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] + mask_color * mask
            target[target >= 255] = 255
            im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] = target
    print filename
    cv2.imwrite(filename, im)

def draw_detection(im_array, boxes_this_image, scale, filename):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    class_names = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'mcycle', 'bicycle')
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array.asnumpy(), config.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.cv.CV_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = boxes_this_image[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    print filename
    cv2.imwrite(filename, im)
