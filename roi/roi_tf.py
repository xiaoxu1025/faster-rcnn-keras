import tensorflow as tf


def roi_pool_tf(args, im_dims):
    feature_maps, rois = args[0], args[1]
    return _roi_pool_tf(feature_maps, rois, im_dims)


def _roi_pool_tf(feature_maps, rois, im_dims):
    """

    :param feature_maps: (batch_size, 36, 36, 512)
    :param rois: shape (batch_size, 128, 5) -> n * (batch_id, x1, y1, x2, y2)
    :param im_dims:
    :return:
    """
    # Image that the ROI is taken from (minibatch of 1 means these will all be 0)
    box_ind = tf.cast(rois[..., 0], dtype=tf.int32)

    # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]
    # box must be normalized
    boxes = rois[..., 1:]
    normalization = tf.cast(tf.stack([im_dims[1], im_dims[0], im_dims[1], im_dims[0]], axis=0),
                            dtype=tf.float32)
    boxes = tf.div(boxes, normalization)
    boxes = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)  # y1, x1, y2, x2

    # ROI pool output size
    crop_size = tf.constant([14, 14])

    # ROI pool
    pooled_features = tf.image.crop_and_resize(image=feature_maps, boxes=boxes[0, ...], box_ind=box_ind[0, ...],
                                               crop_size=crop_size)
    # Max pool to (7x7)
    pooled_features = tf.nn.max_pool(pooled_features, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pooled_features
