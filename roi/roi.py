import tensorflow as tf
import numpy as np
import config as cfg


def roi_pool(feature_maps, rois, im_dims=(cfg.DEFAUTL_IMAGE_SIZE, cfg.DEFAUTL_IMAGE_SIZE)):
    # 将tensor数据转成numpy计算
    pooled_features = tf.py_function(_roi_pool_py, [feature_maps, rois, im_dims], [tf.float32])
    pooled_features = tf.convert_to_tensor(pooled_features)
    return pooled_features


def _roi_pool_py(feature_maps, regions, im_dims):
    """
    roi pooling 真正实现  这里是缩小了16倍
    :param feature_maps: (bath_size, 36, 36, 512)
    :param rois:      (batch_id, x1, y1, x2, y2)
    :param im_dims:
    :return:
    """
    batch_size, height, width, channels = feature_maps.shape
    # assert batch_size == 1, 'mini-batch should be 1'
    # 获得
    region_nums = regions.shape[0]
    arg_top = np.zeros(shape=(region_nums, cfg.POOL_HEIGHT, cfg.POOL_WIDTH, channels), dtype=np.float32)
    for idx, region in enumerate(regions):
        # get image size
        img_w, img_h = im_dims[0], im_dims[1]
        spatial_scale_w = width // img_w
        spatial_scale_h = height // img_h
        roi_batch_ind = region[0]
        # 得到region在特征图上的坐标
        roi_start_w = int(round(region[1] * spatial_scale_w))
        roi_start_h = int(round(region[2] * spatial_scale_h))
        roi_end_w = int(round(region[3] * spatial_scale_w))
        roi_end_h = int(round(region[4] * spatial_scale_h))
        # # roi_batch_ind should be zero
        # if roi_batch_ind < 0 or roi_batch_ind >= batch_size:
        #     continue
        # 得到region在特征图上宽高
        roi_height = max(roi_end_h - roi_start_h + 1, 1)
        roi_width = max(roi_end_w - roi_start_w + 1, 1)
        # 将region在特征图上的宽高进行划分
        sub_roi_width = roi_width / cfg.POOL_WIDTH
        sub_roi_height = roi_height / cfg.POOL_HEIGHT

        batch_data = feature_maps[roi_batch_ind, ...]
        # 遍历batch_data数据进行 roi_pooling
        for c in range(channels):
            for ph in range(cfg.POOL_HEIGHT):
                for pw in range(cfg.POOL_WIDTH):
                    hstart = int(ph * sub_roi_height)
                    wstart = int(pw * sub_roi_width)
                    hend = int((ph + 1) * sub_roi_height)
                    wend = int((pw + 1) * sub_roi_width)
                    # 计算相对于特征图的坐标
                    hstart = min(max(roi_start_h + hstart, 0), height)
                    wstart = min(max(roi_start_w + wstart, 0), width)
                    hend = min(max(roi_start_h + hend, 0), height)
                    wend = min(max(roi_start_w + wend, 0), width)

                    for h in range(hstart, hend):
                        for w in range(wstart, wend):
                            if batch_data[h, w, c] > arg_top[idx, ph, pw, c]:
                                arg_top[idx, ph, pw, c] = batch_data[h, w, c]
    return arg_top
