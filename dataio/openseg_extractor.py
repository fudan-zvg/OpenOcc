import tensorflow as tf2
import tensorflow.compat.v1 as tf
from tensorflow import io
import torch

def read_bytes(path):
    '''Read bytes for OpenSeg model running.'''
    with io.gfile.GFile(path, 'rb') as f:
        file_bytes = f.read()
    return file_bytes

def extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=None, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''
    
    np_image_string = read_bytes(img_dir)
    results = openseg_model.signatures['serving_default'](
            inp_image_bytes=tf.convert_to_tensor(np_image_string),
            inp_text_emb=text_emb)
    img_info = results['image_info']
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1])
    ]
    if regional_pool:
        image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
    else:
        image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
    if img_size is not None:
        feat_2d = tf.cast(tf.image.resize_nearest_neighbor(
            image_embedding_feat, img_size, align_corners=True)[0], dtype=tf.float16).numpy()
    else:
        feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float16).numpy()

    feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1)
    return feat_2d