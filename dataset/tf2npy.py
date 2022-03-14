import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def decode_tfrecord(tfrecord_list):
    raw_image = tf.data.TFRecordDataset(tfrecord_list)
    decomp_feature = {
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        'label_vol': tf.io.FixedLenFeature([], tf.string),
    }
    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, decomp_feature)

    parsed_image = raw_image.map(_parse_image_function)
    for parser in parsed_image:
        data_vol = tf.io.decode_raw(parser['data_vol'], tf.float32).numpy()
        label_vol = tf.io.decode_raw(parser['label_vol'], tf.float32).numpy()
        image_raw = data_vol.reshape((256, 256, 3))
        label_raw = label_vol.reshape((256, 256, 3))
        return image_raw, label_raw

def tfrecord2npy(input_folder, output_folder):
    for name in ['ct_train', 'ct_val', 'mr_train', 'mr_val']:
        with open(f'{input_folder}/{name}_list.txt', 'r') as f:
            data = f.readlines()
        data = [tfrecord[:-1] for tfrecord in data]
        os.makedirs(os.path.join(output_folder, name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, name, 'labels'), exist_ok=True)
        for i, id in enumerate(data):
            if i % 1000 == 0:
                print(id)
            img_path = f'{input_folder}/{id}'
            image, label = decode_tfrecord(img_path)
            np.save(os.path.join(output_folder, name, 'images', id.split('/')[1]), image)
            np.save(os.path.join(output_folder, name, 'labels', id.split('/')[1]), label)
        print(f"Finish {name}.")

if __name__ == '__main__':
    tfrecord2npy('./data/threcords', './data/npy/')