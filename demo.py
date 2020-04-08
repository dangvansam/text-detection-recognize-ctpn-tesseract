# coding=utf-8
import os
import shutil
import sys
import time
import pytesseract
import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #tắt cảnh báo update tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

# tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
# tf.app.flags.DEFINE_string('output_path', 'data/output/', '')
# tf.app.flags.DEFINE_string('gpu', '0', '')
# tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
# FLAGS = tf.app.flags.FLAGS

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


# def get_images():
    # files = []
    # exts = ['jpg', 'png', 'jpeg', 'JPG']
    # for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        # for filename in filenames:
            # for ext in exts:
                # if filename.endswith(ext):
                    # files.append(os.path.join(parent, filename))
                    # break
    # print('Find {} images'.format(len(files)))
    # return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

from rotate_img import rotate_img
def main(argv=None):
    # if os.path.exists(FLAGS.output_path):
        # shutil.rmtree(FLAGS.output_path)
    # os.makedirs(FLAGS.output_path)
    # print(FLAGS.output_path)
    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        print("init sess")
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state('checkpoints_mlt/')
            model_path = os.path.join('checkpoints_mlt/', os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
			
            #im_fn_list = get_images()
            print('===============')
            im = rotate_img('hoadontiendien-3.png')
            print(im.shape)
			
            cv2.imwrite('rotated2.png', im[:, :, :])
            print("write rotate img") 		 
            start = time.time()

            img, (rh, rw) = resize_image(im)
            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])
            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob], feed_dict={input_image: [img], input_im_info: im_info})

            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            textdetector = TextDetector(DETECT_MODE='H')
            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
            boxes = np.array(boxes, dtype=np.int)

            cost_time = (time.time() - start)
            print("cost time: {:.2f}s".format(cost_time))
            min_x, max_x, min_y, max_y = 0,w,0,h
            box_minx = min([b[0] for b in boxes])
            box_miny = min([b[1] for b in boxes])
            box_maxx = max([b[4] for b in boxes])
            box_maxy = max([b[5] for b in boxes])
            print(box_minx,box_miny)
            print(box_maxx,box_maxy)
            crop_img = img[box_miny:box_maxy, box_minx:box_maxx]
            print(crop_img.shape)
			
            # for b in boxes:
			    # if b[0] <
                # texts = []
            for i, box in enumerate(boxes):
                cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)
                #crop_img2 = img[box[1]-5:box[5]+5, box[0]:box[4]]
            img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
            #print(img[:, :, ::-1].shape)
            #cv2.imshow('aaa',img[:, :, ::-1])
            #cv2.waitKey()
			
            cv2.imwrite('rotate_cuted2.png', crop_img[:, :, :])
            #cv2.imwrite('rotate_cuted_box.png', crop_img2[:, :, ::-1])
            # with open(os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt", "w", encoding="UTF-8") as f:
                # for i, box in enumerate(boxes):
                    # line = ",".join(str(box[k]) for k in range(8))
                    # line += "," + str(texts[i]) + "\r\n"
                    # #print(line)
                    # f.writelines(line)

main()
#if __name__ == '__main__':
    #tf.app.run()