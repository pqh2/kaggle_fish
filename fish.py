__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
np.random.seed(2016)


from vgg16bn import Vgg16BN
import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
import PIL
import ujson as json
from keras.preprocessing.image import ImageDataGenerator
import resnet
from sklearn.cross_validation import KFold
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        horizontal_flip=True)

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()
    sizes = []
    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            sizes.append(PIL.Image.open(fl).size)
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id, sizes


def load_test():
    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data, train_target, train_id, sizes = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)
    train_target = train_target.astype('float32')

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id, sizes


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_model():
   input = Input((3, 224, 224))

   x = ZeroPadding2D((1,1))(input)
   x = Convolution2D(16, 3, 3, activation='relu')(x)    
   x = ZeroPadding2D((1,1))(x)
   x = Convolution2D(16, 3, 3, activation='relu')(x)
   x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
   x = ZeroPadding2D((1,1))(input)
   x = Convolution2D(32, 3, 3, activation='relu')(x)    
   x = ZeroPadding2D((1,1))(x)
   x = Convolution2D(32, 3, 3, activation='relu')(x)
   x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  
   x = ZeroPadding2D((1,1))(x)
   x = Convolution2D(64, 3, 3, activation='relu')(x)
   x = ZeroPadding2D((1,1))(x)
   x = Convolution2D(64, 3, 3, activation='relu')(x)
   x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
   x = ZeroPadding2D((1,1))(x)

   x = Flatten()(x)
   x = Dense(128, activation='relu')(x)
   x = Dropout(0.5)(x)
   x = Dense(128, activation='relu')(x)
   x = Dropout(0.5)(x)

   x_bb = Dense(4, name='bb')(x)
   x_class = Dense(8, activation='softmax', name='class')(x)
   model = Model([input], [x_bb, x_class])
   model.compile(optimizer=SGD(lr=3e-3, decay=1e-6, momentum=0.9, nesterov=True), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'])
  
   return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 32
    nb_epoch = 12
    random_state = 51
    train_data, train_target, train_id, sizes = read_and_normalize_train_data()
    anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']
    bb_json = {}
    
    print sizes[:10]
    for c in anno_classes:
        j = json.load(open('/home/patrick/Documents/kaggle/fish/code/annos/{}_labels.json'.format(c), 'r'))
        for l in j:
            if 'annotations' in l.keys() and len(l['annotations'])>0:
                bb_json[l['filename'].split('/')[-1]] = sorted(
                    l['annotations'], key=lambda x: x['height']*x['width'])[-1]
    empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}
    file2idx = {o:i for i,o in enumerate(train_id)}
    bb_params = ['height', 'width', 'x', 'y']
    def convert_bb(bb, size):
        bb = [bb[p] for p in bb_params]
        conv_x = (224. / size[0])
        conv_y = (224. / size[1])
        bb[0] = bb[0]*conv_y
        bb[1] = bb[1]*conv_x
        bb[2] = max(bb[2]*conv_x, 0)
        bb[3] = max(bb[3]*conv_y, 0)
        return bb

    for f in train_id:
        if not f in bb_json.keys(): bb_json[f] = empty_bbox
    trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(train_id, sizes)] ).astype(np.float32)
    print trn_bbox[:10]


    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        model = create_model()
        print(model.summary())

        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]
        boxes_train = trn_bbox[train_index]
        boxes_valid = trn_bbox[test_index]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ]
        print len(X_train)
        print len(trn_bbox)
        print len(Y_train)
        model.fit(X_train, [boxes_train, Y_train], batch_size=batch_size, nb_epoch=12)
        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid[1])
        print('Score log_loss: ', score)

        models.append(model)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
    batch_size = 32
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction[1])

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
