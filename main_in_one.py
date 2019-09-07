import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import time

import numpy as np
import random as rn
import tensorflow as tf
import h5py

from keras.preprocessing import sequence
from keras.models import Sequential, load_model, model_from_yaml
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Reshape, Permute, Embedding
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_normal, glorot_uniform
from keras.regularizers import l1_l2, l1, l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical


from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluation(y_true, y_pred, threshold=.5):
    auc = roc_auc_score(y_true, y_pred)
    y_pred = (y_pred > threshold).astype('int')
    eval_dict = {'acc': accuracy_score,
                 'mcc': matthews_corrcoef,
                 'precision': precision_score,
                 'recall': recall_score,
                 'f1': f1_score}
    for i in eval_dict:
        ei = eval_dict[i](y_true, y_pred)
        print("{:16}{:.3f}".format(i + ":", ei))
    print("{:16}{:.3f}".format("auc:", auc))
    return

def def_kw(D, K, N=None):
    return D[K] if K in D.keys() else N

print_eq = lambda : print('\n', '=' * 36, sep = '')

def random_set(iseed=123):
    np.random.seed(iseed)
    rn.seed(iseed)
    tf.set_random_seed(iseed)
    return

def build_model(in_shape=(1000, 4), **kw):

    ### defaults
    n_classes = 2
    ## Convolution
    n_filter = 64
    conv_d = 1
    # 1D
    filter_length = 23
    pool_length = 8
    sub1 = 1
    # 2D
    conv_size = iter([(4, 23)])
    pool_size = iter([(1, 8)])
    sub2 = (1, 1)
    ## Recurrent
    n_units = 64
    ## Dropout
    idrop = iter([.5, .3, .3])
    ## Dense
    iden = iter([32])
    ## Regularization
    lambda_l1 = .00000001
    lambda_l2 = .0001
    ## Optmizations
    # loss = 'sparse_categorical_crossentropy'
    # loss = 'categorical_crossentropy'
    loss = 'binary_crossentropy'
    opt = 'adam'
    # opt = 'adadelta'
    # opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    n_filter = def_kw(kw, 'n_filter', n_filter)
    filter_length = def_kw(kw, 'filter_length', filter_length)
    pool_length = def_kw(kw, 'pool_length', pool_length)
    loss = def_kw(kw, 'loss', loss)
    out = def_kw(kw, 'out', None)
    opt = def_kw(kw, 'opt', opt)

    ### modeling
    print("Building model...")
    print("input shape:\t", in_shape)
    model = Sequential()
    model.add(Reshape(in_shape, input_shape=in_shape))
    if conv_d == 2:
        model.add(Reshape((1, in_shape[0], in_shape[1])))
        model.add(Permute((1, 3, 2)))
        model.add(Conv2D(n_filter, next(conv_size),
                         strides=sub2,
                         padding='valid',
                         activation='relu',
                         data_format="channels_first",
                         ))
        model.add(MaxPooling2D(pool_size=next(pool_size), padding='same',
                               data_format="channels_first",
                               ))
        model.add(Dropout(next(idrop)))
    elif conv_d == 1:
        model.add(Conv1D(n_filter,
                         kernel_size=filter_length,
                         strides=sub1,
                         padding='valid',
                         activation='relu',
                         ))
        model.add(MaxPooling1D(pool_size=pool_length, padding='same'))
        model.add(Dropout(next(idrop)))
    if n_units:
        if conv_d==2:
            model.add(Reshape((model.output_shape[1],
                               model.output_shape[2] * model.output_shape[3])))
            model.add(Permute((2, 1)))
        model.add(Bidirectional(GRU(n_units), merge_mode='concat'))
    else:
        model.add(Flatten())
    model.add(Dense(next(iden),
                    activation=None))
    model.add(Dropout(next(idrop)))
    if loss == 'binary_crossentropy':
        out = (1, 'sigmoid')
    elif loss == 'categorical_crossentropy':
        out = (n_classes, 'softmax')
    elif loss == 'sparse_categorical_crossentropy':
        out = (n_classes, 'softmax')
    model.add(Dense(out[0], activation=out[1],
                    kernel_regularizer=l1_l2(lambda_l1, lambda_l2)
                    ))
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def get_callbacks(**kw):
    checkprog = def_kw(kw, 'checkprog', None)
    earlystop = def_kw(kw, 'earlystop', None)
    callbacks = []
    if checkprog:
        checkmodel = os.path.exists('./models/')
        checkmodel = './models/' if checkmodel else './'
        checkmodel += 'model_check_' + checkprog
        checkmodel += '.h5'
        checkpoint = ModelCheckpoint(filepath=checkmodel,
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=True)
        callbacks.append(checkpoint)
    if earlystop:
        assert earlystop > 0, "earlystop must be an integer greater than 0"
        earlystopping = EarlyStopping(monitor='val_loss',
                                      patience=earlystop,
                                      verbose=0)
        callbacks.append(earlystopping)
    return callbacks

def get_model(model=None, weights=None):
    if not model:
        model = build_model()
    elif isinstance(model, str):
        if model[-3:] == '.h5':
            model = load_model(model)
        elif model[-5] == '.':
            with open(model) as f:
                model = f.read()
            model = model_from_yaml(model)
        else:
            model = model_from_yaml(model)
    elif isinstance(model, tuple):
        model = model_from_yaml(model[0], model[1])
    elif isinstance(model, list):
        model = model_from_yaml(model[0], model[1])
    model = model
    if weights: model.load_weights(weights)
    return model

def save_model(model, savename=None, string_type='yaml'):
    if not savename and isinstance(model, str):
        savename = model
    model = get_model(model)
    model.save('model_'+savename+'.h5')
    with open(savename+'.'+string_type, 'w') as f:
        if string_type == 'yaml': f.write(model.to_yaml())
        elif string_type == 'json': f.write(model.to_json())
    model.save_weights(savename+'.h5', overwrite=True)
    return

def load_data_1(datafile):
    print("Loading data ", datafile)
    with h5py.File(datafile, 'r') as h5f:
        X_ = h5f['seqs'].value
        y_ = h5f['labels'].value
    y_ = y_.reshape(-1)
    print("X_ & y_:", X_.shape, y_.shape)
    return (X_, y_)

def joint_data(files, saving=None):
    fi = files[0]
    X_, y_ = load_data_1(fi)
    for fi in files[1:]:
        X_i, y_i = load_data_1(fi)
        X_ = np.concatenate((X_, X_i))
        y_ = np.concatenate((y_, y_i))
    if saving:
        with h5py.File(saving, 'w') as h5f:
            h5f.create_dataset('seqs', data=X_)
            h5f.create_dataset('labels', data=y_)
        print("Saved in", saving)
    print("X_ & y_:", X_.shape, y_.shape)
    return (X_, y_)

    _ = time.time()
    results = model.predict(X_train, batch_size=batch_size)
    pt = time.time() - _
    print("predicting time: {0:.2f}s".format(pt))
    results = results[:, 0]
    evaluation(y_train, results)

def validation_1(datafile=None,
                 dataset=None,
                 model=None,
                 batch_size=32):
    model = get_model(model)
    if dataset:
        X, y = dataset
    if datafile:
        X, y = load_data_1(datafile)
    _ = time.time()
    results = model.predict(X, batch_size=batch_size)
    pt = time.time() - _
    print("predicting time: {0:.2f}s".format(pt))
    results = results[:, 0]
    evaluation(y, results)
    return (y, results)

def load_data_3(datafile):
    print("Loading data ", datafile)
    with h5py.File(datafile, 'r') as h5f:
        X_train = h5f['train_in'].value
        y_train = h5f['train_out'].value
        X_valid = h5f['valid_in'].value
        y_valid = h5f['valid_out'].value
        X_test = h5f['test_in'].value
        y_test = h5f['test_out'].value
    y_train = y_train.reshape(-1)
    y_valid = y_valid.reshape(-1)
    y_test = y_test.reshape(-1)
    return (X_train, y_train,
            X_valid, y_valid,
            X_test, y_test)

def validation_3(datafile, model=None,
                 divided=False, fit=False,
                 earlystop=8,
                 checkprog=None,
                 batch_size=32,
                 n_epoch=200):
    model = get_model(model)
    (X_train, y_train,
     X_valid, y_valid,
     X_test, y_test) = load_data_3(datafile=datafile)
    if fit:
        print_eq()
        print("training")
        _ = time.time()
        callbacks = get_callbacks(checkprog=checkprog,
                                  earlystop=earlystop)
        if len(X_valid) > 0 or X_valid.shape != (1, 1):
            model.fit(X_train, y_train,
                      batch_size=batch_size, epochs=n_epoch,
                      validation_data=(X_valid, y_valid),
                      callbacks=callbacks, verbose=0)
        else:
            model.fit(X_train, y_train,
                      batch_size=batch_size, epochs=n_epoch,
                      callbacks=callbacks, verbose=0)
        pt = time.time() - _
        print("training time: {0:.2f}s".format(pt))
    if checkprog:
        checkmodel = './models/model_check_' + checkprog + '.h5'
        try:
            model = get_model(checkmodel)
        except:
            pass
    print_eq()
    print('training set:')
    validation_1(dataset=(X_train, y_train),
                 model=model,
                 batch_size=batch_size)
    if len(X_valid) > 0 or X_valid.shape != (1,1):
        print_eq()
        print('valid set:')
        validation_1(dataset=(X_valid, y_valid),
                     model=model,
                     batch_size=batch_size)
    print_eq()
    print('testing set:')
    validation_1(dataset=(X_test, y_test),
                 model=model,
                 batch_size=batch_size)
    return model

    
def split_p4653():
    datafile = './data/enh_p4653_3sets.h5'
    filehead = './data/enh_p4653_'
    (X_train, y_train,
     X_valid, y_valid,
     X_test, y_test) = load_data_3(datafile=datafile)
    batch_size = 10000
    i = 0
    for si in np.r_[0:X_train.shape[0]:batch_size]:
        ei = min(si + batch_size, X_train.shape[0])
        with h5py.File(filehead + 'train_%s.h5' % (i), 'w') as h5f:
            h5f.create_dataset('seqs', data=X_train[si:ei, ...])
            h5f.create_dataset('labels', data=y_train[si:ei, ...])
        i += 1
    with h5py.File(filehead + 'valid.h5', 'w') as h5f:
        h5f.create_dataset('seqs', data=X_valid)
        h5f.create_dataset('labels', data=y_valid)
    with h5py.File(filehead + '_test.h5', 'w') as h5f:
        h5f.create_dataset('seqs', data=X_test)
        h5f.create_dataset('labels', data=y_test)
    pass

def joint_p4653():
    filehead = './data/enh_p4653_'
    saving = filehead + '3sets.h5'
    fnames = [filehead + 'train_%s.h5' % (i) for i in np.r_[0:5]]
    X_train, y_train = joint_data(fnames)
    X_valid, y_valid = load_data_1(filehead + 'valid.h5')
    X_test, y_test = load_data_1(filehead + 'test.h5')
    with h5py.File(saving, 'w') as h5f:
        h5f.create_dataset('train_in', data=X_train)
        h5f.create_dataset('train_out', data=y_train)
        h5f.create_dataset('valid_in', data=X_valid)
        h5f.create_dataset('valid_out', data=y_valid)
        h5f.create_dataset('test_in', data=X_test)
        h5f.create_dataset('test_out', data=y_test)
    (X_train, y_train,
     X_valid, y_valid,
     X_test, y_test) = load_data_3(datafile=saving)
    pass


def gen_p4653_model():
    model = build_model((1000, 4))
    datafile = './data/enh_p4653_3sets.h5'
    prog = 'p4653'
    model_saving = './models/model_' + prog + '.h5'
    model = validation_3(datafile, model,
                         batch_size=128,
                         checkprog=prog,
                         divided=True, fit=True)
    model.save(model_saving)
    return

def validation_on_p4653():
    datafile = './data/enh_p4653_3sets.h5'
    model = './models/model_pretrained_0.h5'
    print("evaluating model(%s) on data(%s)" % (model, datafile))
    model = validation_3(datafile, model,
                         divided=True, fit=False)

def retraining_on_vista():
    datafile = './data/vista_human_3sets.h5'
    model = './models/model_pretrained_0.h5'
    model_saving = './models/model_vista.h5'
    print("retraining model(%s) with data(%s)" % (model, datafile))
    model = validation_3(datafile, model,
                         divided=True, fit=True,
                         n_epoch=20)
    model.save(model_saving)

if __name__ == "__main__":
    random_set()                # set random seed
    # split_p4653()               # split data for uploading
    # joint_p4653()               # joint splited data 
    # gen_p4653_model()           # generate pre-training model
    validation_on_p4653()       # evaluating pretrained model on training data
    retraining_on_vista()       # retraining on testing set and save retrained model
