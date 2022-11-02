"""
Author :Yashish Maduwantha

Project : Secret Source paper for ICASSP

Model : CNN-BLSTM model with only 6 TVs as Targets
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input, Dropout, Concatenate, Masking, TimeDistributed, BatchNormalization, Conv1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import tanh
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

import datetime
from datetime import date
# from sklearn.preprocessing import StandardScaler
# from tensorflow_addons.optimizers import NovoGrad
# from tensorflow.keras.utils import to_categorical
# from scipy.stats import pearsonr

from utils import correlation_coefficient_loss

from matplotlib import pyplot
pyplot.switch_backend('agg')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def ppmc(x, y):
    xbar = np.nanmean(x)
    ybar = np.nanmean(y)
    num = np.nansum((x - xbar)*(y - ybar))
    den = np.sqrt(np.nansum((x - xbar)**2))*np.sqrt(np.nansum((y - ybar)**2))
    corr = num/den

    return corr

# computing average correlations on test set
def compute_corr_score(y_predict, y_true):
    corr_TVs = np.zeros(6)
    corr_TVs_pc = np.zeros(6)
    tot_samples = y_true.shape[0]
    for j in range(0, y_true.shape[0]):
        for i in range(0, No_TVs):
            corr_TVs[i] += ppmc(y_predict[j,:,i], y_true[j, :, i])
            # corr, _ = pearsonr(y_predict[j,:,i], y_true[j, :, i])
            # corr_TVs_pc[i] += corr
    # for j in range(0, y_val.shape[0]):
    #     for i in range(0, No_TVs):
    #         corr_TVs[i] += ppmc(y_predict_val[j,:,i], y_val[j, :, i])
    corr_TVs_avg = corr_TVs/tot_samples
    avg_corr_tvs = np.mean(corr_TVs_avg)
    # corr_TVs_pc = corr_TVs_pc/tot_samples

    print("Corr_Average_Test_set :", corr_TVs_avg)
    print("Corr_Average_across_TVs:", avg_corr_tvs)
    # print("Corr_pearson_func :", corr_TVs_pc)

    return corr_TVs_avg, avg_corr_tvs    # print("Corr_pearson_func :", corr_TVs_pc)

def scheduler(epoch, lr):
    if epoch < 10 or lr < 1e-5:
        return lr
    elif epoch%5 == 0:
        return lr * tf.math.exp(-0.1)
    else:
        return lr


def mv_norm(X):
    X_samples = X.shape[0]
    X_mfcc = X.flatten()
    mean = np.mean(X)
    std = np.std(X)
    X_mfcc = (X_mfcc - mean) / std
    #print(std)
    return X_mfcc.reshape(X_samples, TIME_STEPS)

def min_max_norm(X):
    X_samples = X.shape[0]
    X_mfcc = X.flatten()
    min_val = np.min(X)
    max_val = np.max(X)
    X_mfcc = (X_mfcc - min_val) / (max_val - min_val)
    #print(std)
    return X_mfcc.reshape(X_samples, TIME_STEPS)

def blstm_model(sample_x):
    """
    Implmentation of the blstm model
    :param sample_x: Input tensor
    :return: Model object
    """
    print("-------------sample x shape---------------")
    print(sample_x.shape)
    inp = Input(shape=(sample_x.shape))

    masking_layer = Masking()
    masked_embedding = masking_layer(inp)

    # embedding = Embedding(input_dim=sample_x.shape[1], output_dim=sample_x.shape[1], mask_zero=True)
    # masked_output = embedding(inp)

    forward_layer_1 = LSTM(256, return_sequences=True, dropout=0.3)
    backward_layer_1 = LSTM(100, activation='relu', return_sequences=True, go_backwards=True)
    forward_layer_2 = LSTM(256, return_sequences=True, dropout=0.3)
    backward_layer_2 = LSTM(100, activation='relu', return_sequences=True, go_backwards=True)
    forward_layer_3 = LSTM(256, return_sequences=True, dropout=0.3)
    # forward_layer_4 = LSTM(100, return_sequences=True, dropout=0.5)

    conv1 = Conv1D(filters=NUM_FILTERS_CONV1, padding= 'same', activation='relu', kernel_size=FILTER_SIZE1, input_shape=(TIME_STEPS, No_Feats))(masked_embedding)
    conv2 = Conv1D(filters=NUM_FILTERS_CONV2, padding= 'same', activation='relu', kernel_size=FILTER_SIZE2)(conv1)
    conv3 = Conv1D(filters=NUM_FILTERS_CONV3, padding= 'same', activation='relu', kernel_size=FILTER_SIZE3)(conv2)
    conv4 = Conv1D(filters=NUM_FILTERS_CONV4, padding= 'same', activation='relu', kernel_size=FILTER_SIZE4)(conv3)
    conv5 = Conv1D(filters=NUM_FILTERS_CONV5, padding= 'same', activation='relu', kernel_size=FILTER_SIZE5)(conv4)

    concat = Concatenate()([conv1, conv2, conv3, conv4, conv5])

    blstm_1 = Bidirectional(forward_layer_1)(concat)
    # bnorm_1 = BatchNormalization()(blstm_1)
    # drp_1 = Dropout(DROPOUT_PROB)(blstm_1)
    blstm_2 = Bidirectional(forward_layer_2)(blstm_1)
    # blstm_3 = Bidirectional(forward_layer_3)(blstm_2)
    # blstm_4 = Bidirectional(forward_layer_4)(blstm_3)
    # drp_2 = Dropout(DROPOUT_PROB)(blstm_2)
    # conv6 = Conv1D(filters=6, padding= 'same', kernel_size=21)(blstm_2)
    dense_1 = TimeDistributed(Dense(128, activation='relu'))(blstm_2)
    bnorm_2 = BatchNormalization()(dense_1)
    drp_3 = Dropout(DROPOUT_PROB)(bnorm_2)
    # # dense_2 = TimeDistributed(Dense(6, activation='linear'))(drp_3)
    # # out = dense_2
    dense_2 = TimeDistributed(Dense(6, kernel_regularizer='l2'))(drp_3)
    out = tanh(dense_2)

    return Model(inputs=inp, outputs=out)


def fine_tune(x_train, y_train, x_test, y_test, x_val, y_val):
    """
    Model training and evaluation
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param x_val:
    :param y_val:
    :return: Test accuracy evaluated on the test data
    """
    classifier = blstm_model(x_train[0])
    # opt = NovoGrad(lr=1e-3, beta_1=0.95, beta_2=0.5)
    earlystop = EarlyStopping(monitor='val_loss', patience=PATIENCE)
    LR_scheduler = LearningRateScheduler(scheduler)
    opt = Adam(lr=LR)
    # corr_loss = correlation_coefficient_loss
    # classifier.compile(optimizer=opt, loss=corr_loss, metrics=["mse"])
    classifier.compile(optimizer=opt, loss=MeanAbsoluteError(), metrics=['mae'])
    # classifier.compile(optimizer=opt, loss=MeanSquaredError(), metrics=['mse'])
    classifier.summary()

    # fit the model
    history = classifier.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                              verbose=1, validation_data=(x_val, y_val), callbacks=[earlystop, LR_scheduler])

    # compute train accuracy
    _, train_MSE = classifier.evaluate(x_train, y_train, verbose=0)
    # test on data
    _, test_MSE = classifier.evaluate(x_test, y_test, verbose=0)

    # save model
    classifier.save(model_dir + 'BLSTM_model.h5')

    y_predict = classifier.predict(x_test, verbose=0)

    # y_predict_val = classifier.predict(x_val, verbose=0)

    corr_avg, avg_corr = compute_corr_score(y_predict, y_test)

    # write a .txt with results and model params
    lines = ['Feature:' + str(Feature), 'Epochs:' + str(EPOCHS), 'Patience:' + str(PATIENCE), 'Batch size:' + str(BATCH_SIZE), 'Corr_Average:' + str(corr_avg), 'Corr_Average_TVs:' + str(avg_corr)]
    with open(out_dir +'log.txt', 'w') as f:
        f.write('\n'.join(lines))

    # plot TVs for 5 samples
    for sample in range(0,5):
        for tv in range(0,6):
            pyplot.plot(y_predict[sample,:,tv], label='predicted')
            pyplot.plot(y_test[sample,:,tv], label='True')
            pyplot.legend()
            pyplot.title('TV Predictions')
            pyplot.savefig(tv_dir + 'sample_' + str(sample + 1) + '_' + 'tv_' + str(tv + 1) + '.png')
            pyplot.close()
            pyplot.show()

    # plots for loss and accuracy change over epochs
    print('Train MSE: %.3f, Test MSE: %.3f' % (train_MSE, test_MSE))
    # plot loss during training
    # pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    # pyplot.subplot(212)
    # pyplot.title('Accuracy')
    # pyplot.plot(history.history['accuracy'], label='train')
    # pyplot.plot(history.history['val_accuracy'], label='test')
    # pyplot.legend()
    pyplot.savefig(loss_dir + 'losses.png')
    pyplot.close()
    pyplot.show()


    return test_MSE

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    out_dir = "./Model_outs/CNN_BLSTM_model_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour) + "/"
    loss_dir = out_dir + "loss/"
    model_dir = out_dir + 'net/'
    tv_dir = out_dir + 'tvs/'

    # Create sub directory is dont exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create weights directory if don't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    if not os.path.exists(tv_dir):
        os.makedirs(tv_dir)

    # define model parameters of CNN layers
    params = {'FILTER_SIZE': (1, 3, 5, 7, 9), 'NUM_FILTERS_CONV': (32, 32, 32, 32, 32)}
    print(params)

    NUM_FILTERS_CONV1 = params['NUM_FILTERS_CONV'][0]
    NUM_FILTERS_CONV2 = params['NUM_FILTERS_CONV'][1]
    NUM_FILTERS_CONV3 = params['NUM_FILTERS_CONV'][2]
    NUM_FILTERS_CONV4 = params['NUM_FILTERS_CONV'][3]
    NUM_FILTERS_CONV5 = params['NUM_FILTERS_CONV'][4]
    FILTER_SIZE1 = params['FILTER_SIZE'][0]
    FILTER_SIZE2 = params['FILTER_SIZE'][1]
    FILTER_SIZE3 = params['FILTER_SIZE'][2]
    FILTER_SIZE4 = params['FILTER_SIZE'][3]
    FILTER_SIZE5 = params['FILTER_SIZE'][4]

    Feature = 'Melspecs'  # or 'MFCCs'
    if Feature == 'MFCCs':
        No_Feats = 13
    elif Feature == 'Melspecs':
        No_Feats = 40

    # No_Feats = 13
    No_TVs = 6
    LR = 1e-3
    DROPOUT_PROB = 0.3
    EPOCHS = 500
    PATIENCE = 10
    BATCH_SIZE = 32
    TIME_STEPS = 200

    if Feature == 'MFCCs':
        # To use MFCCs as inputs
        X_tr = np.load('data/Train_files/x_train_200_13mfcc_postpad.npy')
        X_val = np.load('data/Train_files/x_val_200_13mfcc_postpad.npy')
        X_te = np.load('data/Train_files/x_test_200_13mfcc_postpad.npy')

        Y_tr = np.load('data/Train_files/y_train_200_13mfcc_postpad.npy')
        Y_val = np.load('data/Train_files/y_val_200_13mfcc_postpad.npy')
        Y_te = np.load('data/Train_files/y_test_200_13mfcc_postpad.npy')

    elif Feature == 'Melspecs':
        # To use MelSpecs as inputs
        X_tr = np.load('data/Train_files/x_train_200_melspec_postpad_new.npy')
        X_val = np.load('data/Train_files/x_val_200_melspec_postpad_new.npy')
        X_te = np.load('data/Train_files/x_test_200_melspec_postpad_new.npy')

        Y_tr = np.load('data/Train_files/y_train_200_melspec_postpad_new.npy')
        Y_val = np.load('data/Train_files/y_val_200_melspec_postpad_new.npy')
        Y_te = np.load('data/Train_files/y_test_200_melspec_postpad_new.npy')

    accs = fine_tune(X_tr, Y_tr, X_te, Y_te, X_val, Y_val)

    print('Test set MSE:', accs)



