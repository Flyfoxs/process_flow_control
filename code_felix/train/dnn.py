import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

from code_felix.feature.config import label_name_list
from code_felix.feature.read_file import get_train, get_test, get_label
from code_felix.utils_.other import save_result_for_ensemble
from code_felix.utils_.util_log import logger


def learning(model ,Xtrain ,y ,Xtest,label_name, number_of_folds= 5, seed = 777):
    train_index = Xtrain.index
    test_index = Xtest.index

    Xtrain = Xtrain.reset_index(drop=True)
    Xtest  = Xtest.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_train = y

    logger.debug(f'train:{Xtrain.shape}, label:{y.shape}, test:{Xtest.shape}')
    print( 'Model: %s' % model)

    """ Each model iteration """
    train_predict_y = np.zeros((len(y)))
    test_predict_y = np.zeros((Xtest.shape[0]))
    learn_loss = 0.
    """ Important to set seed """

    tmp_model = './output//model/checkpoint/dnn_best_tmp.hdf5'
    check_best = ModelCheckpoint(filepath=tmp_model,
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    early_stop = EarlyStopping(monitor='val_loss', verbose=1,
                               patience=100,
                               )
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                               patience=30, verbose=1, mode='min')

    logger.debug(f'y_train.shape:{y_train.shape}')



    from sklearn.model_selection import KFold
    skf = KFold(n_splits = number_of_folds ,shuffle=True, random_state=seed)
    """ Each fold cross validation """

    for i, (train_idx, val_idx) in enumerate(skf.split(Xtrain)):
        logger.debug(f'Fold#{i + 1}' )
        #print(train_idx)

        history = model.fit(Xtrain.values[train_idx], y[train_idx],
                            validation_data=((Xtrain.values[val_idx],  y[val_idx])),
                            #callbacks=[check_best, early_stop, reduce],
                            batch_size=8,
                            # steps_per_epoch= len(X_test)//128,
                            epochs=2,
                            verbose=1,
                            )

        best_epoch = np.array(history.history['val_loss']).argmin() + 1
        best_score = np.array(history.history['val_loss']).min()


        logger.debug(f"Fold#{i+1} arrive {best_score} at {best_epoch}")

        scoring = model.predict(Xtrain.values[val_idx])
        #bst.predict(data.data, ntree_limit=bst.best_ntree_limit)
        """ Out of fold prediction """
        train_predict_y[val_idx] = scoring
        l_score = mean_squared_error(y[val_idx], scoring)
        learn_loss += l_score
        logger.debug('Fold %d score: %f' % (i + 1, l_score))

        test_predict_y = test_predict_y + model.predict(Xtest.values)

    test_predict_y = test_predict_y / number_of_folds

    avg_loss = round(learn_loss / number_of_folds, 6)
    print('average val log_loss: %f' % (avg_loss))
    """ Fit Whole Data and predict """
    print('training whole data for test prediction...')

    # np.save('./output/xgb_train.np', train_predict_y)
    # np.save('./output/xgb_test.np', test_predict_y)


    ###Save result for ensemble
    train_bk = pd.DataFrame(train_predict_y,
                            index=train_index,
                            columns=[label_name]
                            )

    test_bk = pd.DataFrame(test_predict_y,
                           index=test_index,
                           columns=[label_name]
                           )

    label_bk = pd.DataFrame({'label': y},
                            columns=[label_name],
                            index=train_index,
                            )
    model_name = type(model).__name__
    save_result_for_ensemble(f'kfold_{label_name}_{model_name}_{avg_loss}',
                             label_name=label_name,
                             train=train_bk,
                             test=test_bk,
                             label=label_bk,
                             )


def get_model(input_dim):
    logger.debug(f'The input size for DNN is:{input_dim}')
    dropout= 0.5
    lr =0.0001
    model = Sequential()
    model.add(Dense(int(input_dim*1.5), input_shape=(input_dim,)))

    # model.add(Dropout(dropout))


    # model.add(Dense(100))
    # model.add(LeakyReLU(alpha=0.01))
    # model.add(BatchNormalization())
    # model.add(Dropout(dropout))


    model.add(Dense(1, kernel_initializer='normal'))

    # model.compile(optimizer="sgd", loss="mse")
    adam = Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=adam,
                    #metrics=['categorical_crossentropy'],
                  )
    model.summary()



    return  model



if __name__ == '__main__':
    for label_name in label_name_list:
        # label_name = 'phosphorus_content'

        train = get_train(4)
        test = get_test(4)
        model = get_model(train.shape[1])
        label = get_label(label_name)

        learning(model, train, label, test, label_name)
