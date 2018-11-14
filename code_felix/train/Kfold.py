from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from code_felix.utils_.other import save_result_for_ensemble
from code_felix.utils_.util_log import *
from code_felix.feature.read_file import *

def learning(model ,Xtrain ,y ,Xtest,label_name, number_of_folds= 5, seed = 777):
    train_index = Xtrain.index
    test_index = Xtest.index

    Xtrain = Xtrain.reset_index(drop=True)
    Xtest  = Xtest.reset_index(drop=True)
    y = y.reset_index(drop=True)

    logger.debug(f'train:{Xtrain.shape}, label:{y.shape}, test:{Xtest.shape}')
    print( 'Model: %s' % model)

    """ Each model iteration """
    train_predict_y = np.zeros((len(y)))
    test_predict_y = np.zeros((Xtest.shape[0]))
    learn_loss = 0.
    """ Important to set seed """
    from sklearn.model_selection import KFold
    skf = KFold(n_splits = number_of_folds ,shuffle=True, random_state=seed)
    """ Each fold cross validation """

    for i, (train_idx, val_idx) in enumerate(skf.split(Xtrain)):
        logger.debug(f'Fold#{i + 1}' )
        #print(train_idx)

        model.fit(Xtrain.values[train_idx], y[train_idx],
                  eval_set=[(Xtrain.values[train_idx], y[train_idx]),
                            (Xtrain.values[val_idx],  y[val_idx])
                            ],

                  early_stopping_rounds=50, verbose=True)

        best_epoch, best_score = evaluate_score(model)
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

def evaluate_score(model):
    if isinstance(model, LGBMRegressor):
        best_epoch = model.best_iteration_
        best_score = model.best_score_

    elif isinstance(model, XGBRegressor):
        results = model.evals_result()
        best_epoch = np.array(results['validation_1']['rmse']).argmin() + 1
        best_score = np.array(results['validation_1']['rmse']).min()
    else:
        return None, None
    return best_epoch, best_score


if __name__ == '__main__':

    from code_felix.train.xgb import get_model

    label_name = 'phosphorus_content'

    model = get_model(200)


    train = get_train(4)
    test = get_test(4)

    label = get_label(label_name)




    learning(model, train, label, test, label_name )
