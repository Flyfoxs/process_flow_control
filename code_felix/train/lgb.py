from lightgbm import LGBMRegressor
from code_felix.feature.read_file import *
from code_felix.train.Kfold import learning

def get_model(n_estimators=200, num_leaves=31):
    gbm = LGBMRegressor(
        #n_estimators=100,
        #learning_rate=0.08, gamma=0, subsample=0.75,
        #colsample_bytree=1,
        #max_depth=7,
        #boosting_type= 'gbdt',
        objective= 'regression_l2',
        metric = [ 'rmse'],
        verbose = -1,
        num_leaves=num_leaves,

        # eval_metric="merror",
        #eval_metric='rmse',
        # booster='gblinear',
        # max_depth=3,
        # reg_alpha=10,
        # reg_lambda=10,
        # subsample=0.7,
        # colsample_bytree=0.6,
        n_estimators=n_estimators,
        #
        # learning_rate=0.01,
        #
        # seed=1,
        # missing=None,
        #
        # # Useless Paras
        # silent=True,
        # gamma=0,
        # max_delta_step=0,
        # min_child_weight=1,
        # colsample_bylevel=1,
        # scale_pos_weight=1,

    )
    return gbm


def train(n_estimators=200, num_leaves=31, gap_hours=4):
    args = locals()
    try:
        for label_name in label_name_list:
            #label_name = 'phosphorus_content'

            model = get_model(n_estimators, num_leaves)


            train = get_train(gap_hours)
            test = get_test(gap_hours)

            label = get_label(label_name)

            loss = learning(model, train, label, test, label_name )
            logger.debug(f"The Loss for {label_name}:{'{:,.6f}'.format(loss)} {args} ")
    except Exception as e:
        logger.error(f"Error is happen with {args}")
        logger.error(e)


if __name__ == '__main__':

    for n_estimators in [100, 200, 300]:
        for num_leaves in [31, 63, 70, 80, 100]:
            for gap_hours in [4, 5, 6]:
                train(n_estimators, num_leaves, gap_hours)
