# reference: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from hyperopt import hp, Trials, fmin, tpe
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
import logging
import warnings

warnings.filterwarnings('ignore')


class lgbm_demo:
    def __init__(self, objective, file_path):
        logging.basicConfig(level=logging.INFO,
                            filename='info.log',
                            filemode='w',
                            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        self.objective = objective
        self.fixed_params = {'metric': 'l1',
                             'feature_fraction': 0.9,
                             'bagging_fraction': 0.8,
                             'bagging_freq': 5,
                             'verbose': 0}
        train_path = file_path + '/train_final.csv'
        train_df = pd.read_csv(train_path, header=0, delimiter=',')

        train_set, validate_set = train_test_split(train_df, test_size=0.2)
        test_path = file_path + '/test_final.csv'
        test_df = pd.read_csv(test_path, header=0, delimiter=',')
        # create train dataset
        train_set = np.array(train_set)
        self.train_label = train_set[:, 15]  # This should be modified
        self.train_data = np.delete(train_set, 15, axis=1)
        self.train_dataset = lgb.Dataset(self.train_data, self.train_label)
        # create validation dataset
        validate_set = np.array(validate_set)
        self.v_label = validate_set[:, 15]  # This should be modified
        self.v_data = np.delete(validate_set, 15, axis=1)
        self.v_dataset = lgb.Dataset(self.v_data, self.v_label, reference=self.train_dataset)
        # create test dataset
        self.test_label = test_df.loc[:, 'loan_status']  # This should be modified
        self.test_data = test_df.drop('loan_status', axis=1)

    def build(self):
        print("Starting building......")

        def fn(params):
            gbm = lgb.LGBMClassifier(boosting_type=params['boosting_type'],
                                     objective=self.objective,
                                     learning_rate=params['learning_rate'],
                                     num_leaves=params['num_leaves'],
                                     max_depth=params['max_depth'],
                                     n_estimators=params['n_estimators'])
            model = self.fit(gbm)
            pred = model.predict(self.v_data, num_iteration=model.best_iteration_).astype(int)
            return 1 - roc_auc_score(self.v_label, pred)

        # # initial
        print('Starting initialization......')
        learning_rate = 0.05
        num_leaves = 31
        max_depth = 5
        n_estimators = 20
        gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective=self.objective,
                                 learning_rate=learning_rate,
                                 num_leaves=num_leaves,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators)
        # first
        print('Starting searching for best max depth & leave number......')
        candidate = {'max_depth': range(3, 10, 1),
                     'num_leaves': range(80, 100, 5)}
        gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=5, n_jobs=-1)
        gsearch.fit(self.train_data, self.train_label)
        max_depth = gsearch.best_params_['max_depth']
        num_leaves = min(2 ** max_depth - 1, gsearch.best_params_['num_leaves'])
        logging.info('Best max depth is ' + str(max_depth))
        # sophisticated modification
        # space_dtree = {
        #     'boosting_type': 'gbdt',
        #     'objective': self.objective,
        #     'learning_rate': learning_rate,
        #     'max_depth': max_depth,
        #     'num_leaves': hp.choice('num_leaves',
        #                             range(max(num_leaves - 10, max_depth),
        #                                   min(num_leaves + 10, 2 ** max_depth - 1), 1)),
        #     'n_estimators': n_estimators
        # }
        # best1 = fmin(fn=fn, space=space_dtree, algo=tpe.suggest, max_evals=100, trials=Trials(), verbose=True)
        # num_leaves = best1['num_leaves']
        logging.info('Best num of leaves ' + str(num_leaves))
        # update model
        gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective=self.objective,
                                 learning_rate=learning_rate,
                                 num_leaves=num_leaves,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators)
        # second
        print('Starting searching for best number of estimators......')
        candidate = {'n_estimators': range(10, 120, 10)}
        gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=5, n_jobs=-1)
        gsearch.fit(self.train_data, self.train_label)
        n_estimators = gsearch.best_params_['n_estimators']
        logging.info('Best num of estimator ' + str(n_estimators))
        # update model
        gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective=self.objective,
                                 learning_rate=learning_rate,
                                 num_leaves=num_leaves,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators)
        # third
        print('Starting searching for best learning rate......')
        step_for_learning_rate = 2
        while step_for_learning_rate > 0.05:
            candidate = {'learning_rate': [learning_rate * (2 ** i) for i in range(-1, 2, 1)]}
            gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=5, n_jobs=-1)
            gsearch.fit(self.train_data, self.train_label)
            step_for_learning_rate = step_for_learning_rate / 2
            learning_rate = gsearch.best_params_['learning_rate']
        logging.info('Best learning rate ' + str(learning_rate))
        gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective=self.objective,
                                 learning_rate=learning_rate,
                                 num_leaves=num_leaves,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators)

        print('Building completed!')
        return gbm

    def fit(self, gbm):
        print('Starting training...')
        # train
        gbm.fit(self.train_data, self.train_label,
                eval_set=[(self.v_data, self.v_label)],
                eval_metric='l1', early_stopping_rounds=5
                )

        return gbm

    def predict(self, gbm):
        print('Starting predicting...')
        # predict
        pred = gbm.predict(self.test_data, num_iteration=gbm.best_iteration_).astype(int)
        return mean_squared_error(self.test_label, pred)


if __name__ == '__main__':
    file_path = "./final"
    lgb_ = lgbm_demo('binary', file_path)
    gbm = lgb_.build()
    lgb_.fit(gbm)
    logging.info('Mean squared error is ' + str(lgb_.predict(gbm)))
    # gbm = lgb.LGBMClassifier(objective=lgb_.objective, num_leaves=31,
    #                          learning_rate=0.05, n_estimators=20)
    # gbm.fit(lgb_.train_data, lgb_.train_label,
    #         eval_set=[(lgb_.v_data, lgb_.v_label)],
    #         eval_metric='l1', early_stopping_rounds=5)
    # y_pred = gbm.predict(lgb_.test_data, num_iteration=gbm.best_iteration_)
    # print(mean_squared_error(lgb_.test_label, y_pred))
