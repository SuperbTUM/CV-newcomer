# reference: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from hyperopt import hp, Trials, fmin, tpe
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import logging
import warnings

warnings.filterwarnings('ignore')


class lgbm_demo:
    def __init__(self, objective, file_path):
        super(lgbm_demo, self).__init__()
        logging.basicConfig(level=logging.INFO,
                            filename='info.log',
                            filemode='w',
                            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        self.objective = objective
        self.fixed_params = {'metric': 'binary_logloss',
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
        print("Starting building...")

        def fn(params):
            model = self.train({**self.fixed_params, **params})
            pred = self.predict(self.v_data, self.v_label, model) # ???
            return 1 - roc_auc_score(self.v_label, pred)

        # initial
        print('Starting initialization...')
        learning_rate = 0.05
        num_leaves = 31
        max_depth = 6
        n_estimators = 100
        gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective=self.objective,
                                 learning_rate=learning_rate,
                                 num_leaves=num_leaves,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators)
        # first
        print('Starting searching for best max depth & leave number...')
        candidate = {'max_depth': range(3, 10, 1),
                     'num_leaves': range(10, 100, 10)}
        gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=5, n_jobs=-1)
        gsearch.fit(self.train_data, self.train_label)
        num_leaves = gsearch.best_params_['num_leaves']
        max_depth = gsearch.best_params_['max_depth']
        logging.info('Best max depth is %d' % max_depth)
        # sophisticated modification
        space_dtree = {
            'boosting_type': 'gbdt',
            'objective': self.objective,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': hp.choice('num_leaves',
                                    range(max(num_leaves - 10, max_depth),
                                          min(num_leaves + 10, 2 ** max_depth - 1), 1)),
            'n_estimators': n_estimators
        }
        best1 = fmin(fn=fn, space=space_dtree, algo=tpe.suggest, max_evals=100, trials=Trials(), verbose=True)
        num_leaves = best1['num_leaves']
        logging.info('Best num of leaves %d' % num_leaves)
        # update model
        gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective=self.objective,
                                 learning_rate=learning_rate,
                                 num_leaves=num_leaves,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators)
        # second
        print('Starting searching for best number of estimators...')
        candidate = {'n_estimators': range(50, 150, 10)}
        gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=5, n_jobs=-1)
        gsearch.fit(self.train_data, self.train_label)
        n_estimators = gsearch.best_params_['n_estimators']
        logging.info('Best num of estimator %d' % n_estimators)
        # update model
        gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective=self.objective,
                                 learning_rate=learning_rate,
                                 num_leaves=num_leaves,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators)
        # third
        print('Starting searching for best learning rate...')
        step_for_learning_rate = 2
        while step_for_learning_rate > 0.1:
            candidate = {'learning_rate': [learning_rate * (2 ** i) for i in range(-1, 2, 1)]}
            gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=5, n_jobs=-1)
            gsearch.fit(self.train_data, self.train_label)
            step_for_learning_rate = step_for_learning_rate / 2
            learning_rate = gsearch.best_params_['learning_rate']
        logging.info('Best learning rate %.3f' % learning_rate)
        params_corr = {'boosting_type': 'gbdt',
                       'objective': self.objective,
                       'learning_rate': learning_rate,
                       'num_leaves': num_leaves,
                       'max_depth': max_depth,
                       'n_estimators': n_estimators
                       }
        try:
            params_corr['learning_rate'] = gsearch.best_params_['learning_rate']
        except Exception:
            warnings.warn('Learning rate not tuned.', RuntimeWarning)
        print('Building completed!')
        return params_corr

    def train(self, params):
        print('Starting training...')
        # train
        gbm = lgb.train(params,
                        self.train_dataset,
                        num_boost_round=20,
                        valid_sets=self.v_dataset,
                        early_stopping_rounds=5)

        return gbm

    def predict(self, data, label, gbm):
        print('Starting predicting...')
        # predict
        # if gbm comes from train, then prediction will be probability
        # otherwise prediction will be pure classification
        pred = gbm.predict(data, num_iteration=gbm.best_iteration)
        pred = list(map(lambda x: 1 if x >= 0.5 else 0, pred))
        logging.info('Accuracy score for final prediction is %.5f' % accuracy_score(label, pred))
        return pred


if __name__ == '__main__':
    file_path = "./final/final"
    lgb_ = lgbm_demo('binary', file_path)
    params_corr = lgb_.build()
    params_corr.update(lgb_.fixed_params)
    gbm = lgb_.train(params_corr)
    final_pred = lgb_.predict(lgb_.test_data, lgb_.test_label, gbm)
