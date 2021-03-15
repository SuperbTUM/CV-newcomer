# reference: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import mean_squared_error
from hyperopt import hp, Trials, fmin, tpe
from sklearn.metrics import roc_auc_score
import math
import warnings
warnings.filterwarnings('ignore')


class lgbm_demo:
    def __init__(self, objective, file_path):
        self.objective = objective
        self.fixed_params = {'metric': {'l2', 'l1'},
                             'feature_fraction': 0.9,
                             'bagging_fraction': 0.8,
                             'bagging_freq': 5,
                             'verbose': 0}
        train = file_path + '/train'
        train_ds = pd.read_csv(train, header=None, delimiter='\t')
        validation = file_path + '/test'
        validation_ds = pd.read_csv(validation, header=None, delimiter='\t')
        # get labels
        self.t_label = train_ds[-1]
        self.t_data = train_ds.drop(train_ds[-1], axis=1)
        self.train_set = lgb.Dataset(self.t_data, self.t_label)
        self.v_label = validation_ds[-1]
        self.v_data = validation_ds.drop(validation_ds[-1], axis=1)
        self.validation_set = lgb.Dataset(self.v_data, self.v_label, reference=self.train_set)

    def build(self):
        def fn(params):
            model = self.train({**self.fixed_params, **params})
            pred = self.predict(gbm=model)
            return 1 - roc_auc_score(pred, self.v_label)

        # first
        learning_rate = 0.05
        num_leaves = 31
        max_depth = 6
        gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective=self.objective,
                                 learning_rate=learning_rate,
                                 num_leaves=num_leaves,
                                 max_depth=max_depth,
                                 n_estimators=100,
                                 n_jobs=1)
        step_for_leaves_num = 20
        step_for_depth = 5

        while step_for_depth <= 2 and step_for_leaves_num <= 5:
            if step_for_depth <= 2:  # fixed max depth
                candidate = {'num_leaves': [num_leaves + step_for_leaves_num * i for i in range(-1, 2, 1)],
                             'max_depth': max_depth}
            elif step_for_leaves_num <= 5:
                candidate = {'num_leaves': num_leaves,
                             'max_depth': [max_depth + step_for_depth * i for i in range(-1, 2, 1)]}
            else:
                candidate = {'num_leaves': [num_leaves + step_for_leaves_num * i for i in range(-1, 2, 1)],
                             'max_depth': [max_depth + step_for_depth * i for i in range(-1, 2, 1)]}
            gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=3)
            gsearch.fit(self.t_data, self.t_label)
            num_leaves = gsearch.best_params_['num_leaves']
            max_depth = gsearch.best_params_['max_depth']
            if step_for_depth > 2:
                step_for_depth = math.ceil(step_for_depth // 2)
            if step_for_leaves_num > 5:
                step_for_leaves_num = step_for_leaves_num // 2

        space_dtree = {
            'num_leaves': hp.choice('num_leaves', [num_leaves + step_for_leaves_num * i for i in range(-1, 2, 1)]),
            'max_depth': hp.choice('max_depth', [max_depth + step_for_depth * i for i in range(-1, 2, 1)])
        }
        best1 = fmin(fn=fn, space=space_dtree, algo=tpe.suggest, max_evals=1000, trials=Trials(), verbose=True)
        # second
        params_corr = {'boosting_type': 'gbdt',
                       'objective': self.objective,
                       'learning_rate': learning_rate,
                       'num_leaves': best1['num_leaves'],
                       'max_depth': best1['max_depth']
                       }
        step_for_learning_rate = 2
        while step_for_learning_rate <= 0.1:
            candidate = {'learning_rate': [learning_rate * (2 ** i) for i in range(-1, 2, 1)]}
            gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=3)
            gsearch.fit(self.t_data, self.t_label)
            step_for_learning_rate = step_for_learning_rate / 2
            learning_rate = gsearch.best_params_['learning_rate']

        params_corr['learning_rate'] = gsearch.best_params_['learning_rate']
        return params_corr

    def train(self, params):
        print('Starting training...')
        # train
        gbm = lgb.train(params,
                        self.t_data,
                        num_boost_round=20,
                        valid_sets=self.t_label,
                        early_stopping_rounds=5)

        print('Saving model...')
        # save model to file
        gbm.save_model('model.txt')
        return gbm

    def predict(self, gbm):
        print('Starting predicting...')
        # predict
        pred = gbm.predict(self.v_data, num_iteration=gbm.best_iteration).astype(int)
        print('Mean square error is ', mean_squared_error(pred, self.v_label) ** 0.5)


if __name__ == '__main__':
    file_path = ""
    lgb_ = lgbm_demo('binary', file_path)
    params_corr = lgb_.build()
    params_corr.update(lgb_.fixed_params)
    gbm = lgb_.train(params_corr)
    # gbm = lgb.Booster(model_file='model.txt')
    lgb_.predict(gbm)
