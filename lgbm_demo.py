# reference: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import mean_squared_error


def load_dataset(file_path: str):
    train = file_path + '/train'
    train_ds = pd.read_csv(train, header=None, delimiter='\t')
    validation = file_path + '/test'
    validation_ds = pd.read_csv(validation, header=None, delimiter='\t')
    # get labels
    t_label = train_ds[-1]
    t_data = train_ds.drop(train_ds[-1], axis=1)
    train_set = lgb.Dataset(t_data, t_label)
    v_label = validation_ds[-1]
    v_data = validation_ds.drop(validation_ds[-1], axis=1)
    validation_set = lgb.Dataset(v_data, v_label, reference=train_set)
    return train_set, validation_set


def build(t_data, t_label):
    # first
    learning_rate = 0.05
    num_leaves = 31
    max_depth = 6
    gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='regression',
                             learning_rate=learning_rate,
                             metrics={'l2', 'l1'},
                             num_leaves=num_leaves,
                             feature_fraction=0.9,
                             bagging_fraction=0.8,
                             bagging_freq=5,
                             verbose=0,
                             max_depth=max_depth)
    step_for_leaves_num = 10
    step_for_depth = 2

    while step_for_depth <= 1 and step_for_leaves_num <= 1:
        candidate = {'num_leaves': [num_leaves + step_for_leaves_num * i for i in range(-1, 2, 1)],
                     'max_depth': [max_depth + step_for_depth * i for i in range(-1, 2, 1)]}
        gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=3)
        gsearch.fit(t_data, t_label)
        num_leaves = gsearch.best_params_['num_leaves']
        max_depth = gsearch.best_params_['max_depth']
        step_for_depth = step_for_depth // 2
        step_for_leaves_num = step_for_leaves_num // 2

    # second
    params_corr = {'boosting_type': 'gbdt',
                   'objective': 'regression',
                   'learning_rate': learning_rate,
                   'metrics': {'l2', 'l1'},
                   'num_leaves': gsearch.best_params_['num_leaves'],
                   'feature_fraction': 0.9,
                   'bagging_fraction': 0.8,
                   'bagging_freq': 5,
                   'verbose': 0,
                   'max_depth': gsearch.best_params_['max_depth']
                   }
    step_for_learning_rate = 2
    while step_for_learning_rate <= 0.1:
        candidate = {'learning_rate': [learning_rate * (2 ** i) for i in range(-1, 2, 1)]}
        gsearch = GridSearchCV(gbm, param_grid=candidate, scoring='roc_auc', cv=3)
        gsearch.fit(t_data, t_label)
        step_for_learning_rate = step_for_learning_rate / 2
        learning_rate = gsearch.best_params_['learning_rate']

    params_corr['learning_rate'] = gsearch.best_params_['learning_rate']
    return params_corr


def train(params, lgb_train, lgb_eval):
    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt')


def predict(v_data, v_label, gbm):
    print('Starting predicting...')
    # predict
    pred = gbm.predict(v_data, num_iteration=gbm.best_iteration)
    error = mean_squared_error(v_label, pred) ** 0.5
    return error


if __name__ == '__main__':
    file_path = ""
    train_set, validation_set = load_dataset(file_path)
    params_corr = build(train_set.data, train_set.label)
    train(params_corr, train_set.data, train_set.label)
    gbm = lgb.Booster(model_file='model.txt')
    error = predict(validation_set.data, validation_set.label, gbm)
