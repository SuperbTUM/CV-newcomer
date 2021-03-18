# Assignment 3 说明

在常规的lightgbm算法下，得到的模型主要参数为：

| Best max depth | Best num of leaves | Best num of estimator | Best learning rate | Mean squared error | Running time             |
| -------------- | ------------------ | --------------------- | ------------------ | ------------------ | ------------------------ |
| 3              | 7                  | 110                   | 0.1                | 0.08266            | 95.8901207447052 seconds |



观察数据集特征后，添加新的特征
$$
payment\underline\ ratio=installment*12 \div annual\underline\ inc
$$
得到结果如下，可以看到Mean squared error有略微下降。

| Best max depth | Best num of leaves | Best num of estimator | Best learning rate | Mean squared error | Running time              |
| -------------- | ------------------ | --------------------- | ------------------ | ------------------ | ------------------------- |
| 3              | 7                  | 110                   | 0.1                | 0.0824             | 96.72293996810913 seconds |



删除特征`continuous_dti_joint`后，可以看到Mean squared error进一步下降。

| Best max depth | Best num of leaves | Best num of estimator | Best learning rate | Mean squared error | Running time              |
| -------------- | ------------------ | --------------------- | ------------------ | ------------------ | ------------------------- |
| 3              | 7                  | 110                   | 0.1                | 0.0819             | 73.64432835578918 seconds |