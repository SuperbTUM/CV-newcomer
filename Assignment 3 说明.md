# Assignment 3 说明

在常规的lightgbm算法下，使用自动化调参，得到的预测结果为：

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



删除特征`continuous_dti_joint`后，可以看到Mean squared error进一步下降，耗时也大幅减少。

| Best max depth | Best num of leaves | Best num of estimator | Best learning rate | Mean squared error | Running time              |
| -------------- | ------------------ | --------------------- | ------------------ | ------------------ | ------------------------- |
| 3              | 7                  | 110                   | 0.1                | 0.0819             | 73.64432835578918 seconds |



额外的，提供一种新的思路：如果不对原数据集的特征进行删减，只是在增加新特征`payment_ratio`的前提下，将onehot编码重构为特征哈希，可以看到Mean squared error再进一步下降。

| Best max depth | Best num of leaves | Best num of estimator | Best learning rate | Mean squared error | Running time              |
| -------------- | ------------------ | --------------------- | ------------------ | ------------------ | ------------------------- |
| 3              | 7                  | 110                   | 0.1                | 0.08174            | 93.75345778465271 seconds |