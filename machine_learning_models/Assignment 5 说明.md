# Assignment 5 说明

### 单一模型的预测指标（KFold=5）

其中Baseline中使用的LightGBM模型处理的数据包含'NaN'并且经历了自动化调参，TabNet模型处理的数据中对空白数据经历了数据预填充，填充策略是补0，其它策略有取平均值、中位数和众数，但补0得到的Accuracy_score值最大。DeepFM模型处理的数据也提前对空白数据进行了相同的数据预填充。

| 模型名称       | LightGBM (Baseline, fine tuning) | TabNet  | DeepFM  | CatBoost          |
| -------------- | -------------------------------- | ------- | ------- | ----------------- |
| Accuracy_score | 0.91762                          | 0.91636 | 0.89364 | 0.91764 (Better!) |



### 多模型堆叠 Stacking

多模型堆叠是基于交叉验证的，需要两个阶段的模型，阶段1模型基于交叉验证，阶段2模型基于阶段1模型特征提取得到的新的特征训练集合。在第一阶段组合为LightGBM和XGBoost时，第二阶段分类模型采用SVM （候选分类器有Logistic Regression, SVM, AdaBoost, Naive Bayes, Random Forest），得到最大Accuracy_score。

| 包含模型       | Stage 1: LightGBM+XGBoost, Stage 2: SVM | Stage 1: LightGBM+XGBoost+TabNet, Stage 2: AdaBoost |
| -------------- | --------------------------------------- | --------------------------------------------------- |
| Accuracy_score | 0.91752                                 | 0.91712                                             |



### 投票机制 Voting

对于单一模型预测，可以使用硬投票 (hard vote)，即少数服从多数的算法进行加权。常规投票机制可以分为三种：少数服从多数（平局时随机指定）、置特定值、置随机值。在本数据集中软投票 (soft vote) 的表现大多不如硬投票 (hard vote)。当堆叠模型为LightGBM+CatBoost+Soft Voting（候选分类器为Logistic Regression, SVM, Gaussian Bayes）时，预测指标最优。

| 包含模型       | TabNet+LightGBM                                    | Stage 1: LightGBM+XGBoost, Stage 2: Hard Voting | Stage 1: LightGBM+CatBoost, Stage 2: Soft Voting |
| -------------- | -------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| Accuracy_score | 0.91712                                            | 0.91752                                         | *<u>0.91826 (Best!)</u>*                         |
| **包含模型**   | **Stage 1: LightGBM+TabNet, Stage 2: Soft Voting** | **Stage 1: LightGBM, Stage 2: Soft Voting**     | **Stage 1: CatBoost, Stage 2: Voting**           |
| Accuracy_score | 0.91726                                            | 0.9178 (Better!)                                | 0.91816 (Better!)                                |



### 多模型混合 Blending

多模型混合与多模型堆叠类似，但设置了留出集作为特定的验证集，降低了过拟合的可能。

| 包含模型       | Stage 1: Logistic Regression+Decision Tree+Naive Bayes, Stage 2: SVM |
| -------------- | ------------------------------------------------------------ |
| Accuracy_score | 0.913                                                        |



### 总结

通过上述分析，最终优化模型为CatBoost+Voting组合的堆叠模型，与Baseline相比，预测指标Accuracy_score相对上升了0.07%。

| 最终模型       | LightGBM (Baseline) | CatBoost+LightGBM+Soft Voting |
| -------------- | ------------------- | ----------------------------- |
| Accuracy_score | 0.91762             | 0.91826 (Best!)               |

