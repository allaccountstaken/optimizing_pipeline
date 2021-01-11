# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains panel data about clients marketing characteristics. It is required to predict if a certain client will make a fixed term deposit. In other words, it is a binary classification problem, as the input values are marketing characteristics and the target output is 1 or 0, i.e “yes” or “no” prediction. The set comes in a tabular format with rows containing information about specific clients and columns being marketing characteristics.

Proposed pipeline optimization can be broken down into several parts. First, Python script is used to clean the dataset and create a baseline model. Second, Notebook environment is used to run Azure hyperdrive functionality to optimize hyper-parameters of the baseline model. After that, AutoML is used to find an even more accurate model. Present research report discusses these steps in  more detail.

![](https://github.com/allaccountstaken/optimizing_pipeline/blob/master/img/Screen%20Shot%202021-01-10%20at%202.06.56%20PM.png)

Overall, VotingEnsemble was the best performing model based on testing accuracy score of  **0.91745**. Important to note, the model was evaluated in-sample only after random train-test-split and class imbalances were not corrected for. Out-of-sample regime changes and data drifts will likely jeopardize the model performance. 


## Scikit-learn Pipeline
Linear logistic regression was fit on cleaned training data to produce the best prediction as measured by testing accuracy. Proposed design with isolated model logic in `train.py` requires a preliminary run of the script in order to fit the model, i.e. instantiate and run `LogisticRegression` object. Only once the model is fitted, training accuracy becomes available because it is a computed value returned from `model.predict()`. Baseline accuracy of the first run was **0.91284**.

![](https://github.com/allaccountstaken/optimizing_pipeline/blob/master/img/Screen%20Shot%202021-01-10%20at%207.31.00%20AM.png)

Hyperdrive allows to vary hyper-parameters of the model. Regularization parameter was randomly sampled from (0, 1) uniform space and iterations were randomly selected from the following range: 50, 100, 150, 200, 250. This approach provided a good coverage of possible parameters.

Bandit policy was employed for early termination based on a slack factor of 0.1, evaluation interval of 1, and a delay evaluation of 5. Policy for preventive termination of unsuccessful runs is important to control for resource allocation and time.

The best model achieved testing accuracy of **0.91551** with regularization of approximately 0.5 and 100 iterations.

![](https://github.com/allaccountstaken/optimizing_pipeline/blob/master/img/Screen%20Shot%202021-01-11%20at%206.39.48%20AM.png)

## AutoML
Automated machine learning takes fundamentally different approach, as it varies functional form of the model. In other words, this tool goes through a library of binary classifiers and fits the data iteratively. 

As expected, even better accuracy was achieved by ensemble-type models: voting produced best testing accuracy of **0.91745**, followed by stacked model’s accuracy of **0.9164**.

![](https://github.com/allaccountstaken/optimizing_pipeline/blob/master/img/Screen%20Shot%202021-01-10%20at%203.12.23%20PM.png)

VotingEnsemble, the most accurate model, combines results of several good, but conceptually different models, by allowing several classifiers to “vote” 1 or 0 for every testing observations.  Such approach effectively minimizes weaknesses of contributing models by reducing variance of individual testing errors. This can potentially result in even better overall testing accuracy at cost of explainability.  

![](https://github.com/allaccountstaken/optimizing_pipeline/blob/master/img/Screen%20Shot%202021-01-10%20at%203.10.18%20PM.png)

## Pipeline comparison
As stated above, two approached are not directly comparable because they follow different knowledge discovery paths. 

Important, varying regularization of a logistic regression only changes the bias of the selected model, not the model itself. Therefore, hyperdrive approach starts with assumptions about appropriate functional form of the best model that are presumably supported by literature review or empirical evidence. 

AutoML, on the other hand, simply iterates through generally appropriate models, in this case, minimizing variance of prediction. Unsurprisingly, an ensemble of voting gives better accuracy with a tradeoff of potentially diminished explainability. 

## Future work
Limitation of the proposed approach stems from the original problem definition and a simplified dataset. It is assumed that panel data, basically a snapshot, can be used to model demand for fixed term deposits.Realistically, decision to place a fixed term deposit is likely time-variant and driven by dynamic utility curves. 

Time series forecast is more appropriate as there could be strong seasonality and trend effects. Moreover, macroeconomic factors, such as growth and interest rates, as well as fixed term deposit rates are important lurking variables. For example, year-end bonuses paid to certain employees will likely increase their interested in placing deposits. On the other hand, fiscal easing and lowering of interest rates will make savings less attractive and thus decreases demand for such products.

Alternative model does not necessarily need a better classification accuracy, but should account for the true nature of phenomena and perform better out-of-sample. Using the snapshot dataset possibly commingles data points from different time periods and economic regimes without proper labeling. Time-series forecasting may fall behind in terms of accuracy, but will better deal with data drift.
