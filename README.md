# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains panel data about clients marketing characteristics. It is required to predict if a certain client will make a fixed term deposit. In other words, it is a binary classification problem, as the input values are marketing characteristics and the target output is 1 or 0, i.e “yes” or “no” prediction. The set comes in a tabular format with rows containing information about specific clients and columns being marketing characteristics.

Proposed pipeline optimization can be broken down into several parts. First, Python script is used to clean the dataset and create a baseline model. Second, Notebook environment is used to run Azure hyperdrive functionality to optimize hyper-parameters of the baseline model. After that, AutoML is used to find even more accurate model.
![](https://github.com/allaccountstaken/optimizing_pipeline/blob/master/img/Screen%20Shot%202021-01-10%20at%202.06.56%20PM.png)
Overall, VotingEnsemble was the best performing model based on testing accuracy score of  0.917. Important to note, the model was evaluated in-sample only after random train-test-split and class imbalances were not corrected for. Out-of-sample regime changes and data drifts will likely jeopardize the model performance. 


## Scikit-learn Pipeline
Linear logistic regression was fit on cleaned training data to produce the best prediction as measured by testing accuracy. Proposed design with isolated model logic in `train.py` requires a preliminary run of the script in order to fit the model, i.e. instantiate and run `LogisticRegression` object. Only once the model is fitted, training accuracy becomes available because it is a computed value returned from `model.predict()`. Baseline accuracy of the first run was **0.913**.
![](https://github.com/allaccountstaken/optimizing_pipeline/blob/master/img/Screen%20Shot%202021-01-10%20at%207.31.00%20AM.png)
Hyperdrive allows to vary hyper-parameters of the model. Regularization parameter was randomly sampled from (0, 1) uniform space and iterations were randomly selected from the following range: 50, 100, 150, 200, 250. This approach provided a good coverage of possible parameters.

Bandit policy was employed for early termination based on a slack factor of 0.1, evaluation interval of 1, and a delay evaluation of 5. Policy for preventive termination of unsuccessful runs is important to control for resource allocation and time.

The best model achieved testing accuracy of **0.916** with regularization of approximately 0.5 and 100 iterations.
![](https://github.com/allaccountstaken/optimizing_pipeline/blob/master/img/Screen%20Shot%202021-01-10%20at%207.31.30%20AM.png)

## AutoML
Automated machine learning takes fundamentally different approach, as it varies functional form of the model. In other words, this tool goes through a library of binary classifiers and fits the data iteratively. 

As expected, the best performance was achieved by a voting ensemble model with testing accuracy of **0.917**.
![](https://github.com/allaccountstaken/optimizing_pipeline/blob/master/img/Screen%20Shot%202021-01-10%20at%207.32.01%20AM.png)
## Pipeline comparison
As stated above, two approached are not directly comparable because they follow different knowledge discovery paths. 

Important, varying regularization of a logistic regression only changes the bias of the selected model, not the model itself. Therefore, hyperdrive approach starts with assumptions about appropriate functional form of the best model that are presumably supported by literature review or empirical evidence. 

AutoML, on the other hand, simply iterates through generally appropriate models, in this case, minimizing entropy at each step. Unsurprisingly, an ensemble of voting or stacked models gives better accuracy with a tradeoff of potentially diminished explainability. 


## Future work
Limitation of this study stems from the original problem definition and a much simplified dataset. 

Decision to place a fixed term deposit is likely time-variant and driven by dynamic utility curves. Time series forecast is more appropriate as there could be strong seasonality and trend effects. Moreover, macroeconomic factors, such as growth and interest rates, as well as fixed term deposit rates are important lurking variables. 

A better model will not necessarily have a better testing accuracy, but will likely perform better out-of-sample. 
