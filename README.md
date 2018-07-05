#### Practicing Machine Learning

##### Introduction

The goal of this repository is to implement machine learning algorithms as taught in the 
[Coursera Machine Learning course by Andrew Ng](https://www.coursera.org/learn/machine-learning). In order 
to focus on the *machine learning* aspect, I chose not not to spend time
on the (very important) step of *feature engineering*. Instead I use publicly available code to 
pre-process the raw data before submitting it to my algorithms. The source for any data wrangling 
code I use is available in the *data_clean_name.py* files.
 
To test my code I use various Kaggle *getting started* prediction competitions:

* Logistic Regression: [Kaggle Titanic Survival Prediction](https://www.kaggle.com/c/titanic)

<br>


##### Logistic Regression

My goal was to implementation logistic regression using **regularized Gradient Descent**. Further, In order 
to evaluate the algorithm I implemented:
 
 * a **cost history curve** to ensure that cost correctly converges to a minimum 
 * a **learning curve** to analyze the algorithm in terms of *bias* and *variance*

<br>

##### Results

* Titanic Logistic Regression: 0.75 accuracy and leaderboard position 9779 (as of 07/2018)

<br>
##### Some Notes on Regularization

* Regularization is applient to computation of gradients **and** of the cost. 
Beware, the formulas are different!

* *L2 regularization* refers to regularization using the sum of the **square** of the weight coefficients.
 

* The regularization term penalizes large parameters.

* each time some parameter is updated to become significantly large,
  * it will increase the value of the cost function by the regularization term, and as a result
  * it will be penalized and updated to a small value.

<br>

##### Further Reading

* Logistic Regression
  * [Logistic Regression Essentials in R](http://www.sthda.com/english/articles/36-classification-methods-essentials/151-logistic-regression-essentials-in-r/#logistic-function)
  * [Fitting a logistic regression model to the iris data set](http://wilkelab.org/classes/SDS348/2015_spring_worksheets/class11_solutions.html)
  * [Notes for Machine Learning - Week 3](https://www.yuthon.com/2016/08/05/Coursera-Machine-Learning-Week-3/)
  * 

* Regularization
  * [Chun's Machine Learning Page](https://chunml.github.io/ChunML.github.io/tutorial/Regularization/)
* Learning Rate
  * [Understanding Learning Rates and How It Improves Performance in Deep Learning](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)
  * [Debugging bias-variance with learning curves](https://github.com/arturomp/coursera-machine-learning-in-python/blob/master/bias-variance-learning-curves.ipynb) 
  (used this page to design logistic regression learning curve implementation) 

* Data Wrangling
  * [Kaggle Notebook](https://www.kaggle.com/netssfy/learning-curve) (used for feature engineering)
  * [Is standardization needed before fitting logistic regression?](https://stats.stackexchange.com/questions/48360/is-standardization-needed-before-fitting-logistic-regression)
.
