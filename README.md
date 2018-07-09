### Practicing Machine Learning: Logistic Regression

#### Introduction

The goal of this repository is to implement **logistic regression** without using 
off-the-shelf machine learning libraries. My DIY implementation of logistic regression is based 
on the 
[Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) 
course by Stanford University. The dataset was sourced from the 
[Kaggle Titanic](https://www.kaggle.com/c/titanic) competition.

**A note on Feature Engineering**

In order to focus on the machine learning aspect, I chose not not to spend time
on the (very important) step of feature engineering. Instead I use publicly available code to 
pre-process the raw data before submitting it to my algorithms. 
The source for any data wrangling code I use is available in the *data_clean_name.py* files.
 
<br>


#### Implementation Notes

I immplemented logistic regression using *regularized gradient descent*. 

To evaluate the algorithm I implemented:
 
 * a *cost history curve* function to check whether cost converges to a minimum 
 * a *learning curve* to analyze the algorithm in terms of *bias* and *variance*

<br>

#### Titanic Survival predictions scored by Kaggle

* hyper-parameters: 
  * num iterations = 1000
  * learning rate = 0.3 
  * regularization term = 1.1
  
* Results: 
  * 0.75 accuracy 
  * leaderboard position 9779 (as of 07/2018)

 

<br>

#### Notes

* Feature normalization is **not** required for logistic regression.

* Regularization is applied to the computation of gradients **and** to the computation of 
the cost. 

* *L2 regularization* refers to regularization using the sum of the **square** of the weight coefficients.
 
* The regularization term penalizes large parameters. Each time some parameter is updated 
  to become significantly large the (large) parameter will increase the value of the cost 
  function by the regularization term. For this reason a reg. term is used in order to 
  penalize the parameter and update it to a small value.

<br>

#### Further Reading

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

