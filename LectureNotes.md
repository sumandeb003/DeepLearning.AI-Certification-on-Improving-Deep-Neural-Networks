1.  In the previous era of machine learning, it was common practice to take all your data and split it according to maybe a 70/30%....people often talk about the 70/30 train test splits. If you don't have an explicit dev set or maybe a 60/20/20% split, in terms of 60% train, 20% dev and 20% test. And several years ago, this was widely considered best practice in machine learning. If you have here maybe 100 examples in total, maybe 1000 examples in total, maybe after 10,000 examples, these sorts of ratios were perfectly reasonable rules of thumb. But in the modern big data era, where, for example, you might have a million examples in total, then the trend is that your dev and test sets have been becoming a much smaller percentage of the total. Because remember, the goal of the dev set or the development set is that you're going to test different algorithms on it and see which algorithm works better. So the dev set just needs to be big enough for you to evaluate, say, two different algorithm choices or ten different algorithm choices and quickly decide which one is doing better. And you might not need a whole 20% of your data for that. So, for example, if you have a million training examples, you might decide that just having 10,000 examples in your dev set is more than enough to evaluate, you know, which one or two algorithms does better. And in a similar vein, the main goal of your test set is, given your final classifier, to give you a pretty confident estimate of how well it's doing. And again, if you have a million examples, maybe you might decide that 10,000 examples is more than enough in order to evaluate a single classifier and give you a good estimate of how well it's doing. So, in this example, where you have a million examples, if you need just 10,000 for your dev and 10,000 for your test, your ratio will be more like...this 10,000 is 1% of 1 million, so you'll have 98% train, 1% dev, 1% test. And I've also seen applications where, if you have even more than a million examples, you might end up with, you know, 99.5% train and 0.25% dev, 0.25% test. Or maybe a 0.4% dev, 0.1% test.
2.  Assuming:
     - Bayes error is quite small
     - Training and validation sets are drawn from the same distribution

    We can define **$\color{red}{\textbf{bias}}$** and **$\color{red}{\textbf{variance}}$** problems as:
     - **High variance => Poor generalizability => Big change in accuracy from training set to test/validation set**
     - Variance is the change in the model's performance (accuracy) with changes in the dataset.
     - In statistics, we don't want the model's performance (accuracy) to change much with little changes in the training dataset. 
     - **High bias => Low Accuracy on training set => Underfitting (poor fitting) the training data by being an oversimplified model (e.g. linear instead of being curve, say, quadratic etc)**
     - In statistics, bias is calculated as the average squared distance between the actual data points and the corresponding points on the learned function (model). Bias is a measure of the inability of the model to capture/match/reproduce the true relationship. High bias leads to loss of training accuracy.
       
3. **Bias-Variance Tradeoff**:
     - To reduce bias, we try to match the learned function with the true/real relationship. In doing so, we end up matching the learned function with the extra peculiarities/complexities of the training data i.e. overfitting the training data. This increases the variance of the model.
     - To reduce variance, we try to reduce the complexity of the learned function. This leads to a simplified model which underfits or poorly fits the training data, thereby, increasing bias.

4. **High Bias (training accuracy) --> Bigger NN (More hidden layers, more hidden units per layer), Training longer --> Lower Bias**
5. **High Variance (validation/test accuracy) --> More training data, Regularization --> Lower Variance**
6. **Bigger network has more parameters (weights) and hence, more flexibility or capacity to learn. This means increased probability of overfitting.**
7. **A weight is the quantification or numerical value of the importance given to an input feature (in case of logistic regression) or neuronal connection (in case of NNs).** High weight means an input has high importance. For example, consider the models: $2x+b$, $5x+b$, $9x+b$. Here, the coefficients of $x$ are the weights. ***Higher the weight or coefficient, more the feature value will get magnified and hence, more the model's behavior will vary with changes in the feature.*** This can be seen from the following figure. When the weight is high, the model's behavior follows the variations in the corresponding input feature more strongly. This is what happens in overfitting. To reduce overfitting, the weights need to be reduced.

![./Overfitting.png](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/main/Overfitting.png)

8. **Regularization is the process of reducing the weights of the model to prevent overfitting.**
9. To regularize the NN model *i.e.*, to reduce the weights, a term containing the weights is added to the loss function. This term is called the regularization constant. Minimizing the loss function during training also minimizes the weights. If the weights go up, the loss value goes up. Training brings down the loss value.
10. Two types of regularization terms are possible:
     - L1: $\frac{\lambda}{2m}\Sigma$
     - L2: $(\frac{\lambda}{2m}\Sigmaw^2$
12. Types of Regularization: L1, L2, Dropout, Early Stopping training
13. 
