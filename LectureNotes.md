1.  In the previous era of machine learning, it was common practice to take all your data and split it according to maybe a 70/30%....people often talk about the 70/30 train test splits. If you don't have an explicit dev set or maybe a 60/20/20% split, in terms of 60% train, 20% dev and 20% test. And several years ago, this was widely considered best practice in machine learning. If you have here maybe 100 examples in total, maybe 1000 examples in total, maybe after 10,000 examples, these sorts of ratios were perfectly reasonable rules of thumb. But in the modern big data era, where, for example, you might have a million examples in total, then the trend is that your dev and test sets have been becoming a much smaller percentage of the total. Because remember, the goal of the dev set or the development set is that you're going to test different algorithms on it and see which algorithm works better. So the dev set just needs to be big enough for you to evaluate, say, two different algorithm choices or ten different algorithm choices and quickly decide which one is doing better. And you might not need a whole 20% of your data for that. So, for example, if you have a million training examples, you might decide that just having 10,000 examples in your dev set is more than enough to evaluate, you know, which one or two algorithms does better. And in a similar vein, the main goal of your test set is, given your final classifier, to give you a pretty confident estimate of how well it's doing. And again, if you have a million examples, maybe you might decide that 10,000 examples is more than enough in order to evaluate a single classifier and give you a good estimate of how well it's doing. So, in this example, where you have a million examples, if you need just 10,000 for your dev and 10,000 for your test, your ratio will be more like...this 10,000 is 1% of 1 million, so you'll have 98% train, 1% dev, 1% test. And I've also seen applications where, if you have even more than a million examples, you might end up with, you know, 99.5% train and 0.25% dev, 0.25% test. Or maybe a 0.4% dev, 0.1% test.

## Bias-Variance Tradeoff

2. Assuming:
 - Bayes error is quite small
 - Training and validation sets are drawn from the same distribution

3. We can define **$\color{red}{\textbf{bias}}$** and **$\color{red}{\textbf{variance}}$** problems as:
 - **High variance => Poor generalizability => Big change in accuracy from training set to test/validation set**
 - Variance is the change in the model's performance (accuracy) with changes in the dataset.
 - In statistics, we don't want the model's performance (accuracy) to change much with little changes in the training dataset. 
 - **High bias => Low Accuracy on training set => Underfitting (poor fitting) the training data by being an oversimplified model (e.g. linear instead of being curve, say, quadratic etc)**
 - In statistics, bias is calculated as the average squared distance between the actual data points and the corresponding points on the learned function (model). Bias is a measure of the inability of the model to capture/match/reproduce the true relationship. High bias leads to loss of training accuracy.
 - To reduce bias, we try to match the learned function with the true/real relationship. In doing so, we end up matching the learned function with the extra peculiarities/complexities of the training data i.e. overfitting the training data. This increases the variance of the model.
 - To reduce variance, we try to reduce the complexity of the learned function. This leads to a simplified model which underfits or poorly fits the training data, thereby, increasing bias.

4. **High Bias (training accuracy) --> Bigger NN (More hidden layers, more hidden units per layer), Training longer --> Lower Bias**
5. **High Variance (validation/test accuracy) --> More training data, Regularization --> Lower Variance**
6. **Bigger network has more parameters (weights) and hence, more flexibility or capacity to learn. This means increased probability of overfitting.**
7. **A weight is the quantification or numerical value of the importance given to an input feature (in case of logistic regression) or neuronal connection (in case of NNs).** High weight means an input has high importance. For example, consider the models: $2x+b$, $5x+b$, $9x+b$. Here, the coefficients of $x$ are the weights. ***Higher the weight or coefficient, more the feature value will get magnified and hence, more the model's behavior will vary with changes in the feature.*** This can be seen from the following figure. When the weight is high, the model's behavior follows the variations in the corresponding input feature more strongly. This is what happens in overfitting. To reduce overfitting, the weights need to be reduced.

![./Overfitting.png](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/main/Overfitting.png)

## Regularization

**Regularization** is the process of simplifying the model to prevent overfitting. A model can be simplified or a simpler model can be obtained by either reducing the size (no. of neurons) of the NN or reducing the weight of certain neurons in the NN.
9. Types of Regularization: L1, L2, Dropout, More training data, Early Stopping training
10. To regularize the NN model by reducing the weights, a term containing the weights is added to the loss function. Minimizing the loss function during training also minimizes the weights. If the weights go up, the loss value goes up. Training brings down the loss value.
11. Two types of regularization terms are possible:
     - L1: $\frac{\lambda}{2m}\Sigma\vert w \vert$; Here, $\lambda$ is called the regularization constant. It is a hyperparameter that needs to be adjusted via hit and trial. As the value of $\lambda$ is set higher, the weights will more be pushed towards zero to keep the loss value low.
     - L2: $\frac{\lambda}{2m}\Sigma w^2$
12. Instead of choosing either one, both L1 and L2 terms can be added to the loss function.
14. In **Dropout regularization**, a keep probability is assigned to the NN. Based on that probability, the neurons are kept or dropped out during training. If a neuron is dropped out, its input and output connections/weights are not considered/used during forward and backward propagation. To compensate for the decrease in $a$ due to the dropped out input neurons, $Z$ is calculated as $Z=W(a/p_k)+b$, where $p_k$ is the keep probability, in the forward propagation.
16. One can also choose to assign different keep probabilities to different layers in the NN. Usually, lower keep probability will be assigned to a layer with more neurons because fatter layers are more prone to overfitting. A keep probability of 1 can also be assigned to a layer, say a layer with one neuron.
17. With dropout regularization, the loss function is different for different dropouts because some weights and neurons get randomly dropped out during each training sample or batch. So, it is not possible to keep track of the convergence of the loss function as training progresses. As a way around this limitation, one can observe the convergence of the loss function with the keep probability = 1. Once convergence of the loss function is confirmed, one can introduce the dropout regularization and train.
18. Having **more training data** can help get a more generalized model, thereby, preventing overfitting. But getting new data can be expensive, time consuming and difficult. In case of image datasets, one can get more training data simply by flipping the training images or cropping the training images or reorient the training images or adding random distortions to the training images. This won't improve the training as much as adding newly clicked images, but, at least, its an easy, inexpensive way to increase the number of training data.
19. **Early stopping** is stopping the training when the error or cost function dtops decreasing on the validation set.

## Weight Normalization

When training a neural network, one of the techniques to speed up your training is **normalization**. Normalizing the inputs speeds up the NN training.

![Data Before Normalization](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/3b601f62cc34b6243bcd2cfd79c2fc60ca35e059/DataBeforeNormalization.png)

Normalizing your inputs corresponds to two steps:
  - The first is to subtract the mean $\mu$ ($=\frac{1}{n}\Sigma x_i$) from each of the training samples. There will be a separate mean for each feature. Subtract the means to the corresponding features of the training and test samples. This subtraction renders the means of all the features of the samples equal to 0.

![Data After Subtracting Mean from Data](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/3b601f62cc34b6243bcd2cfd79c2fc60ca35e059/DataAfterSubtractingMean.png)

  - Then the second step is to normalize the variances. Obtain the feature-wise standard deviations ($\sigma=\sqrt{variance}$; $variance(\sigma ^2)=\frac{1}{n}\Sigma (x_i-\mu)^2$, here, $\mu=0$) of the samples obtained from step 1. Divide the same samples by their standard deviations to normalize them. This renders the variances of all the features of the training samples equal to 1.

![Data After Normalization](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/3b601f62cc34b6243bcd2cfd79c2fc60ca35e059/DataAfterNormalization.png)

  - Now, use the same mean and std. dev. to normalize the test samples also.

If your features came in on similar scales,e.g. if one feature, say $x_1$ ranges from 0-1 and $x_2$ ranges from minus 1-1, and $x_3$ ranges from 1-2, then normalization is less important although performing this type of normalization pretty much never does any harm. Often you'll do it anyway, if you are not sure whether or not it will help with speeding up training for your algorithm.

**Why does normalization speed up the learning?**

To develop an intuition for why normalization speeds up training, consider why a ball takes longer to reach the bottom in an elongated valley landscape, which is a metaphor for optimizing a cost function without normalized inputs in neural network training.

**The Landscape Metaphor**

Imagine a landscape representing the cost function of a neural network, where:
 - The horizontal dimensions represent the parameters (here, $x$ and $y$) of the model. In case of training using unnormalized data, one of the parameters has a large range while the other has a small range. Let's say, $y$ has a large range while $x$ has a small range.
 - The vertical dimension represents the cost $J$ (or error) associated with those parameters.

The goal of training the neural network is to find the lowest point in this landscape (the global minimum of the cost function), which corresponds to the best set of parameters for the model. 
 - **In case of training using unnormalized data, the shape of the cost function is like a folded page with a base having narrow, thin elliptical shadow. The fold lies along the parameter $y$ with large range. The slope of the cost function is very low along the $y$ direction but very steep along the $x$ direction.**

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/7ef31b1b26302a1e7c8a6f71410a8964bcc186c6/CostFunctionUnnormalizedData.png)

![Narrow Elliptical Base](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/cdcc8fa2f7b60b82f803f757f17b8574d52ebc0d/NarrowEllipticalBase.png)

 - **In case of training using normalized direction, the shape of the cost function is like a bowl with a base having circular shadow**

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/7ef31b1b26302a1e7c8a6f71410a8964bcc186c6/CostFunctionNormalizedData.png)



**Elongated Valley Scenario**

In the scenario of an elongated valley:
 - The valley is much steeper in one direction (axis) than in the other. This difference in steepness represents the disparity in the scale of input features. Some weights (parameters) of the neural network must make large adjustments due to large-scale features, while others make tiny adjustments due to small-scale features.
 - When gradient descent is applied to find the minimum, the steps taken in the direction of the steep sides are larger, potentially overshooting the minimum along the narrow axis. Conversely, the steps in the shallow direction are smaller, making progress slow.
 - This mismatch in step sizes can cause the optimization path to zig-zag and take a longer time to converge to the minimum. The optimization path oscillates back and forth across the narrow axis of the valley because the gradient descent algorithm struggles to balance the progress in different directions due to the uneven scales.

**Why It Takes Longer?**

The key reasons for the longer path to the bottom are:
 - Overshooting and Zig-Zagging: Large steps in the steep direction can overshoot the minimum, requiring corrections that send the optimizer back and forth. The need to correct overshoots and the slow progress in the shallow direction lead to a zig-zagging path, which is less efficient than a straight-line descent.
 - Small Learning Rate Compromise: If the learning rate is reduced to prevent overshooting in the steep direction, progress becomes painfully slow in all directions, further delaying convergence.

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/7ef31b1b26302a1e7c8a6f71410a8964bcc186c6/TrainingOnNon-NormalizedData.png)

To visualize why the ball takes longer, picture trying to manually roll a ball down a narrow, elongated trough with one side significantly steeper than the other:
- If you push the ball too hard (large learning rate), it rapidly oscillates from side to side (overshooting), thereby, taking longer.
- If you push too gently (small learning rate), progress is safe but slow, taking a long time to reach the bottom.
- Achieving just the right push to get the ball to the bottom quickly is much more difficult in this uneven trough compared to a more uniformly shaped bowl.

Training using normalized data is smooth, easy and faster.

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/7ef31b1b26302a1e7c8a6f71410a8964bcc186c6/TrainingOnNormalizedData.png)

## Vanishing Gradient Problem/Exploding Gradient Problem

To understand this problem, we first need to understand gradient and backpropagation. We use a simplified case of a NN with 4 layers and 1 neuron per layer. 

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/6be41e6c8a268f6094a0ef6ca9d02d0426c448ea/BP1.png)

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/7cdbd012bfbb4afe81291afaab871d176ba3e93f/BP2.png)

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/7cdbd012bfbb4afe81291afaab871d176ba3e93f/BP3.png)

During learning via backpropagation, to calculate the weight update for $W_1$, we need $\frac{\partial O}{\partial W_1}$ and for $W_2$, we need $\frac{\partial O}{\partial W_2}$. 

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/7cdbd012bfbb4afe81291afaab871d176ba3e93f/BP4.png)

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/7cdbd012bfbb4afe81291afaab871d176ba3e93f/BP5.png)

**As you can see above, $\frac{\partial O}{\partial W_1}$ is the product of the weights in all the following layers and the gradient (slope) of the activation functions in all the following layers of the NN.**
 - The weights are randomly drawn from a normal or gaussian distribution with mean 0 and standard deviation 1. So, they are all fractions between -1 and +1.
 - If the activation function ($f_a$) is sigmoid, its gradient $\frac{\partial f_a}{\partial \theta}$ (slope of the activation function at a point) is given by $\theta(1-\theta)$. Min and max values of $\frac{\partial f_a}{\partial \theta}$ are 0 and 1/4 (for $\theta=0.5$) respectively. So, its values lie between 0 and 1/4.

So, all the weight terms and the gradient terms in the product $\frac{\partial O}{\partial W_1}$ are fractions less than 1. 

**As the number of layers increases,** the number of fractional terms in this product increases and therefore, **the product** $\frac{\partial O}{\partial W_1}$ **decreases exponentially**. 

For large number of layers, the product becomes vanishingly small. As a result, the weight updates for the initial layers become vanishingly small, causing negligible or no learning in the initial layers. This is known as the **problem of vanishing gradient**.

If the terms in the product are greater than 1, the product $\frac{\partial O}{\partial W_1}$ will increase exponentially with increasing layers. For large number of layers, the gradient $\frac{\partial O}{\partial W_1}$ explodes. As a result, the weight updates will get too large causing oscillation or even divergence during gradient descent. This is known as the **problem of exploding gradient**.

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/5ba6699a30106ee128d3ed2b9bd81fe157fda33c/VanishingGradientProblem.png)

**To reduce the vanishing gradient problem:**
 - Use ReLU activation function. It has a gradient/slope of 1 for $x>0$.
 - We initialize the weights of a neuron in an NN by randomly drawing them from a gaussian distribution with mean 0 but variance $1/N$, where $N$ is the number of input nodes of this neuron. This restricts the weights to be smaller. The reasoning behind this is that larger the number of weights of the neuron, the smaller the individual weights we want to prevent $\Sigma w_i x_i$ from exploding. It turns out for $ReLU$ activation, a variance of $2/N$ works little better. For $tanh$ activation, the variance of $1/N$ works better. This is called **Xavier initialization**. Its code implementation would be: `np.random.randn()*np.sqrt(1/N)`

## Gradient Approximation

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/b56e1966e891b0ca072b0ca97ca4180f24bd5011/GradientApproximation.png)

The derivative (slope) at a point $(f(\theta),\theta)$ lying on the above curve: $f(\theta)=\theta^3$ is given by: 
 - Ratio of height to base of the upper small triangle, *i.e.*, $\frac{f(\theta+\epsilon)-f(\theta)}{\epsilon}$ = $\frac{(\theta+\epsilon)^3 - (\theta)^3}{\epsilon}$ = $\frac{(\theta(1+\frac{\epsilon}{\theta}))^3 - (\theta)^3}{\epsilon}$ = $\frac{\theta^3(1+3\frac{\epsilon}{\theta}-1)}{\epsilon}$ = $3\theta^2$
 - Ratio of height to base of the outer large triangle, *i.e.*, $\frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}$  = $\frac{(\theta+\epsilon)^3 - (\theta-\epsilon)^3}{2\epsilon}$ = $\frac{(\theta(1+\frac{\epsilon}{\theta}))^3 - (\theta(1-\frac{\epsilon}{\theta}))^3}{2\epsilon}$ = $\frac{\theta^3(1+3\frac{\epsilon}{\theta}-1+3\frac{\epsilon}{\theta})}{2\epsilon}$ = $3\theta^2$

The derivaitve of a multi-dimensional cost function $J(\theta_1, \theta_2, \theta_3,....\theta_i,...\theta_n)$ w.r.t a component parameter $\theta_i$ at a training point is approximated as: $\frac{\partial J}{\partial \theta_i}$ = $\frac{J(\theta_1, \theta_2, \theta_3,....\theta_i+\epsilon,...\theta_n) - J(\theta_1, \theta_2, \theta_3,....\theta_i-\epsilon,...\theta_n)}{2\epsilon}$

### Backpropagation

**BACKPROPAGATION = CHAIN RULE** for obtaining derivative (gradient or slope) of the loss function w.r.t to a weight as product of derivatives of activation functions ($\sigma$) and derivatives of activations ($z=wx+b$) in the intermediate layers

The gradient/slope/derivative ($\frac{\partial J}{\partial w_i}$) points in the direction of steepest increase of the loss function ($J$) with respect to the weight($w_i$). To minimize the loss, we move in the opposite direction, hence the subtraction in the update rule

## Batch vs Mini-batch vs Stochastic Gradient Descent (GD)

In batch GD (BGD):
- All the training samples are forward passed through the NN.
- Then, the value of gradient of loss function ($\frac{\partial J}{\partial w_i}$) is calculated (from the above step) over the entire training set.
- A fraction (learning rate, $\eta$) of this gradient is subtracted from the weight to get the new weight: $w_i = w_i - \eta\frac{\partial J}{\partial w_i}$
- So, the weights are updated only once, at the end of the epoch

In mini-batch GD:
- The training set is divided into mini batches.
- All the training samples in a mini-batch are forward passed through the NN.
- Then, the value of gradient of loss function ($\frac{\partial J}{\partial w_i}$) is calculated (from the above step) over the entire training set.
- A fraction (learning rate, $\eta$) of this gradient is subtracted from the weight to get the new weight: $w_i = w_i - \eta\frac{\partial J}{\partial w_i}$
- So, the weights are updated at the end of each mini-batch

**Mini-batch GD is faster than batch GD**

**If size of mini-batch is 1, it is called stochastic GD (SGD).**

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/2eb88bf45d3111e5be2776f9e3f8159325e9ecf6/BatchGDvsMiniBatch%20GD.png)

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/2eb88bf45d3111e5be2776f9e3f8159325e9ecf6/BatchGDvsMiniBatchGDvsStochasticGD.png)

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/e1de84a027824ec2955c838fcd5fd83a09ceca79/SGDvsBGD.png)

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/e1de84a027824ec2955c838fcd5fd83a09ceca79/SGDvsBGDvsminiBGD.png)

**Batch GD is stable while SGD is noisy and oscillatory in traversing to the global minima.**

*In BGD as well as SGD, forward propagation happens as many times as the number of samples in the training set. In BGD, the weights of the network are updated (via backpropagation) once per epoch. In SGD, the weights of the network are updated (via propagation) once for every sample. So, per epoch, SGD updates the weights as many times as the number of samples in the entire training set. The number of weights updates (via backpropagation) is more in case of SGD than in the case of BGD while they have the same number of forward propagations. So, according to this logic, SGD should take longer than BGD for the same number of epochs. But, SGD may achieve lower accuracy than BGD in the same number of epochs due to the oscillatory nature of SGD.* - **Why is SGD considered to be computationally more efficient?**

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/7e1311c424f25ca4e80cc7491c199e08fa938fa4/SGDvsBGDvsMiniGD.png)

1. **Can use batch training if the training set has less than 2000 samples. Else, use mini-batch training.**
2. Due to the way how computer memory is layed out and accessed, mini-batches with sizes of the power of 2 work better. **Mini-batch sizes of $2^6$, $2^7$, $2^8$, $2^9$ are more commonly used in machine learning.**
 - **Also, mini-batch size should be such that it fits in the CPU/GPU memory**. If it doesn't fit, the performance will fall sharply.
 - **Start with a larger mini-batch size and lower if performance is poor.**
3. 
4. 

**Network Architecture**
- **Input Layer**: 2 features
- **Hidden Layer**: 3 neurons (with sigmoid activation)
- **Output Layer**: 1 neuron (with sigmoid activation)

![](https://github.com/sumandeb003/DeepLearning.AI-Certification-on-Improving-Deep-Neural-Networks/blob/c10056f676f24f59aa7b5a33196dd2e63f115d80/3layerNN.png)

**Weights and Biases (Assumed)**

**Weights ($W_{ih}$) from Input to Hidden Layer :** 
```math
\begin{bmatrix}0.10 & 0.40\\ 0.20 & 0.50\\0.30 & 0.60\end{bmatrix}
```

**Biases ($b_h$) for Hidde Layer:**

```math
\begin{bmatrix}0.1 \\ 0.2 \\ 0.3\end{bmatrix}
```
**Weights ($W_{ho}$) from Hidden to Output Layer:**

```math
\begin{bmatrix}0.4 & 0.5 & 0.6\end{bmatrix}
```

**Bias for Output Layer ($b_o$):** 
```math
\begin{bmatrix}0.4\end{bmatrix}
```

Let's use a mini-batch of 3 samples for vectorization illustration:

**Inputs ($X$):**

```math
\begin{bmatrix}1&0&1\\0&1&0\end{bmatrix}
```
Each column of $X$ is one sample of the mini-batch

**Output Labels ($Y$):**

```math
\begin{bmatrix}1&0&1\end{bmatrix}
```

**Batchwise Forward Propagation:**

**Hidden Layer:**
```math
\sigma(Z_{h})=\sigma(\begin{bmatrix}0.10 & 0.40\\ 0.20 & 0.50\\0.30 & 0.60\end{bmatrix}\begin{bmatrix}1&0&1\\0&1&0\end{bmatrix}+\begin{bmatrix}0.1&0.1&0.1 \\ 0.2&0.2&0.2 \\ 0.3&0.3&0.3\end{bmatrix})
= \sigma(\begin{bmatrix}0.1&0.4&0.1\\0.2&0.5&0.2\\0.3&0.6&0.3\end{bmatrix})+\sigma(\begin{bmatrix}0.1&0.1&0.1 \\ 0.2&0.2&0.2 \\ 0.3&0.3&0.3\end{bmatrix})
= \sigma(\begin{bmatrix}0.2&0.5&0.2\\0.4&0.7&0.4\\0.6&0.9&0.6\end{bmatrix})
```
```math
= \begin{bmatrix}0.55&0.62&0.55\\0.59&0.67&0.59\\0.65&0.71&0.65\end{bmatrix}
```
The three columns in [0.55 0.62 0.55 .... 0.65 0.71 0.65] represent the three output vectors (of the hidden layer) corresponding to the three samples in the mini-batch.

**Output Layer:**

```math
\sigma(Z_o)=\sigma(\begin{bmatrix}0.4 & 0.5 & 0.6\end{bmatrix}\begin{bmatrix}0.55&0.62&0.55\\0.59&0.67&0.59\\0.65&0.71&0.65\end{bmatrix}+\begin{bmatrix}0.4&0.4&0.4\end{bmatrix})
=\sigma(\begin{bmatrix}0.905&1.009&0.905\end{bmatrix}+\begin{bmatrix}0.4&0.4&0.4\end{bmatrix})
=\sigma(\begin{bmatrix}1.305&1.409&1.305\end{bmatrix})
```
```math
= \begin{bmatrix}0.79&0.80&0.79\end{bmatrix}
```

The three columns in [0.79 0.80 0.79] represent the three outputs corresponding to the three samples in the mini-batch.

**$\sigma(Wx+b)$ is at the core of NN. Thanks to matrix-matrix multiplication, $\sigma(WX+b)$ is used to compute:**
- **not just the output of a neuron for an training sample,**
- **but also, the outputs of an entire layer of neurons for all the samples in a mini-batch.**

**The weight update equation for gradient descent, $w=w-\eta\frac{\partial L}{\partial w}$, is used not only for individual weights but also for the whole weight matrix of a layer: $W=W-\eta\frac{\partial L}{\partial W}$. $\frac{\partial L}{\partial W}$ computed from a mini-batch is a matrix with shape same as that of matrix $W$.**
