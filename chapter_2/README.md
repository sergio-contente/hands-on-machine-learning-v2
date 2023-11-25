# Chapter 2

## Task

Use the California census data to build a model of housing prices in the state. This data includes metrics such as the population, median income and median housing prices for each block group in California. Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data (we will call them "districts").

### Pipeline

A sequence of data processing components is called a data _pipeline_. Pipelines use components that typically run asynchronously, each one of them is fairly self-cotained: the interface between components is simply the data store. With this approach, if a component breaks down, the downstream components can often continue to run normally by just using the last output from the broken component.

Each component pulls in a large amount of data, processes it and spits out the result in another data store.

Keep in mind that a broken component can go unnoticed for some time if proper monitoring is not implemented.

## Frame the problem

It's a clearly supervised learning task, since you are given **labeled** training examples (each district comes with the expected output: i.e., the district's median housing price) and is also a typical regression task, since you are asked to predict a value. More specifically, this is a _multiple regression_ problem since the system will use multiple features to make a prediction (district's population, median income, etc.).

It's also a _univariate regression_ problem since we are trying to predict a single value for each district. If we were trying to predict multiple values per distric, it would be a _multivariate regression_ problem.

Finally, there is no continous flow of data incoming, hence there is no particular need to adjust to changing data rapidly and the data is small enough to fit in the memory. If we were treating huge data, you could either split your batch learning work across multiple servers (MapReduce technique) or use an online learning technique.

## Select a performance measure

A typical perfomance measure for regression problems is the Root Mean Square Error (RMSE). It gives an idea of how much error the system typically makes in its predictions with a higher wight for large errors.

$RMSE(X, h)=\sqrt{\frac{1}{m}\sum_{i=1}^{m} (h(x^{(i)})-y^{(i)})^2}$

Some notations observations:

- $m$ is the number of instances in the dataset you are measuring the RMSE on.
  - If you are evaluating the RMSE on a validation set of 2000 districts, then $m = 2000$
- $\mathbf{x^(i)}$ is a vector of all the feature values (excluding the label) of the $i^{th}$ instance in the dataset, and $y^{(i)}$ is its label (the desired output for that instance). Example:
  
$$  \mathbf{x^{(1)}} = \begin{bmatrix}
    feature 1 \\
    feature 2 \\
    feature 3 \\
    \end{bmatrix}
\\
y^{(1)} = label_{1}
$$

- $\mathbf{X}$ is a matrix containing all the feature values (excluding labels) of all instances in the dataset. There is one row per instance and the $i^{th}$ row is equal to the transpose of $\mathbf{x^{i}}$, noted $\mathbf{(x^{i})^T}$.

$$  \mathbf{X} = \begin{bmatrix}
    (x^{(1)})^T \\
    (x^{(2)})^T  \\
    \vdots \\
    (x^{(1999)})^T\\
    (x^{(2000)})^T
    \end{bmatrix}
$$

- $h$ is the prediction function, also called a _hypothesis_. When your system is given an instance's feature vector $\mathbf{x^(i)}$, it outputs a predicated value  $\hat{y^{(i)}} = h(\mathbf{x^(i)})$ for that instance.
- RMSE(X,h) is the cost function measured on the set of examples using the hypothesis $h$.
- Lowercase italic font -> scalar values and function names
- Lowercase bold font -> vectors
- Uppercase bold font -> matrices

Even though RMSE is generally preferred for regression tasks, you can use another function. Example: supposing that there are many outlier districts, then in this case you may consider using the _mean absolute error_ (MAE, average absolute deviation)

## Check the assumptions

What if the downstream system converts the prices into categories (e.g. "cheap", "medium", "expensive") and then uses categories instead of the prices themselves? In this case, getting the price correctly is not important at all; your system just needs to get the category right. If that so, the problem should have been freamed as a **classification** task and not a regression.

However, this is not the case here.
