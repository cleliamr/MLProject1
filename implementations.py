import numpy as np

### Implementation of ML methods: mean squared error, least squares, ridge regression, logistic regression, and regularized logistic regression ###

### Mean Squared Error ###
def calculate_mse(e):
    """MSE with the 1/2 factor: 1/(2N) * sum(e^2)."""

    return 1 / 2 * np.mean(e**2)


### Loss Function for a single data point ###
def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx.dot(w)

    return calculate_mse(e)


### Computing the Gradient ###
def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    err = y - tx.dot(w)

    grad = -tx.T.dot(err) / len(err)

    return grad


### Linear regression using gradient descent that only returns  last weight vector of the
### method and the corresponding loss value


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Computes linear regression using gradient descent.
    Args:
        y: numpy array of shape=(N,), containing the target values
        tx: numpy array of shape=(N, D), containing the input data
        initial_w: numpy array of shape=(D,), initial weight vector
        max_iters: integer, maximum number of gradient descent iterations
        gamma: float, learning rate

    Returns:
        w: numpy array of shape=(D,), final weight vector.
        loss: float, mean squared error corresponding to the final weights
    """

    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_gradient(y, tx, w)
    return w, compute_loss(y, tx, w)


### Linear regression using stochastic gradient descent that only returns  last weight vector of the
### method and the corresponding loss value


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Computes linear regression using stochastic gradient descent.
    Args:
        y: numpy array of shape=(N,), containing the target values
        tx: numpy array of shape=(N, D), containing the input data
        initial_w: numpy array of shape=(D,), initial weight vector
        max_iters: integer, maximum number of stochastic gradient descent iterations
        gamma: float, learning rate

     Returns:
         w: numpy array of shape=(D,), final weight vector
         loss: float, mean squared error corresponding to the final weights
    """
    w = initial_w
    N = len(y)
    for _ in range(max_iters):
        i = np.random.randint(N)  # batch size 1
        grad = compute_gradient(y[i : i + 1], tx[i : i + 1], w)
        w = w - gamma * grad
    return w, compute_loss(y, tx, w)


def least_squares(y, tx):
    """Least squares via the normal equations.

    Returns:
        w: shape (D,)
        loss: scalar, MSE with the 1/2 factor
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = 0.5 * np.mean((y - tx @ w) ** 2)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression via the normal equations.

    Returns:
        w: shape (D,)
        loss: scalar, MSE with the 1/2 factor, without the penalty term
    """
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    w = np.linalg.solve(A, tx.T @ y)
    loss = 0.5 * np.mean((y - tx @ w) ** 2)
    return w, loss


### Sigmoid Function for Logistic Regression ###
def sigmoid(t):
    """Numerically safe sigmoid."""
    t = np.clip(t, -500, 500)
    return 1 / (1 + np.exp(-t))


### Computing the loss for Logistic Regression ###
def compute_loss_logistic(y, tx, w):
    """Mean negative log-likelihood, y in {0, 1}."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


### Logistic Regression ###


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D,)
        max_iters: number of gradient descent iterations
        gamma: step size

    Returns:
        loss: final logistic loss
        w: final weight vector
    """

    w = initial_w

    for i in range(max_iters):

        # compute predictions
        predictions = sigmoid(tx @ w)

        # compute gradient
        gradient = (tx.T @ (predictions - y)) / len(y)

        # update weights
        w = w - gamma * gradient

    # calculate final loss
    loss = compute_loss_logistic(y, tx, w)

    return w, loss


### Regularized Logistic Regression ###
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N, D)
        lambda_: regularization parameter
        initial_w: numpy array of shape=(D,)
        max_iters: number of gradient descent iterations
        gamma: step size

    Returns:
        loss: final regularized logistic loss
        w: final weight vector
    """

    w = initial_w

    for i in range(max_iters):

        # Compute predictions
        predictions = sigmoid(tx @ w)

        # logistic regression gradient with regularization gradient
        gradient = (tx.T @ (predictions - y)) / len(y) + 2 * lambda_ * w

        # update weights
        w = w - gamma * gradient

    # calculate final loss
    loss = compute_loss_logistic(y, tx, w)

    # add regularization term to loss -> we should not add it
    # loss += lambda_ * np.sum(w**2)

    return w, loss


'''
### Mean Squared Error with Gradient Descent ###
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)

        # update w by gradient descent
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws
    
### Computing Stochastic Gradient ###
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    err = y - tx.dot(w)
    
    grad = -tx.T.dot(err) / len(err)
    
    return grad, err


### Mean Squared Error with Stochastic Gradient Descent ###
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
          
            # update w through the stochastic gradient update
            w = w - gamma * grad
          
            # calculate loss
            loss = compute_loss(y, tx, w)
          
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws
    
### Least Squares ###
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """

    ## from normal equation Aw = b (X^TXw = X^Ty), define A and b
    A = tx.T @ tx
    b = tx.T @ y

    #solve for w
    w = np.linalg.solve(A, b)

    # calculate error (target - predicted)
    error = y - tx @ w
  
    # calculate MSE
    mse = np.mean(error **2)
  
    return w, mse


### Ridge Regression ###
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> e.g. ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> e.g. ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    
    """
  
    # number of samples
    N = tx.shape[0]
  
    # number of features
    D = tx.shape[1]

    # create an identity matrix shape D x D
    I = np.eye(D)

    # define A and b
    # add regularization term to A
    A = tx.T @ tx + 2 * N * lambda_ * I
    b = tx.T @ y

    # solve for w (optimal weights)
    w = np.linalg.solve(A, b)

    return w
    
### Computing the loss for Logistic Regression ###
def compute_loss_logistic(y, tx, w):
    """Calculate the loss for logistic regression."""

    predictions = sigmoid(tx.dot(w))

    loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    return loss
'''
