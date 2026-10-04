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

# ---------------------------------------------------------------------------
# Shared evaluation tools (used by all models)
# ---------------------------------------------------------------------------

# Splits into stratified folds, so each fold keeps the ~8.8% positive rate.
def stratified_kfold_indices(y, k, seed=42):
    """Return a list of k index arrays, each with the same class ratio as y."""
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].append(part)
    return [np.concatenate(f) for f in folds]


# Precision, recall and F1 for the positive class (heart attack).
# F1 is used for all decisions; precision and recall are only for diagnosis/reporting.
def classification_scores(y_true, y_pred):
    """Return (precision, recall, f1) for labels in {0, 1}, positive class = 1."""

    # 1. Count true positives, false positives and false negatives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # 2. Precision: among predicted positives, how many are real positives
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0

    # 3. Recall: among real positives, how many we detected
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    # 4. F1: harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1



# Generic k-fold cross-validation, used for all models.
# Only the training and the scoring depend on the model,
# so they are passed as functions: train_fn and score_fn.
def cross_validate(train_fn, score_fn, x, y01, thresholds, k=5, seed=42):
    """Stratified k-fold cross-validation for any model.

    train_fn(x_tr, y_tr, fold) -> model   : trains the model on the training rows
    score_fn(model, x_val)     -> scores  : one score per validation row
                                            (probability or raw score)

    The best threshold is chosen on the mean F1 only.
    Precision and recall are reported at that threshold (for diagnosis).
    Returns (best_threshold, mean [P, R, F1], std [P, R, F1]) over the folds."""

    # 1. Split row indices into k folds, each with the same % of positives
    folds = stratified_kfold_indices(y01, k, seed)

    # 2. Table to fill in: one row per fold, one column per threshold,
    #    and 3 values per cell: precision, recall, F1
    results = np.zeros((k, len(thresholds), 3))

    for i in range(k):
        # 3. Fold i is the validation set, the other k-1 folds are for training
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        # 4. Train the model on the training rows only (model-specific)
        model = train_fn(x[tr_idx], y01[tr_idx], i)

        # 5. Predict a score for each validation row (model-specific)
        scores = score_fn(model, x[val_idx])

        # 6. For each threshold, turn scores into classes (1 if score > t)
        #    and compare with the true validation labels: precision, recall, F1
        for t_i, t in enumerate(thresholds):
            y_pred = (scores > t).astype(int)
            results[i, t_i] = classification_scores(y01[val_idx], y_pred)

        print("fold", i, "done")

    # 7. Average F1 over the folds and pick the threshold with the highest mean F1
    best = np.argmax(results[:, :, 2].mean(axis=0))

    # 8. Mean and std over the folds of precision, recall and F1 at that threshold
    mean = results[:, best].mean(axis=0)
    std = results[:, best].std(axis=0)
    print(f"best threshold {thresholds[best]:.2f} | "
          f"F1 {mean[2]:.4f} ± {std[2]:.4f} | "
          f"P {mean[0]:.4f} ± {std[0]:.4f} | "
          f"R {mean[1]:.4f} ± {std[1]:.4f}")

    return thresholds[best], mean, std


def best_f1(scores, y_true, thresholds):
    """Highest F1 over the threshold grid for one set of scores."""
    return max(classification_scores(y_true, (scores > t).astype(int))[2] for t in thresholds)