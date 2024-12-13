from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W) # (1, C)
        scores -= np.max(scores)
        # If the maximum value is not subtracted, the calculation of exp may result in very large values, e.g. score=1000.
        # After subtracting max the largest fraction is 0, the rest are negative and the range of exp becomes [0,1], thus avoiding overflow.
        # When all scores are small and approaching negative infinity, the value of exp will be close to 0. 
        # After subtracting max, the range of scores is compressed to a larger value between negative and 0 to avoid loss of precision.
        softmax_P = np.exp(scores)/np.sum(np.exp(scores))
        loss -= np.log(softmax_P[y[i]])
        for j in range(num_classes):
            dW[:,j] += X[i] * softmax_P[j]
        dW[:,y[i]] -= X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W) # (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)
    softmax_P = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)

    loss = np.sum(-np.log(softmax_P[range(num_train),y]))
    softmax_P[range(num_train),y] -= 1
    dW = X.T @ (softmax_P)

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
