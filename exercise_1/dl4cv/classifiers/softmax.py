import numpy as np
from random import shuffle

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
    SF = np.transpose(np.matmul(np.transpose(W), np.transpose(X)))
    for count, imagescore in enumerate(SF):
        sum_unnormal_prob = 0
        softmax = np.array([])
        for classscore in imagescore:
            unnormal_prob = np.exp(classscore)
            softmax = np.append(softmax, unnormal_prob)
            sum_unnormal_prob += unnormal_prob
        for cnt, up in enumerate(softmax):
            softmax[cnt] = up / sum_unnormal_prob
        
        ground_truth_class = y[count]
        
        nor_prob = softmax[ground_truth_class]
        loss += -np.log(nor_prob)

        dscores = softmax
        dscores[ground_truth_class] -= 1
        dW += X[count][:,None].dot(dscores[None,:])

    dW /= X.shape[0]
    dW += reg * W

    loss /= X.shape[0]
    loss += reg * np.sum(W*W)

    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    SF = np.transpose(np.matmul(np.transpose(W), np.transpose(X)))

    unnormal_prob = np.exp(SF)
    sum_unnormal_prob = np.sum(unnormal_prob, axis=1)
    softmax = unnormal_prob / sum_unnormal_prob[:,None]
    #print ((sum_unnormal_prob.shape))
    nor_prob = softmax[np.arange(len(softmax)), y]
    #print (nor_prob.shape)
    losses = -np.log(nor_prob) 
    loss = np.mean(losses)
    loss += reg * np.sum(W*W)

    dscores = softmax
    #print (dscores[0])
    dscores[np.arange(len(softmax)), y] -= 1
    #print (dscores[0])
    dW = (np.transpose(X)).dot(dscores)
    dW /= X.shape[0]
    dW += reg * W
    
    #print (dW)
    """for count, imagescore in enumerate(SF):
        unnormal_prob = np.exp(imagescore)
        sum_unnormal_prob = np.sum(unnormal_prob)
        softmax = unnormal_prob / sum_unnormal_prob
        
        ground_truth_class = y[count]
        nor_prob = softmax[ground_truth_class]
        dscores = softmax
        dscores[ground_truth_class] -= 1
        dW += X[count][:,None].dot(dscores[None,:])
        loss += -np.log(nor_prob)
        
    dW /= X.shape[0]
    dW += reg * W
    
    loss /= X.shape[0]
    loss += reg * np.sum(W*W)
    """
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

