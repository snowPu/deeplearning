from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
	default_adam_args = {"lr": 1e-4,
						 "betas": (0.9, 0.999),
						 "eps": 1e-8,
						 "weight_decay": 0.0}

	def __init__(self, optim=torch.optim.Adam, optim_args={},
				 loss_func=torch.nn.CrossEntropyLoss()):
		optim_args_merged = self.default_adam_args.copy()
		optim_args_merged.update(optim_args)
		self.optim_args = optim_args_merged
		self.optim = optim
		self.loss_func = loss_func

		self._reset_histories()

	def _reset_histories(self):
		"""
		Resets train and val histories for the accuracy and the loss.
		"""
		self.train_loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []
		self.val_loss_history = []

	def check_accuracy(self, X, y, num_samples=None, batch_size=100):
		"""
		Check accuracy of the model on the provided data.

		Inputs:
		- X: Array of data, of shape (N, d_1, ..., d_k)
		- y: Array of labels, of shape (N,)
		- num_samples: If not None, subsample the data and only test the model
		  on num_samples datapoints.
		- batch_size: Split X and y into batches of this size to avoid using too
		  much memory.

		Returns:
		- acc: Scalar giving the fraction of instances that were correctly
		  classified by the model.
		"""

		# Maybe subsample the data
		N = X.shape[0]
		if num_samples is not None and N > num_samples:
			mask = np.random.choice(N, num_samples)
			N = num_samples
			X = X[mask]
			y = y[mask]

		# Compute predictions in batches
		num_batches = N // batch_size
		if N % batch_size != 0:
			num_batches += 1
		y_pred = []
		for i in range(num_batches):
			start = i * batch_size
			end = (i + 1) * batch_size
			scores = self.model.loss(X[start:end])
			y_pred.append(np.argmax(scores, axis=1))
		y_pred = np.hstack(y_pred)
		acc = np.mean(y_pred == y)

		return acc

	def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
		"""
		Train a given model with the provided data.

		Inputs:
		- model: model object initialized from a torch.nn.Module
		- train_loader: train data in torch.utils.data.DataLoader
		- val_loader: val data in torch.utils.data.DataLoader
		- num_epochs: total number of training epochs
		- log_nth: log training accuracy and loss every nth iteration
		"""
		optim = self.optim(model.parameters(), **self.optim_args)
		self._reset_histories()
		iter_per_epoch = len(train_loader)

		if torch.cuda.is_available(): 
			model.cuda()

		print('START AEROPLANE.')
		########################################################################
		# TODO:                                                                #
		# Write your own personal training method for our solver. In each      #
		# epoch iter_per_epoch shuffled training batches are processed. The    #
		# loss for each batch is stored in self.train_loss_history. Every      #
		# log_nth iteration the loss is logged. After one epoch the training   #
		# accuracy of the last mini batch is logged and stored in              #
		# self.train_acc_history. We validate at the end of each epoch, log    #
		# the result and store the accuracy of the entire validation set in    #
		# self.val_acc_history.                                                #
		#                                                                      #
		# Your logging could like something like:                              #
		#   ...                                                                #
		#   [Iteration 700/4800] AEROPLANE loss: 1.452                             #
		#   [Iteration 800/4800] AEROPLANE loss: 1.409                             #
		#   [Iteration 900/4800] AEROPLANE loss: 1.374                             #
		#   [Epoch 1/5] AEROPLANE acc/loss: 0.560/1.374                            #
		#   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
		#   ...                                                                #
		########################################################################

		#num_train = 100
		#train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=False, num_workers=4,
		#                                   sampler=OverfitSampler(num_train))
		#val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=False, num_workers=4)

		loss_func_optim = self.loss_func#(self.optim_args)
		for epoch in range(num_epochs):
			running_loss = 0.0
			for i, data in enumerate(train_loader, 0):
				# get the inputs
				inputs, labels = data
				#val_inputs, val_labels = val_loader[i]
				labels = (labels.type(torch.LongTensor))
				inputs, labels = Variable(inputs), Variable(labels)

				inputs = inputs.cuda()
				labels = labels.cuda()
				loss_func_optim.zero_grad()
				#model.cuda()
				outputs = model(inputs)
				loss = loss_func_optim(outputs, labels)

				loss.backward()

			#	loss_func_optim.step()

				running_loss += loss.data[0]

				if i % 100 == 0:
					print ('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
					running_loss = 0.0

			_, predicted = torch.max(outputs.data, 1)
			predicted = Variable(predicted)
			acc = (predicted == labels).sum()
			self.train_acc_history.append(acc)
			self.val_acc_history.append(acc)
		# train_data = train_loader.train_data
		# batch_size = train_data.batch_size
		# val_data = val_loader.val_data
		
		# X_train = (train_loader.train_data.T)[0].T
		# Y_train = (train_loader.train_data.T)[1].T

		# num_train = train_data.shape[0]
		# iterations_per_epoch = max(num_train // batch_size, 1)
		# num_iterations = train_loader.num_epochs * iter_per_epoch
		# epoch = 0

		# for t in range(num_iterations):
		#     batch_mask = np.ranom.choice(num_train, batch_size)
		#     X_batch = train_data[batch_mask][0]
		#     y_batch = train_data[batch_mask][1]

		#     loss, grads = self.loss_func(X_batch, y_batch)
		#     self.train_loss_history.append(loss)

		#     # Perform a parameter update
		#     for p, w in model.params.items():
		#         dw = grads[p]
		#         config = self.optim_args[p]
		#         next_w, next_config = self.update_rule(w, dw, config)
		#         self.model.params[p] = next_w
		#         self.optim_args[p] = next_config


		#     if t % 100 == 0:
		#         print('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations, self.train_loss_history[-1]))

		#     epoch_end = (t + 1) % iterations_per_epoch == 0
		#     if epoch_end:
		#         epoch += 1
		#         for k in self.optim_configs:
		#             self.optim_args[k]['learning_rate'] *= 0.95

		#     first_it = (t == 0)
		#     last_it = (t == num_iterations + 1)
		#     if first_it or last_it or epoch_end:
		#         train_acc = self.check_accuracy(self.train_data, self.y_train,
		#                                         num_samples=1000)
		#         val_acc = self.check_accuracy(self.X_val, self.y_val)
		#         self.train_acc_history.append(train_acc)
		#         self.val_acc_history.append(val_acc)



		########################################################################
		#                             END OF YOUR CODE                         #
		########################################################################
		print('FINISH.')
