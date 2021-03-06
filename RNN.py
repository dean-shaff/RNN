"""
My first implemenation of recursive neural net (RNN) 
with (hopefully) LSTM architecture for character analysis

to do 07/10/2015
	-implement a checking function 
	-multiprocessing -- scan function is inefficient but does not 
	need to be done in sequence 
	-LSTM -- new weight matrices, new scan implementations (blah!)
	-before LSTM maybe do multi layer RNN?

"""
import theano 
import theano.tensor as T 
import numpy as np 
from character_mapping import Character_Map
import time
try:
	import cPickle as pickle
except:
	import pickle
import os 
from datetime import datetime

class RNNClass(object):

	def __init__(self, nh, nx, ny):
		"""
		This is only set up for a single hidden layer 
		args:
			nh is size of hidden layer vector 
			nx is the size of the input vector 
			ny is the size of the output vector (ny = nx in character example)
		"""

		self.wx = theano.shared(name='wx',
								value=0.2 * np.random.uniform(-1.0, 1.0,
								(nx, nh))
								.astype(theano.config.floatX)) #input weights

		self.wh = theano.shared(name='wh',
								value=0.2 * np.random.uniform(-1.0, 1.0,
								(nh, nh))
								.astype(theano.config.floatX)) #hidden layer weights

		self.wy = theano.shared(name='wy',
							   value=0.2 * np.random.uniform(-1.0, 1.0,
							   (nh, ny))
							   .astype(theano.config.floatX)) #output weights
		
		self.bh = theano.shared(name='bh',
								value=np.zeros(nh,
								dtype=theano.config.floatX)) #hidden layer bias
		
		self.by = theano.shared(name='b',
							   value=np.zeros(ny,
							   dtype=theano.config.floatX)) #output layer bias
		
		self.h0 = theano.shared(name='h0',
								value=np.zeros(nh,
								dtype=theano.config.floatX)) #initial h vector 

		self.sequence_length = 15


	def feed_through(self,x,h_tm1):
		"""
		t_step is the current time step. If t_step == 0, then we use self.h0
		to feed through net.
		basically copied from the theano tutorial
		"""
		h = T.tanh(T.dot(x,self.wx) + T.dot(h_tm1, self.wh) + self.bh)

		y_hat = self.by + T.dot(h,self.wy)

		y_guess = T.nnet.softmax(y_hat) 

		return h, y_guess
	
	# def loss(self,x,y):
	# 	"""
	# 	args:
	# 		- x is a vector containing the first character of a sequence 
	# 		- y is a vector containing the last character of the sequence 
			
	# 	***assuming constance sequence length****
	# 	"""

	# 	[h, s], _ = theano.scan(fn=self.feed_through,
	# 					sequences=x,
	# 					outputs_info=[self.h0,None])
		
	# 	return -T.mean(T.log(s)[T.arange(y.shape[0]), y])

	def cross_entropy_loss(self, x, y):
		"""
		Cross entropy loss function. Average of cross entropy across a minibatch 
		"""

		[h, s], _ = theano.scan(fn=self.feed_through,
						sequences=x,
						outputs_info=[self.h0,None]) 

		y_guess = s[:,0,:]

		return y*T.log(y_guess) + (1.0-y)*T.log(1.0-y_guess)

	def sqr_diff_loss(self, x, y):

		[h, s], _ = theano.scan(fn=self.feed_through,
						sequences=x,
						outputs_info=[self.h0,None])

		y_guess = s[:,0,:]

		return T.sum((y-y_guess)**2)


	def save_param(self,pickle_file):

		pickle_me = {
					'param':[self.wx, self.wh, self.wy, self.bh, self.by, self.h0]
		}

		pickle.dump( pickle_me, open(pickle_file, 'wb') )

	def load_param(self,pickle_file):

		pickle_me = pickle.load(open(pickle_file,'rb'))

		param = pickle_me['param']

		self.wx, self.wh, self.wy, self.bh, self.by, self.h0 = param

	# def train_no_index(self,training_data,learning_rate,n_epochs,mini_batch_size):
	# 	"""
	# 	Right now using cross entropy loss function. This works, albeit very slowly.
	# 	args:
	# 		- training_data: inputs with ideal outputs
	# 		- learning_rate
	# 		- n_epochs: the number of epochs to train the NN for 
	# 		- mini_batch_size: the size of the mini batch to be used for SGD 

	# 	"""
	# 	train_x, train_y = training_data
	# 	train_size_total = train_x.get_value(borrow=True).shape[0]

	# 	n_train_batches = train_size_total/mini_batch_size

	# 	x = T.matrix('x')
	# 	y = T.matrix('y')
	# 	xs = T.tensor3('xs')
	# 	ys = T.itensor3('ys')

	# 	# index = T.iscalar()

	# 	cost = -T.mean(self.cross_entropy_loss(x,y))
	# 	params = [self.wx, self.wh, self.wy, self.bh, self.by, self.h0]
	# 	grads = T.grad(cost,params)
	# 	updates = [(param, param-learning_rate*grad) for param, grad in zip(params,grads)]

	# 	train_model = theano.function(
	# 		inputs = [x,y],
	# 		outputs = cost,
	# 		updates = updates
	# 	)

	# 	train_x_val = train_x.get_value()
	# 	train_y_val = train_y.get_value()
	# 	print("function compiled\n\n")
	# 	for j in xrange(n_epochs):
	# 		sum_epoch = 0 
	# 		t1 = time.time()
	# 		for i in xrange(n_train_batches):	
	# 			sum_mini_batch = 0 
	# 			t3 = time.time()
	# 			x_slice = train_x_val[i*mini_batch_size: (i+1)*mini_batch_size]
	# 			y_slice = train_y_val[i*mini_batch_size: (i+1)*mini_batch_size]
	# 			xy_size = x_slice.shape[0]
				
	# 			for h in xrange(xy_size):
	# 				sum_mini_batch += train_model(x_slice[h], y_slice[h])
	# 			print("Time for minibatch: {}".format(time.time()-t3))
	# 			# print("Time making sum {}".format(time.time()-t4))
	# 			if i % 30 == 0:
	# 				print("Sum for minibatch number {} out of {}: {}".format(i,n_train_batches,sum_mini_batch))
	# 			sum_epoch += sum_mini_batch
	# 		print("Sum for this epoch: {:.3f}, took {:.3f} sec".format(sum_epoch, time.time()-t1))
	# 		if j % 5 == 0:
	# 			t2 = time.time()
	# 			self.save_param("param_epoch{}.dat".format(i))
	# 			print("Pickling epoch number {} took {:.3f} sec".format(j, time.time()-t2))


	def train_index(self,training_data,learning_rate,n_epochs,mini_batch_size):
		"""
		Using a squared difference loss function now. I couldn't get 
		log loss function to work out for me. Dont know how that works.
		args:
			- training_data: inputs with ideal outputs
			- learning_rate
			- n_epochs: the number of epochs to train the NN for 
			- mini_batch_size: the size of the mini batch to be used for SGD 

		"""
		foo = datetime.now()
		param_folder = "param_{}-{}_{}:{}/".format(foo.day, foo.month, foo.hour, foo.minute)
		os.mkdir(param_folder)
		print("Saving initial parameters")
		self.save_param("{}param_epoch{}.dat".format(param_folder,0))
		
		# print("Using train function with indices")
		train_x, train_y = training_data
		train_size_total = train_x.get_value(borrow=True).shape[0]

		n_train_batches = train_size_total/mini_batch_size

		# x = T.matrix('x')
		# y = T.matrix('y')
		xs = T.tensor3('xs')
		ys = T.tensor3('ys')

		index = T.iscalar()

		# cost = self.cross_entropy_loss(x,y)
		results, _ = theano.scan(lambda xi, yi: self.cross_entropy_loss(xi,yi),
									sequences = [xs,ys])
		loss_fn = -T.mean(results) # loss must be a scalar value, not a matrix
		params = [self.wx, self.wh, self.wy, self.bh, self.by, self.h0]
		grads = T.grad(loss_fn,params)
		updates = [(param, param-learning_rate*grad) for param, grad in zip(params,grads)]
	
		train_model = theano.function(
			inputs = [index],
			outputs = loss_fn,
			updates = updates,
			givens = {
				xs: train_x[index*mini_batch_size: (index+1)*mini_batch_size],
				ys: train_y[index*mini_batch_size: (index+1)*mini_batch_size] 
			}
		)
		print("Function compiled!")
		print("Training model")
		for i in xrange(n_epochs):
			t1 = time.time()
			for index in xrange(n_train_batches):
				t2 = time.time()
				train_model(index)
				if index % 30 == 0:
					print("{} out of {} minibatches done, took ~ {:.3f}".format(index,n_train_batches,30*(time.time()-t2)))
			print("Epoch number {}, took {:.3f} sec".format(i,time.time()-t1))
			# if i % 2 == 0:
			t2 = time.time()
			self.save_param("{}param_epoch{}.dat".format(param_folder,i))
			print("Pickling epoch number {} took {:.3f} sec".format(i, time.time()-t2))


	def sequence_guess(self,x_init,sequence_length):
		"""
		Given some x_init vector, this will generate a sequence of 
		characters, 'sequence_length' long. 
		runs x_init through the RNN
		returns a vector containing the generated sequence 
		"""
		x0 = T.vector('x0')
		h0 = T.vector('h0')
		h, y_intermediate = self.feed_through(x0, h0)

		f1 = theano.function(inputs=[x0,h0],
								outputs=[h, y_intermediate])

		f2 = theano.function([x0],T.argmax(x0))

		hi, yi = f1(x_init, self.h0.get_value())
		# ys = [y[0]]
		y_argmax = [f2(yi[0])]

		for i in xrange(1, sequence_length):
				hi, yi = f1(yi[0],hi)
				# ys.append(yi[0])
				y_argmax.append(f2(yi[0]))

		return y_argmax

	# def gen_random_sentence(self,x_init):
	# 	"""
	# 	Run 'x_init' through the RNN, saving the y values at 
	# 	each 'time step'.
	# 	"""
	# 	ys = []
	# 	y, h = self.feed_through(x_init,self.h0)
	# 	ys.append(y)
	# 	for i in xrange(1,self.sequence_length):
	# 		y, h = self.feed_through(y,h)
	# 		ys.append(y)

	# 	# ys = [y.eval() for y in ys]
	# 	# ys_arg_max = [np.argmax(y) for y in ys]

	# 	return ys


	# def compile_gen_sentence(self):
	# 	"""
	# 	compile a theano function that takes the initial x value 
	# 	and returns y vectors for each of the subsequent positions. 
	# 	"""
	# 	x = T.vector('x')
	# 	y = self.gen_random_sentence(x)

	# 	f = theano.function([x],y)

	# 	return f 



if __name__ == '__main__':

	text_test = './../texts/melville.txt'
	char_map_obj = Character_Map(text_test,'mapping.dat',overwrite=True, break_line=None)
	char_map_obj.k_map()
	x, y, shared_x, shared_y = char_map_obj.gen_x_and_y(filename=None)
	# print(shared_x, shared_y.get_value().shape[0])
	nh = 100
	nx = len(char_map_obj.unique_char)
	ny = nx 

	trainer = RNNClass(nh,nx,ny)
	# jobs = []
	# for i in xrange(2):
	# 	p = multiprocessing.Process(target=trainer.train, args=((shared_x,shared_y),0.03,1000,10,))
	# 	jobs.append(p)
	# 	p.start()
	# trainer.load_param('param_epoch95.dat')
	trainer.train_index(training_data=(shared_x,shared_y),
						learning_rate=0.01,
						n_epochs=100,
						mini_batch_size=1000)
	# f = trainer.compile_gen_sentence()




	



