"""
My first implemenation of recursive neural net (RNN) 
with LSTM architecture for character analysis


To do 02/09/2015 
	 - get scan function working 
	 - convert x and y to theano shared variables to do fast math 

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
import multiprocessing

class RNN(object):

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
	
	def loss(self,x,y):
		"""
		args:
			- x is a vector containing the first character of a sequence 
			- y is a vector containing the last character of the sequence 
			
		***assuming constance sequence length****
		"""

		[h, s], _ = theano.scan(fn=self.feed_through,
						sequences=x,
						outputs_info=[self.h0,None])
		
		return -T.mean(T.log(s)[T.arange(y.shape[0]), y])

	def cross_entropy_loss(self, x, y):
		"""
		Cross entropy loss function. Average of cross entropy across a minibatch 
		"""

		[h, s], _ = theano.scan(fn=self.feed_through,
						sequences=x,
						outputs_info=[self.h0,None]) 

		y_guess = s[:,0,:]

		return -T.mean(y*T.log(y_guess) + (1.0-y)*T.log(1-y_guess))

	def sqr_diff_loss(self, x, y):

		[h, s], _ = theano.scan(fn=self.feed_through,
						sequences=x,
						outputs_info=[self.h0,None])

		y_guess = s[:,0,:]

		return T.sum((y-y_guess)**2)

	def 

	def save_param(self,pickle_file):

		pickle_me = {
					'param':[self.wx, self.wh, self.wy, self.bh, self.by, self.h0]
		}

		pickle.dump( pickle_me, open(pickle_file, 'wb') )

	def load_param(self,pickle_file):

		pickle_me = pickle.load(open(pickle_file,'rb'))

		param = pickle_me['param']

		self.wx, self.wh, self.wy, self.bh, self.by, self.h0 = param

	def train(self,training_data,learning_rate,n_epochs,mini_batch_size):
		"""
		Using a squared difference loss function now. I couldn't get 
		log loss function to work out for me. Dont know how that works.
		args:
			- training_data: inputs with ideal outputs
			- learning_rate
			- n_epochs: the number of epochs to train the NN for 
			- mini_batch_size: the size of the mini batch to be used for SGD 

		"""
		train_x, train_y = training_data
		cast_y = T.cast(train_y,'int32')
		train_size_total = train_x.get_value(borrow=True).shape[0]

		n_train_batches = train_size_total/mini_batch_size

		x = T.matrix('x')
		y = T.imatrix('y')
		xs = T.tensor3('xs')
		ys = T.itensor3('ys')

		index = T.iscalar()

		[h, s], _ = theano.scan(fn=self.feed_through,
						sequences=x,
						outputs_info=[self.h0,None])

		y_guess = s[:,0,:]
		loss = T.sum((y-y_guess)**2)
		# total, _ = theano.scan(fn=loss,
		# 					sequences=y)

		# loss = T.sum(T.sum((y-y_guess)**2,axis=1),axis=1)
		# train_model1 =  theano.function([x,y],loss)#,mode='DebugMode')

		cost = self.sqr_diff_loss(x,y)
		params = [self.wx, self.wh, self.wy, self.bh, self.by, self.h0]
		grads = T.grad(cost,params)
		updates = [(param, param-learning_rate*grad) for param, grad in zip(params,grads)]

		train_model = theano.function(
			inputs = [x,y],
			outputs = cost,
			updates = updates
			# givens = {
			# 	x: train_x[index*mini_batch_size: (index+1)*mini_batch_size],
			# 	y: train_y[index*mini_batch_size: (index+1)*mini_batch_size] 
			# }
		)
		# for i in xrange(n_epochs):
		# 	t1 = time.time()
		# 	for index in xrange(n_train_batches):
		# 		train_model(index)
		# 		# print("Minibatch done")
		# 	print("Epoch number {}, took {:.3f} sec".format(i,time.time()-t1))
		train_x_val = train_x.get_value()
		cast_y_val = cast_y.eval()
		print("function compiled\n\n\n")
		for j in xrange(n_epochs):
			sum_epoch = 0 
			t1 = time.time()
			for i in xrange(n_train_batches):	
				sum_mini_batch = 0 
				t3 = time.time()
				x_slice = train_x_val[i*mini_batch_size: (i+1)*mini_batch_size]
				y_slice = cast_y_val[i*mini_batch_size: (i+1)*mini_batch_size]
				# print("Time making slice {}".format(time.time()-t3))
				xy_size = x_slice.shape[0]
				t4 = time.time()
				for h in xrange(xy_size):
					sum_mini_batch += train_model(x_slice[h], y_slice[h])
				# print("Time making sum {}".format(time.time()-t4))
				if i % 30 == 0:
					print("Sum for minibatch number {} out of {}: {}".format(i,n_train_batches,sum_mini_batch))
				sum_epoch += sum_mini_batch
			print("Sum for this epoch: {:.3f}, took {:.3f} sec".format(sum_epoch, time.time()-t1))
			if j % 5 == 0:
				t2 = time.time()
				self.save_param("param_epoch{}.dat".format(i))
				print("Pickling epoch number {} took {:.3f} sec".format(j, time.time()-t2))

		# train_model = theano.function(
		# 	inputs = [index],
		# 	outputs = cost,
		# 	updates = updates,
		# 	givens = {
		# 		x: train_x[index*mini_batch_size: (index+1)*mini_batch_size],
		# 		y: train_y[index*mini_batch_size: (index+1)*mini_batch_size] 
		# 	}
		# )
		# print("Function compiled")
		# for i in xrange(n_epochs):
		# 	t1 = time.time()
		# 	for index in xrange(n_train_batches):
		# 		train_model(index)
		# 		# print("Minibatch done")
		# 	print("Epoch number {}, took {:.3f} sec".format(i,time.time()-t1))
		# 	if i % 5 == 0:
		# 		t2 = time.time()
		# 		self.save_param("param_epoch{}.dat".format(i))
		# 		print("Pickling epoch number {} took {:.3f} sec".format(i, time.time()-t2))


	def gen_random_sentence(self,x_init):
		"""
		Run 'x_init' through the RNN, saving the y values at 
		each 'time step'.
		"""
		ys = []
		y, h = self.feed_through(x_init,self.h0)
		ys.append(y)
		for i in xrange(1,self.sequence_length):
			y, h = self.feed_through(y,h)
			ys.append(y)

		# ys = [y.eval() for y in ys]
		# ys_arg_max = [np.argmax(y) for y in ys]

		return ys


	def compile_gen_sentence(self):
		"""
		compile a theano function that takes the initial x value 
		and returns y vectors for each of the subsequent positions. 
		"""
		x = T.dvector('x')
		y = self.gen_random_sentence(x)

		f = theano.function([x],y)

		return f 



if __name__ == '__main__':

	text_test = './../texts/melville.txt'
	char_map_obj = Character_Map(text_test,'mapping.dat',overwrite=True, break_line=None)
	char_map_obj.k_map()
	x, y, shared_x, shared_y = char_map_obj.gen_x_and_y(filename=None)
	# print(shared_x, shared_y.get_value().shape[0])
	nh = 100
	nx = len(char_map_obj.unique_char)
	ny = nx 

	trainer = RNN(nh,nx,ny)
	# jobs = []
	# for i in xrange(2):
	# 	p = multiprocessing.Process(target=trainer.train, args=((shared_x,shared_y),0.03,1000,10,))
	# 	jobs.append(p)
	# 	p.start()
	# trainer.load_param('param_epoch95.dat')
	trainer.train((shared_x,shared_y),0.01,100,100)
	# f = trainer.compile_gen_sentence()




	



