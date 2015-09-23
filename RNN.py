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

class RNN(object):

	def __init__(self, nh, nx, ny):
		"""
		This is only set up for a single hidden layer 
		args:
			nh is size of hidden layer vector 
			nx is the size of the input vector 
			ny is the size of the output vector (ny = nx in character example)
		"""
		
		# self.emb = theano.shared(name='embeddings',
		# 						 value=0.2 * numpy.random.uniform(-1.0, 1.0,
		# 						 (ne+1, de))
		# 						 # add one for padding at the end
		# 						 .astype(theano.config.floatX))

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
		# y = y[-1]

		# y_intermediate, h = self.feed_through_dean(x,self.h0)
		# y_total = y_intermediate
		# for i in xrange(1,self.sequence_length):
		# 	y_intermediate, h = self.feed_through_dean(y_intermediate,h)
		# 	y_total += y_intermediate

		[h, s], _ = theano.scan(fn=self.feed_through,
						sequences=x,
						outputs_info=[self.h0,None])
						# n_steps=self.sequence_length)
		
		# return -T.mean(T.log(p_y_given_x_sentence)[T.arange(y.shape[0]), y])
		return -T.mean(T.log(s)[T.arange(y.shape[0]), y])

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
		args:
			- training_data: inputs with ideal 
		"""
		# self.char_sequence_length = char_sequence_length
		train_x, train_y = training_data
		train_y = T.cast(train_y,'int32')
		train_size_total = train_x.get_value(borrow=True).shape[0]

		# n_train = len(train_x)
		# print(train_size_total, n_train)
		n_train_batches = train_size_total/mini_batch_size

		x = T.matrix('x')
		y = T.imatrix('y')
		index = T.iscalar()

		cost = self.loss(x,y)
		params = [self.wx, self.wh, self.wy, self.bh, self.by, self.h0]
		grads = T.grad(cost,params)
		updates = [(param, param-learning_rate*grad) for param, grad in zip(params,grads)]

		train_model = theano.function(
			inputs = [index],
			outputs = cost,
			updates = updates,
			givens = {
				x: train_x[index*mini_batch_size: (index+1)*mini_batch_size],
				y: train_y[index*mini_batch_size: (index+1)*mini_batch_size] 
			}
		)
		print("Function compiled")
		for i in xrange(n_epochs):
			t1 = time.time()
			for index in xrange(n_train_batches):
				train_model(index)
				# print("Minibatch done")
			print("Epoch number {}, took {:.3f} sec".format(i,time.time()-t1))
			if i % 5 == 0:
				t2 = time.time()
				self.save_param("param_epoch{}.dat".format(i))
				print("Pickling epoch number {} took {:.3f} sec".format(i, time.time()-t2))


	def gen_random_sentence(self,x_init):
		"""
		Run 'x_init' through the RNN, saving the y values at 
		each 'time step'.
		"""
		ys = []
		y, h = self.feed_through_dean(x_init,self.h0)
		ys.append(y)
		for i in xrange(1,self.sequence_length):
			y, h = self.feed_through_dean(y,h)
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
	char_map_obj = Character_Map(text_test,'mapping.dat',overwrite=True, break_line=5000)
	char_map_obj.k_map()
	x, y, shared_x, shared_y = char_map_obj.gen_x_and_y(filename=None)
	# print(shared_x, shared_y.get_value().shape[0])
	nh = 100
	nx = len(char_map_obj.unique_char)
	ny = nx 

	trainer = RNN(nh,nx,ny)
	# trainer.load_param('param_epoch95.dat')
	trainer.train((shared_x,shared_y),0.0001,100,10)
	f = trainer.compile_gen_sentence()




	



