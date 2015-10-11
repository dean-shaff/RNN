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

class RNN_multilayer(object):

	def __init__(self, nh_vector, nx, ny, sequence_length=15):
		"""
		Setting this up for multiple layers
		args:
			- nh_vector is a list or tuple containing 
				the dimensions of the hidden layers 
			- nx is the size of the input vector 
			- ny is the size of the output vector (ny = nx in character example)
		"""
		self.nh_vector = nh_vector
		wx_n = [] #W_{x h^n} matrix 
		whh_n = [] #W_{h^n h^n} matrix
		whm1h_n = [] #W_{h^{n-1} h^n} matrix 
		bh_n = [] 
		h0_n = [] 
		for i in xrange(len(nh_vector)):
			wx_n.append(theano.shared(name='wx',
						value=0.2 * np.random.uniform(-1.0, 1.0,(nx, nh_vector[i])).astype(theano.config.floatX))) #input weights
			
			whh_n.append(theano.shared(name='wh',
						value=0.2 * np.random.uniform(-1.0, 1.0,(nh_vector[i], nh_vector[i])).astype(theano.config.floatX)))

			bh_n.append(theano.shared(name='bh',
						value=np.zeros(nh_vector[i], dtype=theano.config.floatX)))

			h0_n.append(theano.shared(name='h0',
								value=np.zeros(nh_vector[i],dtype=theano.config.floatX)))

			if i == 0:
				continue
			else: 
				#my understanding is that whm1_n should have one less dimension than the whh_n list
				whm1h_n.append(theano.shared(name='wh',
							value=0.2 * np.random.uniform(-1.0, 1.0,(nh_vector[i-1], nh_vector[i])).astype(theano.config.floatX)))

		self.wx_n = wx_n 
		self.whh_n = whh_n
		self.whm1h_n = whm1h_n
		self.bh_n = bh_n
		self.h0_n = h0_n 

		self.wy = theano.shared(name='wy',
							   value=0.2 * np.random.uniform(-1.0, 1.0,
							   (nh_vector[-1], ny))
							   .astype(theano.config.floatX)) #output weights
		
		self.by = theano.shared(name='b',
							   value=np.zeros(ny,
							   dtype=theano.config.floatX)) #output layer bias
		
		self.sequence_length = sequence_length

	def save_param(self,pickle_file):

		pickle_me = {
					'param':[self.wx, self.wh, self.wy, self.bh, self.by, self.h0]
		}

		pickle.dump( pickle_me, open(pickle_file, 'wb') )

	def load_param(self,pickle_file):

		pickle_me = pickle.load(open(pickle_file,'rb'))

		param = pickle_me['param']

		self.wx, self.wh, self.wy, self.bh, self.by, self.h0 = param


	def feed_through_1(self,x,h_tm1):
		"""
		this is for n = 0 (the first layer)
		"""
		ht = T.tanh(T.dot(x,self.wx_n[0]) + T.dot(h_tm1, self.whh_n[0]) + self.bh_n[0])

		# y_hat = self.by + T.dot(h,self.wy) 

		# y_guess = T.nnet.softmax(y_hat) 

		return ht

	def feed_through_n(self, x, h_nm1, h_tm1, n_step):
		"""
		feed through the RNN , n>0
		"""
		# if n_step == 0:
		# 	hnt = T.tanh(T.dot(x,self.wx_n[0]) + 
		# 		T.dot(h_tm1, self.whh_n[0]) + 
		# 		self.bh_n[0])

		# elif n_step > 0:
		hnt = T.tanh(T.dot(x,self.wx_n[n_step]) + 
					T.dot(h_nm1, self.whm1h_n[n_step-1]) + 
					T.dot(h_tm1, self.whh_n[n_step]) +
					self.bh_n[n_step])

		return hnt

	def cross_entropy_loss(self, x):#, y):
		"""
		Cross entropy loss function. Average of cross entropy across a minibatch 
		"""
		# write it with a for loop first. 
		# h = []
		# for n in xrange(len(self.nh_vector)):
		# 	hs = [self.h0_n[n]] 
		# 	for t in xrange(1,self.sequence_length):
		# 		if n == 0:
		# 			hs.append(self.feed_through_1(x[t],hs[t-1]))
		# 		else:
		# 			hs.append(self.feed_through_n(x[t],hs[t-1],h[n-1][t],n))
		# 	h.append(hs)



		# h_final = h[-1]
		# return h_final
		h_init, _ = theano.scan(fn=self.feed_through_1,
						sequences=x,
						outputs_info=self.h0_n[0]) 

		return h_init

		# h_next, _ = 



		# for i in xrange(1,len(self.nh_vector)):
		# 	h, _ = theano.scan(fn = self.feed_through_n,
		# 						sequences = [x, h],
		# 						outputs_info=[self.h0_n[i],i,None])

		# h_final = h[:,0,:]

		# y_guess = T.nnet.softmax(self.by + T.dot(h,self.wy))

		# return y*T.log(y_guess) + (1.0-y)*T.log(1.0-y_guess)


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




	



