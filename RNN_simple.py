from character_mapping import Character_Map
from theano import tensor as T 
import theano 
import numpy as np 


filename = './../texts/melville.txt'
foo = Character_Map(filename,'mapping.dat',overwrite=True, break_line=5000)
map_matrix = foo.k_map()
x,y,shared_x, shared_y = foo.gen_x_and_y(filename=None)

# define weight matrices 

wx = theano.shared(name='wx',
					value=0.2 * np.random.uniform(-1.0, 1.0,
					(nx, nh))
					.astype(theano.config.floatX)) #input weights

wh = theano.shared(name='wh',
				value=0.2 * np.random.uniform(-1.0, 1.0,
				(nh, nh))
				.astype(theano.config.floatX)) #hidden layer weights

wy = theano.shared(name='wy',
			   value=0.2 * np.random.uniform(-1.0, 1.0,
			   (nh, ny))
			   .astype(theano.config.floatX)) #output weights

bh = theano.shared(name='bh',
				value=np.zeros(nh,
				dtype=theano.config.floatX)) #hidden layer bias

by = theano.shared(name='b',
			   value=np.zeros(ny,
			   dtype=theano.config.floatX)) #output layer bias

h0 = theano.shared(name='h0',
			value=np.zeros(nh,
			dtype=theano.config.floatX)) #initial h vector 

sequence_length = 15