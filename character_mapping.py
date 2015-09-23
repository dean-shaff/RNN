# -*- coding: utf-8 -*-
"""
character_mapping.py
"""
try:
	import cPickle as pickle 
except:
	import pickle
import os
import time
import numpy as np 
import theano

class Character_Map(object):

	def __init__(self,text_file, pickle_file,break_line=5000,overwrite=False):
		"""
		This function creates a dictionary lookup table for the characters present in a text. 
		if 'a' is the first charcter in the character set, then it will be given index 1, 'b' will be 2
		and so on. 
		This function follows the "text_to_tensor" method in samim23's implemenation, found here:
		https://github.com/samim23/char-rnn-api/blob/master/util/CharSplitLMMinibatchLoader.lua

		args:
			-text_file: the text_file you want to read in 
			-pickle_file: the name of the pickle file made with this function, containing 
				relevant info
			-overwrite: if True, overwrites current pickle file. if false, doesn't overwrite.
		"""
		def gen_char_map():
			t1 = time.time()
			dump = str()	
			with open(text_file,'r') as reader:
				for index,line in enumerate(reader):
					if index == break_line:
						break
					else:
						dump += line
			dump = dump.translate(None,'\t\n') #get rid of formatting. I don't know if I actually want to do this, however.
			char_list = list(dump)
			unique_char = list(set(char_list))
			unique_char.sort()
			char_dict = {char:i for (char, i) in zip(unique_char,xrange(len(unique_char)))} # I don't know how zeros will behave, hence i+1
			mapping = [char_dict[char] for char in char_list]

			# now pickle the result:

			pickle_me = {'char_dict':char_dict,
						'mapping':mapping, 
						'char_list':char_list, 
						'unique_char':unique_char}

			pickle.dump( pickle_me , open( pickle_file, "wb" ) )
			print("Time creating character mapping and pickling: {:.4f} sec".format(time.time()-t1))

			return pickle_me

		if os.path.isfile(pickle_file) and overwrite == False:
			t1 = time.time()
			pickle_me = pickle.load(open(pickle_file,'rb'))
			print("Time loading in pickle file: {:.4f} sec".format(time.time()-t1))

		elif overwrite == True or not os.path.isfile(pickle_file):
			pickle_me = gen_char_map()

		self.pickle_me = pickle_me
		self.mapping = pickle_me['mapping']
		self.char_dict = pickle_me['char_dict']
		self.unique_char = pickle_me['unique_char']
		self.char_list = pickle_me['char_list']


	def k_map(self):
		"""
		mapping from a list of indices to a k-map. For each character in the text
		we create a vector of length K, where K is the number of unique characters in the text 
		the kth position in the vector is equal to one, and the rest are zeros. 

		mapping_vector -- the vector containing the data, now mapped to indices
		unique_char -- the list of unique characters present in the text 
		"""
		t1 = time.time()
		mapping_matrix = [] 
		for index in self.mapping:
			vector = np.zeros(len(self.unique_char),dtype=float)
			vector[index] = 1.0
			mapping_matrix.append(vector)
		print("Time creating k map {:.3f} sec".format(time.time()-t1))
		self.mapping_matrix = mapping_matrix
		return mapping_matrix

	def gen_x_and_y(self,borrow=True,filename='x_y.dat',sequence_length=15):
		"""
		Generates two lists, each of length N-1, where N is the total number of characters
		in the input text. This means we miss the last character
		x is the input for NN, and y is the theoretical output. y is just x, but shifted one index over.
		the shared variables are so they are compatible with theano.
		"""
		t1 = time.time()

		def make():
			x = self.mapping_matrix[:-1] #missing last index
			y = self.mapping_matrix[1:] #missing first index
			length_x = len(x)
			# print(length_x//sequence_length)
			x = [sequence for sequence in [x[i:i+sequence_length] for i in xrange(length_x//sequence_length)]]
			y = [sequence for sequence in [y[i:i+sequence_length] for i in xrange(length_x//sequence_length)]]
			shared_x = theano.shared(np.asarray(x,dtype=theano.config.floatX),borrow=borrow)
			shared_y = theano.shared(np.asarray(y,dtype=theano.config.floatX),borrow=borrow)
			return (x, y, shared_x, shared_y)

		if filename == None:
			x,y, shared_x, shared_y = make()
			print("Time creating arrays: {:.3f} sec".format(time.time()-t1))

		elif filename != None and not os.path.isfile(filename):
			x,y, shared_x, shared_y = make()
			pickle_dic = {'x':x,
							'y':y,
							'shared_x':shared_x,
							'shared_y':shared_y}
			pickle.dump(pickle_dic , open(filename, "wb" ))
			print("Time creating arrays and pickling: {:.3f} sec".format(time.time()-t1))

		elif os.path.isfile(filename):
			pickled = pickle.load(open(filename,'rb'))
			x = pickled['x']
			y = pickled['y']
			shared_x = pickled['shared_x']
			shared_y = pickled['shared_y']
			print("Time loading in arrays: {:.3f} sec".format(time.time()-t1))
		
		return (x, y, shared_x, shared_y)


def test():
	filename = './../texts/melville.txt'
	foo = Character_Map(filename,'mapping.dat',overwrite=True)
	# print(len(foo.mapping))
	map_matrix = foo.k_map()
	x,y,shared_x, shared_y = foo.gen_x_and_y(filename=None)
	# print(shared_x.get_value()[:10])
	# print(shared_y.get_value()[:10])
	# print(shared_y.get_value().shape)

	# print(len(x))
	print(shared_x.get_value().shape)
	# print(shared_x.get_value()[0])

if __name__ =='__main__':
	test()