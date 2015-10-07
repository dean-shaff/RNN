from character_mapping import Character_Map 
from RNN import RNNClass 
import numpy as np 

def train_NN(mu, n_epoch, mini_batch):
	"""
	Train the neural net 
	"""
	text_test = './../texts/melville.txt'
	char_map_obj = Character_Map(text_test,'mapping.dat',overwrite=True, break_line=None)
	char_map_obj.k_map()
	x, y, shared_x, shared_y = char_map_obj.gen_x_and_y(filename=None)

	nh = 100
	nx = len(char_map_obj.unique_char)
	ny = nx 

	trainer = RNNClass(nh,nx,ny)
	trainer.train_index((shared_x,shared_y),mu,n_epoch,mini_batch)

def load_shit():
	text_test = './../texts/melville.txt'
	char_map_obj = Character_Map(text_test,'mapping.dat',overwrite=True, break_line=None)
	unique_char = char_map_obj.unique_char
	char_map_obj.k_map()
	x, y, shared_x, shared_y = char_map_obj.gen_x_and_y(filename=None)
	# print(shared_x, shared_y.get_value().shape[0])
	nh = 100
	nx = len(char_map_obj.unique_char)
	ny = nx 

	trainer = RNN(nh,nx,ny)
	trainer.load_param('param_6-10_17:52/param_epoch199.dat')

	f = trainer.compile_gen_sentence()

	for xi in x[100:150]:
		y_guess = f(xi[0])
		y_argmax = [np.argmax(y) for y in y_guess] 
		char_y = [unique_char[int(yi)] for yi in y_argmax]
		print(char_y)
		print(''.join(char_y))


	return trainer, char_map_obj, x, y, shared_x, shared_y

if __name__ == '__main__':
	# train_NN(mu=0.001,n_epoch=200,mini_batch=1000) #change the learning rate
	load_shit()









