import numpy as np
import math
from sklearn.utils import shuffle


def to_numpy(x,y):
	#takes pandas dataframe and converts them to numpy array
	x = x.to_numpy()
	y = y.to_numpy()
	return x, y

def scale_cost(y):
	s = sum(y)
	norm = [float(i)/s for i in y]
	return norm

def configure_data(x_train, y_train, x_test, y_test):
	x_train, y_train = to_numpy(x_train, y_train)
	x_test, y_test = to_numpy(x_test, y_test)
	y_train = y_train.reshape(y_train.shape[0], 1)
	y_test = y_test.reshape(y_test.shape[0],1)
	return x_train, y_train, x_test, y_test 



class layer:

	def __init__(self, n_neurons, input_shape, activation):
		self.weights = np.random.randn(n_neurons, input_shape)*(math.sqrt(2.0/input_shape))
		self.biases = np.zeros(n_neurons)
		self.biases = self.biases.reshape((self.biases.shape[0],1))
		self.activation = activation

	def Relu(self, input):
		output = input
		output[output<0] = 0
		return output

	def forward(self, input):
		if(self.activation == "relu"):
			return layer.Relu(self, (np.dot(self.weights, input)+np.dot(self.biases, input.shape[1])) )
		else:
			print("Sorry, other activation functions are still in development!")



class create_model:
	def __init__ (self, layers):
		self.layers = layers
		self.A = []
		self.learning_rate = 0.001
		self.epochs = 100
		self.clip = 1000
		self.plot_x = []
		self.plot_y = []
		self.u = 0
		self.optimizer = 'GD'
		self.loss = 'mse'
		self.batch_size = 100

	def printLayerWeights(self):
		for layer in self.layers:
			print(layer.weights)


	def compile(self, optimizer, learning_rate, loss, gradient_clip):		
		self.optimizer = optimizer
		self.learning_rate = learning_rate
		self.loss = loss
		self.clip = gradient_clip


	def clip(self, dw):
		norm = np.linalg.norm(dw)
		if(norm< self.clip and norm > -1 * self.clip):
			pass
		elif(norm> self.clip):
			dw = 1.6 * (dw/norm)
		elif(norm< -1 * self.clip):
			dw =  -1.6 * (dw/norm)
		return dw

	def dRelu(self , z):
   		return np.where(z <= 0, 0, 1)
	
	def predict(self, x):
		y_prediction = []
		if(self.optimizer == "MBGD"):
			for i in range(x.shape[0]%self.batch_size):
				x = np.delete(x, x.shape[0]-1, axis= 0)
			x = x.reshape(x.shape[0]//self.batch_size,-1,x.shape[1])
			for batch in x:
				a = batch.T
				for layer in self.layers:
					temp = layer.forward(a)
					a = temp
				for i in range(self.batch_size):
					y_prediction.append(np.array([a[0][i]]))

		elif(self.optimizer == "SGD"):
			for sample in x:
				a = sample.reshape(1, sample.shape[0]).T
				for layer in self.layers:
					temp = layer.forward(a)
					a = temp
				y_prediction.append(a[0])
		elif(self.optimizer == "GD"):
			row = []
			for _ in range(x.shape[1]):
				row.append(0)
			row = np.array([row])
			for i in range(self.u-x.shape[0]):
				x = np.concatenate((x,row), axis = 0)
			a = x.T
			for layer in self.layers:
				temp = layer.forward(a)
				a = temp
			return a.T
		return np.array(y_prediction)
		

	def create_batch(self, x, y, size):
		for i in range(x.shape[0]%size):
			x = np.delete(x, i, axis= 0)
			y = np.delete(y, i, axis= 0)
		x = x.reshape(x.shape[0]//size,-1,x.shape[1])
		y = y.reshape(y.shape[0]//size,-1,y.shape[1])
		return x, y


	def mse(self, y_pred, y_actual):
		return (1/2)*1/y_pred.shape[1] * np.sum(np.square(y_pred-y_actual))


	def fit(self, x, y, epochs, batch_size):
		self.u = y.shape[0]
		self.batch_size = batch_size
		if(self.optimizer == "GD"):
			print("Optimizing with batch gradient descent!", "\n", "\n")
			self.GD_optimizer(x, y, epochs)
		if(self.optimizer == "SGD"):
			print("Optimizing with stochastic gradient descent!", "\n", "\n")
			self.SDG_optimizer(x, y, epochs)
		if(self.optimizer == "MBGD"):
			print("Optimizing with mini batch gradient descent!", "\n", "\n")
			self.MBGD_optimizer(x, y, epochs, batch_size)


	def GD_optimizer(self, x, y, epochs):
		for epoch in range(epochs):
			x, y = shuffle(x, y)
			cost = self.forward_propagation(x.T, y.T)
			print("Cost : ", cost)
			self.plot_y.append(cost)
			self.plot_x.append(epoch)
		self.plot_y = scale_cost(self.plot_y)


	def SDG_optimizer(self, x, y, epochs):
		for epoch in range(epochs):
			x, y = shuffle(x,y)
			print(x.shape)
			for i in range(x.shape[0]):
				x1 = x[i].reshape(1, x[i].shape[0])
				y1 = y[i].reshape(1, y[i].shape[0])
				cost = self.forward_propagation(x1.T, y1.T)
				print("cost : ",cost)
				self.plot_y.append(cost)
				self.plot_x.append(epoch)
		self.plot_y = scale_cost(self.plot_y)


	def MBGD_optimizer(self, x, y, epochs, batch_size):
		x, y = self.create_batch(x, y, batch_size)
		count = 0
		for epoch in range(epochs):
			x, y = shuffle(x, y)
			for i in range(x.shape[0]):
				cost = self.forward_propagation(x[i].T, y[i].T)
				print("Cost : ", cost)
				count+=1
				self.plot_y.append(cost)
				self.plot_x.append(count)
		self.plot_y = scale_cost(self.plot_y)



	def forward_propagation(self, x, y):
		a = x
		self.A.clear()
		for layer in self.layers:
			a = layer.forward(a)
			self.A.append(a)
		cost = self.mse(a, y)
		self.backward_propagation(x,y)
		return cost


	def backward_propagation(self, x, y):
		n = len(self.A)
		del_C = -1*(y-self.A[n-1])
		delta = np.multiply(del_C, create_model.dRelu(self, self.A[n-1]))
		dw = np.zeros((1,1))
		db = np.zeros((1,1))

		for i in range(n):
			if(i == 0):
				dw = create_model.clip(self, np.dot(delta,self.A[n-2].T))
				self.layers[n-i-1].weights = self.layers[n-i-1].weights - (self.learning_rate * dw)
				self.layers[n-i-1].biases = self.layers[n-i-1].biases - (self.learning_rate * create_model.clip(self, delta))

			else:
				delta_curr = np.multiply(np.dot(self.layers[n-i].weights.T, delta), create_model.dRelu(self, self.A[n-i-1]))
				if (i < n-1):
					dw_curr = create_model.clip(self, np.dot(delta_curr, self.A[n-i-2].T))

				elif (i == n-1):
					dw_curr = create_model.clip(self, np.dot(delta_curr, x.T))
				
				self.layers[n-i-1].weights = self.layers[n-i-1].weights - (self.learning_rate*dw_curr)
				self.layers[n-i-1].biases = self.layers[n-i-1].biases - (self.learning_rate* create_model.clip(self, delta_curr))
				delta = delta_curr
				dw = dw_curr
		return self.A[len(self.A)-1]

