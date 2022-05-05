# Neural Network from Scratch

I have successfully implemented the Neural Network from scratch in python.      
## Modules used: 
Numpy, Pandas, Maths, Scikit learn.

## Description:
This neural network is able to solve the regression based problems. Three Optimizers have been implemented:
1. Batch Gradient Descent:  Whole training sample feeds at a time and update the weight parameters 
2. Mini Batch Gradient Descent: One training sample at a time, calculates cost for one particular sample and updates the weight parameters.
3. Stochastic Gradient Descent: Creates a mini batches of samples and feeds them to network, calculates cost for one particular mini batch and updates the weight parameters

Syntax for creaing the model, adding the layers, number of neurons in each layers are almost same. Rectified linear unit (ReLu) activation function has been implemented for solving. 

## Results: 
I have used this neural network framework for the time series stock marcket price prediction. The predicted price is quite close to the testing data. Please find the attached jupyter notebbok file for detailed view of the results. 


## Problems faced during implementation:

1. Implementing the back propagation algorithm recursively, 
![Backpropagation algorithm](https://github.com/ujjawalmodanwal/Neural_Network_from_scratch/blob/main/Images/Screenshot%20from%202022-05-05%2023-33-03.png)


2. Generalization in model prediction. This was solved by shuffling the train data after each epoch. 
3. Weight parameters exploding or vanishing problem during the training process, in both the cases network becomes dead. This occurs due to large loss value, which makes the gradients to take large step. During back -propagation these gradients get accumulated in weight matrices, resulting in explosion or overflow on integer values.

**Initial Weights**       
![intial weights](https://github.com/ujjawalmodanwal/Neural_Network_from_scratch/blob/main/Images/Screenshot%20from%202021-11-23%2016-38-55.png)       
**Exploded Weights**       
![exploded weights](https://github.com/ujjawalmodanwal/Neural_Network_from_scratch/blob/main/Images/Screenshot%20from%202021-11-23%2016-39-07.png)       
**Final weights**       
![final weights](https://github.com/ujjawalmodanwal/Neural_Network_from_scratch/blob/main/Images/Screenshot%20from%202021-11-23%2016-39-16.png)

This can be removed by gradient clipping method. If values in gradient matrices goes beyond a specified range, we insert new values in that place. 

## Batch Gradient Descent 
**Cost Vs Epoch**          
![cost_v_epoch](https://github.com/ujjawalmodanwal/Neural_Network_from_scratch/blob/main/Images/Screenshot%20from%202021-11-23%2017-37-58.png)
![prediction](https://github.com/ujjawalmodanwal/Neural_Network_from_scratch/blob/main/Images/Screenshot%20from%202021-11-23%2018-18-13.png)


## Mini Batch Gradient Descent
**Cost Vs Epoch**       
![cost v epoch](https://github.com/ujjawalmodanwal/Neural_Network_from_scratch/blob/main/Images/Screenshot%20from%202021-11-23%2019-04-55.png)
![prediction](https://github.com/ujjawalmodanwal/Neural_Network_from_scratch/blob/main/Images/Screenshot%20from%202021-11-23%2018-26-53.png)
