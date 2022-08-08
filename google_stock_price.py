#####################################################################################################################
#   Project : Stock Price Prediction using Recurrent Neural Network 
#   This is a starter code in Python 3.6 for a Stock Price Prediction using Recurrent Neural Network and back propagation
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################

#importing the headers
#importing matplotlib for plotting

from math import sqrt
import pandas
import numpy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as matplt
import seaborn as seaborn
import logging
logging.basicConfig(filename="google_stock_price_RNN.log", level=logging.INFO, format='%(message)s')
#Declaring the class Recurrent_neural_network 
#Initializing various functions for obtaining the values to predict the stocks

class Recurrent_neural_network:
    def __init__(self):
        pass
    
    #Performed Initializing a function get_data
    #Used pandas to obtain the data from the csv file
    def get_data(self,URL):
        if URL == '':
            URL = "https://drive.google.com/uc?id=1jYUvEV7fHEyCQiw5Fm71JXfLIiji_JBb"
        data = pandas.DataFrame(pandas.read_csv(URL))
        print(data.head(3))
        return data

    #Performed Initializing a function check_null_duplicates
    #Fucntion is created to find out on duplicate records
    def check_null_duplicates(self,processed_data):
        if processed_data.isnull().sum().sum() == 0:
            print("No Null values found")
        else:
            value = processed_data.isnull().sum().sum()
            print(f"Number of Null records found = {value}" )

        if processed_data.duplicated().sum() == 0:
            print("No duplicates found")
        else:
            value = processed_data.duplicated().sum()
            print(f"Number of duplicate records found = {value}")

    #Performed Initializing a function data_processing
    #Fucntion is defined for pre_processing the data
    def data_processing(self,data):
        print("\nPre-Processing the Data:\n")
        processed_data = data
        self.check_null_duplicates(processed_data)

        print("correlation_matrix ",processed_data.corr())
        matplt.figure(figsize = (5, 5))
        seaborn.heatmap(processed_data.corr(), cmap = "YlGnBu")

        #                                    ["Date", "Open", "Close", "High", "Low", "Volume","Adj Close"] 
        processed_data = processed_data.drop(["Date", "Adj Close", "Close", "High",  "Low", "Volume"], axis = 1)
        
        print(" Column: is used for our precidtion and analysis ")
        print(processed_data.head())

        min_max_scaler = MinMaxScaler()
        scaled_data = min_max_scaler.fit_transform(processed_data)
        return scaled_data,processed_data,min_max_scaler

    #Performed Initializing a function train_test_split_data
    #Fucntion is defined for splitting of the data into various parameter on basis of test and train it.
    def train_test_split_data(self,scaled_data,train_split_percentage = 0.80):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        t_split_index = round((len(scaled_data) - 1) * train_split_percentage)

        # data = [n-2, n-1]
        # input data is [Price on 1st Day, Price on 2nd Day] and [Price on 2nd Day, Price on 3rd Day] the data to predict is [Price on 3rd Day, Price on 4th Day]

        for index in range(2, t_split_index):
            tTemp = [scaled_data[index - 2], scaled_data[index - 1]]
            x_train.append(tTemp)

        for element in scaled_data[2 : t_split_index]:
            y_train.append(element)

        for index in range(t_split_index + 2, len(scaled_data)):
            tTemp = [scaled_data[index - 2], scaled_data[index - 1]]
            x_test.append(tTemp)

        for element in scaled_data[t_split_index + 2 : ]:
            y_test.append(element)

        y_train = numpy.array(y_train)
        y_test = numpy.array(y_test)
        x_train = numpy.array(x_train)
        x_test = numpy.array(x_test)

        print('shapes of xTrain,yTrain,xTest,yTest', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        return x_train, y_train, x_test, y_test

    #Performed Initializing a function calculate_loss
    #Fucntion is defined for calucating the loss
    def calculate_loss(self, actual_value, model_predicted):
        return numpy.mean(numpy.square(actual_value - model_predicted))

    #Performed Initializing a function relu_activation_function
    #Fucntion is defined for obtaing relu_activation_function
    def relu_activation_function(self, input, differentitate = False):
        if(False == differentitate):
            return numpy.maximum(input, 0)
        else:
            return (input > 0)

    #Performed Initializing a function tanH_activation_function
    #Fucntion is defined for obtaing tanH_activation_function
    def tanH_activation_function(self,input, differentitate = False):
        if(False == differentitate):
            return numpy.tanh(input)
        else:
            return (1 - numpy.square(numpy.tanh(input)))

    #Performed Initializing a function tanH_activation_function
    #Fucntion is defined for obtaing tanH_activation_function
    def forward_propagation(self,hidden_neurons,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights, activation_function,input):
        output_hidden_States = []
        output_hidden_States.append(numpy.zeros((hidden_neurons, 1)))
        step = 0

        while(step < input.shape[0]):
            tWeightsSum = (input_to_hidden_weights @ input[[step]].T) + (hidden_to_output_weights @ output_hidden_States[-1])
            tNextHiddenStage = activation_function(tWeightsSum)
            output_hidden_States.append(tNextHiddenStage)
            step = step + 1

        hidden_output = hidden_to_hidden_weights @ output_hidden_States[-1]
        return output_hidden_States, hidden_output

    #Performed Initializing a function backward_propagation
    #Fucntion is defined for performing backpropagation to neural network
    def backward_propagation(self,input, output, hidden_states, hidden_output,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights,lr,activation_func):
        i_loss = self.calculate_loss(output, hidden_output)

        i_input_to_hidden_weights = numpy.zeros(input_to_hidden_weights.shape)
        i_hidden_to_hidden_weights = numpy.zeros(hidden_to_hidden_weights.shape)
        i_hidden_to_output_weights = numpy.zeros(hidden_to_output_weights.shape)

        i_error_slope = numpy.dot(hidden_to_hidden_weights.T, i_loss)
        i_hidden_states_slope = i_error_slope * activation_func(hidden_states[-1], differentitate = True)

        for step in reversed(range(input.shape[0])):
            i_temp = i_hidden_states_slope @ hidden_states[step-1].T
            i_hidden_to_output_weights = i_hidden_to_output_weights + i_temp
            i_temp = i_hidden_states_slope @ input[[step-1]]
            i_input_to_hidden_weights = i_input_to_hidden_weights + i_temp

        i_temp = (hidden_output - output) @ hidden_states[-1].T
        i_hidden_to_hidden_weights = i_hidden_to_hidden_weights + i_temp
        input_to_hidden_weights = input_to_hidden_weights - lr * i_input_to_hidden_weights
        hidden_to_hidden_weights = hidden_to_hidden_weights - lr * i_hidden_to_hidden_weights
        hidden_to_output_weights = hidden_to_output_weights - lr * i_hidden_to_output_weights
        return input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights


    #Performed Initializing a function train_model
    #Fucntion is defined to train the model
    
    def train_model(self, x_train, y_train, x_test, y_test, i_input_neurons = 1, i_hidden_neurons = 10, i_output_neurons = 1, lr = 0.01, activation_function_name = "TanH", iterations = 50):
        input_neurons = i_input_neurons
        hidden_neurons = i_hidden_neurons
        output_neurons = i_output_neurons
        learning_rate = lr
        #iterations = iterations
        activation_func = self.relu_activation_function
        if("relu" == activation_function_name):
            activation_func = self.relu_activation_function
            
        if("TanH" == activation_function_name):
            activation_func = self.tanH_activation_function

        input_to_hidden_weights = (numpy.random.uniform(0, 1, (hidden_neurons, input_neurons)) / 2)
        hidden_to_hidden_weights = (numpy.random.uniform(0, 1, (output_neurons, hidden_neurons)) / 2)
        hidden_to_output_weights = (numpy.random.uniform(0, 1, (hidden_neurons, hidden_neurons)) / 2)

        index = 0
        while(index < iterations):
            if(index == iterations - 1):
                train_results = []

            for stage in range(x_train.shape[0]):
                t_hidden_states, t_hidden_output = self.forward_propagation(hidden_neurons,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights, activation_func,x_train[stage])
                if(index == iterations - 1):
                    train_results.append(t_hidden_output.tolist()[0])
                input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights = self.backward_propagation(x_train[stage], y_train[stage], t_hidden_states, t_hidden_output,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights,learning_rate,activation_func)
            index = index + 1
        train_results = numpy.array(train_results).T[0]
        return train_results,t_hidden_output,hidden_neurons,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights, activation_func


    #Performed Initializing a function test_model
    #Fucntion is defined to test the model
    def test_model(self,x_test,t_hidden_output,hidden_neurons,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights, activation_func):
        test_results = []
        for stage in range(x_test.shape[0]):
            t_hidden_states, t_hidden_output = self.forward_propagation(hidden_neurons,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights, activation_func,x_test[stage])
            test_results.append(t_hidden_output.tolist()[0])
        test_results = numpy.array(test_results).T[0]
        return test_results

    #Performed Initializing a function model_error
    #Fucntion is defined to obtain the model error
    #Obtain Training Values for the model
    #Obatin Test Values for the model
    def model_error(self,y_train,train_results,test_results,y_test):
        print("Train MSE = ", mean_squared_error(y_train, train_results))
        print("Train RMSE = ", sqrt(mean_squared_error(y_train, train_results)))
        print("Train MAE = ", mean_absolute_error(y_train, train_results))
        print("Train R2 = ", r2_score(y_train, train_results))

        print("Test MSE = ", mean_squared_error(y_test, test_results))
        print("Test RMSE = ", sqrt(mean_squared_error(y_test, test_results)))
        print("Test MAE = ", mean_absolute_error(y_test, test_results))
        print("Test R2 = ", r2_score(y_test, test_results))

    #Performed Initializing a function model_error_performance
    #Fucntion is defined to obtain the performance of model error
    #Obtain Training Error for the model
    #Obatin Test Error for the model
    #History Curve (Plot of Accuracy against training steps) 
    def model_error_performance(self,train_results,test_results,y_train,y_test,processed_data,min_max_scaler):
        train_results = min_max_scaler.inverse_transform(train_results.reshape(-1,1))
        test_results = min_max_scaler.inverse_transform(test_results.reshape(-1,1))
        y_train = min_max_scaler.inverse_transform(y_train.reshape(-1,1))
        y_test = min_max_scaler.inverse_transform(y_test.reshape(-1,1))

        self.model_error(y_train,train_results,test_results,y_test)

        matplt.figure(figsize = (20, 15))
        matplt.ylabel("Google Stock Price")
        matplt.xlabel("Number of days")
        matplt.title("Google Stock Price Prediction using Recurrent neural network ")

        matplt.plot(train_results, label = "Training o/p")
        matplt.plot(processed_data, label = "Actual Price")
        test_results = [tDataPoint for tDataPoint in test_results]
        test_results.insert(0, train_results[-1])
        matplt.plot([tDataPoint for tDataPoint in range(len(train_results) - 1, len(train_results) + len(test_results) - 1)], test_results, label = "Test prediction Output")

        matplt.legend()
        matplt.grid()
        matplt.show()


    #Performed Initializing a function fit
    #Fucntion is defined to data fit
    def fit(self,data):
        scaled_data,processed_data,min_max_scaler = self.data_processing(data)
        x_train, y_train, x_test, y_test= self.train_test_split_data(scaled_data,train_split_percentage = 0.80)

        train_results,t_hidden_output,hidden_neurons,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights, activation_func = self.train_model( x_train, y_train, x_test, y_test)
        test_results = self.test_model(x_test,t_hidden_output,hidden_neurons,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights, activation_func)
        
        self.model_error_performance(train_results,test_results,y_train,y_test,processed_data,min_max_scaler)

    #Function defined to generate logs for 
    #all values of iterations, neurons, lrs, and errors.
    def logGenerator(self,data):
        #Testing on the following parameters
        tItns = [100, 200]
        tActFuns = [ "Relu","TanH"]
        tLRs = [0.001, 0.01 ]
        tTrain = [0.80, 0.90]
        tNeurons = [10, 12]

        activation = lr = epochs = neurons_list = train = list()

        # Create the recurrent neural network and be sure to keep track of the performance
        for A_fnx in tActFuns:
            #print("function --> ",function)
            for rate in tLRs:
                #print("rate --> ",rate)
                for iterations in tItns:
                    #print("iterations --> ",iterations)
                    for neurons in tNeurons:
                        #print("neurons --> ",neurons)
                        for split in tTrain:
                            #print("split --> ",split)

                            # Store training parameters
                            activation.append(A_fnx)
                            lr.append(rate)
                            epochs.append(iterations)
                            neurons_list.append(neurons)
                            train.append(split)

                            logging.info(f'activation_function: {A_fnx}')
                            logging.info(f'Learning rate: {rate}')
                            logging.info(f'iterations: {iterations}')
                            logging.info(f'Split: {split}')

                            print(f'activation_function: {A_fnx}')
                            print(f'Learning rate: {rate}')
                            print(f'iterations: {iterations}')
                            print(f'Split: {split}')

                            # Split Data into Train and Test Data
                            scaled_data,processed_data,min_max_scaler = self.data_processing(data)
                            x_train, y_train, x_test, y_test= self.train_test_split_data(scaled_data,train_split_percentage = split)
                            # Train the model
                            train_results,t_hidden_output,hidden_neurons,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights, activation_func = self.train_model( x_train, y_train, x_test, y_test, i_input_neurons = 1, i_hidden_neurons = neurons, i_output_neurons = 1, lr = rate, activation_function_name = A_fnx, iterations = iterations)
                            # Test the model
                            test_results = self.test_model(x_test,t_hidden_output,hidden_neurons,input_to_hidden_weights,hidden_to_hidden_weights,hidden_to_output_weights, activation_func)
                            # generates graphs    
                            self.model_error_performance(train_results,test_results,y_train,y_test,processed_data,min_max_scaler)
                            
                            # Update the result metrics
                            train_results = min_max_scaler.inverse_transform(train_results.reshape(-1,1))
                            test_results = min_max_scaler.inverse_transform(test_results.reshape(-1,1))
                            y_train = min_max_scaler.inverse_transform(y_train.reshape(-1,1))
                            y_test = min_max_scaler.inverse_transform(y_test.reshape(-1,1))

                            train_mse = mean_squared_error(y_train, train_results)
                            test_mse = mean_squared_error(y_test, test_results)
                            train_RMSE = sqrt(mean_squared_error(y_train, train_results))
                            test_RMSE = sqrt(mean_squared_error(y_test, test_results))
                            train_MAE = mean_absolute_error(y_train, train_results)
                            test_MAE = mean_absolute_error(y_test, test_results)
                            train_R2 = r2_score(y_train, train_results)
                            test_R2 = r2_score(y_test, test_results)

                            logging.info(f'Train Mean Squared Error: {train_mse}')
                            logging.info(f'Test Mean Squared Error: {test_mse}')
                            logging.info(f'Train RMSE: {train_RMSE}')
                            logging.info(f'Test RMSE: {test_RMSE}')
                            logging.info(f'Train  Mean absolute Error: {train_MAE}')
                            logging.info(f'Test Mean absolute Error: {test_MAE}')
                            logging.info(f'Train R2 score: {train_R2}')
                            logging.info(f'Test R2 score: {test_R2}')
                            logging.info(f'\n----------------------------\n')

if __name__ == "__main__":
    RNN = Recurrent_neural_network()
    data = RNN.get_data('https://drive.google.com/uc?id=1jYUvEV7fHEyCQiw5Fm71JXfLIiji_JBb')
    RNN.fit(data)
    RNN.logGenerator(data)
    #exit()