import math
import time
import sys
import numpy as np
import pandas as pd


def read_input(train_image_filename, train_label_filename, test_image_filename):
    train_image_data, train_image_label, test_image_data = pd.read_csv(train_image_filename, header=None), \
                                                           pd.read_csv(train_label_filename, header=None), \
                                                           pd.read_csv(test_image_filename, header=None)
    return train_image_data, train_image_label, test_image_data


def get_init_weight(nodes_in_layer, nodes_in_next_layer):
    limit = math.sqrt(6/(nodes_in_layer+nodes_in_next_layer))
    return np.random.uniform(low=-limit, high=limit, size=(nodes_in_next_layer, nodes_in_layer))


def get_cross_entropy_loss(predicted, actual):
    return -1*np.sum(actual*np.log(predicted))/predicted.shape[0]


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def softmax(values):
    exps = np.exp(values - values.max())
    return exps / np.sum(exps, axis=0)


def sigmoid_derivative(values):
    return (np.exp(-values))/((np.exp(-values)+1)**2)


def softmax_derivative(values):
    exps = np.exp(values - values.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))


class NeuralNetwork:
    def __init__(self):
        self.input_size = 784
        self.output_size = 10
        self.batch = 32
        self.epochs = 1000
        self.learning_rate = 0.01
        self.hidden_layer_1_size = 128
        self.hidden_layer_2_size = 64
        self.loss_list = []
        self.accuracy_list = []
        self.split_ratio = 0.8

    def create_encoding(self):
        for i in range(1, self.output_size+1):
            self.complete_data[self.input_size+i] = 0
        for i in range(self.complete_data.shape[0]):
            column = self.input_size+int(self.complete_data.loc[i, self.input_size])
            self.complete_data.loc[i, column+1] = 1

    def split_data(self, train_image_data, train_image_label):
        self.complete_data = train_image_data
        self.complete_data = self.complete_data/255.0
        self.complete_data[self.input_size] = train_image_label[0]
        self.create_encoding()
        self.train = self.complete_data.sample(frac=self.split_ratio, random_state=42)
        self.valid = self.complete_data.loc[~self.complete_data.index.isin(self.train.index)].reset_index(drop=True)
        self.train = self.train.reset_index(drop=True)

        exclude_columns = [self.input_size + temp for temp in range(0, self.output_size + 1)]
        encoded_column = [self.input_size + temp2 for temp2 in range(1, self.output_size + 1)]
        rename_column_dict = dict()
        for i in range(0, self.output_size):
            rename_column_dict[self.input_size+1+i] = i
        self.X = np.asarray(self.train.loc[:, ~self.train.columns.isin(exclude_columns)])
        self.Y_encoded = np.asarray(self.train.loc[:,  self.train.columns.isin(encoded_column)].rename(columns=rename_column_dict))
        self.Y = np.asarray(self.train.loc[:, self.train.columns.isin([self.input_size])])
        self.valid_X = np.asarray(self.valid.loc[:, ~self.valid.columns.isin(exclude_columns)])
        self.valid_Y_encoded = np.asarray(self.valid.loc[:,  self.valid.columns.isin(encoded_column)].rename(
            columns=rename_column_dict))
        self.valid_Y = np.asarray(self.valid.loc[:, self.valid.columns.isin([self.input_size])])

    def initialize_weight(self):
        self.W1 = get_init_weight(self.input_size, self.hidden_layer_1_size)
        self.W2 = get_init_weight(self.hidden_layer_1_size, self.hidden_layer_2_size)
        self.W3 = get_init_weight(self.hidden_layer_2_size, self.output_size)

    def feedforward(self, x):
        #input_layer
        self.z1 = self.W1.dot(x)
        self.a1 = sigmoid(self.z1)

        #hidden_layer_1
        self.z2 = self.W2.dot(self.a1)
        self.a2 = sigmoid(self.z2)

        #hidden_layer_2
        self.z3 = self.W3.dot(self.a2)
        self.a3 = softmax(self.z3)

        return self.a3

    def backpropagation(self):
        cost_derivative = (self.a3 - self.y_encoded)
        dw3 = np.dot(cost_derivative, self.a2.T)

        cost_derivative = np.dot(self.W3.T, cost_derivative) * sigmoid_derivative(self.z2)
        dw2 = np.dot(cost_derivative, self.a1.T)

        cost_derivative = np.dot(self.W2.T, cost_derivative) * sigmoid_derivative(self.z1)
        dw1 = np.dot(cost_derivative, self.x.T)

        self.W3 = self.W3 - self.learning_rate * dw3
        self.W2 = self.W2 - self.learning_rate * dw2
        self.W1 = self.W1 - self.learning_rate * dw1

    def get_accuracy(self, input, target):
        correct_predictions = []

        for x, y in zip(input, target):
            predicted = self.feedforward(x)
            pred = np.argmax(predicted)
            correct_predictions.append(pred == np.argmax(y))
        return np.mean(correct_predictions)

    def shuffle(self):
        idx = [i for i in range(self.X.shape[0])]
        np.random.shuffle(idx)
        self.X = self.X[idx]
        self.Y_encoded = self.Y_encoded[idx]

    def train(self, train_image_data, train_image_label):
        self.initialize_weight()
        self.split_data(train_image_data, train_image_label)
        for epoch in range(self.epochs):
            self.shuffle()
            start_time = time.time()
            loss = 0
            for batch_num in range(self.X.shape[0]//self.batch - 1):
                start_index = batch_num*self.batch
                last_index = (batch_num+1)*self.batch
                self.x = self.X[start_index:last_index].T
                self.y_encoded = self.Y_encoded[start_index:last_index].T
                self.y_predicted = self.feedforward(self.x)
                self.backpropagation()
                loss = loss + get_cross_entropy_loss(self.y_predicted, self.y_encoded)
            accuracy = self.get_accuracy(self.valid_X, self.valid_Y_encoded)
            print("epoch:", epoch, "time_diff:", time.time()-start_time, "loss:", loss, "accuracy:", accuracy*100)
            if abs(accuracy) < 0.05:
                break




if __name__ == "__main__":
    #train_image_filename, train_label_filename, test_image_filename = sys.argv[1], sys.argv[2], sys.argv[3]
    train_image_filename, train_label_filename, test_image_filename = "train_image.csv", "train_label.csv", "test_image.csv"
    train_image_data, train_image_label, test_image_data = read_input(train_image_filename, train_label_filename,
                                                                      test_image_filename)
    nn = NeuralNetwork()
    nn.train(train_image_data, train_image_label)
    normalized_test_data = test_image_data/255.0
    normalized_test_data = np.asarray(normalized_test_data)
    test_predicted = []
    for x in normalized_test_data:
        predicted = nn.feedforward(x)
        pred = np.argmax(predicted)
        test_predicted.append(pred)
    np.savetxt('test_predictions.csv', test_predicted, delimiter=',', fmt='%d')

