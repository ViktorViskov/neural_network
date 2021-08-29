# import libs
import numpy as np

# class for creating neuron


class Neuron:
    def __init__(self, weight):
        self.weight = weight

    def Sigmoid(self, summator):
        return 1 / (1 + np.exp(- summator))

    def Delta_Sigmoid(self, sigmoid):
        return sigmoid * (1 - sigmoid)

    def Predict(self, data_array):
        summator = np.dot(data_array, self.weight)
        return self.Sigmoid(summator)

    def Answer(self, data_array):
        return 1 if self.Predict(data_array) >= 0.5 else 0

# error checking
def MSE(n_answer, r_answer):
    return np.mean((n_answer - r_answer) ** 2)


data = np.array([1])
value = 0.5
# neurons
h1 = Neuron(np.array([np.random.random()]))
h2 = Neuron(np.array([np.random.random()]))
h3 = Neuron(np.array([np.random.random()]))
h4 = Neuron(np.array([np.random.random()]))

o1 = Neuron(np.array([np.random.random(), np.random.random(),np.random.random(), np.random.random()]))
o2 = Neuron(np.array([np.random.random(), np.random.random(),np.random.random(), np.random.random()]))
o3 = Neuron(np.array([np.random.random(), np.random.random(),np.random.random(), np.random.random()]))
o4 = Neuron(np.array([np.random.random(), np.random.random(),np.random.random(), np.random.random()]))

last = Neuron(np.array([np.random.random(), np.random.random(),np.random.random(), np.random.random()]))

# predirection
result_1 = h1.Predict(data)
result_2 = h2.Predict(data)
result_3 = h3.Predict(data)
result_4 = h4.Predict(data)
o1_result = o1.Predict(np.array([result_1, result_2, result_3, result_4]))
o2_result = o2.Predict(np.array([result_1, result_2, result_3, result_4]))
o3_result = o3.Predict(np.array([result_1, result_2, result_3, result_4]))
o4_result = o4.Predict(np.array([result_1, result_2, result_3, result_4]))

print("N1 -> %f  N2 -> %f   N3 -> %f   N4 -> %f" % (o1_result, o2_result, o3_result, o4_result))



# function for learn network
def Training(train_array):

    # step for changing result
    learn_rate = 0.3

    # preprocess data
    data = np.array([train_array[0]])

    # predirection
    result_1 = h1.Predict(data)
    result_2 = h2.Predict(data)
    result_3 = h3.Predict(data)
    result_4 = h4.Predict(data)
    o1_result = o1.Predict(np.array([result_1, result_2, result_3, result_4]))
    o2_result = o2.Predict(np.array([result_1, result_2, result_3, result_4]))
    o3_result = o3.Predict(np.array([result_1, result_2, result_3, result_4]))
    o4_result = o4.Predict(np.array([result_1, result_2, result_3, result_4]))

    last_result = last.Predict(np.array([o1_result, o2_result, o3_result, o4_result]))

    # 
    # Last neuron
    # 
    last_error_size = last_result - train_array[1]
    last_weight_delta = last_error_size * last.Delta_Sigmoid(last_result)
    last.weight[0] = last.weight[0] - o1_result * last_weight_delta * learn_rate
    last.weight[1] = last.weight[1] - o2_result * last_weight_delta * learn_rate
    last.weight[2] = last.weight[2] - o3_result * last_weight_delta * learn_rate
    last.weight[3] = last.weight[3] - o4_result * last_weight_delta * learn_rate

    

    #
    # o1 neuron
    #
    # correcting weight
    o1_error_size = last.weight[0] * last_weight_delta
    o1_weight_delta = o1_error_size * o1.Delta_Sigmoid(o1_result)

    # change weight values
    o1.weight[0] = o1.weight[0] - result_1 * o1_weight_delta * learn_rate
    o1.weight[1] = o1.weight[1] - result_2 * o1_weight_delta * learn_rate
    o1.weight[2] = o1.weight[2] - result_3 * o1_weight_delta * learn_rate
    o1.weight[3] = o1.weight[3] - result_4 * o1_weight_delta * learn_rate

    #
    # o2 neuron
    #
    o2_error_size = last.weight[1] * last_weight_delta
    o2_weight_delta = o2_error_size * o2.Delta_Sigmoid(o2_result)
    o2.weight[0] = o2.weight[0] - result_1 * o2_weight_delta * learn_rate
    o2.weight[1] = o2.weight[1] - result_2 * o2_weight_delta * learn_rate
    o2.weight[2] = o2.weight[2] - result_3 * o2_weight_delta * learn_rate
    o2.weight[3] = o2.weight[3] - result_4 * o2_weight_delta * learn_rate

    #
    # o3 neuron
    #
    o3_error_size = last.weight[2] * last_weight_delta
    o3_weight_delta = o3_error_size * o3.Delta_Sigmoid(o3_result)
    o3.weight[0] = o3.weight[0] - result_1 * o3_weight_delta * learn_rate
    o3.weight[1] = o3.weight[1] - result_2 * o3_weight_delta * learn_rate
    o3.weight[2] = o3.weight[2] - result_3 * o3_weight_delta * learn_rate
    o3.weight[3] = o3.weight[3] - result_4 * o3_weight_delta * learn_rate

    #
    # o4 neuron
    #
    o4_error_size = last.weight[3] * last_weight_delta
    o4_weight_delta = o4_error_size * o4.Delta_Sigmoid(o4_result)
    o4.weight[0] = o4.weight[0] - result_1 * o4_weight_delta * learn_rate
    o4.weight[1] = o4.weight[1] - result_2 * o4_weight_delta * learn_rate
    o4.weight[2] = o4.weight[2] - result_3 * o4_weight_delta * learn_rate
    o4.weight[3] = o4.weight[3] - result_4 * o4_weight_delta * learn_rate


    #
    # h1 neuron
    #
    h1_o1_error_size = ((o1.weight[0] + o2.weight[0] + o3.weight[0] + o4.weight[0]) / 4) * ((o1_weight_delta * o2_weight_delta * o3_weight_delta * o4_weight_delta) / 4)
    h1_o1_weight_delta = h1_o1_error_size * h1.Delta_Sigmoid(result_1)
    h1.weight[0] = h1.weight[0] - train_array[0] * h1_o1_weight_delta * learn_rate


    #
    # h2 neuron
    #
    h2_o1_error_size = ((o1.weight[1] + o2.weight[1] + o3.weight[1] + o4.weight[1]) / 4) * ((o1_weight_delta * o2_weight_delta * o3_weight_delta * o4_weight_delta) / 4)
    h2_o1_weight_delta = h2_o1_error_size * h1.Delta_Sigmoid(result_2)
    h2.weight[0] = h2.weight[0] - train_array[0] * h2_o1_weight_delta * learn_rate

    #
    # h3 neuron
    #
    h3_o1_error_size = ((o1.weight[2] + o2.weight[2] + o3.weight[2] + o4.weight[2]) / 4) * ((o1_weight_delta * o2_weight_delta * o3_weight_delta * o4_weight_delta) / 4)
    h3_o1_weight_delta = h3_o1_error_size * h1.Delta_Sigmoid(result_3)
    h3.weight[0] = h3.weight[0] - train_array[0] * h3_o1_weight_delta * learn_rate

    #
    # h4 neuron
    #
    h4_o1_error_size = ((o1.weight[3] + o2.weight[3] + o3.weight[3] + o4.weight[3]) / 4) * ((o1_weight_delta * o2_weight_delta * o3_weight_delta * o4_weight_delta) / 4)
    h4_o1_weight_delta = h4_o1_error_size * h1.Delta_Sigmoid(result_4)
    h4.weight[0] = h4.weight[0] - train_array[0] * h4_o1_weight_delta * learn_rate
    
    print("N1 error %f" % (MSE(last_result, train_array[1])))


# result
print("N1 -> %f  N2 -> %f   N3 -> %f   N4 -> %f" % (o1_result, o2_result, o3_result, o4_result))

train = np.array([
    [-3, 1],
    [5, 1],
    [12, 1],
    [15, 0],
    [26, 0],
    [32, 0],
    [63, 1],
])

# learning
for i in range(10000):
    for train_set in train:
        Training(train_set)

    print(i)

# speaking
while True:
    mounth = int(input("Введіть температуру: "))

    data = np.array([mounth])

    # predirection
    result_1 = h1.Predict(data)
    result_2 = h2.Predict(data)
    result_3 = h3.Predict(data)
    result_4 = h4.Predict(data)
    o1_result = o1.Predict(np.array([result_1, result_2, result_3, result_4]))
    o2_result = o2.Predict(np.array([result_1, result_2, result_3, result_4]))
    o3_result = o3.Predict(np.array([result_1, result_2, result_3, result_4]))
    o4_result = o4.Predict(np.array([result_1, result_2, result_3, result_4]))

    last_result = last.Predict(np.array([o1_result, o2_result, o3_result, o4_result]))
    answer = last.Answer(np.array([o1_result, o2_result, o3_result, o4_result]))
    
    print(last_result)
    if answer == 1:
        print("Не літо.")

    else:
        print("Літо!")


