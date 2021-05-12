import numpy as np
import time

nn_arch = [
    { "in_dim": 3 , "out_dim": 2  , "activation": "relu"},
    {"in_dim": 2, "out_dim": 2, "activation": "relu"},
]

class NN:
    @staticmethod
    def sigmoid(x):
        return 1 / 1 + np.exp(-x)

    @staticmethod
    def relu(x):
        return np.max(0,x)

    @staticmethod
    def d_sigmoid(x):
        return NN.sigmoid(x)*(1 - NN.sigmoid(x))

    def __init__(self , nn_arch , seed = 14):
        nb_layer = len(nn_arch)
        self.params = {}
        self.nn_arch = nn_arch

        np.random.seed(seed)

        for lyr_idx , lyr_arch in enumerate(nn_arch):
            lyr_idx = lyr_idx + 1

            self.params["W" + str(lyr_idx)] = np.random.rand(
                lyr_arch["out_dim"] , lyr_arch["in_dim"]
            )

            self.params["b" + str(lyr_idx)] = np.random.rand(
                lyr_arch["out_dim"],1
            )

        def single_lyr_feedfp(A_prev , W_curr , b_curr , activation_func):
            Z_curr = np.dot(W_curr,A_prev) + b_curr

            if activation_func is "relu":
                activation = NN.relu
            elif activation_func is "sigmoid":
                activation = NN.sigmoid
            else:
                raise Exception('Non-supported activation function')

            return activation(Z_curr),Z_curr

        def full_forward(self,X):
            A_curr = X

            memory = {}

            for lyr_idx , lyr_arch in self.nn_arch:
                lyr_idx = lyr_idx + 1
                A_curr,Z_curr = single_lyr_feedfp(A_curr,lyr_arch["W" + str(lyr_idx)] , lyr_arch["b"+str(lyr_idx)],
                                           lyr_arch["activation"])
                memory["A" + str(lyr_idx)] = A_curr
                memory["Z" + str(lyr_idx)] = Z_curr

            return A_curr , memory

        def single_backpropagation(dA_curr,A_prev,Z_curr,W_curr,activation_func):

            if activation_func is "relu":
                pass
            elif activation_func is "sigmoid":
                dActivation_func = NN.d_sigmoid
            else:
                raise Exception('Non-supported activation function')

            m = A_prev.shape[1]

            dZ_curr = np.dot(dA_curr,dActivation_func(Z_curr))
            dA_prev = np.dot(dZ_curr,W_curr.T)
            dB_curr = np.sum(dZ_curr,axis=1,keepdims=True) / m
            dW_curr = np.dot(dZ_curr,A_prev.T) / m

            return dA_prev,dW_curr,dB_curr

        def full_backpropagation(memory,y_hat,y):


            for prev_lyr_idx , layer in reversed(list(enumerate(self.nn_arch))):
                A_prev = memory["A" + str(prev_lyr_idx)]
                W_curr = self.params["W" + str(prev_lyr_idx + 1)]
                B_curr = self.params["b" + str(prev_lyr_idx + 1)]
                Z_curr = memory["Z" + str(prev_lyr_idx + 1)]

                dA_prev, dW_curr, dB_curr = single_backpropagation()

                dA_curr = dA_prev


