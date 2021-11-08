import numpy as np
import time
import sys


def load_data(mode):
    if mode == 0:
        data_source = 'train'
    elif mode == 1:
        data_source = 'test'
    else:
        print("load_data usage: mode = 0 is train and mode = 1 is test")
        return
    x_train_set = np.genfromtxt(f'../data/{data_source}/X_{data_source}.csv', delimiter=',')
    y_train_set = np.genfromtxt(f'../data/{data_source}/Y_{data_source}.csv', delimiter=',')
    x_train_set = np.delete(x_train_set, 0, axis=0)
    y_train_set = np.delete(y_train_set, 0, axis=0)
    return x_train_set, y_train_set


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def train():
    x_train_set, y_train_set = load_data(mode=0)
    train_size = x_train_set.shape[0]
    feature_size = x_train_set.shape[1]
    # formalize feature vector
    for feature_it in x_train_set.T:
        max_value = feature_it.max()
        if max_value != 1 and max_value != 0:
            feature_it /= max_value
    # initialize w array
    w = np.random.rand(feature_size + 1)
    w *= 0.1
    w -= 0.05
    # begin train
    eta = 0.0001
    x_test_set, y_test_set = load_data(mode=1)
    test_size = x_test_set.shape[0]
    # formalize feature vector
    for feature_it in x_test_set.T:
        max_value = feature_it.max()
        if max_value != 1 and max_value != 0:
            feature_it /= max_value
    round_count = 1
    while True:
        for j in range(500):
            delta_w = np.zeros(feature_size + 1)
            # version 1: using loop
            # for i in range(train_size):
            #     a = x_train_set[i] @ w[1:] + w[0]
            #     y = sigmoid(x_train_set @ w[1:] + w[0])
            #     delta_w[1:] += (y_train_set - y) * x_train_set[i]
            #     delta_w[0] += (y_train_set - y)
            # w += eta * delta_w

            # version 2: using matrix multiply
            y = sigmoid(x_train_set @ w[1:] + w[0])
            delta_w[1:] = (y_train_set - y) @ x_train_set
            delta_w[0] = (y_train_set - y).sum()
            w += eta * delta_w
        accurate_rate = test(w, x_test_set, y_test_set, test_size, round_count)
        round_count += 1
        if accurate_rate > 0.85:
            break


def test(w, x_test_set, y_test_set, test_size, round_count):
    accurate_count = 0
    for i in range(test_size):
        y = sigmoid(w[1:] @ x_test_set[i] + w[0])
        if y > 0.5:
            result = 1.0
        else:
            result = 0.0
        if result == y_test_set[i]:
            accurate_count += 1
    accurate_rate = accurate_count / test_size
    print(f'round {round_count}: accurate rate is {accurate_rate}')
    return accurate_rate


def main():
    start_time = time.time()
    train()
    print(f'training time is {time.time() - start_time}')


if __name__ == '__main__':
    # f = open('output.txt', 'w')
    # sys.stdout = f
    # sys.stderr = f
    main()
