from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
import numpy as np


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


def train():
    x_train_set, y_train_set = load_data(mode=0)
    # for feature_it in x_train_set.T:
    #     max_value = feature_it.max()
    #     if max_value != 1 and max_value != 0:
    #         feature_it /= max_value
    gs_clf = GaussianNB()
    mu_clf = MultinomialNB()
    br_clf = BernoulliNB()
    cp_clf = ComplementNB()
    gs_clf.fit(x_train_set, y_train_set)
    mu_clf.fit(x_train_set, y_train_set)
    br_clf.fit(x_train_set, y_train_set)
    cp_clf.fit(x_train_set, y_train_set)
    x_test_set, y_test_set = load_data(mode=1)
    # for feature_it in x_test_set.T:
    #     max_value = feature_it.max()
    #     if max_value != 1 and max_value != 0:
    #         feature_it /= max_value
    gs_predict_set = gs_clf.predict(x_test_set)
    mu_predict_set = mu_clf.predict(x_test_set)
    br_predict_set = br_clf.predict(x_test_set)
    cp_predict_set = cp_clf.predict(x_test_set)
    gs_accurate_rate = (gs_predict_set == y_test_set).sum() / gs_predict_set.shape[0]
    mu_accurate_rate = (mu_predict_set == y_test_set).sum() / mu_predict_set.shape[0]
    br_accurate_rate = (br_predict_set == y_test_set).sum() / br_predict_set.shape[0]
    cp_accurate_rate = (cp_predict_set == y_test_set).sum() / cp_predict_set.shape[0]
    print(f'Gaussian Naive Bayes accurate rate is {gs_accurate_rate}')
    print(f'Multinomial Naive Bayes accurate rate is {mu_accurate_rate}')
    print(f'Bernoulli Naive Bayes accurate rate is {br_accurate_rate}')
    print(f'Complement Naive Bayes accurate rate is {cp_accurate_rate}')


if __name__ == '__main__':
    train()
