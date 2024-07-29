from sklearn.metrics import accuracy_score


def training_accuracy(model, X_train, Y_train):
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
    print("Accuracy score on the training data : ", training_data_accuracy)


def testing_accuracy(model, X_test, Y_test):
    X_test_prediction = model.predict(X_test)
    testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    print("Accuracy score on the testing data : ", testing_data_accuracy)
