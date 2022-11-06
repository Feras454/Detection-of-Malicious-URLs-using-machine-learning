import pandas as pd
import numpy as np
import tensorflow
from matplotlib import pyplot
import re
from urllib.parse import urlparse
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (check report i put link of dataset)
data = pd.read_csv('malicious_phish.csv')

print(data.type.value_counts())

# print(data.type.value_counts())
# print(data.columns)

# Prepare label encoder for type, so it can be understandble for keras.
lb_make = LabelEncoder()
data["class"] = lb_make.fit_transform(data["type"])
print(data['class'].head())


def having_ip_address(url):
    # regex for IPv4 and IPv6 to check if url contain IP
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


def digit_count(url):
    # counting digit in url
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


def NumSensitiveWords(url):
    match = re.search(
        'PayPal|Bitcoin|login|signin|bank|banking|account|update|free|lucky|service|bonus|ebayisapi|webscr',
        url)
    if match:
        return 1
    else:
        return 0


def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


# Extraction and Selection of URL Attributes
data['use_of_ip'] = data['url'].apply(lambda i: having_ip_address(i))
data['url_length'] = data['url'].apply(lambda i: len(str(i)))
data['numOf-https'] = data['url'].apply(lambda i: i.count('https'))
data['numOf-http'] = data['url'].apply(lambda i: i.count('http'))
data['hostname_length'] = data['url'].apply(lambda i: len(urlparse(i).netloc))
data['count-digits'] = data['url'].apply(lambda i: digit_count(i))
data['count-letters'] = data['url'].apply(lambda i: letter_count(i))
data['NumSensitiveWords'] = data['url'].apply(lambda i: NumSensitiveWords(i))
data['numOf.'] = data['url'].apply(lambda i: i.count('.'))
data['numOf%'] = data['url'].apply(lambda i: i.count('%'))
data['numOf?'] = data['url'].apply(lambda i: i.count('?'))
data['numOf-'] = data['url'].apply(lambda i: i.count('-'))
data['numOf='] = data['url'].apply(lambda i: i.count('='))
data['abnormal_url'] = data['url'].apply(lambda i: abnormal_url(i))
data['binary'] = data['type'].apply(lambda x: 0 if x == 'benign' else 1)

# Predictor Variables
X = data[['use_of_ip', 'abnormal_url', 'numOf.', 'numOf%', 'numOf-', 'numOf?', 'numOf=', 'url_length', 'count-digits',
     'count-letters',
     'NumSensitiveWords', 'hostname_length', 'binary', 'numOf-https', 'numOf-http']]
# Target Variable
y = data['class']

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)
# Converts a class vector (integers) to binary class matrix

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=4)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=4)

print(data.head())

print(X_train.shape[1])
print(y_train.shape[1])
# Create hidden layers
model = Sequential()
model.add(Dense(128, input_shape=(15,), input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))
print(model.summary())
#  Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

ypred = model.predict(X_test)
# to fix confusion_matrix can't handle multiple classification
ypred = np.argmax(ypred, axis=1)
y_test = np.argmax(y_test, axis=1)

# show the accuracy score, classificaiton report and confusion matrix
print("*" * 60)
print(f'accuracy score: {accuracy_score(y_test, ypred):.3}')
print(classification_report(y_test, ypred))
print(confusion_matrix(y_test, ypred))
print("*" * 60)

# accuracy plot
pyplot.title('The Accuracy')
pyplot.plot(history.history['categorical_accuracy'], label="training")
pyplot.plot(history.history['val_categorical_accuracy'], label="testing")
pyplot.legend()
pyplot.show()
# loss plot
pyplot.title('The Loss')
pyplot.plot(history.history['loss'], label="training")
pyplot.plot(history.history['val_loss'], label="testing")
pyplot.legend()
pyplot.show()
