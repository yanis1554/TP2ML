import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

#print(mnist)
#print (mnist.data)
#print (mnist.target)
#len(mnist.data)
#help(len)   
#print (mnist.data.shape)
#print (mnist.target.shape)
#mnist.data[0]
#mnist.data[0][1]
#mnist.data[:,1]
#mnist.data[:100]


images = mnist.data.reshape((-1, 28, 28))
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()


# Max iter Value = 10

import time

# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurone 50
mlp=  MLPClassifier(hidden_layer_sizes=(50),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='adaptive',
                    learning_rate_init=0.01,
                    max_iter=10,
                    tol=10**(-4),
                    verbose=True,
                    warm_start=False,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=50)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)

# prediction des valeurs de test et affichage pour la classe d'image 4
y_test_predict = mlp.predict(x_test)
print("La valeur prédite est:",y_test_predict[3])
print("La valeur réel est:",y_test[3])

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))
print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta,"s")

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


# Max iter Value = 50

import time

# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,random_state=0, test_size = 0.20) 


# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurone 50
mlp=  MLPClassifier(hidden_layer_sizes=(50),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='adaptive',
                    learning_rate_init=0.01,
                    max_iter=50,
                    tol=10**(-4),
                    verbose=True,
                    warm_start=False,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=50)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)

# prediction des valeurs de test et affichage pour la classe d'image 4
y_test_predict = mlp.predict(x_test)
print("La valeur prédite est:",y_test_predict[3])
print("La valeur réel est:",y_test[3])

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))
print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta)

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


# Max iter Value = 300

import time

# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,random_state=0, test_size = 0.20) 


# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurone 50
mlp=  MLPClassifier(hidden_layer_sizes=(50),
                    max_iter=300,
                    )  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)

# prediction des valeurs de test et affichage pour la classe d'image 4
y_test_predict = mlp.predict(x_test)
print("La valeur prédite est:",y_test_predict[3])
print("La valeur réel est:",y_test[3])

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))
print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta)

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


# Sample = 20000

import numpy as np
from sklearn.datasets import fetch_openml
import time
import matplotlib.pyplot as plt


# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=20000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 


# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurone 50
mlp=  MLPClassifier(hidden_layer_sizes=(50),max_iter=300,)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)

# prediction des valeurs de test et affichage pour la classe d'image 4
y_test_predict = mlp.predict(x_test)
print("La valeur prédite est:",y_test_predict[3])
print("La valeur réel est:",y_test[3])

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))
print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta)

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


# Sample = 50000

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=50000)
data = mnist.data[sample]
target = mnist.target[sample]

# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurone 50
mlp=  MLPClassifier(hidden_layer_sizes=(50),max_iter=300,)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)

# prediction des valeurs de test et affichage pour la classe d'image 4
y_test_predict = mlp.predict(x_test)
print("La valeur prédite est:",y_test_predict[3])
print("La valeur réel est:",y_test[3])

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))
print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta)

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


# hidden layer = 20

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=50000)
data = mnist.data[sample]
target = mnist.target[sample]

# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,random_state=0, test_size = 0.20) 


# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurone 20
mlp=  MLPClassifier(hidden_layer_sizes=(20),)

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)

# prediction des valeurs de test et affichage pour la classe d'image 4
y_test_predict = mlp.predict(x_test)
print("La valeur prédite est:",y_test_predict[3])
print("La valeur réel est:",y_test[3])

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))
print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta)

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


# hidden layer = 100

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=50000)
data = mnist.data[sample]
target = mnist.target[sample]

# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,random_state=0, test_size = 0.20) 


# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurone 100
mlp=  MLPClassifier(hidden_layer_sizes=(100),)

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)

# prediction des valeurs de test et affichage pour la classe d'image 4
y_test_predict = mlp.predict(x_test)
print("La valeur prédite est:",y_test_predict[3])
print("La valeur réel est:",y_test[3])

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))
print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta)

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


# hidden layer = 300

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=50000)
data = mnist.data[sample]
target = mnist.target[sample]

# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,random_state=0, test_size = 0.20) 


# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurone 300
mlp=  MLPClassifier(hidden_layer_sizes=(300),)

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)

# prediction des valeurs de test et affichage pour la classe d'image 4
y_test_predict = mlp.predict(x_test)
print("La valeur prédite est:",y_test_predict[3])
print("La valeur réel est:",y_test[3])

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))
print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta)

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


# hidden layer = 500

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=50000)
data = mnist.data[sample]
target = mnist.target[sample]

# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,random_state=0, test_size = 0.20) 


# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurone 500
mlp=  MLPClassifier(hidden_layer_sizes=(500),)

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)

# prediction des valeurs de test et affichage pour la classe d'image 4
y_test_predict = mlp.predict(x_test)
print("La valeur prédite est:",y_test_predict[3])
print("La valeur réel est:",y_test[3])

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))
print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta)

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()
