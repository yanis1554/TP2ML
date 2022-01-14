#%% Réseau (50,50)



import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(50,50),)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))

print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta,"s")

# Loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("itération")
plt.title("loss curve")
plt.show()

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()



#%% Réseau (70,70)


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70),)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))

print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta,"s")


# Loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("itération")
plt.title("loss curve")
plt.show()

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


#%% Réseau (100,100)


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(100,100),)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))

print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta,"s")


# Loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("itération")
plt.title("loss curve")
plt.show()

# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()



#%% Réseau (50,50,50,50,50)


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(50,50,50,50,50),)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))

print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta,"s")


# Loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("itération")
plt.title("loss curve")
plt.show()


# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()



#%% Réseau (70,70,70,70,70)


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))

print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta,"s")

# Loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("itération")
plt.title("loss curve")
plt.show()


# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()



#%% Réseau (100,100,100,100,100)


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(100,100,100,100,100),)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

# Calcul de precision
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("La precision est de:",precision_score(y_test,y_test_predict,
                                                  average="micro"))

print("L'accuracy est de:",accuracy_score(y_test,y_test_predict))
print("Le run time est de:",delta,"s")

# Loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("itération")
plt.title("loss curve")
plt.show()


# Confusion Matrix
cm=confusion_matrix(y_test,y_test_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()


#%% Réseau (70,70,70,70,70) Adam


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),solver='adam')  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

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



#%% Réseau (70,70,70,70,70) L-BFGS


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),solver='lbfgs')  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

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


#%% Réseau (70,70,70,70,70) SGD


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),solver='sgd')  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

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


#%% Réseau (70,70,70,70,70) L-BFGS logistic


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),solver='lbfgs',
                    activation = 'logistic')  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

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

#%% Réseau (70,70,70,70,70) L-BFGS relu


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),solver='lbfgs',
                    activation = 'relu')  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

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


#%% Réseau (70,70,70,70,70) L-BFGS tanh


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),solver='lbfgs',
                    activation = 'tanh')  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

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



#%% Réseau (70,70,70,70,70) L-BFGS relu  alpha = 10-1


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),solver='lbfgs',
                    activation = 'relu',alpha = 10**(-1))  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

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



#%% Réseau (70,70,70,70,70) L-BFGS relu  alpha = 10-2


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),solver='lbfgs',
                    activation = 'relu',alpha = 10**(-2))  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

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


#%% Réseau (70,70,70,70,70) L-BFGS relu  alpha = 10-6


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    random_state=0, 
                                                    test_size = 0.20) 

# MLP Regression
from sklearn.neural_network import MLPClassifier

# creation du reseau de neurones
mlp=  MLPClassifier(hidden_layer_sizes=(70,70,70,70,70),solver='lbfgs',
                    activation = 'relu',alpha = 10**(-6))  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train)
delta = time.time()-start

# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
y_test_predict = mlp.predict(x_test)

# score train/test
print("Le score train vaut:",score_train)
print("La score test vaut:",score_test)

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
