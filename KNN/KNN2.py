# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# Подробнее со следующими библиотеками вы познакомитесь на следующих занятиях. 
# Для решения задания потребуется лишь знания чистого python.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
digits = load_digits()
plt.matshow(digits.images[22])
plt.gray()
display(digits.images.shape)
X = digits.images.reshape(len(digits.images), -1)
display(X.shape)
Y = digits.target
X_train, X_test = X[:1500], X[1500:]
Y_train, Y_test = Y[:1500], Y[1500:]


# %%
def dist(vec1, vec2):
    assert len(vec1) == len(vec2)
    ans = 0
    for x1 in vec1:
        for x2 in vec2:
            ans += (x1 - x2) ** 2
    


def classify_KNN(X_train, Y_train, X_test):
    for elem in X_test:
        pass


# %%



# %%



