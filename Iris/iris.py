# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape -- We can get a quick idea of how many instances (rows) 
# and how many attributes (columns) the data contains with the shape property.
print("\n\n")
print(dataset.shape,"--->(Instances and attributes)")

# head -- Peeking at your data, always a good idea
print("\n\n")
print("The first 20 rows of data are printed below")
print(dataset.head(20))

# descriptions -- Now we can take a look at a summary of each attribute.
# This includes the count, mean, the min and max values as well as some percentiles.
print("\n\n")
print("All of the numerical values have the same scale and similar ranges btn 0 & 8 cm.")
print(dataset.describe())

# class distribution -- Let’s now take a look at the number of instances (rows) 
# that belong to each class. We can view this as an absolute count.
print("\n\n")
print("We see that each class has the same number of instances (50 or 33.33% of the dataset)")
print(dataset.groupby('class').size())

# box and whisker plots -- We start with some univariate plots, that is, plots of each individual variable.
# Given that the input variables are numeric, we can create box and whisker plots of each.
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms -- We can also create a histogram of each input variable to get an idea of the distribution.
dataset.hist()
plt.show()

# scatter plot matrix -- Now we can look at the interactions between the variables.
# First, let’s look at scatterplots of all pairs of attributes. 
# This can be helpful to spot structured relationships between input variables.
print("\n\n")
print("""Note the diagonal grouping of some pairs of attributes. 
This suggests a high correlation and a predictable relationship.""")
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset -- We will split the loaded dataset into two, 
# 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric -- We will use 10-fold cross validation to estimate accuracy.
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.
# We are using the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances 
# in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). 
# We will be using the scoring variable when we run build and evaluate each model next.
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms -- Let’s build and evaluate our five models
print("\n\n")
models = []
models.append(('LR', LogisticRegression()))           # Logistic Regression (LR)
models.append(('LDA', LinearDiscriminantAnalysis()))  # Linear Discriminant Analysis (LDA)
models.append(('KNN', KNeighborsClassifier()))        # K-Nearest Neighbors (KNN).
models.append(('CART', DecisionTreeClassifier()))     # Classification and Regression Trees (CART).
models.append(('NB', GaussianNB()))                   # Gaussian Naive Bayes (NB).
models.append(('SVM', SVC()))                         # Support Vector Machines (SVM).
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# We now have 6 models and accuracy estimations for each. 
# We need to compare the models to each other and select the most accurate.


# We can also create a plot of the model evaluation results and compare 
# the spread and the mean accuracy of each model. There is a population of accuracy measures 
# for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
print("\n\n")

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset -- Now we want to get an idea of the accuracy of the model on our validation set.
# This will give us an independent final check on the accuracy of the best model. 
# It is valuable to keep a validation set just in case you made a slip during training, 
# such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.
# We can run the KNN model directly on the validation set and summarize the results 
# as a final accuracy score, a confusion matrix and a classification report.
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print("""We can see that the accuracy is 0.93 or 93%. The confusion matrix 
provides an indication of the three errors made. Finally, the classification 
report provides a breakdown of each class by precision, 
recall, f1-score and support showing excellent results\n""")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

