# Machine Learning Model

This repo is where I document my journey through various machine learning models. It's a mix of classic algorithms, tuning techniques, and quick experiments – all written in Jupyter Notebooks. Nothing fancy, just working code and practical examples.
## What's inside

```text
.
├── Data/
│   └── Titanic-Dataset.csv
├── DecisionTree/
│   └── TitanicPredictionModel.ipynb
├── K-Nearest Neighbors (KNN) /
│   └── TitanicPredictionModel.ipynb
├── Logistic Regression /
│   └── TitanicPredictionModel.ipynb
├── Naive Bayes/
│   └── TitanicPredictionModel.ipynb
├── Notebook/
│   └── titanic_analysis.ipynb
└──  hyperparameter tuning techniques/
    └── GridSearchCV&RandomizedSearchCV.ipynb
```

> A quick note: a few folder names currently include spaces, including one leading space before `hyperparameter tuning techniques`. If you use the command line, wrap those paths in quotes.

## Folder guide

### `Data/`

This folder stores the local Titanic CSV file used for the exploratory statistics notebook. The file includes passenger-level fields such as `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.

Example row structure from the dataset:

```csv
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
```

Use this folder when you want to work with a real CSV file instead of loading a dataset directly from Seaborn or KaggleHub.

---

### `Notebook/`

This folder contains the main exploratory analysis notebook: `titanic_analysis.ipynb`.

It focuses less on training models and more on understanding the Titanic data through statistics and visualizations. The notebook looks at things like:

- missing values in `Age`
- mean, median, mode, and skewness
- age distribution plots
- fare variance and standard deviation
- fare outliers beyond two standard deviations
- survival probabilities by gender and passenger class
- a binomial probability example using second-class survival rate

A real example from the notebook calculates basic statistics for passenger age:

```python
print('mean : ', df['Age'].mean())
print('median : ', df['Age'].median())
print('mode : ', df['Age'].mode()[0])
print('skew: ',df['Age'].skew())
```

Another nice part of this notebook is the conditional survival comparison:

```python
p1_female_survival = df[(df['Sex'] == 'female') & (df['Pclass'] == 1)]['Survived'].mean()
p3_male_survival = df[(df['Sex'] == 'male') & (df['Pclass'] == 3)]['Survived'].mean()
print(f"P(Survived | Pclass=1 & Female): {p1_female_survival:.2f}")
print(f"P(Survived | Pclass=3 & Male): {p3_male_survival:.2f}")
```

---

### `DecisionTree/`

This folder contains a Titanic survival prediction notebook using a Decision Tree classifier.

The notebook loads the Titanic dataset from Seaborn, removes columns that are either redundant or not needed for the model, fills missing ages with the mean, drops rows with missing `embarked`, label-encodes categorical columns, splits the data, scales the features, trains a decision tree, and evaluates it with accuracy, a confusion matrix, and a classification report.

A real preprocessing step from the notebook:

```python
df.drop(['who','deck','embark_town','alive','class','adult_male'],axis = 1 ,inplace=True)
df['age'].fillna(df['age'].mean(),inplace=True)
df.dropna(subset=['embarked'],inplace=True)
```

The actual Decision Tree training code:

```python
from sklearn.tree import DecisionTreeClassifier

model_DT = DecisionTreeClassifier(random_state=42)
model_DT.fit(X_train_scaled,y_train)
y_pred_DT = model_DT.predict(X_test_scaled)
```

And the model is evaluated like this:

```python
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,y_pred_DT)
confusion_matrix(y_test,y_pred_DT)
print(classification_report(y_test,y_pred_DT))
```

---

### `K-Nearest Neighbors (KNN) /`

This folder contains a Titanic survival prediction notebook using the K-Nearest Neighbors classifier.

KNN is sensitive to feature scale, so this notebook includes `StandardScaler` before fitting the model. The workflow is similar to the other Titanic model notebooks: clean the data, encode `sex` and `embarked`, split into train/test sets, scale the features, train the model, and evaluate predictions.

A real example from the notebook shows the scaling step:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```

The actual KNN model uses 5 neighbors:

```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors= 5)
knn_model.fit(X_train_scaled,y_train)
y_pred = knn_model.predict(X_test_scaled)
```

Evaluation is done with the standard classification metrics:

```python
from sklearn.metrics import accuracy_score ,confusion_matrix , classification_report

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
```

---

### `Logistic Regression /`

This folder contains a Titanic survival prediction notebook using Logistic Regression.

The notebook follows a straightforward classification pipeline: load Titanic data from Seaborn, remove columns that are not being used, handle missing values, label-encode categorical variables, split the dataset, train a logistic regression model, and evaluate the predictions.

A real feature/target split from the notebook:

```python
X = df.drop('survived',axis = 1)
y = df['survived']
```

The Logistic Regression model is trained directly on the train split:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
```

The notebook then checks model performance with:

```python
from sklearn.metrics import accuracy_score ,confusion_matrix , classification_report

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
```

---

### `Naive Bayes/`

This folder contains a Titanic survival prediction notebook using Gaussian Naive Bayes.

The data preparation is very similar to the Decision Tree and KNN notebooks. The main difference is the classifier: this notebook uses `GaussianNB`, which is a good fit for trying a simple probabilistic baseline on numeric features.

A real encoding step from the notebook:

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])
```

The actual Naive Bayes model code:

```python
from sklearn.naive_bayes import GaussianNB

model_NB = GaussianNB()
model_NB.fit(X_train,y_train)
y_pred_NB = model_NB.predict(X_test)
```

The notebook evaluates the predictions with:

```python
from sklearn.metrics import accuracy_score ,confusion_matrix , classification_report

accuracy_score(y_test,y_pred_NB)
confusion_matrix(y_test,y_pred_NB)
print(classification_report(y_test,y_pred_NB))
```

---

### ` hyperparameter tuning techniques/`

This folder contains `GridSearchCV&RandomizedSearchCV.ipynb`, a notebook focused on tuning model parameters instead of building another Titanic classifier.

Here, the Iris dataset is loaded from Seaborn. The notebook first trains a KNN model and an SVM model, then uses both `GridSearchCV` and `RandomizedSearchCV` to compare SVM parameter combinations.

The notebook starts by preparing the Iris features and target:

```python
df = sns.load_dataset("iris")

X = df.drop('species',axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Before tuning, it tries a KNN model:

```python
from sklearn.neighbors import KNeighborsClassifier
model_KNN = KNeighborsClassifier(n_neighbors=9)

model_KNN.fit(X_train,y_train)
y_pred = model_KNN.predict(X_test)
```

Then it creates an SVM model and tunes `C` and `kernel` with grid search:

```python
from sklearn.svm import SVC
model_svm = SVC(gamma='auto')

from sklearn.model_selection import GridSearchCV

classifier = GridSearchCV((model_svm),{
    'C' : [1,10,20,30],
     'kernel':['rbf','linear']
},cv = 5, return_train_score=False)

classifier.fit(X,y)
results = pd.DataFrame(classifier.cv_results_)
results[['param_C','param_kernel','mean_test_score']]
```

The same search space is also tested with randomized search:

```python
from sklearn.model_selection import RandomizedSearchCV

classifier_random = RandomizedSearchCV((model_svm),{
    "C":[1,10,20,30],
    'kernel':['rbf','linear']
},n_iter = 4,cv=5,return_train_score=False)

classifier_random.fit(X,y)
results = pd.DataFrame(classifier_random.cv_results_)
results[['param_C','param_kernel','mean_test_score']]
```

## Common workflow across the model notebooks

Most Titanic model notebooks follow this same pattern:

1. Load the Titanic dataset with Seaborn.
2. Drop columns that are not needed for training.
3. Fill missing `age` values with the mean.
4. Drop rows where `embarked` is missing.
5. Convert categorical columns with `LabelEncoder`.
6. Split the data with `train_test_split`.
7. Train a classifier.
8. Evaluate the model with accuracy, confusion matrix, and classification report.

The shared train/test split looks like this in the notebooks:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Setup

This repository is notebook-based, so the easiest way to run it is with Jupyter Notebook or JupyterLab.

Install the main Python packages:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter kagglehub
```

Then start Jupyter:

```bash
jupyter notebook
```

Open the notebook you want to explore and run the cells from top to bottom.
