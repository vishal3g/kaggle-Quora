# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Importing the dataset
dataset = pd.read_csv('train.csv')

dataset = dataset.drop(['qid'],axis=1)






# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#nltk.download('corpora/wordnet')
ps = PorterStemmer()

corpus = []
stopword_set = set(stopwords.words('english'))
for i in range(0, 1306122):
    review = re.sub('[^a-zA-Z]', ' ', dataset['question_text'][i])
    review = review.lower()
    review = review.split()
    #review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = [ps.stem(word) for word in review if not word in stopword_set]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model

from sklearn.feature_extraction.text import TfidfVectorizer
#cv = CountVectorizer(max_features = None)

cv= TfidfVectorizer(ngram_range=(1, 3),min_df=1,max_features=500,analyzer='word')
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values



# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                    random_state = 0,
                                                    stratify=y,
                                                    shuffle=True)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB,MultinomialNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


from sklearn.model_selection import GridSearchCV
parameters = [{'alpha':[0.1,0.2,0.3,0.4,0.5]}]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


classifier.score(X_test,y_test)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)