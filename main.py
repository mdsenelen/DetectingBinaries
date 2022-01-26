from os import listdir
import pandas as pd
import numpy as np
import entropy
# Machine learning tools From scikit-learn library.
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
# Classifiers used
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Calculates different entropy values for a binary executable and creates a feature list.
def get_feature_list(content):
	v0 = entropy.shannon_entropy(content)
	v1 = sum(entropy.sample_entropy(content, 3))
	v2 = entropy.permutation_entropy(content)
	#v3 = entropy.weighted_permutation_entropy(content)
	#v4 = sum(entropy.multiscale_entropy(content, 3, maxscale=3))
	return [v0, v1, v2] # , v3, v4

# Create dataset by calculating entropy values for each file under './Packed' and './Not Packed' directories.
if "collection.csv" not in listdir():
	HEADERS = ["Shannon Entropy", "Sample Entropy", "Permutation Entropy", "Target"] #, "Weighted Permutation Entropy" , "Multiscale Entropy"
	entries = []

	for category, class_val in zip(["Not Packed", "Packed"], [0, 1]):
		for file in listdir(f"./{category}/"):
			print(f"Calculating values for './{category}/{file}'")
			with open(f"./{category}/{file}", "rb") as f:
				content = list(f.readline()) 
				entries.append(get_feature_list(content)+[class_val])
				
	data_frame = pd.DataFrame(entries, columns=HEADERS)
	data_frame.to_csv("collection.csv", sep=",", header=True)

# If dataset is created and ready to use, Start training multiple classifiers for machine learning.
data_frame = pd.read_csv("collection.csv",  sep=",")
data_frame = data_frame.iloc[: , 1:] # Dropping the Index column as it confuses the learning algorithms.
data_frame = data_frame.replace([np.inf, -np.inf], np.nan)
data_frame = data_frame.dropna(axis="index")
print("Number of entries in the dataframe: ", len(data_frame))
print("Class Distrubition")
print(data_frame["Target"].value_counts())

target_vals = data_frame["Target"]				# Only target column for every entry : y
entries = data_frame.drop(["Target"], axis=1)	# Every column except the target column for every entry : x
	
# Dictionary containing all the classifier methods that will be used to train our dataset.
# This dictionary is iterated and its inside will be filled with relative data and will contain the trained model for later use if needed.
classifiers = {
	"GaussianNB" : {
		"Constructor" : GaussianNB 
	},
	"MultinomialNB" : {
		"Constructor" : MultinomialNB 
	},
	"KNeighborsClassifier" : {
		"Constructor" : KNeighborsClassifier 
	},
	"RandomForestClassifier" : {
		"Constructor" : RandomForestClassifier 
	},
	"GradientBoostingClassifier" : {
		"Constructor" : GradientBoostingClassifier 
	},
	"LinearDiscriminantAnalysis" : { 
		"Constructor" : LinearDiscriminantAnalysis
	},
	"SGDClassifier" : {
		"Constructor" : SGDClassifier
	},
	"DecisionTreeClassifier" : {
		"Constructor" : DecisionTreeClassifier
	} 
}

most_accurate_classifier = ["classifier", 0]	# Holds the most accurate classifier and it's cross val score.
x_train, x_test, y_train, y_test = train_test_split(entries, target_vals, test_size=0.25, random_state=105) # Dividing the data into training and test sets with 75/25 ratio

# For each classifier in classifiers dictionary, train the classifier, store the trained model as well as the calculated data about the trained model.
for classifier in classifiers:
	classifier_object = classifiers[classifier]["Constructor"]()
	classifier_object.fit(x_train, y_train)
	classifiers[classifier]["Model"] = classifier_object
	prediction = classifier_object.predict(x_test)
	classifiers[classifier]["Confusion_Matrix"] = confusion_matrix(prediction, y_test) # Left diagonal values are the correct guesses and other values are mispredictions.
	classifiers[classifier]["Accuracy"] = accuracy_score(prediction, y_test) # Compares what model guessed and what it actually is to get the accuracy.
	classifiers[classifier]["Cross_Validation_Score"] = cross_val_score(classifier_object, x_test, y_test, cv=10).mean()
	if most_accurate_classifier[1] <  classifiers[classifier]["Cross_Validation_Score"]:
		most_accurate_classifier = classifier, classifiers[classifier]["Cross_Validation_Score"]
	print("Classifier: {classifier} --> Accuracy: {accuracy:.4f}, Cross Validation Score: {cvs:.4f}".format(classifier=classifier, accuracy=classifiers[classifier]["Accuracy"], cvs=classifiers[classifier]["Cross_Validation_Score"]))

print(f"The most accurate classifier is: '{most_accurate_classifier[0]}' with highest percentage of cross_val_score: {round(most_accurate_classifier[1]*100, 4)}%")

