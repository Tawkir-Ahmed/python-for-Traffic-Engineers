#Book: Hands on Machine learning
#https://github.com/tuitet/Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow-3rd-Edition/tree/main
#Part 1: Chapter 3
#Example 1-1: Training and running a linear model suing Scikit-Learn
# MNIST dataset
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', as_frame=False)

# Assign the data and target variables
X, y = mnist.data, mnist.target

# Check the shapes of X and y
print(X.shape)
print(y.shape)

# Define the plot_digit function
def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')

# Visualize a digit (for example, the first one)
some_digit = X[0]
plot_digit(some_digit)
plt.show()

y[0]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:] #spliting data

# Training a Binary Classifier
y_train_5 = (y_train == '5') # True for all 5s, False for all other digits 
y_test_5 = (y_test == '5')

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])

#Performance Measure
#Measuring Accuracy Unsing Cross-Validation
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=5, scoring='accuracy')

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train))) # prints fAlse: no 5s detected

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring='accuracy')

#Implementing Cross-Validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3) # add shuffle=Ture if the dataset is
                                      # not already shuffled

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))  # prints 0.95, 0.96, and 0.96

# confusion matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_pred)
cm

y_train_perfect_predictions = y_train_5  # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)

# preecision and recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) # == 3530 / (687 + 3530)
recall_score(y_train_5, y_train_pred) # == 3530 / (1891 + 3530)

from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

# the precision/recall trade-off
y_scores = sgd_clf.decision_function([some_digit])
y_scores
threshold = 0
y_some_digit_pred = (y_scores > threshold)
threshold = 3000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, 
                             method = 'decision_function')

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)

# Replace 'threshold' with the actual threshold value you want to highlight
highlight_threshold = 0.5  # Change this value to the threshold you want to highlight

plt.vlines(highlight_threshold, 0, 1.0, 'k', 'dotted', label='Threshold')

plt.xlabel('Threshold')
plt.ylabel('Precision / Recall')
plt.grid(True)
plt.legend(loc='best')
plt.title('Precision and Recall vs. Threshold')

# Add circles to highlight the threshold value
plt.plot([highlight_threshold], [precisions[np.argmax(thresholds >= highlight_threshold)]], 'bo')  # Red circle for precision
plt.plot([highlight_threshold], [recalls[np.argmax(thresholds >= highlight_threshold)]], 'go')     # Blue circle for recall
plt.show()

# precison vs recall plot
plt.plot(recalls, precisions, linewidth=2, label='Precision/Recall curve')
plt.xlabel('Trcall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend(loc='best')
plt.title('Precision vs Recall')
plt.show()


import matplotlib.pyplot as plt

# Your precision, recall, and threshold values here...
# Plot the precision-recall curve
plt.plot(recalls, precisions, linewidth=2, label='Precision/Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend(loc='best')
plt.title('Precision vs Recall')

# Specify the threshold value you want to highlight
highlight_threshold = 0.5  # Change this to your desired threshold

# Find the index corresponding to the threshold in the thresholds array
threshold_index = next((i for i, value in enumerate(thresholds) if value >= highlight_threshold), None)

# Highlight the point on the curve corresponding to the threshold
if threshold_index is not None:
    plt.plot(recalls[threshold_index], precisions[threshold_index], 'ro', label=f'Threshold {highlight_threshold}', markersize=8)

    # Add horizontal dotted line
    plt.axhline(y=precisions[threshold_index], color='gray', linestyle='--', xmax=recalls[threshold_index], linewidth=1)
    # Add vertical dotted line
    plt.axvline(x=recalls[threshold_index], color='gray', linestyle='--', ymax=precisions[threshold_index], linewidth=1)

    # Add an arrow with text "Higher Threshold"
    plt.annotate('Higher Threshold', xy=(recalls[threshold_index], precisions[threshold_index]), xytext=(0.2, 0.6),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='black'), fontsize=12)

plt.show()

# page= 114

idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision

y_train_pred_90 = (y_scores >= threshold_for_90_precision)
precision_score(y_train_5, y_train_pred_90)

recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
recall_at_90_precision

# ROC Curve: the receiver operating characteristic (binary classifier)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

import matplotlib.pyplot as plt
# Plot the ROC curve
plt.plot(fpr, tpr, linewidth=2, label='ROC Curve')
# Plot the ROC curve for a random classifier
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
# Specify the threshold point for 90% precision
#fpr_90 = ...  # Replace with your calculated value
#tpr_90 = ...  # Replace with your calculated value
plt.plot([fpr_90], [tpr_90], "ko", label='Threshold for 90% precision')
# Beautify the figure: add labels, grid, legend, arrow, and text
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.grid(True)
plt.legend(loc='lower right')
plt.annotate('90% Precision', xy=(fpr_90, tpr_90), xytext=(fpr_90 + 0.1, tpr_90 - 0.1),  # Adjust the position
             arrowprops=dict(arrowstyle='->'))
plt.text(0.5, 0.4, 'Random Classifier', color='black', rotation=45)
plt.xlim([0, 1])
plt.ylim([0, 1])
# Show the plot
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)

#117
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method ='predict_proba')

y_probas_forest[:2]

y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label= "Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend(loc="lower right")
# need to see how to add SGD
plt.show()

y_train_pred_forest = y_probas_forest[:, 1] >= 0.5 # positive proba >=50%
f1_score(y_train_5, y_train_pred_forest) # 
roc_auc_score(y_train_5, y_scores_forest)

# Multiclass classification
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000]) # y_train, not y-trian_5

svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)

class_id = some_digit_scores.argmax()
class_id

svm_clf.classes_

svm_clf.classes_[class_id]

from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

sgd_clf.decision_function([some_digit]).round()

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring= 'accuracy')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# Error Analysis


from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
plt.rc('font', size=9)  # extra code – make the text smaller
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize='true', values_format='.0%')
plt.show()

sample_weight = (y_train_pred != y_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight = sample_weight,
                                        normalize='true', values_format=".0%")
plt.show()


# Analyze the errors
cl_a, cl_b = '3', '5'
X_aa = X_train[(y_train ==cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

# extra code – this cell generates and saves Figure 3–11
size = 5
pad = 0.2
plt.figure(figsize=(size, size))
for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)),
                                       (X_aa, (0, 1)), (X_ab, (1, 1))]:
    for idx, image_data in enumerate(images[:size*size]):
        x = idx % size + label_col * (size + pad)
        y = idx // size + label_row * (size + pad)
        plt.imshow(image_data.reshape(28, 28), cmap="binary",
                   extent=(x, x + 1, y, y + 1))
plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# Multilabel Classification
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average='macro')

from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])

chain_clf.predict([some_digit])

# Multioutput Classification
np.random.seed(42) # to make this code example reproducible
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

#clean image
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()

#Pg: 12
#Read and Write



