import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    'attendance': [85, 60, 78, 90, 50, 72, 95, 40, 88, 67,
                   92, 55, 80, 70, 45, 98, 65, 75, 82, 58],
    'study_hours': [4, 2, 3, 5, 1, 3, 6, 1, 4, 2,
                    5, 2, 4, 3, 1, 6, 2, 3, 4, 2],
    'previous_marks': [75, 50, 65, 88, 40, 60, 92, 35, 85, 55,
                       90, 45, 70, 68, 38, 95, 58, 72, 80, 52],
    'result': [1, 0, 1, 1, 0, 1, 1, 0, 1, 0,
               1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

X = df[['attendance', 'study_hours', 'previous_marks']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_log))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred_log))

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

cm = confusion_matrix(y_test, y_pred_log)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Example new student
new_student = pd.DataFrame({
    'attendance': [80],
    'study_hours': [4],
    'previous_marks': [70]
})

prediction = log_model.predict(new_student)

if prediction[0] == 1:
    print("Prediction: Student is likely to PASS")
else:
    print("Prediction: Student is likely to FAIL")
