
# ==========================================
# STUDENT PASS/FAIL PREDICTION SYSTEM
# ==========================================

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import joblib

# 1. HISTORICAL STUDENT DATA
# Features: Study hours, Attendance %, Previous exam marks

X = [
    [1, 40, 20],
    [2, 45, 25],
    [2, 50, 30],
    [3, 55, 35],
    [3, 60, 38],
    [4, 60, 40],
    [4, 65, 45],
    [4, 70, 42],
    [5, 70, 50],
    [5, 75, 55],
    [6, 80, 60],
    [6, 85, 65],
    [7, 85, 70],
    [7, 90, 75],
    [8, 95, 80],
    [8, 98, 90],
    [2, 40, 28],
    [3, 50, 32],
    [5, 65, 48],
    [6, 75, 58]
]

# 0 = Fail, 1 = Pass
y = [
    0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 0, 1, 1
]

# 2. SPLIT DATA INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 3. CREATE AND TRAIN THE MODEL
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("=" * 45)
print(" STUDENT PASS/FAIL PREDICTION SYSTEM")
print("=" * 45)

print("\nModel trained successfully!")

# 4. SAVE THE TRAINED MODEL
joblib.dump(model, "student_model.pkl")
print("Model saved as student_model.pkl")

# 5. EVALUATE THE MODEL
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)

# 6. PREDICT FOR A NEW STUDENT
print("\n" + "=" * 45)
print(" PREDICT FOR A NEW STUDENT")
print("=" * 45)

try:
    hours = float(input("Study hours per day (0-24): "))
    attendance = float(input("Attendance percentage (0-100): "))
    marks = float(input("Previous exam marks (0-100): "))

    if not (
        0 <= hours <= 24
        and 0 <= attendance <= 100
        and 0 <= marks <= 100
    ):
        print("Invalid input. Please check the allowed ranges.")

    else:
        prediction = model.predict(
            [[hours, attendance, marks]]
        )[0]

        if prediction == 1:
            print("Prediction: Student may PASS!")
        else:
            print("Prediction: Student may FAIL!")

except ValueError:
    print("Invalid input! Please enter numeric values.")

# 7. VISUALIZE HISTORICAL STUDENT DATA
plt.figure(figsize=(8, 5))

for label, name, color in [
    (0, "Fail", "red"),
    (1, "Pass", "green")
]:
    indices = [
        i for i, result in enumerate(y)
        if result == label
    ]

    plt.scatter(
        [X[i][0] for i in indices],
        [X[i][2] for i in indices],
        color=color,
        label=name,
        s=70
    )

plt.xlabel("Study Hours per Day")
plt.ylabel("Previous Exam Marks")
plt.title("Student Results: Historical Examples")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. VISUALIZE THE TRAINED DECISION TREE
plt.figure(figsize=(16, 8))

plot_tree(
    model,
    feature_names=[
        "Study Hours",
        "Attendance",
        "Previous Marks"
    ],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Student Pass/Fail Prediction")
plt.tight_layout()
plt.show()

print("\nProject completed successfully!")