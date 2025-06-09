from Helper.AI_Helper import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report

Iris = Get_Dataset("himanshunakrani/iris-dataset", "iris.csv")


y = Iris["species"]
x = Iris.drop("species", axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
# shows the difference between predicted categories and actual categories
print(score)
print(classification_report(y_test, y_pred))