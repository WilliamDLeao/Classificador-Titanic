import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("train.csv")

features = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
]

X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Acuracia com k=5: {acc:.3f}")

k_values = range(1, 31)
scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    temp_model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])
    
    score = cross_val_score(
        temp_model, X_train, y_train,
        cv=cv,
        scoring="accuracy"
    ).mean()
    scores.append(score)

plt.plot(list(k_values), scores, marker="o")
plt.xlabel("Valor de k")
plt.ylabel("Acuracia media de validacao")
plt.title("KNN - Titanic: Acuracia por K")
plt.grid(True)
plt.show()