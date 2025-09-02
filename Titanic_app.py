import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load & preprocess dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\padal\model_Deploy\Titanic_train.csv")

# Encode categorical columns
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# Drop missing values (basic cleanup)
df = df.dropna(subset=["Age", "Embarked", "Fare"])

# Features & target
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate accuracy
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

# -----------------------------
# 2. Streamlit App
# -----------------------------
st.title("🚢 Titanic Survival Prediction (Logistic Regression)")

st.sidebar.header("User Input Parameters")

def user_input_features():
    Pclass = st.sidebar.selectbox("Pclass", (1, 2, 3))
    Sex = st.sidebar.selectbox("Sex", ("male", "female"))
    Age = st.sidebar.slider("Age", 0, 100, 25)
    SibSp = st.sidebar.slider("SibSp (Siblings/Spouses Aboard)", 0, 8, 0)
    Parch = st.sidebar.slider("Parch (Parents/Children Aboard)", 0, 6, 0)
    Fare = st.sidebar.slider("Fare", 0, 500, 50)
    Embarked = st.sidebar.selectbox("Embarked", ("C", "Q", "S"))

    data = {
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked
    }

    features = pd.DataFrame(data, index=[0])

    # Encode categorical
    features["Sex"] = features["Sex"].map({"male": 0, "female": 1})
    features["Embarked"] = features["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    return features

# Collect user input
input_df = user_input_features()

st.subheader("🔹 User Input Features")
st.write(input_df)

# -----------------------------
# 3. Make Predictions
# -----------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("🔹 Prediction")
st.write("✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive")

st.subheader("🔹 Prediction Probability")
st.write(prediction_proba)

st.subheader("🔹 Model Accuracy")
st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Test Accuracy: {test_acc:.2f}")