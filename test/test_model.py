from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def test_model_accuracy():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.33, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[Test] Model Accuracy: {accuracy * 100:.2f}%")

    # Test: Ensure accuracy is above 90%
    assert accuracy > 0.9, "Model accuracy is below 90%"

if __name__ == "__main__":
    test_model_accuracy()