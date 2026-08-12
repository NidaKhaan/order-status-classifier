from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_and_clean, encode_features

def main():
    df = load_and_clean('../data/orders.csv')
    df_encoded, le = encode_features(df)

    X = df_encoded.drop(columns=['OrderStatus'])
    y = df_encoded['OrderStatus']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
    print(classification_report(y_test, lr_pred, target_names=le.classes_))

    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

if __name__ == '__main__':
    main()