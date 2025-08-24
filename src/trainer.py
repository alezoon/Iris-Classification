from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_model(model, X, y, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test



def model_evaluation(model, X_test, y_test):

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    return acc, cm, cr



def prediction(model, datapoint, target_names=None):
    predicted_variant = model.predict(datapoint)
    variant_probability = model.predict_proba(datapoint)

    if target_names is not None:
        predicted_name = target_names[predicted_variant[0]]
        return predicted_variant, predicted_name, variant_probability
    return predicted_variant, variant_probability