import pandas as pd
import numpy as np
import mlflow
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression

def get_data():
    try:
        df = pd.read_csv('./heart_failure.csv')
    except:
        df = pd.read_csv('https://raw.githubusercontent.com/dimikara/Survival-Prediction-of-Patients-with-Heart-Failure/master/heart_failure_clinical_records_dataset.csv',encoding='latin1')
        df.to_csv('data/heart_failure.csv',index=False)

    return df

    
def eval_metrics(X_test,y_test):
  acc = accuracy_score(y_test,X_test)
  precision = precision_score(y_test,X_test)
  recall = recall_score(y_test,X_test)
  f1 = f1_score(y_test,X_test)

  return acc,precision,recall,f1


def train_model(C,max_iter):
    # The predicted column is "quality" which is a scalar from [3, 9]
    df = get_data()
    X = df.drop(['DEATH_EVENT'],axis=1)
    y = df['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.3)

    with mlflow.start_run():
      lr = LogisticRegression(C=C,max_iter=max_iter)
      lr.fit(X_train, y_train)

      predicted_qualities = lr.predict(X_test)

      (acc,precision,recall,f1) = eval_metrics(y_test, predicted_qualities)
      # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
      print("Accuracy: %s" % acc)
      print("Precision: %s" % precision)
      print("Recall: %s" % recall)
      print("F1: %s" % f1)

      mlflow.log_param("C", C)
      mlflow.log_param("max_iter", max_iter)

      mlflow.log_metric("acc", acc)
      mlflow.log_metric("precision", precision)
      mlflow.log_metric("recall", recall)
      mlflow.log_metric("f1", f1)

      mlflow.sklearn.log_model(lr, "model")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--c", "-C_", type=float, default=4)
    args.add_argument("--max_iter", "-iter", type=float, default=100)
    parsed_args = args.parse_args()

    train_model(C=parsed_args.c, max_iter=parsed_args.max_iter)