import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate  # For generating markdown tables
import mlflow
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

os.environ["MLFLOW_TRACKING_URI"] = "https://gitlab-codecamp24.obss.io/api/v4/projects/129/ml/mlflow/"
os.environ["MLFLOW_TRACKING_TOKEN"] = "glpat-cxSaGfZ6sy-ifJaYJAmB"



df=pd.read_csv("tips.csv")
df.head()


print(df.isna().sum())



df['smoker'] = df['smoker'].map({'Yes': 1, 'No': 0})
df['sex'] = df['sex'].map({'Female': 1, 'Male': 0})
df['time'] = df['time'].map({'Dinner': 1, 'Lunch': 0})



correlation = df['total_bill'].corr(df['tip'])
print('Pearson Korelasyon Katsayısı:', correlation)


correlation = df['size'].corr(df['tip'])
print('Pearson Korelasyon Katsayısı:', correlation)



day_mapping = {
    'Sun': 0,
    'Mon': 1,
    'Tue': 2,
    'Wed': 3,
    'Thur': 4,
    'Fri': 5,
    'Sat': 6
}

df['day'] = df['day'].map(day_mapping)



sns.boxplot(x='sex', y='tip', data=df)
plt.title('Cinsiyete Göre tip Dağılımı')
plt.show()

cosine_sim_matrix = cosine_similarity(df.T)

cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=df.columns, columns=df.columns)

print(cosine_sim_df)



df['combined_feature'] = df['size'] * df['total_bill']
df = df.drop(columns=['size'])
df = df.drop(columns=['total_bill'])


Q1 = df['tip'].quantile(0.25)
Q3 = df['tip'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mean_tip = df['tip'].mean()

df.loc[(df['tip'] < lower_bound) | (df['tip'] > upper_bound), 'tip'] = mean_tip

sns.boxplot(x='sex', y='tip', data=df)
plt.title('Cinsiyete Göre Tip Dağılımı')
plt.show()


X = df.drop(columns=['tip'])
y = df['tip']

RANDOM_SEED = 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
model = ElasticNet(alpha=0.6, l1_ratio=1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Elastic Net MSE: {mse}')


df['integer_part'] = df['tip'].apply(lambda x: int(x))
df['tip'] = df['integer_part']
df = df.drop(columns=['integer_part'])
y = df['tip']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = SVC(kernel='linear') 
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
param_grid_tree = {
    "max_depth": [3, 5, 7, 9, 11, 13],
    'criterion': ["gini", "entropy"],
}

grid_tree = GridSearchCV(
        estimator=dt,
        param_grid=param_grid_tree, 
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0
    )
dt_model = grid_tree.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)



lr = LogisticRegression(random_state=RANDOM_SEED)
param_grid_log = {
    'C': [150, 10, 1.0, 0.2, 0.05],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_log = GridSearchCV(
        estimator=lr,
        param_grid=param_grid_log, 
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0
    )
model_log = grid_log.fit(X_train, y_train)


mlflow.set_experiment("tip-prediction")
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1,average='macro')
    fpr, tpr, _ = metrics.roc_curve(actual, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return accuracy, f1, auc


def save_model_report(metrics, params, name, report_path="metrics_report.md"):
    report = []
    if os.getenv('MLFLOW_TRACKING_URI'):
        if os.getenv('GITLAB_CI'):
            ci_job_id = os.getenv('CI_JOB_ID')
    else:
        ci_job_id = "Undefined"
    
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = f.readlines()
            
    report.append(f"\n\n# Model Report for {name}\n\n")
    report.append(f"#### CI Job ID: {ci_job_id}\n\n")
    report.append("## Model Parameters\n\n")
    for key, value in params.items():
        report.append(f"- **{key}** : {value}\n")
    report.append("\n\n## Metrics\n\n")

    if isinstance(metrics, dict):
        metrics_list = [metrics]
    else:
        metrics_list = metrics

    report.append(tabulate(metrics_list, headers="keys", tablefmt="pipe"))

    with open(report_path, "w") as f:
        f.write("".join(report))

    return report_path

def mlflow_logging(model, X, y, name):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        if os.getenv('GITLAB_CI'):
            mlflow.set_tag('gitlab.CI_JOB_ID', os.getenv('CI_JOB_ID'))
        
        pred = model.predict(X)
        # Metrics
        accuracy, f1, auc = eval_metrics(y, pred)
        
        metrics_data = {
            "Mean CV score": model.best_score_,
            "Accuracy": accuracy,
            "f1-score": f1,
            "AUC": auc
        }
        params = model.best_params_
        
        # Logging best parameters from GridSearchCV
        mlflow.log_params(params)
        mlflow.log_params({"Class": name})
        # Log the metrics
        mlflow.log_metrics(metrics_data)
        
        # Logging artifacts and model
        mlflow.log_artifact("plots/ROC_curve.png")
        
        # Save and log model report
        report_path = save_model_report(metrics_data, params, name)

        mlflow.sklearn.log_model(model, name) 

        mlflow.end_run()


mlflow_logging(dt_model, X_test, y_test, "DecisionTreeClassifier")
mlflow_logging(model_log, X_test, y_test, "LogisticRegression")



