from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from clearml import Task


task = Task.init(project_name='IRIS HPO TEST', task_name='experiment with HPO form repo py file')
task.set_repo(
    repo="https://github.com/BalogneseCoder/clearml_testing.git",
    branch='main',
)

task.execute_remotely(queue_name="ml_ops_stage")

iris = datasets.load_iris()
x, y = iris.data, iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)
hyperparameters = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_leaf': 1
}

task.connect(hyperparameters)
classifier = RandomForestClassifier(**hyperparameters)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

logger = task.get_logger()

train_accuracy = accuracy_score(y_train, classifier.predict(x_train))
test_accuracy = accuracy_score(y_test, predictions)

logger.report_scalar(title='accuracy', series='train', value=train_accuracy, iteration=0)
logger.report_scalar(title='accuracy', series='test', value=test_accuracy, iteration=0)

task.close()
