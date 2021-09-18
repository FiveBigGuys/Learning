from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
print('======')
print("数据的键")
print(mnist.keys())
print("数据概况")
print(mnist.data.shape[0])
print(mnist.data.shape[1])
print(mnist['DESCR'])
n
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(mnist['data'],mnist['target'])
print('======')
print(f"X_train_shape{X_train.shape}")
print(f"X_test_shape{X_test.shape}")
print(f"y_train_shape{y_train.shape}")
print(f"y_test_shape{y_test.shape}")

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print(knn)
knn.fit(X_train,y_train)
print(f"测试得分：{knn.score(X_test,y_test):.2f}")
print(f"测试分类：{knn.score(X_test,y_test):.2f}")