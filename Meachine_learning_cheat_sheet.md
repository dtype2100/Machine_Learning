# Meachine_learning_cheat_sheet

# train_test_split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=66)
```

# KNeighborsClassifier

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print('테스트 세트에 대한 예측값: \n', y_pred)

print('테스트 세트 예측', clf.predict(X_test))

print('테스트 세트의 정확도:{:.2f}'.format(
    np.mean(y_pred == y_test)))

# 얼마나 잘 일반화되었는지 score 메서드 호출
print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, y_test)))
```

# LinearRegression

```python
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print('lr.coef_:', lr.coef_) # 기울기(w) coef_ 속성에 저장됨
print("lr.intercept_:", lr.intercept_) # 편향/절편(b) intercept_ 저장됨

print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
```

# Ridge

```python
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print('훈련 세트 점수: {:.2f}'.format(ridge.score(X_train, y_train)))
print('테스트 세트 점수: {:.2f}'.format(ridge.score(X_test, y_test)))

# alpha매개변수로 훈련 세트의 성능 대비 모델을 얼마나 단순화할지 지정
# (기본값 alpha=1.0)
# alpha값을 높이면 계수를 0에 더 가깝게 만들어 훈련 세트의 성능은 나빠지지만
# 일반화는 쉬워진다.
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))

# 반대로 alpha 값을 줄이면 계수에 대한 제약이 그만큼 풀리면서
# LinearRegression(0.95, 0.61)으로 만든 모델과 거의 같아진다.

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))
```

# Lasso

[Machine_Learning/220802_KNN.ipynb at master · dtype2100/Machine_Learning](https://github.com/dtype2100/Machine_Learning/blob/master/sac_machine_learning/220802_KNN.ipynb)

![Untitled](Meachine_learning_cheat_sheet%2070516c183e4846d18e975bea28d49ecf/Untitled.png)

- 위 그림을 보면 모든 데이터셋에 대해 릿지와 선형 회귀 모두 훈련세트의 점수가 테스트 세트의 점수보다 높다.
- 릿지 회귀에는 규제가 적용되므로 릿지의 훈련 데이터 점수가 전체적으로 선형 회귀의 훈련 데이터 점수보다 낮다.
- 그러나 테스트 데이터에서는 릿지의 점수가 더 높으며, 특별히 작은 데이터셋에서는 더 그렇다.
- 두 모델의 성능은 데이터가 많아질수로 좁아지고 마지막에는 선형 회귀가 릿지 회귀를 따라잡는다.
- => 데이터를 충분히 주면 규제 항은 중요성이 떨어진다.

```python
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso.score(X_test, y_test)))
print("사용한 특성의 개수:", np.sum(lasso.coef_ != 0))

lasso001 = Lasso(alpha=0.01, max_iter=50000).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso001.score(X_test, y_test)))
print("사용한 특성의 개수:", np.sum(lasso001.coef_ != 0))
```

# QuantileRegressor

```python
from sklearn.linear_model import QuantileRegressor

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pred_up = QuantileRegressor(quantile=0.9, alpha=0.01).fit(X_train, y_train).predict(X_test)
pred_med = QuantileRegressor(quantile=0.5, alpha=0.01).fit(X_train, y_train).predict(X_test)
pred_low = QuantileRegressor(quantile=0.1, alpha=0.01).fit(X_train, y_train).predict(X_test)

plt.scatter(X_train, y_train, label='훈련 데이터')
plt.scatter(X_test, y_test, label='테스트 데이터')
plt.plot(X_test, pred_up, label='백분위:0.9')
plt.plot(X_test, pred_med, label='백분위:0.5')
plt.plot(X_test, pred_low, label='백분위:0.1')
plt.legend()
plt.show()
```

# LogisticRegression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(max_iter=5000), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title(clf.__class__.__name__)
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend()
plt.show() # 책에는 없음

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)

logreg = LogisticRegression(max_iter=5000).fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(logreg.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg.score(X_test, y_test)))

logreg001 = LogisticRegression(C=100, max_iter=5000).fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(logreg100.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg100.score(X_test, y_test)))
```

# DecisionTreeClassifier

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                                                    stratify=iris.target,
                                                    random_state=42,
                                                   test_size = 0.2)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

dot_data = export_graphviz(tree,
                          out_file=None,
                          feature_names = iris.feature_names,
                          class_names = iris.target_names,
                          filled=True,
                          rounded=True,
                          special_characters  = True)
graph = graphviz.Source(dot_data)
graph

print("특성 중요도:\n{}".format(tree.feature_importances_))

def plot_feature_importances_cancer(model):
    n_features = iris.data.shape[1]
    plt.barh(np.arange(n_features), tree.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), iris.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)

plot_feature_importances_cancer(tree)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                                                    stratify=iris.target,
                                                    random_state=42,
                                                   test_size = 0.2)
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

pred = tree.predict(X_test)
(pred == y_test).mean() # Accuracy
```

# PCA

[Machine_Learning/220804_practice.ipynb at master · dtype2100/Machine_Learning](https://github.com/dtype2100/Machine_Learning/blob/master/sac_machine_learning/220804_practice.ipynb)

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                   random_state=0)
print(X_train.shape)
print(X_test.shape)

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df.head()

scaler = StandardScaler()
scaler.fit(iris.data)
X_scaled = scaler.transform(iris.data)

pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

print('원본 데이터:', str(X_scaled.shape))
print('축소된 데이터', str(X_pca.shape))
print('PCA 주성분:', str(pca.components_.shape))
print('PCA 주성분:', pca.components_)

plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ['첫 번째 주성분', '두 번째 주성분'])
plt.colorbar()
plt.xticks(range(len(iris.feature_names)),
          iris.feature_names, rotation=60, ha='left')
plt.xlabel('특성')
plt.ylabel('주성분')
plt.show()
```

# KMeans

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(random_state=1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

######

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
plt.show()

#######

from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)

linkage_array = ward(X)

dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' 두 개 클러스터', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' 세 개 클러스터', va='center', fontdict={'size': 15})
plt.xlabel("샘플 인덱스")
plt.ylabel("클러스터 거리")
plt.show()
```

# DBSCAN

```python
from sklearn.cluster import DBSCAN
X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("클러스터 레이블:\n", clusters)

##################################

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors='black')
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.show()

##################################

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN(eps=0.2) # shift + tab
clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=60, 
            edgecolors='black')
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.show()
```

# cross_val_score(교차검증)

[Machine_Learning/220805_machine_learning.ipynb at master · dtype2100/Machine_Learning](https://github.com/dtype2100/Machine_Learning/blob/master/sac_machine_learning/220805_machine_learning.ipynb)

```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression(max_iter=1000)

# cross_val_score의 기본값은 3겹(5겹) 교차 검증이므로 정확도 값이 3개 반환
scores = cross_val_score(logreg, iris.data, iris.target)
print('교차 검증 점수:', scores) # 임의로 설정하지 않아서 3겹

##################################

# cv=10: 교차검증 => 폴드 수는 cv매개변수를 사용하여 변경
# 모델, 훈련 데이터, 타깃 레이블
scores = cross_val_score(logreg, iris.data, iris.target, cv=10) 
print('교차 검증 점수:', scores)

print('교차 검증 평균 점수: {:.2f}'.format(scores.mean()))

from sklearn.model_selection import cross_validate
res = cross_validate(logreg, iris.data, iris.target,
                      return_train_score=True)
res
```

# KFold

```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

kf_tree_clf = DecisionTreeClassifier()
sco = 'accuracy'
score = cross_val_score(kf_tree_clf, X_train, y_train, cv=k_fold, scoring=sco)
print(score)
```