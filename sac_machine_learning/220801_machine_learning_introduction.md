# machine_learning_introduction

# 지도학습 예시

1. 편지 볻투에 손으로 쓴 우편번호 숫자 판별
2. 의료 영상 이미지에 기반한 종양판다
3. 의심되는 신용카드 거래 감지

# 비지학습 예시

1. 블로그 글의 주제 구분
2. 고객들을 취향이 비슷한 그룹으로 묶기
3. 비정상적인 웹사이트 접근 탐지

# 라이브러리

1. scikit-learn
2. NumPy
3. Scipy
4. matplotlib
5. pandas

# 성과측정

1. training set: 머신러닝 모델을 만들 때 사용
2. test set: 모델이 잘 작동하는지 측정할 때 사용
3. scikit-learn에서 데이터는 대문자 X, 레이블은 소문자 y로 표기
4. train_test_split: 전체 행 중, 75%를 레이블 데이터와 함께 training set. 나머지 25%는 레이블 데이터와 함께 test set.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)
```

1. train_test_split 함수로 데이터를 나누기 전, 데이터를 섞어야함
2. 데이터를 섞지 않고 25%를 test set로 사용하면 레이블은 모두 2가 됨. 데이터 포인트가 레이블 순서로 정렬되어 있기 때문.

```python
# print, shape 활용하여 확인
print("X_train 크기:", X_train.shape)
print("y_train 크기:", y_train.shape)

print("X_test 크기:", X_test.shape)
print("y_test 크기:", y_test.shape)

# 데이터 프레임에 넣어서 확인

iris_dataset = load_iris()
iris_data =  iris_dataset.data

# 붓꽅 데이터 셋을 자세히 보기 위해 DataFrame으로 변환
iris_df = pd.DataFrame(data=iris_data,
                      columns=iris_dataset.feature_names)
iris_df['label'] = iris_dataset.target
iris_df.head(3)
```

# 데이터 시각화

```python
iris_datasframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                          hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()
```

![Untitled](machine_learning_introduction%200750768777ef4c0e94c9d7a8198fe873/Untitled.png)

# KNeighborsClassifier(K-최근접 이웃 알고리즘)

- 개요
    1. k는 가장 가까운 이웃 ‘하나’가 아니라
    2. 훈련 데이터에서 새로운 데이터 포인트에 가장 가까운 ‘k개’의 이웃을 찾는다는 의미
    3. ex) 가장 가까운 3개 또는 5개의 이웃
    4. 그 다음 이웃들의 클래스 중 빈도가 가장 높은 클래스를 예측값으로 사용
    5. 모델 사용을 위해 클래스로부터 객체를 만들어야함
    6. 모델에 필요한 매개변수 입력

- 모델 훈련

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) # n_neighbors = 이웃의 개수. 직접 설정.

knn.fit(X_train, y_train) # 매개변수 입력 및 모델 훈련
```

# 예측하기

- NumPy 배열 생성

```python
X_new = np.array([[5, 2.9, 1, 0.2]])
print('X_new.shape:', X_new.shape)
```

- 예측

```python
prediction = knn.predict(X_new)
print('에측:', prediction)
print('예측한 타깃의 이름:', 
     iris_dataset['target_names'][prediction])

# out
에측: [0]
예측한 타깃의 이름: ['setosa']
```

# 모델 평가

1. 모델을 평가할 때, test set 사용 

```python
y_pred = knn.predict(X_test)
print('테스트 세트에 대한 예측값: \n', y_pred)

print('테스트 세트의 정확도:{:.2f}'.format(
    np.mean(y_pred == y_test)))

# score 사용
print("테스트 세트의 정확도: {:.2f}".format(
knn.score(X_test, y_test)))
```

# 정리

1. 각 품종을 ‘클래스’, 개별 붓꽃을 ‘레이블’
2. 붓꽃 데이터셋은 두 개의 NumPy 배열
3. 하나는 데이터를 답고 있으며 Scikit-learn에서 X 표기
4. 다른 하나는 기대하는 출력을 가지고 있으며 y 표기
5. 배열 X는 2차원 배열로 각 데이터 포인트는 행 하나, 각 특성은 열 하나
6. 배열 y는 1차원 배열로 각 샘플의 클래스 레이블에 해당하는 0~2 사이의 정수를 담고 있음
7. 모델이 새로운 데이터에 잘 적용되는지 평가하기 위해 training set, test set 으로 나눔
8. KNeighborsClassifier: 가장 가까운 이웃 선택
9. n_neighbors 매개변수를 지정해 클래스 객체 생성
10. fit 메서드를 호출해 모델 생성
11. score 메서드를 사용해 모델 평가 

```python
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print('테스트 세트의 정확도: {:.2f}'.format(knn.score(X_test, y_test)))

# out
테스트 세트의 정확도: 0.97
```

- 보충 설명
1. training set으로 훈련해서 test set으로 예측
2. 이미 test set의 답을 알고 있기 때문.
3. 예측값과 실제값을 비교하고 score를 구할 수 있음
4. 최종 목표는 진짜 데이터를 맞추는 것. 하지만 실제로 맞출 데이터가 없어서 ML의 성능을 높이기 위해 데이터를 나눠서 사용