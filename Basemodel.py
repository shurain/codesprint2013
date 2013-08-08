# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
from IPython.core.display import HTML
import matplotlib as mtp
from pylab import *

from datetime import datetime, timedelta
from StringIO import StringIO

# <codecell>

parse = lambda x: datetime.strptime(x, '%Y%m%d %H%M')

# <codecell>

april = pd.read_csv('data/round2-4.csv', names=['date', 'time', 'direction', 'index', 'source', 'destination', 'distance', 'speed'], parse_dates=[[0, 1]], date_parser=parse, header=None)
may = pd.read_csv('data/round2-5.csv', names=['date', 'time', 'direction', 'index', 'source', 'destination', 'distance', 'speed'], parse_dates=[[0, 1]], date_parser=parse, header=None)
june = pd.read_csv('data/round2-6.csv', names=['date', 'time', 'direction', 'index', 'source', 'destination', 'distance', 'speed'], parse_dates=[[0, 1]], date_parser=parse, header=None)

# <markdowncell>

# 간단한 검증을 거쳐서 모델을 선택하기로 한다. 4, 5월을 모델 학습을 위한 데이터로 삼고 6월을 이를 검증하는 데이터로 삼는다.

# <codecell>

train = pd.concat([april, may])
test = pd.concat([june])

train = train.sort(['direction', 'index', 'date_time'])
test = test.sort(['direction', 'index', 'date_time'])

# <markdowncell>

# Data analysis에서 평일과 주말을 분리하여 보기로 하였는데, 검증을 해보자. 일단 전체 (평일과 주말) 데이터를 사용하여 median을 구해보자.

# <codecell>

whole_week = train.copy()
whole_week['time'] = whole_week.date_time.apply(lambda x: "{:02d}{:02d}".format(x.hour, x.minute))
group = whole_week.groupby(['direction', 'index', 'time'])
df = group.median()
median_model = df.reset_index()

# <codecell>

print median_model
display(HTML(median_model[:10].to_html()))

# <markdowncell>

# Test를 어떻게 하느냐도 문제가 되지만 일단 화요일에 대한 검증만 해보도록 하자. 2013년 6월의 화요일은 6/4, 6/11, 6/18, 6/25일이다.

# <codecell>

def test_june(prediction, dow='tue'):
    week = ['sat', 'sun', 'mon', 'tue', 'wed', 'thu', 'fri']
    i = week.index(dow.lower()) 
    testing_days = range(i+1, 31, 7)

    result = []
    for k in testing_days:
        test_data = june.copy()
        test_data['day'] = test_data.date_time.apply(lambda x: int(x.day))
        test_data['time'] = test_data.date_time.apply(lambda x: "{:02d}{:02d}".format(x.hour, x.minute))
        test_data = test_data[test_data['day'] == k]
        assert(len(test_data) == 2*126*288)
        test_data = test_data.sort(['direction', 'index', 'time'])
        prediction = prediction.sort(['direction', 'index', 'time'])
        
        result.append(np.mean(np.abs(prediction.speed.values - test_data.speed.values)))
        
    return result

# <codecell>

median_res = test_june(median_model, 'tue')
print np.mean(median_res)
print median_res

# <markdowncell>

# 주중의 데이터만 활용한 모델을 만들어보자.

# <codecell>

weekdays = train.copy()
weekdays['weekday'] = weekdays['date_time'].apply(lambda x: x.weekday())
weekdays = weekdays[weekdays['weekday'] < 5]
del weekdays['weekday']
weekdays['time'] = weekdays.date_time.apply(lambda x: "{:02d}{:02d}".format(x.hour, x.minute))
group = weekdays.groupby(['direction', 'index', 'time'])
df = group.median()
weekday_median_model = df.reset_index()

# <codecell>

weekday_median_res = test_june(weekday_median_model, 'tue')
print np.mean(weekday_median_res)
print weekday_median_res

# <markdowncell>

# 일단 화요일에 대해서는 주중 데이터만 활용하는 것이 더 좋다. 다만 데이터를 자세히 살펴보면 2:2의 결과이며 하루는 값이 좀 튀는 경향이 있다. 일단 데이터 포인트가 4개 밖에 안 되기 때문에 통계적으로 안정적인 결과라 할 수는 없다.

# <codecell>

for i in range(7):
    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    print days[i]
    res1 = test_june(median_model, days[i])
    res2 = test_june(weekday_median_model, days[i])
    print np.mean(res1), np.mean(res2)

# <markdowncell>

# 주말의 결과는 전체 데이터를 사용한 것이 월등하다. 평일에는 조금 갈리는 경향을 보인다. 월, 목, 금에는 전체 데이터를 사용한 편이 좋고 화, 수에는 평일 데이터만 활용하는 것이 좋다.

# <markdowncell>

# 조금 더 나은 분석을 위해 일종의 cross validation을 해보자.

# <codecell>

whole_data = pd.concat([april, may, june])

# <codecell>

whole_data['date'] = whole_data.date_time.apply(lambda x: x.date())
whole_data['time'] = whole_data.date_time.apply(lambda x: "{:02d}{:02d}".format(x.hour, x.minute))
whole_data['weekday'] = whole_data['date_time'].apply(lambda x: x.weekday())
whole_data = whole_data.sort(['date', 'direction', 'index', 'time'])

# <codecell>

import random

def crossvalidate():
    # 91 days
    days = range(91)
    random.shuffle(days)

    STRIDE = 2 * 126 * 288
    
    test_range = days[:10]
    train_range = days[10:]
    
    train_data = []
    for x in train_range:
        train_data.append(whole_data[x * STRIDE:(x + 1) * STRIDE])
        
    test_data = []
    for x in test_range:
        test_data.append(whole_data[x * STRIDE:(x + 1) * STRIDE])
        
    cv_train = pd.concat(train_data)
    cv_test = pd.concat(test_data)

    return cv_train, cv_test

# <markdowncell>

# Crossvalidate 함수는 말 그대로 k-fold cross validation을 하기 위한 함수이다. 데이터를 10:81으로 나누도록 하드코딩 되어 있으니 9-fold CV라 할 수 있겠다. 이런식으로 사용하기 위해 몇 가지 가정이 뒷받침되어야 하지만 이는 된다고 가정하고 분석을 해보자.

# <codecell>

def test_cv(prediction, test_data, dow='tue'):
    week = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    i = week.index(dow.lower())
    
    test_data = test_data[test_data['weekday'] == i]
    
    STRIDE = 2 * 126 * 288
    stepsize = len(test_data) / STRIDE
    
    result = []
    for k in range(stepsize):
        temp_data = test_data[k * STRIDE:(k + 1) * STRIDE]
        temp_data = temp_data.sort(['direction', 'index', 'time'])
        prediction = prediction.sort(['direction', 'index', 'time'])
        
        result.append(np.mean(np.abs(prediction.speed.values - temp_data.speed.values)))
        
    return result

# <codecell>

for x in range(10):
    train, test = crossvalidate()

    group = train.groupby(['direction', 'index', 'time'])
    df = group.median()
    cv_median_model = df.reset_index()
    
    weekdays = train[train['weekday'] < 5]
    group = weekdays.groupby(['direction', 'index', 'time'])
    df = group.median()
    cv_weekday_median_model = df.reset_index()
    
    cv_median_model_res = test_cv(cv_median_model, test, 'tue')
    cv_weekday_median_model_res = test_cv(cv_weekday_median_model, test, 'tue')
    
    print np.mean(cv_median_model_res), np.mean(cv_weekday_median_model_res)
    print np.mean(cv_median_model_res) - np.mean(cv_weekday_median_model_res)

# <markdowncell>

# 화요일 기준으로는 평일 데이터를 사용한 것이 거의 항상 우월하다.

# <codecell>

for y in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
    print y
    result = []
    for x in range(10):
        train, test = crossvalidate()
    
        group = train.groupby(['direction', 'index', 'time'])
        df = group.median()
        cv_median_model = df.reset_index()
        
        weekdays = train[train['weekday'] < 5]
        group = weekdays.groupby(['direction', 'index', 'time'])
        df = group.median()
        cv_weekday_median_model = df.reset_index()
        
        cv_median_model_res = test_cv(cv_median_model, test, y)
        cv_weekday_median_model_res = test_cv(cv_weekday_median_model, test, y)
        
        result.append(np.mean(cv_median_model_res) - np.mean(cv_weekday_median_model_res))
    print result

# <markdowncell>

# 전체 요일에 대해 비슷하게 cross validation 분석을 해보면 전체 데이터를 사용하는 편이 주말은 물론이고 월요일에도 더 우월한 전략이다. 화요일과 수요일, 목요일 그리고 금요일에는 평일 데이터만 사용하는 편이 더 우월하다. 이는 따로 cross validation을 하지 않은 결과와 비슷해 보인다. 비록 10회 밖에 반복을 하지 않아 통계적인 안정성을 말할 수는 없지만, 적어도 화요일에는 평일 데이터만 사용하는 편이 더 나은 것으로 보인다.
# 
# 요일별로 양상이 다른 것을 고려한다면 목표 예측 요일별 데이터를 뽑아내는 모집단도 더 세밀하게 나눠보는 것을 고려할 수 있을 것이다.

# <markdowncell>

# 최종 loss function이 MAE (mean absolute error) 이므로 평균값 (mean) 보다는 중앙값 (median) 을 사용하는 편이 더 성능이 좋을 것이라고 생각할 수 있다. 이를 검증하는 것은 쉬운 문제이다.

# <codecell>

whole_week = pd.concat([april, may])
whole_week['time'] = whole_week.date_time.apply(lambda x: "{:02d}{:02d}".format(x.hour, x.minute))
group = whole_week.groupby(['direction', 'index', 'time'])
df = group.mean()
mean_model = df.reset_index()

mean_res = test_june(mean_model, 'tue')
print np.mean(mean_res)
print mean_res

# <markdowncell>

# Median을 사용한 모델의 에러는 5.86801621748 였는데, mean을 사용한 모델은 6.04020650371 로 크게 차이난다.

# <codecell>

weekdays = pd.concat([april, may])
weekdays['weekday'] = weekdays['date_time'].apply(lambda x: x.weekday())
weekdays = weekdays[weekdays['weekday'] < 5]
del weekdays['weekday']
weekdays['time'] = weekdays.date_time.apply(lambda x: "{:02d}{:02d}".format(x.hour, x.minute))
group = weekdays.groupby(['direction', 'index', 'time'])
df = group.mean()
weekday_mean_model = df.reset_index()

weekday_mean_res = test_june(weekday_mean_model, 'tue')
print np.mean(weekday_mean_res)
print weekday_mean_res

# <markdowncell>

# 주중 데이터만 사용한 경우에도 마찬가지의 결과를 얻을 수 있다. Median 기반은 5.85363191689 인데 mean 기반은 5.9240332326 이다.

# <codecell>


