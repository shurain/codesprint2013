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
    # dow는 0부터 6까지의 값으로 0이 일요일이다. 화요일은 2이다.
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

