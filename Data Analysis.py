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

# <markdowncell>

# 주어진 데이터 형식에서 날짜와 시간이 분리되어 있는데, 이를 동시에 처리하기 위한 유틸리티 함수를 만든다.

# <codecell>

parse = lambda x: datetime.strptime(x, '%Y%m%d %H%M')

# <codecell>

april = pd.read_csv('data/round2-4.csv', names=['date', 'time', 'direction', 'index', 'source', 'destination', 'distance', 'speed'], parse_dates=[[0, 1]], date_parser=parse, header=None)
may = pd.read_csv('data/round2-5.csv', names=['date', 'time', 'direction', 'index', 'source', 'destination', 'distance', 'speed'], parse_dates=[[0, 1]], date_parser=parse, header=None)
june = pd.read_csv('data/round2-6.csv', names=['date', 'time', 'direction', 'index', 'source', 'destination', 'distance', 'speed'], parse_dates=[[0, 1]], date_parser=parse, header=None)

# <markdowncell>

# 전체 데이터의 추이를 살펴보고 어떤 접근을 취할지 고민한다.

# <codecell>

whole_data = pd.concat([april, may, june])

# <markdowncell>

# 도로가 다양하므로 시각화가 어려우니 한 개의 도로가 어떤 모양을 그리는지 보자.

# <codecell>

df = whole_data[(whole_data['direction'] == 'D') & (whole_data['index'] == 1)]
df = df.sort(['date_time'])

del df['source']
del df['destination']
del df['direction']
del df['distance']
del df['index']

df = df.set_index('date_time')

# <markdowncell>

# 데이터가 5분 단위로 떨어져 있으니 이를 다루기 위한 유틸리티 함수를 만들자.

# <codecell>

from datetime import datetime, timedelta

def sec2timestr(sec):
    sec = timedelta(seconds=sec)
    d = datetime(1,1,1) + sec

    return "{:02}:{:02}".format(d.hour, d.minute)

# <markdowncell>

# 전체적인 추이를 살피기 위해 시간의 흐름에 따라 어떻게 도로 상황이 변하는지 애니메이션을 만들어보자.

# <codecell>

import matplotlib.animation as animation

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111)
line, = plt.plot(df[::288])

for k in range(1,14):
    l = plt.axvline(x=7*k-1, color='r')  # mark weekend
            
plt.ylim(0, 120)
plt.xlabel('Day')
plt.ylabel('Speed')
plt.title('Speed of D 001 as time progresses')

def animate(i):
    line.set_ydata(df[i::288])  # update the data
    
    timestr = sec2timestr(5 * 60 * i)
    
    plt.legend([line], [timestr])
    return line,

line_ani = animation.FuncAnimation(fig, animate, 288, interval=100)
line_ani.save('figure/daily_traffic_D_001.mp4')

# <markdowncell>

# 생성된 애니메이션을 보면 아마도 평일과 주말이 양상이 많이 다른 것 같다. 그 외에도 평일 나름의 특징이 있는 것 같은데, 일단 이건 무시한다.

# <codecell>


