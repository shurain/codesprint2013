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

import pymc as mc

# <codecell>

parse = lambda x: datetime.strptime(x, '%Y%m%d %H%M')

# <codecell>

april = pd.read_csv('data/round2-4.csv', names=['date', 'time', 'direction', 'index', 'source', 'destination', 'distance', 'speed'], parse_dates=[[0, 1]], date_parser=parse, header=None)
may = pd.read_csv('data/round2-5.csv', names=['date', 'time', 'direction', 'index', 'source', 'destination', 'distance', 'speed'], parse_dates=[[0, 1]], date_parser=parse, header=None)
june = pd.read_csv('data/round2-6.csv', names=['date', 'time', 'direction', 'index', 'source', 'destination', 'distance', 'speed'], parse_dates=[[0, 1]], date_parser=parse, header=None)

# <codecell>

april = april.sort(['direction', 'index', 'date_time'])

# <codecell>

april['time'] = april.date_time.apply(lambda x : "{:02d}{:02d}".format(x.hour, x.minute))

# <codecell>

df = april[(april['direction'] == 'D') & (april['index'] == 1) & (april['time'] == '0800')]
scatter(range(30), df.speed)
plt.ylim(0, 120)
plt.xlim(0, 30)

# <codecell>

p = mc.Uniform("p", 0, 1)

assignment = mc.Categorical("assignment", [p, 1-p], size=len(df.speed)) 

taus = 1.0/mc.Uniform("stds", 0, 100, size=2)**2
mus = mc.Uniform("mus", 0, 120, size=2)

# <codecell>

taus.value

# <codecell>

@mc.deterministic 
def mu_i(assignment=assignment, mus=mus):
    return mus[assignment]

@mc.deterministic
def tau_i(assignment=assignment, taus=taus):
    return taus[assignment]

# <codecell>

observations = mc.Normal("obs", mu_i, tau_i, value=df.speed, observed=True)
model = mc.Model([p, assignment, mus, taus])

# <codecell>

mcmc = mc.MCMC(model)
mcmc.sample(500000)

# <codecell>

mu_trace = mcmc.trace('mus')[:]
std_trace = mcmc.trace('stds')[:]
p_trace = mcmc.trace('p')[:]

# <codecell>

figsize( 12.5, 9 )
subplot(311)

colors = ["#348ABD", "#A60628"]
lw = 1

plot(mu_trace[:,0], label = "trace of mu 0", c = colors[0], lw = lw)
plot(mu_trace[:,1], label = "trace of mu 1", c = colors[1], lw = lw)
plt.title( "Traces of unknowns" )
leg = plt.legend(loc = "upper right")

subplot(312)
plot(std_trace[:,0], label = "trace of standard deviation of cluster 0", c = colors[0], lw=lw)
plot(std_trace[:,1], label = "trace of standard deviation of cluster 1", c = colors[1], lw=lw)
leg = plt.legend(loc = "upper right")

subplot(313)
plot( p_trace, label = "$p$: frequency of assignment to mu 0", color = "#467821", lw = lw)
leg = plt.legend(loc = "upper right")

# <codecell>

figsize( 11.0, 4 )

_i = [ 1,2,3,0]
for i in range(2):
    subplot(2, 2, _i[ 2*i ])
    plt.title("Posterior of mu %d"%i)
    plt.hist(mu_trace[:, i], color = colors[i],bins = 30, histtype="stepfilled" )
    
    subplot(2, 2, _i[ 2*i + 1])
    plt.title("Posterior of standard deviation %d"%i)
    plt.hist( std_trace[:, i], color = colors[i],  bins = 30, histtype="stepfilled"  )
    
plt.tight_layout()

# <codecell>

plt.hist(p_trace, bins=30, histtype="stepfilled")

# <codecell>

weekdays = pd.concat([april, may, june])
weekdays['weekday'] = weekdays['date_time'].apply(lambda x: x.weekday())
weekdays = weekdays[weekdays['weekday'] < 5]
weekdays['time'] = weekdays.date_time.apply(lambda x: "{:02d}{:02d}".format(x.hour, x.minute))

# <codecell>

road_101 = weekdays[(weekdays['direction'] == 'D') & (weekdays['index'] == 1)]

# <codecell>

display(HTML(road_101[100::288].to_html()))

# <codecell>

n_var = 2
timepoint = 103

# <codecell>

print len(road_101[100::288])

# <codecell>

road_p = mc.Uniform("road_p", 0, 1)

road_assign = mc.Categorical("road_assign", [road_p, 1-road_p], size=road_101[0::288].shape[0])
print road_assign.value

# <codecell>

road_p = mc.Uniform("road_p", 0, 1)

road_assign = mc.Categorical("road_assign", [road_p, 1-road_p], size=road_101[::288].shape[0])

road_taus = 1.0/mc.Uniform("road_stds", 0, 100, size=2*n_var)**2
road_mus = mc.Uniform("road_mus", 0, 120, size=2*n_var)

@mc.deterministic 
def road_mu_i(road_assign=road_assign, road_mus=road_mus):
    return np.transpose([road_mus[road_assign+i] for i in range(0,2*n_var,2)])

@mc.deterministic
def road_tau_i(road_assign=road_assign, road_taus=road_taus):
    return np.transpose([road_taus[road_assign+i] for i in range(0, 2*n_var, 2)])

#FIXME hardcoded road_101 zip
road_obs = mc.Normal("road_obs", road_mu_i, road_tau_i, value=zip(*[road_101[timepoint+i::288].speed for i in range(n_var)]), observed=True)
road_model = mc.Model([road_p, road_assign, road_mus, road_taus, road_obs])

# <codecell>

road_mcmc = mc.MCMC(road_model)
%time road_mcmc.sample(500000, 400000)

# <codecell>

road_mu_trace = road_mcmc.trace('road_mus')[:]
road_std_trace = road_mcmc.trace('road_stds')[:]
road_p_trace = road_mcmc.trace('road_p')[:]

# <codecell>

print road_mu_trace[:,3]

# <codecell>

figsize( 11.0, 4 )

subplot(2, 2, 1)
plt.title("Posterior of mu 0 at {}:{}".format(timepoint*5/60, timepoint*5%60))
plt.hist(road_mu_trace[:, 0],bins = 30, color = colors[0], histtype="stepfilled")
subplot(2, 2, 2)
plt.hist(road_mu_trace[:, 1],bins = 30, color = colors[0], histtype="stepfilled")
plt.title("Posterior of mu 1 at {}:{}".format(timepoint*5/60, timepoint*5%60))
subplot(2, 2, 3)
plt.title("Posterior of mu 0 at {}:{}".format((timepoint+1)*5/60, (timepoint+1)*5%60))
plt.hist(road_mu_trace[:, 2],bins = 30, color = colors[0], histtype="stepfilled")
subplot(2, 2, 4)
plt.hist(road_mu_trace[:, 3],bins = 30, color = colors[0], histtype="stepfilled")
plt.title("Posterior of mu 1 at {}:{}".format((timepoint+1)*5/60, (timepoint+1)*5%60))

plt.tight_layout()

# <codecell>

print len(road_mu_trace)

# <markdowncell>

# 3개를 제출할 때, 1st model을 가정한 경우, 2nd model을 가정한 경우, 양쪽 모델을 모두 고려한 경우를 제출하는 게 좋지 않을까?

# <codecell>

plt.hist(road_p_trace, bins=30, histtype="stepfilled")

# <codecell>

road_assign_trace = road_mcmc.trace('road_assign')

# <codecell>

road_assign_trace[1]

# <codecell>

road_assign_trace[:].shape

# <codecell>

generated_road = []

for i, j  in enumerate(road_assign_trace[:]):
    generated_road.extend(np.transpose(mc.rnormal([road_mu_trace[i][j + k] for k in range(0, 2*n_var, 2)], [1.0/road_std_trace[i][j + k] for k in range(0, 2*n_var, 2)])))

# <codecell>

print generated_road[0]

# <markdowncell>

# 음수인 결과가 나오면 안 된다. 모델을 수정해야 한다.

# <codecell>

def loss_func(guessed_speed, true_speed):
    loss = np.abs(true_speed - guessed_speed)

    return np.sum(loss, 1)

# <codecell>

expected_loss = lambda guess: loss_func(guess, np.array(generated_road)).mean()    

# <codecell>

import scipy.optimize as sop

# <codecell>

%time min_res = sop.fmin(expected_loss, x0=road_101[timepoint:timepoint+n_var].speed)
print min_res

# <codecell>

road_101 = weekdays[(weekdays['direction'] == 'D') & (weekdays['index'] == 10)]

answers = []
for timepoint in range(0, 288-1):
    n_var = 2
    
    road_p = mc.Uniform("road_p", 0, 1)
    
    road_assign = mc.Categorical("road_assign", [road_p, 1-road_p], size=road_101[::288].shape[0])
    
    road_taus = 1.0/mc.Uniform("road_stds", 0, 100, size=2*n_var)**2
    road_mus = mc.Uniform("road_mus", 0, 120, size=2*n_var)
    
    @mc.deterministic
    def road_mu_i(road_assign=road_assign, road_mus=road_mus):
        return np.transpose([road_mus[road_assign+i] for i in range(0,2*n_var,2)])
    
    @mc.deterministic
    def road_tau_i(road_assign=road_assign, road_taus=road_taus):
        return np.transpose([road_taus[road_assign+i] for i in range(0, 2*n_var, 2)])
    
    #FIXME hardcoded road_101 zip
    road_obs = mc.Normal("road_obs", road_mu_i, road_tau_i, value=zip(*[road_101[timepoint+i::288].speed for i in range(n_var)]), observed=True)
    road_model = mc.Model([road_p, road_assign, road_mus, road_taus, road_obs])
    
    road_mcmc = mc.MCMC(road_model)
    %time road_mcmc.sample(50000, 40000)
    
    road_mu_trace = road_mcmc.trace('road_mus')[:]
    road_std_trace = road_mcmc.trace('road_stds')[:]
    road_p_trace = road_mcmc.trace('road_p')[:]
    road_assign_trace = road_mcmc.trace('road_assign')[:]
    
    generated_road = []
    
    for i, j  in enumerate(road_assign_trace):
        generated_road.extend(np.transpose(mc.rnormal([road_mu_trace[i][j + k] for k in range(0, 2*n_var, 2)], [1.0/road_std_trace[i][j + k] for k in range(0, 2*n_var, 2)])))
    
    def loss_func(guessed_speed, true_speed):
        loss = np.abs(true_speed - guessed_speed)
    
        return np.sum(loss, 1)
    
    expected_loss = lambda guess: loss_func(guess, np.array(generated_road)).mean()    
    
    %time min_res = sop.fmin(expected_loss, x0=road_101[timepoint:timepoint+n_var].speed)
    answers.append(min_res)
    
print answers

# <codecell>

final_answer = [(a[1]+b[0])/2 for a, b in zip(answers, answers[1:])]
final_answer.insert(0, answers[0][0])
final_answer.append(answers[-1][1])
print final_answer

# <codecell>


