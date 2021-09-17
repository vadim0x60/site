+++
title = "Neurogenetic optimization"
date = "2021-09-17"
author = "Vadim Liventsev"
+++


What's your favourite approach to gradient-free optimization? (You do have a favourite approach to gradient-free optimization, right?) Chances are that your answer is one of the following:
* *Evolutionary algorithms*: simulated annealing or anything based on a "generate - mutate - crossover - select" loop
* *Machine learning*: modeling the unknown function with a neural network or another machine learning model

I am here to make the case that a combination of the two is often desired.

Make sure you have the libraries installed:


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
```

Consider the task of finding the minimum of a simple function:


```python
def reward_f(x):
    return np.sin(x) + np.sin(x * 10) - 0.01 * x ** 2
```

We're allowed to query the values of the function for any x, but (please pretend for the sake of the argument that) we don't have access to it's derivative


```python
min_x = -10
max_x = 10
```


```python
plot_x = np.linspace(min_x, max_x, 1000)
plot_y = reward_f(plot_x)

sns.lineplot(plot_x, plot_y)
```

    /home/vadim0x60/.pyenv/versions/3.9.5/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(





    <AxesSubplot:>




    
![png](output_7_2.png)
    


## Evolutionary approach


```python
xs = []
rs = []
rs_exp = []
```


```python
def genetic_step():
    parent1, parent2 = np.random.choice(xs, size=2, p=rs_exp/np.sum(rs_exp))
    new_x = (parent1 + parent2) / 2
    new_r = reward_f(new_x)
    
    xs.append(new_x)
    rs.append(new_r)
    rs_exp.append(np.exp(new_r))
```


```python
xs = [min_x, max_x]
rs = [reward_f(min_x), reward_f(max_x)]
rs_exp = [np.exp(reward_f(min_x)), np.exp(reward_f(max_x))]

for _ in range(10000):
    genetic_step()
    
best_idx = np.argmax(rs)
```


```python
xs = np.array(xs)
rs = np.array(rs)
rs_exp = np.array(rs_exp)
```


```python
xs[best_idx], rs[best_idx]
```




    (1.4150118827819824, 1.9677836634587074)




```python
sns.lineplot(x=plot_x, y=plot_y)
sns.scatterplot(x=xs, y=rs, hue=range(len(xs)))
sns.scatterplot(x=[xs[best_idx]], y=[rs[best_idx]], marker='x', s=300)
```

    /home/vadim0x60/.pyenv/versions/3.9.5/lib/python3.9/site-packages/seaborn/relational.py:651: UserWarning: You passed a edgecolor/edgecolors ('w') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
      points = ax.scatter(*args, **kws)





    <AxesSubplot:>




    
![png](output_14_2.png)
    


## Neural approach

Learn a mapping from the normal distribution to high reward distribution and sample from it


```python
import torch
from torch import nn

actor = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

actor_opt = torch.optim.Adam(actor.parameters())

critic = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
critic_opt = torch.optim.Adam(critic.parameters())
```


```python
def neural_step():
    z = torch.normal(mean=torch.tensor([0.0]),
                     std=torch.tensor([1.0]))
    x = actor(z).detach().numpy()[0]
    r = reward_f(x)
    
    xs.append(x)
    rs.append(r)
    rs_exp.append(np.exp(r))
    
    critic_loss = (critic(torch.tensor(xs).float().reshape(-1,1)) 
                        - torch.tensor(rs).float().reshape(-1,1)) ** 2
    critic_loss = critic_loss.sum()
    
    critic_loss.backward(retain_graph=True)
    critic_opt.step()
    critic_opt.zero_grad()
    
    z = torch.normal(mean=torch.zeros(10),
                     std=torch.ones(10))
    z = z.reshape(-1, 1)
    actor_loss = - critic(actor(z)).sum()
    actor_loss.backward()
    actor_opt.step()
    actor_opt.zero_grad()
```


```python
xs = []
rs = []
rs_exp = []

for _ in range(10000):
    neural_step()
    
best_idx = np.argmax(rs)
```


```python
xs = np.array(xs)
rs = np.array(rs)
critic_rs = critic(torch.tensor(plot_x.reshape(-1, 1).astype('float32'))).detach().numpy().reshape(-1)
```


```python
sns.lineplot(x=plot_x, y=plot_y)
sns.scatterplot(x=xs, y=rs, hue=range(len(xs)))
sns.scatterplot(x=[xs[best_idx]], y=[rs[best_idx]], marker='x', s=300)
sns.lineplot(x=plot_x, y=critic_rs)
plt.ylim(-3, 3)
```

    /home/vadim0x60/.pyenv/versions/3.9.5/lib/python3.9/site-packages/seaborn/relational.py:651: UserWarning: You passed a edgecolor/edgecolors ('w') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
      points = ax.scatter(*args, **kws)





    (-3.0, 3.0)




    
![png](output_20_2.png)
    


## Neurogenetic approach


```python
xs = []
rs = []
rs_exp = []

neural_step()
for _ in range(10000):
    neural_step()
    genetic_step()
    
best_idx = np.argmax(rs)
```


```python
sns.lineplot(x=plot_x, y=plot_y)
sns.scatterplot(x=xs, y=rs, hue=range(len(xs)))
sns.scatterplot(x=[xs[best_idx]], y=[rs[best_idx]], marker='x', s=300)
sns.lineplot(x=plot_x, y=critic_rs)
plt.ylim(-3, 3)
```

    /home/vadim0x60/.pyenv/versions/3.9.5/lib/python3.9/site-packages/seaborn/relational.py:651: UserWarning: You passed a edgecolor/edgecolors ('w') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
      points = ax.scatter(*args, **kws)





    (-3.0, 3.0)




    
![png](output_23_2.png)
    



```python

```
