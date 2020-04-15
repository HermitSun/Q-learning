## RL小作业

### 1. Q-learning的过程

#### 公式理解

Q-learning 是一个基于**值**的强化学习算法，目的是最大化 Q 函数的值；使用Q表来寻找每个状态的“最优”操作。

参考公式，每个循环内，会进行以下操作：
$$
NewQ(s,a) = Q(s,a) + \alpha[r + \gamma max Q(s',a') - Q(s,a)]
$$

其中，$s$为Q表，$a$为动作。公式右边：

- $Q(s,a)$是当前Q表项的值；
- $\alpha$为学习率（保留新值的比例）；
- $r$为奖励值；
- $\gamma$为衰减率（对未来的“预测”总会有衰减）；
- $max_{a'}Q(s',a')$为下一步Q表项中的最大值；
- 计算完成后，更新左边的值。

同时，由于采用$\epsilon$-greedy的策略，每次会有一定概率进行随机选择，而不是选取Q表中的最优值。通常情况下，初始值为1（全随机），并且随着智能体的学习过程而下降，因为智能体应该对这个环境更加“熟悉”，更加“有把握”。

#### 过程描述

以下是简单的用伪码描述的过程：

```python
def rl():
    q_table = build_q_table()				# 建立Q表
    for episode in range(MAX_EPISODES):		# 循环
        while not terminated:
            choose_best_action()			# 选择最优的操作
            get_env_feedback()				# 获得环境反馈
            calc_and_update_q_table()		# 使用公式进行计算，并且更新Q表
            move_to_next_state()			# 转移到下一个状态
        decrease_epsilon()					# 降低epsilon值
```

### 2. 代码解释

#### 参数

```python
from enum import Enum

ENV_SIZE = 6  							# 场景大小
REACHABLE = 1  							# 距离多远即可获得奖励
MAX_DISTANCE = ENV_SIZE - REACHABLE		# 智能体能达到的最远距离

EPSILON = 1								# greedy策略选择需要的参数epsilon
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13						# 最大循环次数

FRESH_TIME = 0.3						# 控制台输出的刷新时间


class Actions(Enum):					# 可选择的行为
    LEFT = 'left',
    RIGHT = 'right'


TERMINATED = -1							# 终止状态
```

其中，枚举类Actions纯属个人喜好。

初始Q表为空，通过不断迭代来对Q表进行更新：

```python
def build_q_table(size: int, actions: type(Actions)) -> pd.DataFrame:
    action_names = actions.__members__
    table = pd.DataFrame(
        np.zeros((size, len(action_names))),
        columns=action_names.keys(),
    )
    return table
```

#### 策略选择

主要采用了三种策略：线性下降，指数下降，阶梯下降。

参数是随便选的……试了几组数据，选了一组效果比较好的数据。

##### 线性下降

$$
f(x) = \epsilon - \frac{x}{10}
$$

##### 指数下降

$$
f(x) = \epsilon - 0.000045 * e^{x}
$$

##### 阶梯下降

$$
f(x) = \epsilon - \frac{\lceil x \rceil}{5}
$$

##### 对应代码

```python
# epsilon.py
from numpy import exp, ceil


def linear(episode: int, epsilon: float) -> float:
    return epsilon - 0.1 * episode


def exponent(episode: int, epsilon: float) -> float:
    return epsilon - 0.000045 * exp(episode)


def slider(episode: int, epsilon: float) -> float:
    slide = ceil(episode / 2) * 2
    return epsilon - 0.1 * slide
```

#### Q值更新

参考之前的公式进行实现：

```python
def rl(strategy: Callable[[int, float], float]) -> pd.DataFrame:
    q_table = build_q_table(ENV_SIZE, Actions)									# 建立Q表
    epsilon = EPSILON
    for episode in range(MAX_EPISODES):											# 循环
        step_counter = 0
        state = 0
        has_terminated = False
        update_env(state, episode, step_counter)
        while not has_terminated:
            action = choose_best_action(state, Actions, q_table, epsilon)		# 选择最优的操作
            new_state, reward = get_env_feedback(state, action)					# 获得环境反馈

            q_predict = q_table.loc[state, action]
            if new_state != TERMINATED:
                q_target = reward + GAMMA * q_table.iloc[new_state, :].max()
            else:
                q_target = reward
                has_terminated = True
            q_table.loc[state, action] += ALPHA * (q_target - q_predict)		# 计算并更新Q表

            state = new_state
            update_env(state, episode, step_counter + 1)						# 转移到下一个状态
            step_counter += 1

        epsilon = strategy(episode, epsilon)									# 根据不同的策略更新epsilon

    return q_table

q_table_final = rl(linear)
# q_table_final = rl(exponent)
# q_table_final = rl(slider)
```

### 3. 实验结果

$\epsilon$线性下降的Q表如下：

```python
           LEFT     RIGHT
0  8.393074e-07  0.001653
1  3.029900e-06  0.007750
2  7.294353e-05  0.032695
3  5.684493e-04  0.119401
4  2.013518e-03  0.348161
5  3.653100e-03  0.745813
```

$\epsilon$指数下降的Q表如下：

```python
       LEFT     RIGHT
0  0.028163  0.047689
1  0.024547  0.079797
2  0.042114  0.132250
3  0.062793  0.227192
4  0.072144  0.429606
5  0.106823  0.745813
```

$\epsilon$阶梯下降的Q表如下：

```python
       LEFT     RIGHT
0  0.000000  0.001665
1  0.000000  0.008798
2  0.000000  0.037248
3  0.000317  0.128329
4  0.004076  0.355049
5  0.009237  0.745813
```

### 4. 参考资料

1. [什么是 Q Leaning](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-A-q-learning/)
2. [通过 Q-learning 深入理解强化学习](https://www.jiqizhixin.com/articles/2018-04-17-3)