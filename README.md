# Create a new gym environment

ref: <https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952>

## 创建过程

### 文件结构

```text
gym-basic/           # 主文件夹
  README.md      # 说明文档
  setup.py              # 安装(指定包名和依赖)
  gym_basic/            # 次文件夹
    __init__.py           # 
    envs/
      __init__.py
      basic_env.py
      basic_env_2.py
```

### `setup.py`

安装pip包时使用，指定包名和依赖，例如：

```python
from setuptools import setup 
setup(name=’gym_basic’, version=’0.0.1', install_requires=[‘gym’] )
```

### `gym_basic/__init__.py`

```python
from gym.envs.registration import register 
register(id='basic-v0',entry_point='gym_basic.envs:BasicEnv',) 
register(id='basic-v2',entry_point='gym_basic.envs:BasicEnv2',)
```

### `gym_basic/envs/__init__.py`

```python
from gym_basic.envs.basic_env import BasicEnv
from gym_basic.envs.basic_env_2 import BasicEnv2
```

### `gym_basic/envs/basic_env.py`

```python
import gym

class BasicEnv(gym.Env):
  def __init__(self):
      self.action_space = gym.spaces.Discrete(5)
      self.observation_space = gym.spaces.Discrete(2)

  def step(self, action):
      state = 1
  
      if action == 2:
          reward = 1
      else:
          reward = -1
          
      done = True
      info = {}
      return state, reward, done, info

  def reset(self):
      state = 0
      return state
  ```

## 使用

1. 安装包`pip install -e gym-basic`
2. 在其他程序中调用，例如：

```python
import gym
import gym_basic

env = gym.make('basic-v0')

print('init env ok')
```

## 注意

安装后直接调用会出现环境未注册问题，需要将`gym_basic`文件夹放置在调用文件的子目录中，或者将安装位置加入python路径!

具体见：<https://github.com/openai/gym/issues/626#issuecomment-310642853>
