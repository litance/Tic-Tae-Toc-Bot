# Tic-Tae-Toc-Bot
Tic-Tae-Toc game model based on DQN neural network deep learning

### 所需要第三方库 Third-party libraries needed to run this program
- torch
- numpy

## 1.首先, 安装第三方库 Install Third-party libraries
```
pip install requirements.txt
```

## 2.下载代码 Download code
```
git clone https://github.com/litance/Tic-Tae-Toc-Bot.git
```

## 2.配置好代码 Configure the code(train.py)
Episodes, batch_size, gamma, lr, epsilon_decay, min_epsilon(line:55-56)
```
def __init__(self, episodes=200000, batch_size=1024, gamma=0.995, lr=0.0005, epsilon_decay=0.9999,
             min_epsilon=0.01):
```
