# RL Learn

I'm implementing algorithms from this book (Mathematical Foundation of Reinforcement Learning). I used the [GridWorld environment code](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) provided by the author and made a few changes.

![result](https://github.com/co-gy/rl-learn/blob/main/assets/combine.png?raw=true)
## Menu

- Chapter 4
  - [x] Value Iteration
  - [x] Policy Iteration
  - [x] Truncated Policy Iteration
- Chapter 5
  - [x] MC Basic
  - [x] MC Exploring Starts
  - [x] MC $\epsilon$-greedy
- Chapter 6
  - [x] RM Algorithm example
- Chapter 7
  - [x] Sarsa
  - [x] Q-learning (on-policy version)
  - [x] Q-learning (off-policy version)
- Chapter 8
  - [ ] DQN

## Run

It is easy to run the code by following the steps.

1. activate your venv
```zsh
conda activate base
# or
python -m venv venv
. venv/bin/activate
```

2. install packages
```zsh
pip install -r requirements.txt
```

3. clone this repo
```zsh
git clone git@github.com:co-gy/rl-learn.git
```
4. run

make sure your running directory is rl-learn
```zsh
cd rl-learn
python algorithm/value_iteration.py
```