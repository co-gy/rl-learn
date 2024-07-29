# RL Learn

I'm implementing algorithms from this book (Mathematical Foundation of Reinforcement Learning). I used the [GridWorld environment code](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) provided by the author and made a few changes.

![result](https://github.com/co-gy/rl-learn/blob/main/assets/combine.png)
## Menu

- Chapter 4
  - [x] Value Iteration
  - [x] Policy Iteration
  - [x] Truncated Policy Iteration
- Chapter 5
  - [x] MC Basic
  - [ ] MC Exploring Starts
  - [ ] MC $\epsilon$-greedy
- Chapter 7
  - [ ] sarsa
  - [ ] n-step sarsa
  - [ ] O-learning
- Chapter 8
  - [ ] DQN

## Run
```zsh
# activate your venv
pip install -r requirements.txt
git clone git@github.com:co-gy/rl-learn.git
cd rl-learn
python algorithm/value_iteration.py
```
