from dataclasses import dataclass, astuple

@dataclass(slots=True)
class Step:
    state: tuple
    action: int
    reward: int = 0

    def set_state(self, state):
        self.state = state

    def set_action(self, action):
        self.action = action

    def set_reward(self, reward):
        self.reward = reward

    def __iter__(self):
        return iter(astuple(self))


class Episode:
    def __init__(self):
        self.episode = []
    
    def add(self, step: Step):
        self.episode.append(step)

    def __iter__(self):
        return iter(self.episode)
    
    def __next__(self):
        return next(iter(self))
        

if __name__ == "__main__":
    e = Episode()
    e.add(Step((1, 0), 1, -1))
    e.add(Step((0, 0), 1, 1))
    for s in e:
        print(s)
