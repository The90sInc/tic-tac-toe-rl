"""
Q-learning agent for tic-tac-toe.
 
Core idea
---------
Q(state, action) estimates: "how good is it to play `action` from `state`,
assuming both players continue playing optimally afterward?"
 
The tricky part for a two-player, zero-sum, turn-alternating game like
tic-tac-toe: standard single-agent Q-learning assumes the SAME agent
controls what happens at the next state. Here, after our move, it's the
OPPONENT's turn — and their best response is chosen to help THEM, which
is bad for us. Since the game is zero-sum, we adapt the Bellman update
by negating the value of the opponent's best response before
bootstrapping from it. This is what allows a single shared Q-table to
correctly learn to play both sides of the game through self-play.
"""

import pickle
import random
from typing import Optional

from app.game.symmetry import canonical_form

class QLearningAgent:
    def __init__(self, alpha: float = 0.3, gamma: float = 0.95, epsilon: float = 0.2):
        # alpha:   learning rate -- how much each new experience shifts the old estimate (0 < alpha < 1)
        # gamma:   discount factor -- how much a future reward is worth vs an immediate one (0 < gamma < 1)
        # epsilon: exploration rate -- probability of a random move instead of the best-known one (0 < epsilon < 1)

        self.q_table: dict[tuple, float] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state: tuple, action: int) -> float:
        """
        Return the Q-value for a given state-action pair (state, action), defaulting to 0.0 if unseen.
        Internally canonicalizes `state` so that all 8 symmetric variants of a
        position share the same learned value.
        """
        canon_board, perm = canonical_form(state)
        return self.q_table.get((canon_board, perm[action]), 0.0)

    def choose_action(self, state: tuple, valid_actions: list[int], training: bool = True) -> int:
        """
        Choose an action for the given state, either by exploring (random) or exploiting (best-known).
        If `training` is False, always exploit (no exploration).

        epsilon-greedy policy:
        - training= True: explore randomly with probability epsilon, else exploit
        - training=False: always exploit (choose the best-known action) -- used for the deployed agent
        """
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
 
        canon_board, perm = canonical_form(state)
        q_values = [self.q_table.get((canon_board, perm[a]), 0.0) for a in valid_actions]
        best_q = max(q_values)
        # Tie-break randomly rather than always taking the first-found max --
        # otherwise the agent develops a predictable bias toward low-numbered cells.
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == best_q]
        return random.choice(best_actions)

    def update(self, state: tuple, action: int, reward: float, next_state: tuple, next_valid_actions: list[int], done: bool) -> None:
        """
        TD(0) Q-learning update, adapted for an adversarial 2-player game:
 
            Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
        if the game just ended
            target = reward
        else
            target = reward - gamma * max_a' Q(next_state, a')
 
        The minus sign is the key adaptation: next_state is the OPPONENT's
        turn, so their best move is bad for us -- we subtract it rather
        than add it. 
        Since tic tac toe is a Zero sum game, and our opponent's gain is our loss, 
        this is the correct way to update our Q-values.
        """
        cannon_state, perm = canonical_form(state)
        cannon_action = perm[action]
        old_q = self.q_table.get((cannon_state, cannon_action), 0.0)

        if done or not next_valid_actions:
            target = reward
        else:
            opponents_best_response = max(self.get_q(next_state, action) for action in next_valid_actions)
            target = reward - self.gamma * opponents_best_response

        self.q_table[(cannon_state, cannon_action)] = old_q + self.alpha * (target - old_q)

    def save(self, path: str) -> None:
        """Save the Q-table to a file using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path: str) -> None:
        """Load the Q-table from a file using pickle."""
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)