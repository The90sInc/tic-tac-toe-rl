"""
Self-play training loop for the Q-learning tic-tac-toe agent.

How self-play works here
-------------------------
A single QLearningAgent plays BOTH sides of every training game. This
works cleanly because a given board position uniquely determines whose
turn it is (X and O counts can only differ by at most one) -- so the
same Q-table entry is never ambiguous about which player it describes.

Each move, in order, triggers one online TD update (see agent.py for
the update math). Exploration (epsilon) is decayed smoothly across
training: lots of randomness early to discover the state space, almost
none late so the agent settles into its best-known strategy.
"""
import sys
from pathlib import Path

import random

# Add the 'backend' folder (2 levels up from train.py) to Python's import path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.game.engine import (
    new_board, valid_moves, apply_move, winner, is_draw, other_player,
)
from app.rl.agent import QLearningAgent


def run_self_play_episode(agent: QLearningAgent) -> None:
    """Play one full game with the agent controlling both sides, learning as it goes."""
    board = new_board()
    current_player = "X"

    while True:
        state = board
        action = agent.choose_action(state, valid_moves(state), training=True)
        board = apply_move(board, action, current_player)

        game_winner = winner(board)
        if game_winner == current_player:
            reward, done = 1.0, True
        elif is_draw(board):
            reward, done = 0.0, True
        else:
            reward, done = 0.0, False

        next_state = board
        next_valid = valid_moves(next_state) if not done else []
        agent.update(state, action, reward, next_state, next_valid, done)

        if done:
            return
        current_player = other_player(current_player)


def evaluate_against_random(agent: QLearningAgent, num_games: int = 200) -> dict:
    """
    Play `num_games` with exploration OFF against a purely random opponent,
    alternating who moves first. Returns win/loss/draw counts from the
    agent's perspective. This is a quick sanity check, not proof of
    optimal play -- beating random opponents is a low bar.
    """
    results = {"agent_win": 0, "agent_loss": 0, "draw": 0}

    for i in range(num_games):
        agent_mark = "X" if i % 2 == 0 else "O"
        board = new_board()
        current_player = "X"

        while True:
            valid = valid_moves(board)
            if current_player == agent_mark:
                action = agent.choose_action(board, valid, training=False)
            else:
                action = random.choice(valid)
            board = apply_move(board, action, current_player)

            game_winner = winner(board)
            if game_winner == agent_mark:
                results["agent_win"] += 1
                break
            elif game_winner is not None:
                results["agent_loss"] += 1
                break
            elif is_draw(board):
                results["draw"] += 1
                break
            current_player = other_player(current_player)

    return results


def train(
    num_episodes: int = 20_000,
    eval_every: int = 2_000,
    eval_games: int = 200,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    alpha: float = 0.3,
    gamma: float = 0.95,
) -> QLearningAgent:
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon_start)

    for episode in range(1, num_episodes + 1):
        # Smooth exponential decay from epsilon_start down to epsilon_end.
        progress = episode / num_episodes
        agent.epsilon = epsilon_start * (epsilon_end / epsilon_start) ** progress

        run_self_play_episode(agent)

        if episode % eval_every == 0:
            results = evaluate_against_random(agent, num_games=eval_games)
            print(
                f"Episode {episode:>6} | epsilon={agent.epsilon:.3f} | "
                f"vs random -> win={results['agent_win']:>3} "
                f"loss={results['agent_loss']:>3} draw={results['draw']:>3}"
            )

    return agent

if __name__ == "__main__":
    trained_agent = train()
    trained_agent.save("backend/app/models/q_table.pkl")
    print(f"\nSaved Q-table with {len(trained_agent.q_table)} entries to backend/app/models/q_table.pkl")