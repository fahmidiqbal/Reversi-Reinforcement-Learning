Required Libaries:
    pygame
    numpy

Files:
    -reversi_server.py  Server of the game
    -reversi.py         Base class of the game
    -greedy_player.py   A local greedy AI
    -rl_player.py       Reinforcement Learning DQN

How to run:
    1. Open 3 terminals
    2. Run reversi_server.py in terminal 1
    3. Run player 1 in terminal 2 (rl_player.py or greedy_player.py)
    4. Run player 2 in terminal 3 (greedy_player.py or rl_player.py)
    5. Watch the AIs fight.

Note:
Ran in Python 11
model30.pt must remain in the same directory as rl_player.py
Win-Example.png shows a screenshot where White player was rl_player and Black is greedy_player