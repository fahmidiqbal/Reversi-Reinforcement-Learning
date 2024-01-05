import numpy as np
import math
import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import random
from reversi import reversi


position = [[150, -20,  10,  5,  5, 10, -20, 150],
                       [-20, -50,  -2, -2, -2, -2, -50, -20],
                       [10,   -2,  -1, -1, -1, -1,  -2,  10],
                       [5,    -2,  -1,  0,  0, -1,  -2,   5],
                       [5,    -2,  -1,  0,  0, -1,  -2,   5],
                       [10,   -2,  -1, -1, -1, -1,  -2,  10],
                       [-20,  -50, -2, -2, -2, -2, -50, -20],
                       [150,  -20, 10,  5,  5, 10, -20, 150]]
# Define a simple deep Q-network
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

n_actions = 64
n_observations = 64

q_net = DQN(n_actions,n_observations)
q_net.load_state_dict(torch.load("model30.pt"))
q_net.eval()

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
    
def select_action(state):
    global steps_done
    sample = random.random()
    if sample > 0.1:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action_index = q_net(state).max(1)[1].item()
            x, y = divmod(q_net(state).max(1)[1].view(1, 1).item(), 8)
            return torch.tensor([[action_index]], device=device, dtype=torch.long), position[x][y]
    else:
        action_index = np.random.randint(64)
        x, y = divmod(action_index, 8)
        return torch.tensor([[action_index]], device=device, dtype=torch.long), position[x][y]

MAX_DEPTH = 1

# return is in the form (x, y, heuristic, [next turns])
def next_turns(game, turn, depth):
    # base case
    if(depth == MAX_DEPTH):
        return []
    
    p_turns = []

    # player turn
    for i in range(8):
        for j in range(8):
            temp_game = reversi()
            temp_game.board = np.copy(game.board)
            cur_p = temp_game.step(i, j, turn, False)

            if cur_p > 0:
                temp_game.step(i, j, turn, True)
                o_turns = next_turns(temp_game, (-1)*turn, depth+1)                 # gets the next level of opponent turns
                num = turn

                # our turn
                if(depth%2 == 0):
                    if(num == -1):
                        num = 1
                        #h = (-1)*h
                # opponents turn 
                if(depth%2 == 1):
                    if(num == 1):
                        num = -1
                        #h = (-1)*h

                p_turns.append((i,j))

    return p_turns

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:
        # Receive play request from the server
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)
        game.board = board

        if turn == 0:
            game_socket.close()
            return

        # Convert board to a flattened numpy array for input to the model
        state = np.array(board).flatten()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        #Debug info
        print(turn)
        print(board)
        x,y = -1,-1
        flipped = -1

        # find a legal move to make
        while flipped <= 0:
            num, r = select_action(state)
            x,y = divmod(num.item(),8)
            flipped = game.step(x = x, y = y, piece = turn, commit= False)
            turns = next_turns(game, turn, 0)
            if len(turns) == 0:
                x,y = -1,-1
                break
            if flipped == -5 or flipped == -4:
                x,y = -1,-1
                break

        game_socket.send(pickle.dumps([x, y]))
    
    
if __name__ == '__main__':
    main()