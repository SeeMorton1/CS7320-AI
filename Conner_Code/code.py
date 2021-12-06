import matplotlib.pyplot as plt
import random
import numpy as np
import math

#TODO - Change actions to update based on the height of the row in the board
#TODO - Change action calculations for the remove action
#TODO - 
def empty_board(shape=(6, 7)):
    return np.full(shape=shape, fill_value=0)


def visualize(board):
    plt.axes()
    rectangle=plt.Rectangle((-0.5,len(board)*-1+0.5),len(board[0]),len(board),fc='blue')
    circles=[]
    for i,row in enumerate(board):
        for j,val in enumerate(row):
            color='white' if val==0 else 'red' if val==1 else 'yellow'
            circles.append(plt.Circle((j,i*-1),0.4,fc=color))

    plt.gca().add_patch(rectangle)
    for circle in circles:
        plt.gca().add_patch(circle)

    plt.axis('scaled')
    plt.show()

'''
Transition Model for "Mean" Connect-4 using results(s,a)
action could be a tuple like ('drop',player, column_number) or ('remove',player, column_to_remove, column_to_move_to)

The "lowest" position would be the largest row number in the board

The action 'drop' drops a piece in the column column_number to the lowest position that is currently unoccupied, for the current player. 
If there are other pieces in the column, the piece added will stack on top of them.

An opponent's piece is defined as a piece that is the current player's piece multiplied by -1.
The action 'remove' removes an opponent's piece from the column column_to_remove and moves it to column_to_move_to. The action must take the piece at largest row number in column_to_remove and 'drop' it in the column_to_move_to.
When 'remove' is used, all other pieces in the column_to_remove are moved down one row.

If the action would place a piece outside the board, raise a ValueError.
An example board could look like this:
board = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0,-1,-1, 1,-1, 0, 0]]

If the action was ('drop', 1, 3), the board would look like this:
board = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0,-1,-1, 1,-1, 0, 0]]
         
If the action was ('remove', -1, 3, 2), the board would look like this:
board = [   [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0,-1,-1, 1,-1, 0, 0]]

If the action was ('remove', 1, 1, 2), the board would look like this:
board = [   [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0,-1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0,-1, 1,-1, 0, 0]]
            
            
If the action was ('drop', -1, 1), the board would look like this:
board = [   [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0,-1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0,-1,-1, 1,-1, 0, 0]]

If the action was ('drop', 1, 0), the board would look like this:
board = [   [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0,-1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [1,-1,-1, 1,-1, 0, 0]]
'''
def drop_piece(board, player, column):
    if column < 0 or column >= len(board[0]):
        raise ValueError("Column out of bounds")
    if board[0][column] != 0:
        raise ValueError("Column is full")
    for i in range(len(board)-1, -1, -1):
        if board[i][column] == 0:
            board[i][column] = player
            return board
    raise ValueError("Column is full")

def recursive_move_down(board, column, row):
    if row == 0:
        return board
    else:
        board[row][column] = board[row-1][column]
        return recursive_move_down(board, column, row-1)

def remove_piece(board, player, column_to_remove, column_to_move_to):
    if column_to_remove < 0 or column_to_remove >= len(board[0]):
        raise ValueError("Column out of bounds")
    if column_to_move_to < 0 or column_to_move_to >= len(board[0]):
        raise ValueError("Column out of bounds")
    if board[-1][column_to_remove] == 0:
        raise ValueError("Column is empty")
    if board[-1][column_to_remove] == player:
        raise ValueError("Can't remove own piece")
    
    if board[board.shape[0]-1][column_to_remove] == -player:
        recursive_move_down(board, column_to_remove, board.shape[0]-1)
        return drop_piece(board, -player, column_to_move_to)
    raise ValueError("No opponent piece found")

def transition_model(board, action):
    if action[0] == 'drop':
        return drop_piece(board, action[1], action[2])
    elif action[0] == 'remove':
        return remove_piece(board, action[1], action[2], action[3])
    else:
        raise ValueError("Unknown action")

'''
Defines the utility for the current board state.
The utility is defined as the number of pieces in a row for the agent. A row can be either vertical, horizontal, or diagonal. If the agent was player 1 and the board looked like this:
board = [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0,-1,-1, 1,-1, 0, 0]]
The utility would be the maximum chance of winning which is four pieces in a row. This means that for the board above, the utility would be 3.
'''

def utility(board, player):
    utility = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == player:
                utility += 1
            elif board[i][j] == -player:
                utility -= 1
    return utility

'''the function terminal_test tests for the final state in a board. If in a board a player(being either 1's or -1's) has four consecutive pieces in a row, column, or diagonal, then that player has won the game'''
def terminal_test(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != 0:
                if i < len(board)-3:
                    if board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j]:
                        return True
                if j < len(board[0])-3:
                    if board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3]:
                        return True
                if i < len(board)-3 and j < len(board[0])-3:
                    if board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3]:
                        return True
                if i < len(board)-3 and j > 3:
                    if board[i][j] == board[i+1][j-1] == board[i+2][j-2] == board[i+3][j-3]:
                        return True
    return False

'''Return a list of available actions that the AI could make at the time using actions(s) that returns a list of all that includes all of the columns that are not already full with pieces, and also includes all of the columns that the user could swap pieces with
'''
def actions(board,player):
    actions = []
    for i in range(len(board[0])):
        if board[0][i] == 0:
            actions.append(('drop', player, i))
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == -player:
                for k in range(len(board[0])):
                    if board[i][k] == 0:
                        actions.append(('remove', player, j, k))
    return actions

'''simple_random_agent that randomly selects an action from the actions list'''
def simple_random_agent(board, player):
    actions_list = actions(board, player)
    return random.choice(actions_list)


#Using the simple_random_agent to determine each action. Simulate 1000 games and take statistics on which side wins more often, 1 or -1.
def play_game():
    board = [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]]
    player = 1
    while not terminal_test(board):
        if player == 1:
            action = simple_random_agent(board, player)
        else:
            action = simple_random_agent(board, player)
        board = transition_model(board, action)
        player = -player
    visualize(board)
    return utility(board, player)


'''Now simulate a 1000 play games'''
def simulate_games(n):
    wins = 0
    for i in range(n):
        if play_game() > 0:
            wins += 1
    return wins/n
        

'''
The recursive_minimax function is the main function that is called to get the best action for the agent.
The function will call the recursive_minimax_value function to get the best action for the agent.
The function will return the best action for the agent.
'''
def recursive_minimax_value(new_board,player):
    if terminal_test(new_board):
        return utility(new_board, player)
    v = -math.inf
    for action in actions(new_board, player):
        v = max(v, recursive_minimax_value(transition_model(new_board, action), -player))
    return v


def recursive_minimax(board, player):
    if terminal_test(board):
        return None
    actions_list = actions(board, player)
    best_action = None
    best_value = -math.inf
    for action in actions_list:
        new_board = transition_model(board, action)
        value = recursive_minimax_value(new_board, player)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action

def max_value_ab(state,player, alpha, beta):
    if terminal_test(state):
        return utility(state, player)
    v = -math.inf
    for action in actions(state, player):
        v = max(v, min_value_ab(transition_model(state, action), -player, alpha, beta))
        if v >= beta:
            return v
        alpha = max(alpha, v)
def min_value_ab(state,player, alpha, beta):
    if terminal_test(state):
        return utility(state, player)
    v = math.inf
    for action in actions(state, player):
        v = min(v, max_value_ab(transition_model(state, action), -player, alpha, beta))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v

def alpha_beta_search(board,player):
    if terminal_test(board):
        return None
    actions_list = actions(board, player)
    best_action = None
    best_value = -math.inf
    for action in actions_list:
        new_board = transition_model(board, action)
        value = min_value_ab(new_board, player, -math.inf, math.inf)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action



board = empty_board(shape=(6, 7))
visualize(board)
action = ('drop', 1, 3)
board = transition_model(board, action)
visualize(board)
action = ('drop', -1, 3)
board = transition_model(board, action)
visualize(board)
action = ('remove',-1, 3, 2)
visualize(transition_model(board, action))
action = alpha_beta_search(board, 1)
visualize(transition_model(board, action))



