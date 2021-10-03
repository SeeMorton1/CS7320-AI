from typing import Deque
from collections import deque
import maze_helper as mh


class Node:
    def __init__(self, pos, parent, action, cost):
        self.pos = tuple(pos)  # the state; positions are (row,col)
        self.parent = parent  # reference to parent node. None means root node.
        self.action = action  # action used in the transition function (root node has None)
        self.cost = cost  # for uniform cost this is the depth. It is also g(n) for A* search

    def __str__(self):
        return f"Node - pos = {self.pos}; action = {self.action}; cost = {self.cost}"

    def get_path_from_root(self):
        """returns nodes on the path from the root to the current node."""
        node = self
        path = [node]

        while not node.parent is None:
            node = node.parent
            path.append(node)

        path.reverse()

        return (path)


f = open("medium_maze.txt", "r")
maze_str = f.read()
print(maze_str)

maze = mh.parse_maze(maze_str)

print("Position(0,0):", maze[0, 0])
print("Position(8,1):", mh.look(maze, (8, 1)))

mh.show_maze(maze)

print("Start location:", mh.find_pos(maze, what="S"))
print("Goal location:", mh.find_pos(maze, what="G"))


class Node:
    def __init__(self, pos, parent, action, cost):
        self.pos = tuple(pos)
        self.parent = parent
        self.action = action
        self.cost = cost

    def __str__(self):
        return f"Node - pos = {self.pos}; action = {self.action}; cost = {self.cost}"

    def get_path_from_root(self):
        node = self
        path = [node]
        while not node.parent is None:
            node = node.parent
            path.append(node)
        path.reverse()
        return (path)


'''
Initial State: Starting Location(S) of the agent

Actions: Turn(N,E,S,W)

Transition Model: If at S(0,0) and Agent moves east then Agent is at S(1,0)

Goal state: Goal Loaction and Agent location are the same

Path cost: Number of steps taken to get to the goal


'''

'''
N State space size: The state space would be |x1|*|x2|*....*|xn| where xn is the number of free squares the agent has to move 
That is the actions * the number of free squares, so the options of states it has is action*free squares 

D Depth of optimal solution: The depth of the optimal solution would be whatever the final solution costs, so the goal state

M Maximum depth of tree: The maximum depth of the tree is the list of actions that an agent might take to reach the goal state

B Maximum branching factor: The maximum branching factor at any position would be 4, meaning that the agent could move N,E,S, or W

'''


def find_actions(curr_pos):
    s = []

    if maze[curr_pos[0] - 1][curr_pos[1]] != "X":  # Check N
        s.append((curr_pos[0] - 1, curr_pos[1]))
    if maze[curr_pos[0] + 1][curr_pos[1]] != "X":  # Check S
        s.append((curr_pos[0] + 1, curr_pos[1]))
    if maze[curr_pos[0]][curr_pos[1] - 1] != "X":  # Check W
        s.append((curr_pos[0], curr_pos[1] - 1))
    if maze[curr_pos[0]][curr_pos[1] + 1] != "X":  # Check E
        s.append((curr_pos[0], curr_pos[1] + 1))
    return s


def determine_action(curr_pos, next_pos):
    if next_pos[1] - curr_pos[1] == -1:
        return "W"
    if next_pos[1] - curr_pos[1] == 1:
        return "E"
    if next_pos[0] - curr_pos[0] == -1:
        return "N"
    if next_pos[0] - curr_pos[0] == 1:
        return "S"


def breadth_first_search():
    frontier = deque([Node(mh.find_pos(maze, what="S"), parent=None, action=None, cost=0)])
    goalPos = mh.find_pos(maze, what="G")
    posTraveled = []
    reached = []

    while frontier:
        curr = frontier.pop()

        if curr.pos == goalPos:
            return curr
        s = find_actions(curr.pos)

        for i in s:

            n = Node(i, parent=curr, action=determine_action(curr.pos, i), cost=1)

            if n.pos == mh.find_pos(maze, what="G"):
                return n
            if i not in posTraveled and i not in reached:
                posTraveled.append(i)
                frontier.append(n)
            if i not in reached:
                reached.append(i)
    return None


def is_cycle(n):
    curr = n
    traveled = []
    while curr.parent is not None:
        if curr not in traveled:
            traveled.append(curr)
            curr = curr.parent
        else:
            return True
    return False


def depth_first_search():
    frontier = [Node(mh.find_pos(maze, what="S"), parent=None, action=None, cost=0)]
    start = mh.find_pos(maze, what="S")
    goal_pos = mh.find_pos(maze, what="G")
    pos_traveled = []
    while frontier:

        n = frontier.pop()

        if n.pos == goal_pos:
            return n
        elif not is_cycle(n):
            s = find_actions(n.pos)
            for i in s:
                if i not in pos_traveled:
                    new_node = Node(i, parent=n, action=determine_action(n.pos, i), cost=1)
                    frontier.append(new_node)
                    pos_traveled.append(i)

    return None


n = depth_first_search()
curr = n.parent
print("DEPTH FIRST SEARCH")
dfc_count = 0
while curr:
    dfc_count += 1
    print(curr.pos)
    curr = curr.parent
print()
print("BREADTH FIRST SEARCH")
bfs = breadth_first_search()
currbfs = bfs
bfsCount = 0
while currbfs:
    bfsCount += 1
    print(currbfs.pos)
    currbfs = currbfs.parent
print()

print("DFS Node: ")
print(dfc_count)
print(n)
print("BFS Node: ")
print(bfsCount)
print(bfs)

