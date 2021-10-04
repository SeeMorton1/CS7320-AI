from typing import Deque
from collections import deque
import maze_helper as mh
import numpy as np
import pandas as pd
import tabulate as tb

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


f = open("open_maze.txt", "r")
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

    def __gt__(self, other):
        if self.pos > other.pos:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.pos < other.pos:
            return True
        else:
            return False

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


bfs_nodes_expanded = 0
bfs_max_frontier = 0


def breadth_first_search():
    s = Node(mh.find_pos(maze, what="S"), parent=None, action=None, cost=0)
    goal_pos = mh.find_pos(maze, what="G")
    if s.pos == goal_pos:
        return s
    frontier = deque([s])
    global bfs_nodes_expanded
    global bfs_max_frontier
    pos_traveled = []  # Keep track so we don't go backwards
    reached = []  # Keep track so we don't explore any nodes that we have looked at but not taen

    while frontier:
        curr = frontier.popleft()
        if curr.pos == goal_pos:
            return curr
        s = find_actions(curr.pos)

        for i in s:

            n = Node(i, parent=curr, action=determine_action(curr.pos, i), cost=curr.cost + 1)

            if n.pos == mh.find_pos(maze, what="G"):
                bfs_nodes_expanded = len(set(reached))
                return n
            if i not in pos_traveled and i not in reached:
                pos_traveled.append(i)
                frontier.append(n)
                if len(frontier) > bfs_max_frontier:
                    bfs_max_frontier = len(frontier)
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


dfs_nodes_expanded = 0
dfs_max_frontier = 0


def depth_first_search():
    frontier = [Node(mh.find_pos(maze, what="S"), parent=None, action=None, cost=0)]
    start = mh.find_pos(maze, what="S")
    goal_pos = mh.find_pos(maze, what="G")
    global dfs_nodes_expanded
    global dfs_max_frontier
    pos_traveled = []
    while frontier:

        n = frontier.pop()

        if n.pos == goal_pos:
            dfs_nodes_expanded = len(pos_traveled)
            return n
        elif not is_cycle(n):
            s = find_actions(n.pos)
            for i in s:
                if i not in pos_traveled:
                    new_node = Node(i, parent=n, action=determine_action(n.pos, i), cost=n.cost + 1)
                    frontier.append(new_node)
                    if len(frontier) > dfs_max_frontier:
                        dfs_max_frontier = len(frontier)
                    pos_traveled.append(i)

    return None


def euclidean_distance(curr_pos, goal_pos):
    c = np.array(curr_pos)
    g = np.array(goal_pos)
    return np.linalg.norm(c - g)


def manhattan_distance(curr_pos, goal_pos):
    return sum(abs(val1 - val2) for val1, val2 in zip(curr_pos, goal_pos))


def sort_bfs(frontier):
    goal_pos = mh.find_pos(maze, what="G")
    return sorted(frontier, key=lambda x: manhattan_distance(x.pos, goal_pos))


def sort_astar(frontier):
    goal_pos = mh.find_pos(maze, what="G")
    return sorted(frontier, key=lambda x: (manhattan_distance(x.pos, goal_pos) + 1), reverse=False)


def path_cost(node):
    n = 0
    while node:
        n += node.cost
        node = node.parent
    return n


bestf_nodes_expanded = 0
bestf_frontier = 0


def best_first_search():
    s = Node(mh.find_pos(maze, what="S"), parent=None, action=None, cost=0)
    global bestf_nodes_expanded
    global bestf_frontier
    goal_pos = mh.find_pos(maze, what="G")
    if s.pos == goal_pos:
        return s
    reached = {}
    frontier = [s]
    while frontier:
        curr_node = frontier.pop()
        if curr_node.pos == goal_pos:
            bestf_nodes_expanded = len(reached)
            return curr_node
        act = find_actions(curr_node.pos)
        for i in act:
            if i not in reached or manhattan_distance(i, goal_pos) < manhattan_distance(reached[i].pos, goal_pos):
                new_node = Node(i, parent=curr_node, action=determine_action(curr_node.pos, i), cost=curr_node.cost + 1)
                reached[i] = new_node
                frontier.append(new_node)
                if len(frontier) > bestf_frontier:
                    bestf_frontier = len(frontier)
                frontier = sort_bfs(frontier)

    return None


astar_nodes_expanded = 0
astar_frontier = 0


def astar_search():
    s = Node(mh.find_pos(maze, what="S"), parent=None, action=None, cost=0)
    global astar_nodes_expanded
    global astar_frontier
    goal_pos = mh.find_pos(maze, what="G")
    if s.pos == goal_pos:
        return s
    reached = {}
    frontier = [s]
    while frontier:
        curr_node = frontier.pop()
        if curr_node.pos == goal_pos:
            astar_nodes_expanded = len(reached)
            return curr_node
        act = find_actions(curr_node.pos)
        for i in act:
            if i not in reached or (manhattan_distance(i, goal_pos) + 1) < (
                    manhattan_distance(reached[i].pos, goal_pos) + 1):
                new_node = Node(i, parent=curr_node, action=determine_action(curr_node.pos, i), cost=curr_node.cost + 1)
                reached[i] = new_node
                frontier.append(new_node)
                if len(frontier) > astar_frontier:
                    astar_frontier = len(frontier)
                frontier = sort_astar(frontier)

    return None


f = open("small_maze.txt", "r")
maze_str = f.read()
print(maze_str)

maze = mh.parse_maze(maze_str)

depth_first_small = depth_first_search()
best_fs_small = best_first_search()

bfs_small = breadth_first_search()
astar_small = astar_search()

small = [["BFS", bfs_small.cost, bfs_nodes_expanded, len(bfs_small.get_path_from_root()), "NA", bfs_max_frontier],
         ["DFS", depth_first_small.cost, dfs_nodes_expanded,
          len(depth_first_small.get_path_from_root()), "NA", dfs_max_frontier],
         ["GBS", best_fs_small.cost, bestf_nodes_expanded,
          len(best_fs_small.get_path_from_root()), "NA", bestf_frontier],
         ["A*", astar_small.cost, astar_nodes_expanded, len(astar_small.get_path_from_root()), "NA", astar_frontier]
         ]
small_df = pd.DataFrame(data=small,
                        columns=["Algorithm", "Path Cost", "Nodes Expanded", "Max Tree Depth", "Max Tree Size",
                                 "Max Frontier Size"])
print()
print("SMALL MAZE")
print(small_df.to_markdown)


f = open("medium_maze.txt", "r")
maze_str = f.read()
maze = mh.parse_maze(maze_str)

depth_first_medium = depth_first_search()
best_fs_medium = best_first_search()
bfs_medium = breadth_first_search()
astar_medium = astar_search()

medium = [["BFS", bfs_medium.cost, bfs_nodes_expanded, len(bfs_medium.get_path_from_root()), "NA", bfs_max_frontier],
          ["DFS", depth_first_medium.cost, dfs_nodes_expanded,
           len(depth_first_medium.get_path_from_root()), "NA", dfs_max_frontier],
          ["GBS", best_fs_medium.cost, bestf_nodes_expanded,
           len(best_fs_medium.get_path_from_root()), "NA", bestf_frontier],
          ["A*", astar_medium.cost, astar_nodes_expanded, len(astar_medium.get_path_from_root()), "NA", astar_frontier]
          ]
medium_df = pd.DataFrame(data=medium,
                        columns=["Algorithm", "Path Cost", "Nodes Expanded", "Max Tree Depth", "Max Tree Size",
                                 "Max Frontier Size"])
print()
print("MEDIUM MAZE")
print(medium_df.to_markdown)

f = open("large_maze.txt", "r")
maze_str = f.read()
maze = mh.parse_maze(maze_str)

depth_first_large = depth_first_search()
best_fs_large = best_first_search()
bfs_large = breadth_first_search()
astar_large = astar_search()

large = [["BFS", bfs_large.cost, bfs_nodes_expanded, len(bfs_large.get_path_from_root()), "NA", bfs_max_frontier],
          ["DFS", depth_first_large.cost, dfs_nodes_expanded,
           len(depth_first_large.get_path_from_root()), "NA", dfs_max_frontier],
          ["GBS", best_fs_large.cost, bestf_nodes_expanded,
           len(best_fs_large.get_path_from_root()), "NA", bestf_frontier],
          ["A*", astar_large.cost, astar_nodes_expanded, len(astar_large.get_path_from_root()), "NA", astar_frontier]]
large_df = pd.DataFrame(data=large,
                        columns=["Algorithm", "Path Cost", "Nodes Expanded", "Max Tree Depth", "Max Tree Size",
                                 "Max Frontier Size"])
print()
print("LARGE MAZE")

print(large_df.to_markdown)



print()
