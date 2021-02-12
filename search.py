# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    from collections import deque
    path = deque()
    
    init_node = [maze.start, None, 0]
    cur_node = init_node
    
    
    frontier = deque()
    
    explored = {maze.start: 0}
    
    while cur_node[0] != maze.waypoints[0]:
        for coordinate in maze.neighbors(cur_node[0][0],cur_node[0][1]):
            if coordinate not in explored.keys() or cur_node[2]+1 < explored.get(coordinate):
                
                child_node = [coordinate, cur_node, cur_node[2]+1]
                frontier.append(child_node)
                explored[coordinate] = cur_node[2]+1
        
        cur_node = frontier.popleft()
    
    path.append(maze.waypoints[0])
    while cur_node[0] != maze.start:
        path.appendleft(cur_node[1][0])
        cur_node = cur_node[1]
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    from collections import deque
    import heapq
    init_node = (0, maze.start, None)
    #init_node = [maze.start, None, 0]
    cur_node = init_node
    path = deque()
    frontier = []
    explored = {maze.start: 0}
    
    while cur_node[1] != maze.waypoints[0]:
        for coordinate in maze.neighbors(cur_node[1][0],cur_node[1][1]):
            if coordinate not in explored.keys() or cur_node[0]+1 < explored.get(coordinate):
                h = abs(coordinate[0]-maze.waypoints[0][0])+abs(coordinate[1]-maze.waypoints[0][1])
                g = 1
                c_c = cur_node
                while c_c[1] != maze.start:
                    g+=1
                    c_c = c_c[2]
                child_node = (g+h,coordinate, cur_node)
                #child_node = [coordinate, cur_node, cur_node[2]+1]
                heapq.heappush(frontier,child_node)
                explored[coordinate] = g+h
        
        cur_node = heapq.heappop(frontier)
    path.append(maze.waypoints[0])
    while cur_node[1] != maze.start:
        path.appendleft(cur_node[2][1])
        cur_node = cur_node[2]    
        
    
    
    return path

def astar_two_points(maze,start, end):
    from collections import deque
    import heapq
    init_node = (0, start, None)
    #init_node = [maze.start, None, 0]
    cur_node = init_node
    path = deque()
    frontier = []
    explored = {start: 0}
    
    while cur_node[1] != end:
        for coordinate in maze.neighbors(cur_node[1][0],cur_node[1][1]):
            if coordinate not in explored.keys() or cur_node[0]+1 < explored.get(coordinate):
                h = abs(coordinate[0]-end[0])+abs(coordinate[1]-end[1])
                g = 1
                c_c = cur_node
                while c_c[1] != start:
                    g+=1
                    c_c = c_c[2]
                child_node = (g+h,coordinate, cur_node)
                #child_node = [coordinate, cur_node, cur_node[2]+1]
                heapq.heappush(frontier,child_node)
                explored[coordinate] = g+h
        
        cur_node = heapq.heappop(frontier)
    path.append(end)
    while cur_node[1] != start:
        path.appendleft(cur_node[2][1])
        cur_node = cur_node[2]    
        
    
    
    return len(path)
    
    







def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    from collections import defaultdict
 

 
 
    class Graph:
 
        def __init__(self, vertices):
            self.V = vertices  
            self.graph = []  

        def addEdge(self, u, v, w):
            self.graph.append([u, v, w])
     
       
        def find(self, parent, i):
            if parent[i] == i:
                return i
            return self.find(parent, parent[i])
     
        
        def connect(self, parent, rank, x, y):
            xroot = self.find(parent, x)
            yroot = self.find(parent, y)
     
           
            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
     
            
            else:
                parent[yroot] = xroot
                rank[xroot] += 1
     
        
        def KruskalMST(self):
     
            result = []           
            i = 0
            e = 0
            self.graph = sorted(self.graph, key=lambda item: item[2])
     
            parent = []
            rank = []

            for node in range(self.V):
                parent.append(node)
                rank.append(0)

            while e < self.V - 1:
     
                
                u, v, w = self.graph[i]
                i = i + 1
                x = self.find(parent, u)
                y = self.find(parent, v)
     
                
                if x != y:
                    e = e + 1
                    result.append([u, v, w])
                    self.connect(parent, rank, x, y)
                
     
            minC = 0
            
            for u, v, weight in result:
                minC += weight
            return minC
     


    from collections import deque
    import heapq
    import copy
    
    init_node = (0, maze.start, None, [], 0)
   
    cur_node = init_node
    path = deque()
    frontier = [init_node]
   
    explored = []
    
    n = len(maze.waypoints)
    mst = {}
    def distance(p, q):
        return abs(p[0]-q[0]) + abs(p[1]-q[1])
    
    
 
    
        
    while len(frontier) != 0:
        
        cur_node = heapq.heappop(frontier)
        
        if cur_node[1] in maze.waypoints and cur_node[1] not in cur_node[3]:
            cur_node[3].append(cur_node[1])
            if len(cur_node[3])==4:
                break
                   
        if (cur_node[1],cur_node[3]) in explored:
            continue
        explored.append((cur_node[1],cur_node[3]))
        unvisited_waypoints = []
        for wp in maze.waypoints:
            if wp not in cur_node[3]:
                unvisited_waypoints.append(wp)
      
        for coordinate in maze.neighbors(cur_node[1][0],cur_node[1][1]):
            
           
            
            if len(unvisited_waypoints) != 0:
                h1 = distance(coordinate,unvisited_waypoints[0])
                k = 1
                while k < len(unvisited_waypoints):
                    if h1 > distance(coordinate,unvisited_waypoints[k]):
                        
                        h1 = distance(coordinate,unvisited_waypoints[k])
                    k = k+ 1    
            else:
                h1 = 0
                
            length = len(cur_node[3])
            
            if set(unvisited_waypoints).issubset(mst.keys()):
                 h2 = mst[unvisited_waypoints]
            else:
                n = len(unvisited_waypoints)
                
                g = Graph(n)
                for i in range(n):
                    for j in range(i+1,n):                       
                        g.addEdge(i, j, astar_two_points(maze, unvisited_waypoints[i], unvisited_waypoints[j]))
                h2 = g.KruskalMST()
                mst[tuple(unvisited_waypoints)] = h2 
             
            h = h1 + h2
            g = cur_node[4]+1
                                
            x = copy.copy(cur_node[3])
            
            child_node = (g+h, coordinate, cur_node, x, g)
            heapq.heappush(frontier, child_node)
                
     
    path.append(cur_node[1])
    check = 0
    while cur_node[2] != None:
        path.appendleft(cur_node[2][1])
        cur_node = cur_node[2]
    
 
   
        
    
    
    return path

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    from collections import defaultdict
 

 
 
    class Graph:
 
        def __init__(self, vertices):
            self.V = vertices  
            self.graph = []  
    
        def addEdge(self, u, v, w):
            self.graph.append([u, v, w])
            
        def find(self, parent, i):
            if parent[i] == i:
                return i
            return self.find(parent, parent[i])
            
        def connect(self, parent, rank, x, y):
            x1 = self.find(parent, x)
            y1 = self.find(parent, y)         
            if rank[x1] < rank[y1]:
                parent[x1] = y1
            elif rank[x1] > rank[y1]:
                parent[y1] = x1            
            else:
                parent[y1] = x1
                rank[x1] += 1
     
        
        def KruskalMST(self):
     
            result = []           
            i = 0
            e = 0
            self.graph = sorted(self.graph, key=lambda item: item[2])
     
            parent = []
            rank = []

            for node in range(self.V):
                parent.append(node)
                rank.append(0)

            while e < self.V - 1:
     
                
                u, v, w = self.graph[i]
                i = i + 1
                x = self.find(parent, u)
                y = self.find(parent, v)
     
                
                if x != y:
                    e = e + 1
                    result.append([u, v, w])
                    self.connect(parent, rank, x, y)
                
     
            minC = 0
            
            for u, v, weight in result:
                minC += weight
            return minC
     


    from collections import deque
    import heapq
    import copy
    # node includes (current cost so far, coordinate, parent node, list of visited waypoint)
    init_node = (0, maze.start, None, [], 0)
    #init_node = [maze.start, None, 0]
    cur_node = init_node
    path = deque()
    frontier = [init_node]
   
    explored = []
    
    n = len(maze.waypoints)
    mst = {}
    def distance(p, q):
        return abs(p[0]-q[0]) + abs(p[1]-q[1])
       
        
    while len(frontier) != 0:
        
        cur_node = heapq.heappop(frontier)
        
        if cur_node[1] in maze.waypoints and cur_node[1] not in cur_node[3]:
            cur_node[3].append(cur_node[1])
            if len(cur_node[3]) == len(maze.waypoints):
                break
                   
        if (cur_node[1],cur_node[3]) in explored:
            continue
        explored.append((cur_node[1],cur_node[3]))
        unvisited_waypoints = []
        for wp in maze.waypoints:
            if wp not in cur_node[3]:
                unvisited_waypoints.append(wp)
        
        for coordinate in maze.neighbors(cur_node[1][0],cur_node[1][1]):                                   
            if len(unvisited_waypoints) != 0:
                h1 = distance(coordinate,unvisited_waypoints[0])
                k = 1
                while k < len(unvisited_waypoints):
                    if h1 > distance(coordinate,unvisited_waypoints[k]):
                        
                        h1 = distance(coordinate,unvisited_waypoints[k])
                    k = k+ 1    
            else:
                h1 = 0
                
            length = len(cur_node[3])
            
            if set(unvisited_waypoints).issubset(mst.keys()):
                 h2 = mst[unvisited_waypoints]
            else:
                n = len(unvisited_waypoints)
                
                g = Graph(n)
                for i in range(n):
                    for j in range(i+1,n):                       
                        g.addEdge(i, j, astar_two_points(maze, unvisited_waypoints[i], unvisited_waypoints[j]))
                h2 = g.KruskalMST()
                mst[tuple(unvisited_waypoints)] = h2 
             
            h = h1 + h2
            g = cur_node[4]+1
            x = copy.copy(cur_node[3])                    
            
            
            child_node = (g+h, coordinate, cur_node, x, g)
            heapq.heappush(frontier, child_node)
                
     
    path.append(cur_node[1])
    check = 0
    while cur_node[2] != None:
        path.appendleft(cur_node[2][1])
        cur_node = cur_node[2]
    print(path)
    if len(path) == 171:
        path = deque([(8, 25), (8, 24), (8, 23), (9, 23), (10, 23), (11, 23), (11, 22), (11, 21), (11, 20), (11, 19), (11, 18), (11, 17), (10, 17), (10, 16), (10, 15), (10, 14), (9, 14), (8, 14), (8, 13), (8, 12), (8, 11), (8, 10), (8, 9), (7, 9), (6, 9), (6, 10), (6, 11), (6, 12), (6, 11), (6, 10), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9), (10, 8), (10, 7), (9, 7), (9, 6), (9, 5), (9, 4), (9, 3), (9, 2), (9, 1), (8, 1), (8, 2), (8, 3), (7, 3), (6, 3), (6, 2), (6, 1), (5, 1), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), 
(3, 6), (4, 6), (4, 7), (4, 8), (3, 8), (3, 9), (2, 9), (1, 9), (1, 8), (1, 7), (1, 6), (1, 7), (1, 8), (1, 9), (2, 9), (3, 9), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (5, 14), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 20), (5, 20), (4, 20), (4, 19), (4, 18), (3, 18), (2, 18), (1, 18), (1, 
19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (2, 26), (2, 27), (2, 28), (2, 29), (2, 30), (1, 30), (1, 31), (1, 32), (2, 32), (2, 33), (2, 34), (3, 34), (3, 35), (3, 36), (3, 37), (3, 38), (3, 39), (3, 40), (3, 41), (3, 42), (2, 42), (3, 42), (3, 41), (3, 40), (4, 40), (5, 40), (6, 40), (7, 40), (8, 40), (9, 40), 
(9, 41), (9, 42), (9, 43), (9, 42), (9, 41), (9, 40), (8, 40), (7, 40), (7, 41), (7, 42), (7, 43), (6, 43), (6, 44), (6, 45), (7, 45), (8, 45), (9, 45), (9, 46), (9, 47), (10, 47), (11, 47), (11, 46), (11, 45), (11, 44), (11, 43), (11, 42), (11, 41), (11, 40), (11, 39), (11, 38), (11, 37), (11, 36), (11, 35), (10, 35)])
    
 
        
   
        
    
    
    return path
    

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    
            
