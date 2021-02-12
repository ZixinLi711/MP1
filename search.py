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

#这次作业主要内容就是用A* search 找经过多点的最短路径。因为前两问都是经过一个点，在把一个点扩展成多个点的时候，我的主循环逻辑非常混乱。
#作业网站：https://courses.grainger.illinois.edu/ece448/sp2021/assignment1.html

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #算 MST length 的 Krusakl 是我直接网上copy的
    class Graph:
 
        def __init__(self, vertices):
            self.V = vertices  # No. of vertices
            self.graph = []  # default dictionary
            # to store graph
     
        # function to add an edge to graph
        def addEdge(self, u, v, w):
            self.graph.append([u, v, w])
     
        # A utility function to find set of an element i
        # (uses path compression technique)
        def find(self, parent, i):
            if parent[i] == i:
                return i
            return self.find(parent, parent[i])
     
        # A function that does union of two sets of x and y
        # (uses union by rank)
        def union(self, parent, rank, x, y):
            xroot = self.find(parent, x)
            yroot = self.find(parent, y)
     
            # Attach smaller rank tree under root of
            # high rank tree (Union by Rank)
            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
     
            # If ranks are same, then make one as root
            # and increment its rank by one
            else:
                parent[yroot] = xroot
                rank[xroot] += 1
     
        # The main function to construct MST using Kruskal's
            # algorithm
        def KruskalMST(self):
     
            result = []  # This will store the resultant MST
             
            # An index variable, used for sorted edges
            i = 0
             
            # An index variable, used for result[]
            e = 0
     
            # Step 1:  Sort all the edges in 
            # non-decreasing order of their
            # weight.  If we are not allowed to change the
            # given graph, we can create a copy of graph
            self.graph = sorted(self.graph, 
                                key=lambda item: item[2])
     
            parent = []
            rank = []
     
            # Create V subsets with single elements
            for node in range(self.V):
                parent.append(node)
                rank.append(0)
     
            # Number of edges to be taken is equal to V-1
            while e < self.V - 1:
     
                # Step 2: Pick the smallest edge and increment
                # the index for next iteration
                u, v, w = self.graph[i]
                i = i + 1
                x = self.find(parent, u)
                y = self.find(parent, v)
     
                # If including this edge does't
                #  cause cycle, include it in result 
                #  and increment the indexof result 
                # for next edge
                if x != y:
                    e = e + 1
                    result.append([u, v, w])
                    self.union(parent, rank, x, y)
                # Else discard the edge
     
            minimumCost = 0
            
            for u, v, weight in result:
                minimumCost += weight
            return minimumCost
     

    #从这里开始是我写的：
    from collections import deque
    import heapq
    import copy
    # node includes (current cost so far, coordinate, parent node, list of visited waypoint)
    init_node = (0, maze.start, None, [])
    #init_node = [maze.start, None, 0]
    cur_node = init_node
    path = deque()
    frontier = []
    # explored includes ((coordinate of explored node, current cost so far): list of visited waypoint)
    explored = {maze.start:[]}
    
    # find the fartherest waypoint:
    far_waypoint = maze.waypoints[0]
    
    
    
    
    # compute MST for remaining waypoints and store them in the dic called mst: turple for the index of the visited waypoints; There are in total 2^n keys. 
    
    mst = {}
    def distance(p, q):
        return abs(p[0]-q[0]) + abs(p[1]-q[1])
    


   #我就是觉得这个循环这里非常乱，还很麻烦
    while 1:   
        for coordinate in maze.neighbors(cur_node[1][0],cur_node[1][1]):
            #if coordinate not in explored.keys() or cur_node[0]+1 < explored.get(coordinate):
            if coordinate not in explored.keys() or 0 < len(explored[coordinate]) <= 4:    
                unvisited_waypoints = []
                for wp in maze.waypoints:
                    if wp not in cur_node[3]:
                        unvisited_waypoints.append(wp)
                print(unvisited_waypoints)
                h1 = distance(coordinate,unvisited_waypoints[0])
                if len(unvisited_waypoints) != 0:
                    k = 1
                    while k < len(unvisited_waypoints):
                        if h1 > distance(coordinate,unvisited_waypoints[k]):
                            h1 = distance(coordinate,unvisited_waypoints[k])
                        k += 1
                length = len(cur_node[3])
                
                
                
                if set(unvisited_waypoints).issubset(mst.keys()):
                     h2 = mst[unvisited_waypoints]
                else:
                    n = len(unvisited_waypoints)
                    g = Graph(n)
                    for i in range(n):
                        for j in range(i+1,n):
                            g.addEdge(i, j, distance(unvisited_waypoints[i], unvisited_waypoints[j]))
                    h2 = g.KruskalMST()
                    mst[unvisited_waypoints] = h2 
               
                        
                h = h1 + h2
                
                
                g = 1
                
                c_c = cur_node
                
                while c_c[1] != maze.start:
                    g+=1
                    c_c = c_c[2]
                    
                x = copy.deepcopy(cur_node[3]) 
                if coordinate in maze.waypoints:
                    x.append(coordinate)
                
                child_node = (0, coordinate, cur_node, x)
                #print(child_node)
                #child_tuple = (h + child_node[0], child_node)
                heapq.heappush(frontier, child_node)
                
                explored[coordinate] = x
                
            
        
        
        if len(frontier) != 0:
            cur_node = heapq.heappop(frontier)
           
        if cur_node[1] in maze.waypoints and cur_node[1] not in cur_node[3]:
            print(1)
            print(cur_node)
            print(explored)
            cur_node[3].append(cur_node[1])
            explored[cur_node[1]] = cur_node[3]
            print(cur_node)
            print(explored)
        
        #通过这个来让循环停：
        if len(maze.waypoints) == len(cur_node[3]):
            break
        
        
    #print(cur_node)    
    path.append(cur_node[1])
    #print(explored)
    check = 0
    while cur_node[2] != None:
        path.appendleft(cur_node[2][1])
        cur_node = cur_node[2]
    
    
  
    
    
    return path


            
