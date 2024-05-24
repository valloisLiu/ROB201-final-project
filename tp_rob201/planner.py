import numpy as np
import heapq
import itertools
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):

        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        # TODO for TP5

        # path = [start, goal]  # list of poses

        start = self.grid.conv_world_to_map(start[0], -start[1])
        goal  = self.grid.conv_world_to_map(goal[0], -goal[1])

        path = self.A_Star(start, goal)
        return path

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
    
    # def get_neighbors(self, current_cell):
    #     """ Get the neighbors of a cell """
    #     directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, -1], [-1, 1]])
    #     neighbors = []
    #     for direction in directions:
    #         neighbor = (current_cell[0] + direction[0], current_cell[1] + direction[1])
    #         if (0 <= neighbor[0] < self.grid.x_max_map) and (0 <= neighbor[1] < self.grid.y_max_map):
    #             neighbors.append(neighbor)
    #     return neighbors
    
    def get_neighbors(self, current):
        x, y = current
        neighbors = {(x+i, y+j) for i, j in itertools.product([-1, 0, 1], repeat=2) if (i != 0 or j != 0) and
                 0 <= x+i < self.grid.x_max_map and 0 <= y+j < self.grid.y_max_map and self.grid.occupancy_map[x+i][-(y+j)] < -3}

        return neighbors
    
    def heuristic(self, cell_1, cell_2):
        """ Compute the heuristic between two cells """
        return np.linalg.norm(np.array(cell_1) - np.array(cell_2))
    
    def reconstruct_path(self, cameFrom, current):
        total_path = [current]
        while current in cameFrom.keys() and cameFrom[current] != None:
            current = cameFrom[current]
            total_path.append(current)
        return total_path
    
    def A_Star(self, start, goal):
        
        # Initialisation
        openSet = [(self.heuristic(start,goal), start)]         
        # On définit la liste de priorités : de base, c'est juste le start, de fScore h(start, goal)

        # Utiliser un set et non pas une liste est beaucoup plus rapide ( gain de temps : x10)
        visited_nodes = set()
        
        cameFrom = {start: None}             
        # Le point de départ n'a pas de prédécesseur
        
        # J'ai essayé d'initialiser les dictionnaires avec des valeurs infinies, 
        # mais cela s'est révélé extremement couteux en temps
        # J'ai donc intitialisé des dictionnaires "vides" 
        # et on compare les valeurs de gScore seulement quand il y a une valeur à comparer
        # C'est à dire quand le noeud a déjà été visité.
        
        gScore= {start : 0}                     # Distance start -> start : 0
        fScore = {start : self.heuristic(start, goal)}  # Distance start -> goal : h(start, goal)
        
        # Tant que l'on a des noeuds à explorer
        while openSet is not []:
            # On pop le noeud avec le fScore le plus faible
            current = heapq.heappop(openSet)[1]
            
            # Si c'est le goal, banco
            if current == goal:
                path = self.reconstruct_path(cameFrom, current)
                return path
            
            # Sinon, on regarde ses voisins
            visited_nodes.add(current)
            neighbors = self.get_neighbors(current)

            # Pour chacun des voisins
            for neighbor in neighbors:
                
                    # On regarde son gScore a travers le noeud actuel
                    tentative_gScore = gScore[current] + self.heuristic(current, neighbor)

                    # Si le voisin a déjà été visité et si la tentative de gScore n'améliore rien, on passe
                    if neighbor in visited_nodes and tentative_gScore >= gScore[neighbor] - 10e-3:
                        continue
                    
                    # Sinon
                    cameFrom[neighbor] = current                                 
                    # On note le voisin comme issu du noeud courant
                    gScore[neighbor] = tentative_gScore                          
                    # On actualise le gScore de ce voisin
                    fScore[neighbor] = tentative_gScore + self.heuristic(neighbor, goal) 
                    # On note le fScore de ce voisin
                    if ((fScore[neighbor], neighbor) not in openSet):
                        # On rajoute ce voisin dans openSet, avec son fScore
                        heapq.heappush(openSet, (fScore[neighbor], neighbor))

        return None
