import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from aad.graphs import tools

num_node = 27
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 3), (1,2),  
                    (2, 3), (2, 5), (2, 6), (2, 7),
                    (3, 5), (3, 4), 
                    (4, 5), (4, 11),(5,6),(5,10),(6,7),(6,9),(6,10),(7,8),(7,9)
                    ,(8,9),(8,15),(9,10),(9,14),(10,11),(10,13),(11,12),
                    (12,19),(12,13),(13,18),(13,14),(14,15),(14,17),(15,16),
                    (16,17),(16,23),(17,18),(17,23),(17,22),(18,19),(18,21),(18,22),(19,20),
                    (20,21),(20,26),(21,22),(21,25),(21,26),(22,23),(22,25),(23,24),(23,25),
                    (25,26),(25,27),(26,27)
                    ] #left
# inward_ori_index = [(1, 33), (1, 3), (1,2), 
#                     (2, 3), (2, 5), (2, 6), (2, 7),
#                     (3, 5), (3, 4), (3, 37), 
#                     (4, 5), (4, 11), (4, 38),(5,6),(5,10),(6,7),(6,9),(6,10),(7,8),(7,9)
#                     ,(8,9),(8,15),(9,10),(9,14),(10,11),(10,13),(11,12),(11,47),
#                     (12,48),(12,19),(12,13),(13,18),(13,14),(14,15),(14,17),(15,16),
#                     (16,17),(16,23),(17,18),(17,23),(17,22),(18,19),(18,21),(18,22),(19,20),(19,32),
#                     (20,21),(20,26),(20,31),(21,22),(21,25),(21,26),(22,23),(22,25),(23,24),(23,25),
#                     (25,26),(25,27),(26,27),(26,30),(27,29),
#                     (28,29),(29,30),(30,31),(31,32),(32,48),(48,47),(47,38),(38,37),(37,33)] #left

# inward_ori_index = [(34, 33), (34, 36), (34,35), 
#                     (35, 36), (35, 40), (35, 41), (35, 42),
#                     (36, 37), (36, 39), (36, 40), 
#                     (39, 38), (39,46 ), (39, 40),(40,41),(40,45),(41,42),(41,44),(41,40),(42,43),(42,44)
#                     ,(43,44),(43,52),(44,45),(44,51),(45,46),(45,50),(46,47),(46,49),
#                     (49,48),(49,50),(49,56),(50,51),(50,55),(51,52),(51,54),(52,53),
#                     (53,54),(53,60),(54,55),(54,59),(54,60),(55,56),(55,58),(55,59),(56,32),(56,57),
#                     (57,31),(57,58),(57,63),(58,59),(58,62),(58,63),(59,60),(59,62),(60,61),(60,62),
#                     (62,63),(62,64),(63,30),(63,64),(64,29),
#                     (28,29),(29,30),(30,31),(31,32),(32,48),(48,47),(47,38),(38,37),(37,33)] #right
                   

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
