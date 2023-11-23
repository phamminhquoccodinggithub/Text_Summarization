# pylint: disable=invalid-name
"""
    This class to implement pagerank algorithm
"""

import numpy as np

class PageRank():
    """PageRank
    Contains function to calculate the rank of pages using PageRank algorithm

    Parameters
    ----------
    numpy_array: numpy array with kind of
        [[0. 1. 1. 1. 1. 0.]
        [1. 0. 1. 1. 1. 0.]
        [1. 1. 0. 1. 0. 0.]
        [1. 1. 1. 0. 0. 1.]
        [1. 1. 0. 0. 0. 1.]
        [0. 0. 0. 1. 1. 0.]]

    num_iterations: int, number of iterations

    threshold: float, threshold that will break the loop

    damping_factor: int, default is 0.85

    Returns
    -------
    list
        a list of pagerank scores of each sentence
    """
    def __init__(self, numpy_array, num_iterations=100, threshold=0.000000001, damping_factor=0.85):
        self.matrix = self.convert_2d_array_to_numpy_array(numpy_array)
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.d = damping_factor

    def convert_2d_array_to_numpy_array(self, arr_2d):
        """
            @note: Return a numpy array

            @param: None
        """
        if isinstance(np.ndarray, type(arr_2d)):
            return arr_2d
        return np.array(arr_2d)

    def get_degree_of_vertices(self):
        """
            @note: Return a list degree of vertices from square matrix

            @param: None
        """
        N = self.matrix.shape[0]
        deg = [0] * N
        for i in range(N):
            for j in range(N):
                if self.matrix[i][j] != 0:
                    deg[i] += 1
        return deg

    def get_sum_of_incoming_page_rank_scores(self, page_rank, deg):
        """
            @note: Return a list sum of the incoming PageRank scores of adjacent vertices

            @param pageRank: List initial pagerank
            @param deg: Degree of vertices
        """
        N = self.matrix.shape[0]
        s = [0] * N
        for i in range(N):
            for j in range(N):
                if self.matrix[i][j] != 0:
                    s[i] += page_rank[j] / deg[j]
        return s

    def get_page_rank(self):
        """
            @note: Count and return pagerank based on pagerank algorithm

            @param None
        """
        N = self.matrix.shape[0]
        page_rank = [1] * N
        deg = self.get_degree_of_vertices()

        for _ in range(self.num_iterations):
            new_page_rank = [0] * N
            diff = [0] * N
            s = self.get_sum_of_incoming_page_rank_scores(page_rank=page_rank, deg=deg)

            for j in range(N):
                new_page_rank[j] = (1 - self.d) / N + self.d * s[j]
                diff[j] = abs(new_page_rank[j] - page_rank[j])

            stopPoint = sum(diff) / int(N)
            page_rank = new_page_rank

            if stopPoint <= self.threshold:
                break

        return page_rank
# Test
# A = [[0, 1, 1, 1, 1, 0],
#      [1, 0, 1, 1, 1, 0],
#      [1, 1, 0, 1, 0, 0],
#      [1, 1, 1, 0, 0, 1],
#      [1, 1, 0, 0, 0, 1],
#      [0, 0, 0, 1, 1, 0]
#      ]

# pr = PageRank(numpy_array=A, num_iterations=100)
# rank = pr.get_page_rank()
# print(rank)
