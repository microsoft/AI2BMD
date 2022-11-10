# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# cython: language_level=3


import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy
import networkx as nx

def floyd_warshall(adjacency_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(long, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    cdef numpy.ndarray[long, ndim=2, mode='c'] path = numpy.zeros([n, n], dtype=numpy.int64)

    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj
    cdef long* M_ptr = &M[0,0]
    cdef long* M_i_ptr
    cdef long* M_k_ptr

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path
def all_shortest_path_count(adjacency_matrix, shortest_path_distance_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(long, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    cdef numpy.ndarray[long, ndim=2, mode='c'] path = numpy.zeros([n, n], dtype=numpy.int64)

    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj
    cdef long* M_ptr = &M[0,0]
    cdef long* M_i_ptr
    cdef long* M_k_ptr

    # set unreachable nodes distance to 510
    nG = nx.from_numpy_array(adjacency_matrix)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif shortest_path_distance_matrix[i][j] == 510:
                M[i][j] = 510
            else:
                M[i][j] = len(list(nx.all_shortest_paths(nG, source=i, target=j)))

    return M

def all_shortest_path_count_custom(adjacency_matrix, shortest_path_distance_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adjacency_matrix
    cdef numpy.ndarray[long, ndim=2, mode='c'] MD = shortest_path_distance_matrix
    cdef numpy.ndarray[long, ndim=2, mode='c'] step = numpy.zeros([n, n], dtype=numpy.int64) - 1
    cdef numpy.ndarray[long, ndim=2, mode='c'] path_num = numpy.zeros([n, n], dtype=numpy.int64)
    cdef numpy.ndarray[long, ndim=2, mode='c'] queue = numpy.zeros([n + 1, n + 1], dtype=numpy.int64)

    cdef unsigned int j, k, s
    cdef unsigned int q_begin, q_end
    cdef unsigned int cur_node

    cdef long* step_ptr = &step[0, 0]
    cdef long* queue_ptr = &queue[0, 0]
    cdef long* path_num_ptr = &path_num[0, 0]
    cdef long* M_ptr = &M[0, 0]
    cdef long* MD_ptr = &MD[0, 0]

    cdef long* step_k_ptr
    cdef long* queue_k_ptr
    cdef long* path_num_k_ptr
    cdef long* M_k_ptr
    cdef long* MD_k_ptr

    for k in range(n):
        queue_k_ptr = queue_ptr + (n + 1) * k
        queue_k_ptr[0] = k

        path_num_k_ptr = path_num_ptr + n * k
        path_num_k_ptr[k] = 1

    for k in range(n):
        q_begin = 0
        q_end = 1
        queue_k_ptr = queue_ptr + (n + 1) * k
        step_k_ptr = step_ptr + n * k
        path_num_k_ptr = path_num_ptr + n * k
        MD_k_ptr = MD_ptr + n * k
        while(q_begin < q_end):
            cur_node = queue_k_ptr[q_begin]
            q_begin += 1
            s = step_k_ptr[cur_node] + 1
            M_k_ptr = M_ptr + n * cur_node
            for j in range(n):
                if M_k_ptr[j] == 1:
                    if step_k_ptr[j] == -1 or step_k_ptr[j] > s:
                        step_k_ptr[j] = s
                        path_num_k_ptr[j] = path_num_k_ptr[cur_node]
                        queue_k_ptr[q_end] = j
                        q_end += 1
                    elif s == step_k_ptr[j]:
                        path_num_k_ptr[j] += path_num_k_ptr[cur_node]
                elif MD_k_ptr[j] == 510:
                    path_num_k_ptr[j] = 510

    for k in range(n):
        path_num_k_ptr = path_num_ptr + n * k
        path_num_k_ptr[k] = 0

    return path_num




def get_all_edges(path, i, j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(max_dist, path, edge_feat):

    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(long, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[long, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    cdef unsigned int i, j, k, num_path, cur

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]

    return edge_fea_all

def gen_edge_input_with_node(max_dist, path, edge_feat, node_feat):

    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(long, order='C', casting='safe', copy=True)
    node_feat_copy = node_feat.astype(long, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[long, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    cdef numpy.ndarray[long, ndim=4, mode='c'] node_fea_all = -1 * numpy.ones([n, n, max_dist_copy + 1, node_feat.shape[-1]], dtype=numpy.int64)
    cdef unsigned int i, j, k, num_path, cur

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]
                node_fea_all[i, j, k, :] = node_feat_copy[path[k], :]
                if k == num_path - 1:
                    node_fea_all[i, j, k + 1, :] = node_feat_copy[path[k + 1], :]

    return edge_fea_all, node_fea_all