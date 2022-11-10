import numpy as np
from rdkit import Chem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from OGB_ViSNet.datasets import algos

import torch


def ReorderAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum

def data2graph(data):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    smiles_string, mol_3d_obj, AddHs = data
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if AddHs:
            mol = Chem.AddHs(mol)
        mol = ReorderAtoms(mol)
        
        positions = None
        if mol_3d_obj is not None:
            mol_3d_obj = ReorderAtoms(mol_3d_obj)
            sdf_atom_list = np.array([atom.GetAtomicNum() for atom in mol_3d_obj.GetAtoms()])
            smiles_atom_list = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            assert np.all(sdf_atom_list == smiles_atom_list), "Atom order in SDF and SMILES string is different"
            positions = mol_3d_obj.GetConformer().GetPositions()
            
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        graph['position'] = positions if positions is not None else np.zeros((len(x), 3))
        
        return graph
    except:
        return None
    
def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x  # edge_attr: [E, 3], edge_index: [2, E], x: [N, 9]
    N = x.size(0)

    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr + 1 # '+ 1' to avoid 0
    
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())

    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

    # combine
    item.x = x # [N, 9]
    item.attn_bias = attn_bias # [N + 1, N + 1]
    item.attn_edge_type = attn_edge_type # [N, N, 3]
    item.spatial_pos = spatial_pos # [N, N]
    item.in_degree = adj.long().sum(dim=1).view(-1) # [N]
    item.out_degree = item.in_degree # [N]
    item.edge_input = torch.from_numpy(edge_input) # [N, N, distance, 3]
    
    return item

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree, item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos) for item in items]
    attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
        
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    node_type_edges = []
    for idx in range(len(items)):
        node_atom_type = items[idx][5][:, 0]
        n_nodes = items[idx][5].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,
        x=x,
        edge_input=edge_input,
        y=y,
        pos=pos,
        node_type_edge=node_type_edge,
    )