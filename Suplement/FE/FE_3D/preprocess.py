import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, SDMolSupplier
from torch_geometric.data import Data, Batch
from subword_nmt.apply_bpe import BPE
import pandas as pd
import codecs 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

# allowable multiple choice node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    # "possible_atomic_num_list": list(range(1, 119)), 
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",     
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_bond_direction": [
        "NONE",
        "ENDUPRIGHT",
        "ENDDOWNRIGHT",
    ], 
    "possible_is_conjugated_list": [False, True],
}

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    # print(len(allowable_features["possible_atomic_num_list"]))
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"], atom.GetNumRadicalElectrons()
        ),
        safe_index(allowable_features["possible_hybridization_list"], str(atom.GetHybridization())),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(allowable_features["possible_bond_type_list"], str(bond.GetBondType())),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        safe_index(allowable_features["possible_bond_direction"], str(bond.GetBondDir())),
        # allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature

def smiles_to_data(smiles, xyz_path=None):

    if xyz_path:  
        mol = next(SDMolSupplier(xyz_path))
        pos = mol.GetConformer(0).GetPositions()
    else:       
        mol = Chem.MolFromSmiles(smiles)
        pos = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
    
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_feature_vector(atom))
    x = np.array(atom_features, dtype=np.int64)
    
    edges = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_feat = bond_to_feature_vector(bond)
        
        edges.extend([(i, j), (j, i)])
        edge_attrs.extend([edge_feat, edge_feat])
    
    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.array(edge_attrs, dtype=np.int64)
    
    smiles_encoded, mask = drug2emb_encoder(smiles)
    
    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        pos=torch.from_numpy(pos),
        smiles=smiles_encoded,
        mask=mask,
        ori_smiles=smiles,
        num_nodes=mol.GetNumAtoms()
    )
    return data
    
def prepare_batch(data):
    """
    处理单个Data对象为批处理格式（batch_size=1）
    Args:
        data: 单个Data对象（来自smiles_to_data）
    Returns:
        Batch对象（模拟批次维度为1的情况）
    """
    batch = Batch.from_data_list([data])
    smiles_padded = np.zeros((1, len(data.smiles)), dtype=np.int64)
    mask_padded = np.zeros((1, len(data.mask)), dtype=np.int64)
    
    smiles_padded[0, :] = data.smiles
    mask_padded[0, :] = data.mask
    
    batch.smiles = smiles_padded
    batch.mask = mask_padded
    return batch

def drug2emb_encoder(smile):

    vocab_path = "data/ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("data/ESPF/subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    print(f"BPE tokens for {smile}: {t1}")
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)
