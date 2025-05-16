import numpy as np
import json
import pickle
from collections import OrderedDict
import math
import os
from rdkit import Chem
import networkx as nx

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

CHARPROTLEN = 25

CHARCANSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
                 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
                 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
                 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
                 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
                 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
                 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
                 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
                 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, "p": 65}

CHARISOSMILEN = 65





def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()




## ######################## ##
#
#  DATASET Class
#
## ######################## ##
class DataSet(object):
    def __init__(self, seqlen, smilen, need_shuffle=False):
        fpath = './data/train/'
        self.path = fpath
        self.SEQLEN = seqlen
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN



    def parse_data(self):
        """Parse the data from given directory."""
        print(f"\nParsing data from path: {self.path}\n")

        try:
            # Check if the directory exists
            if not os.path.exists(self.path):
                print(f"Error: Directory {self.path} does not exist")
                raise FileNotFoundError(f"Directory {self.path} does not exist")

            # List available files in directory
            print("\nAvailable files in directory:")
            for file in os.listdir(self.path):
                print(f"- {file}")


            print("\nLoading ligands and proteins data...")
            ligands_train_path = os.path.join(self.path, "ligands_train.txt")
            proteins_train_path = os.path.join(self.path, "proteins_train.txt")

            print("\nProcessing training data...")
            XD_t = []
            XT_t = []

            try:
                # Load training ligands
                with open(ligands_train_path, 'r') as f:
                    content = f.read().strip()
                    if content.rstrip().endswith(','):
                        content = content.rstrip().rstrip(',')
                    if not content.rstrip().endswith('}'):
                        content += '}'
                    ligands_train = json.loads(content)

                # Process training SMILES
                for smile in ligands_train.values():
                    try:
                        mol = Chem.MolFromSmiles(smile)
                        if mol is not None:
                            XD_t.append(smile)
                    except:
                        continue

                # Load training proteins
                with open(proteins_train_path, 'r') as f:
                    proteins_train = json.load(f, object_pairs_hook=OrderedDict)

                # Convert training proteins to features
                for seq in proteins_train.values():
                    try:
                        features = label_sequence(seq, self.SEQLEN, self.charseqset)
                        XT_t.append(features)
                    except:
                        continue

            except FileNotFoundError:
                print("Warning: Training data files not found. Using empty lists for training data.")



            # Convert lists to numpy arrays
            XD_t = np.array(XD_t)
            XT_t = np.array(XT_t)

            print("\nData processing completed:")
            print(f"- Training drugs (XD_t): {len(XD_t)}")
            print(f"- Training proteins (XT_t): {len(XT_t)}")


            return  XD_t, XT_t

        except Exception as e:
            print(f"Error in parse_data: {str(e)}")
            raise


def atom_features(atom):
    """
    Convert atom to 78-dimensional feature vector
    Features include:
    - Atom symbol (44-dim)
    - Degree (11-dim)
    - Number of Hydrogens (11-dim)
    - Implicit Valence (11-dim)
    - Aromatic (1-dim)
    """
    # Atom symbol (44 types)
    symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
               'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
               'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge',
               'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg',
               'Pb', 'Unknown']

    # Concatenate all features
    features = (
            one_of_k_encoding_unk(atom.GetSymbol(), symbols) +  # 44-dim
            one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11-dim
            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11-dim
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11-dim
            [atom.GetIsAromatic()]  # 1-dim
    )

    # Convert to numpy array and normalize
    features = np.array(features, dtype=np.float32)
    return features


def one_of_k_encoding(x, allowable_set):
    """Converts x to one-hot encoding given allowable set"""
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def smile_to_graph(smile):
    """
    Convert SMILES to molecular graph with 78-dim node features
    Args:
        smile: SMILES string
    Returns:
        - node_features: list of 78-dim numpy arrays
        - edge_index: list of [src_idx, dst_idx] edges
    """
    try:
        # Input validation
        if not isinstance(smile, str):
            print(f"Invalid input type: {type(smile)}, expected string")
            return None, None

        # 清理SMILES字符串
        smile = smile.strip()
        if not smile:
            print("Empty SMILES string")
            return None, None

        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print(f"Failed to parse SMILES: {smile}")
            return None, None

        # 检查分子是否有原子
        if mol.GetNumAtoms() == 0:
            print(f"No atoms found in molecule: {smile}")
            return None, None

        # Get node features
        node_features = []
        for atom in mol.GetAtoms():
            features = atom_features(atom)
            node_features.append(features)

        # Get edge indices
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])  # 添加反向边

        # 如果没有边，但有原子
        if not edges and len(node_features) > 0:
            print(f"Warning: No bonds found in molecule {smile}, adding self-loops")
            # 为每个原子添加自环
            edges = [[i, i] for i in range(len(node_features))]

        if not node_features:
            print(f"No valid atoms found in molecule: {smile}")
            return None, None

        return np.array(node_features), np.array(edges)

    except Exception as e:
        print(f"Error converting SMILES to graph: {str(e)}")
        print(f"Problematic SMILES: {smile}")
        return None, None
