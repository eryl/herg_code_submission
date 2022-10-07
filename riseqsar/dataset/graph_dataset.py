import copy
import itertools
import multiprocessing.dummy as multiprocessing
import pickle
from collections import defaultdict
from textwrap import indent
from typing import Union, Dict, List, FrozenSet
from pathlib import Path

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from riseqsar.dataset.rdkit_dataset import RDKitMolDataset, RDKitMolDatasetConfig, DatasetSpec


class CategoricalFeatureMeta(type):
    def __new__(cls, name, bases, dct):
        obj = super().__new__(cls, name, bases, dct)
        obj.index = None
        obj.inv_index = None
        if hasattr(obj, 'values') and obj.values is not None:
            obj.inv_index = {i: x for i, x in enumerate(obj.values)}
            obj.index = {x: i for i, x in obj.inv_index.items()}
        return obj


class CategoricalFeature(metaclass=CategoricalFeatureMeta):
    name = None
    values = None

    def __init__(self, value):
        self.value = value

    @classmethod
    def from_rdobj(self, rd_obj: Union[Chem.Atom, Chem.Bond]):
        raise NotImplementedError()

    @classmethod
    def from_index(cls, index):
        return cls(cls.inv_index[index])

    @classmethod
    def from_value(cls, value):
        return cls(cls.index[value])

    @classmethod
    def n_values(cls):
        return len(cls.index)

    def to_index(self):
        return self.index[self.value]


class SymbolFeature(CategoricalFeature):
    name = 'Symbol'
    values = (None, 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
              'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
              'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
              'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
              'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re',
              'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm',
              'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', '*', 'UNK')

    @classmethod
    def from_rdobj(cls, rd_obj: Union[Chem.Atom, Chem.Bond]):
        assert isinstance(rd_obj, Chem.Atom)
        value = rd_obj.GetSymbol()
        return cls(value)


class FormalChargeFeature(CategoricalFeature):
    name = 'Formal Charge'
    values = (None, -2, -1, 0, 1, 2)
    @classmethod
    def from_rdobj(cls, rd_obj: Union[Chem.Atom, Chem.Bond]):
        assert isinstance(rd_obj, Chem.Atom)
        value = rd_obj.GetFormalCharge()
        return cls(value)

class ExplicitValanceFeature(CategoricalFeature):
    name = 'Explicit Valance'
    values = (None, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    @classmethod
    def from_rdobj(cls, rd_obj: Union[Chem.Atom, Chem.Bond]):
        assert isinstance(rd_obj, Chem.Atom)
        value = rd_obj.GetExplicitValence()
        return cls(value)

class ImplicitValanceFeature(CategoricalFeature):
    name = 'Implicit Valance'
    values = (None, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    @classmethod
    def from_rdobj(cls, rd_obj: Union[Chem.Atom, Chem.Bond]):
        assert isinstance(rd_obj, Chem.Atom)
        value = rd_obj.GetImplicitValence()
        return cls(value)

# Maximum number of neighbors for an atom
class DegreeFeature(CategoricalFeature):
    name = 'Degree'
    MAX_NEIGHBORS = 10
    values = (None,) + tuple(range(MAX_NEIGHBORS))
    @classmethod
    def from_rdobj(cls, rd_obj: Union[Chem.Atom, Chem.Bond]):
        assert isinstance(rd_obj, Chem.Atom)
        value = rd_obj.GetDegree()
        return cls(value)

class AromaticityFeature(CategoricalFeature):
    name = 'Aromaticity'
    values = (None, True, False)
    @classmethod
    def from_rdobj(cls, rd_obj: Union[Chem.Atom, Chem.Bond]):
        assert isinstance(rd_obj, Chem.Atom)
        value = rd_obj.GetIsAromatic()
        return cls(value)


class BondTypeFeature(CategoricalFeature):
    name = 'Bond type'
    values = (None,
              Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC,)

    @classmethod
    def from_rdobj(cls, rd_obj: Union[Chem.Atom, Chem.Bond]):
        assert isinstance(rd_obj, Chem.Bond)
        value = rd_obj.GetBondType()
        return cls(value)

class ConjugatedFeature(CategoricalFeature):
    name = 'Conjugated'
    values = (None, True, False)
    @classmethod
    def from_rdobj(cls, rd_obj: Union[Chem.Atom, Chem.Bond]):
        assert isinstance(rd_obj, Chem.Bond)
        value = rd_obj.GetIsConjugated()
        return cls(value)

class InRingFeature(CategoricalFeature):
    name = 'In Ring'
    values = (None, True, False)
    @classmethod
    def from_rdobj(cls, rd_obj: Union[Chem.Atom, Chem.Bond]):
        assert isinstance(rd_obj, Chem.Bond)
        value = rd_obj.IsInRing()
        return cls(value)


ATOM_FEATURES = [SymbolFeature, FormalChargeFeature, DegreeFeature, ExplicitValanceFeature, ImplicitValanceFeature, AromaticityFeature]
BOND_FEATURES = [BondTypeFeature, ConjugatedFeature, InRingFeature]


class MolecularGraph(object):
    def __init__(self, *,
                 smiles: str,
                 atom_features: Dict[int, List[CategoricalFeature]],
                 bond_features: Dict[FrozenSet[int], List[CategoricalFeature]],
                 shortest_paths,
                 rings):
        self.smiles = smiles
        self.atom_features = atom_features
        self.bond_features = bond_features
        any_atom = next(iter(self.atom_features.values()))  # This just takes the first value from the dict values
        self.atom_feature_spec = [type(feature) for feature in any_atom]
        any_bond = next(iter(self.bond_features.values()))
        self.bond_feature_spec = [type(feature) for feature in any_bond]
        # This dict allow us to correctly order the atoms by atom index
        self.ordered_atoms = {atom_idx: i for i, atom_idx in enumerate(sorted(self.atom_features.keys()))}
        # Since we don't assume the bond atom indices are directly related to the order of atoms, we create a total
        # ordering of the bonds according to the ordered_atoms dict
        self.ordered_bonds = {atom_idx_pair: i
                              for i, atom_idx_pair
                              in enumerate(sorted(self.bond_features.keys(),
                                                  key=lambda bond_pair: frozenset(self.ordered_atoms[atom_idx]
                                                                                  for atom_idx in bond_pair)))}
        self.shortest_paths = shortest_paths
        self.rings = rings

    @classmethod
    def from_rdmol(cls, rd_mol, atom_features=None, bond_features=None, smiles=None):
        if atom_features is None:
            atom_features = ATOM_FEATURES
        if bond_features is None:
            bond_features = BOND_FEATURES
        atoms = make_atom_features(rd_mol, atom_features)
        bonds = make_bond_features(rd_mol, bond_features)
        #shortest_paths = make_shortest_paths(rd_mol)
        #ring_paths = make_ring_paths(rd_mol)
        shortest_paths = None
        ring_paths = None
        if smiles is None:
            smiles = Chem.MolToSmiles(rd_mol, canonical=True)
        return MolecularGraph(smiles=smiles,
                              atom_features=atoms,
                              bond_features=bonds,
                              shortest_paths=shortest_paths,
                              rings=ring_paths
                              )
    
    @classmethod
    def from_smiles(cls, smiles, atom_features=None, bond_features=None):
        rdmol = Chem.MolFromSmiles(smiles)
        return cls.from_rdmol(rdmol, atom_features=atom_features, bond_features=bond_features, smiles=smiles)

    def get_atom_features(self, atom_feature_spec=None):
        if atom_feature_spec is None:
            atom_feature_spec = self.atom_feature_spec
        n_atoms = len(self.atom_features)
        n_features = len(atom_feature_spec)
        feature_matrix = np.zeros((n_atoms, n_features), dtype=np.long)
        for atom_idx, atom_features in self.atom_features.items():
            i = self.ordered_atoms[atom_idx]
            return_features = []
            for feature_class in atom_feature_spec:
                for feature in atom_features:
                    if isinstance(feature, feature_class):
                        return_features.append(feature.to_index())
            feature_matrix[i] = np.array(return_features)
        return feature_matrix

    def get_coo_bonds(self):
        n_bonds = 2*len(self.bond_features) # times 2 since to specify undirected graphs we need to duplicate the edges
        bond_coo_matrix = np.zeros((2, n_bonds), dtype=np.long)
        for bond_pair in self.bond_features.keys():
            atom_idx_u, atom_idx_v = bond_pair
            atom_i_u = self.ordered_atoms[atom_idx_u]
            atom_i_v = self.ordered_atoms[atom_idx_v]
            bond_i = self.ordered_bonds[bond_pair]
            first_direction_i = 2*bond_i
            second_direction_i = 2*bond_i + 1
            bond_coo_matrix[:, first_direction_i] = [atom_i_u, atom_i_v]
            bond_coo_matrix[:, second_direction_i] = [atom_i_v, atom_i_u]
        return bond_coo_matrix

    def get_bond_features(self, bond_feature_spec=None):
        if bond_feature_spec is None:
            bond_feature_spec = self.bond_feature_spec
        n_bonds = 2 * len(self.bond_features)  # times 2 since to specify undirected graphs we need to duplicate the edges
        n_features = len(bond_feature_spec)
        bond_feature_matrix = np.zeros((n_bonds, n_features), dtype=np.long)
        for bond_pair, bond_features in self.bond_features.items():
            bond_i = self.ordered_bonds[bond_pair]
            first_direction_i = 2 * bond_i
            second_direction_i = 2 * bond_i + 1
            return_features = []
            for feature_class in bond_feature_spec:
                for feature in bond_features:
                    if isinstance(feature, feature_class):
                        return_features.append(feature.to_index())
            return_features = np.array(return_features)
            bond_feature_matrix[first_direction_i] = return_features
            bond_feature_matrix[second_direction_i] = return_features
        return bond_feature_matrix


def make_atom_features(rd_mol, atom_feature_spec):
    atom_features = dict()
    for rd_atom in rd_mol.GetAtoms():
        features = [feature.from_rdobj(rd_atom) for feature in atom_feature_spec]
        idx = rd_atom.GetIdx()
        atom_features[idx] = features
    return atom_features


def make_bond_features(rd_mol, bond_features_spec):
    features = dict()
    for bond in rd_mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        pair = frozenset((begin_atom, end_atom))
        features[pair] = [feature.from_rdobj(bond) for feature in bond_features_spec]
    return features


def make_shortest_paths(rd_mol):
    shortest_paths = defaultdict(dict)
    for fragment in Chem.rdmolops.GetMolFrags(rd_mol):
        for a1, a2 in itertools.combinations(fragment, 2):
            shortest_path = Chem.rdmolops.GetShortestPath(rd_mol, a1, a2)
            shortest_paths[(a1, a2)] = shortest_path
            shortest_paths[(a2, a1)] = list(reversed(shortest_path))
    return shortest_paths


def make_ring_paths(rd_mol):
    rings_dict = defaultdict(set)
    ssr = [list(x) for x in Chem.GetSymmSSSR(rd_mol)]
    for ring in ssr:
        ring_size = len(ring)
        is_aromatic = True
        for atom_idx in ring:
            if not rd_mol.GetAtomWithIdx(atom_idx).GetIsAromatic():
                is_aromatic = False
                break
        for ring_idx, atom_pair in enumerate(itertools.combinations(ring, 2)):
            atom_pair = frozenset(atom_pair)
            rings_dict[atom_pair].add((ring_size, is_aromatic))
    return rings_dict


def mol2graph(work_package):
    """Returns a graph representation of a given RDKit Mol object"""
    rd_mol_bytes = work_package['rd_mol']
    atom_features = work_package['atom_features']
    bond_features = work_package['bond_features']
    #smiles = work_package['smiles']
    rd_mol = Chem.Mol(rd_mol_bytes)
    graph = MolecularGraph.from_rdmol(rd_mol, atom_features=atom_features, bond_features=bond_features)
    return graph

def mol2pickled_graph(work_package):
    graph = mol2graph(work_package)
    path = work_package['path']
    with open(path, 'wb') as fp:
        pickle.dump(graph, fp)
    return path

def mols2graphs(rd_mols, atom_features=None, bond_features=None):
    rd_mol_bytes = ({'rd_mol': rd_mol.ToBinary(),
                     'atom_features': atom_features,
                     'bond_features': bond_features,} for rd_mol in rd_mols)
                     #'smiles': smiles} for rd_mol, smiles in zip(rd_mols, smiles_list))
    #return [graph for graph in tqdm(map(mol2graph, rd_mol_bytes), desc='Running mol2graph', total=len(rd_mols))]
    with multiprocessing.Pool() as pool:
        for graph in tqdm(pool.imap(mol2graph, rd_mol_bytes), desc='Running mol2graph', total=len(rd_mols)):
            yield graph

def mols2pickled_graphs(rd_mols, output_dir, atom_features=None, bond_features=None):
                     #'smiles': smiles} for rd_mol, smiles in zip(rd_mols, smiles_list))
    graphs_paths = []
    with multiprocessing.Pool() as pool:
        k = int(np.ceil(np.log10(len(rd_mols))))
        path_pattern = f'graph_{{i:0{k}}}'
        work_packages = []
        for i, rd_mol in enumerate(rd_mols):
            graph_output_path = output_dir / path_pattern.format(i=i)
            work_package = {'rd_mol': rd_mol.ToBinary(),
                            'atom_features': atom_features,
                            'bond_features': bond_features,
                            'path': graph_output_path}
            graphs_paths.append(graph_output_path)
            work_packages.append(work_package)
        # We just make this a loop to simply keep track of progress
        for i in tqdm(pool.imap_unordered(mol2pickled_graph, work_packages), desc='Processing graphs', total=len(work_packages)):
            pass  
    return graphs_paths
    

class MolecularGraphDatasetConfig(RDKitMolDatasetConfig):
    def __init__(self, *args, atom_features=None, bond_features=None, cache_graphs=True, **kwargs):
        super(MolecularGraphDatasetConfig, self).__init__(*args, **kwargs)
        if atom_features is None:
            atom_features = ATOM_FEATURES
        if bond_features is None:
            bond_features = BOND_FEATURES

        self.atom_features = atom_features
        self.bond_features = bond_features
        self.cache_graphs = cache_graphs


class MolecularGraphDataset(RDKitMolDataset):
    def __init__(self, *args, graphs_paths: List[Path], config: MolecularGraphDatasetConfig, **kwargs):
        super(MolecularGraphDataset, self).__init__(*args, config=config, **kwargs)
        self.config = config
        self.graphs_paths = graphs_paths
        self.atom_feature_spec = config.atom_features
        self.bond_feature_spec = config.bond_features
        self.cache_graphs = config.cache_graphs
        
        if self.indices is not None:
            new_graph_paths = [self.graphs_paths[i] for i in self.indices]
            self.graphs_paths = new_graph_paths
        
        if self.cache_graphs:
            self.graphs = [None for i in range(len(graphs_paths))]

    def make_copy(self, dataset_spec=None, identifier=None, tag=None):
        "Return a copy of this dataset"
        if dataset_spec is None:
            dataset_spec = copy.deepcopy(self.dataset_spec)
        dataset_class = type(self)
        dataset_copy = dataset_class(graphs_paths=copy.deepcopy(self.graphs_paths),
                                     molecules = copy.deepcopy(self.molecules), 
                                     properties = copy.deepcopy(self.properties), 
                                     target_lists = copy.deepcopy(self.target_lists), 
                                     config = copy.deepcopy(self.config),
                                     dataset_spec = dataset_spec,
                                     identifier = identifier,
                                     tag=tag)
        return dataset_copy


    def __getitem__(self, item):
        if self.cache_graphs:
            graph = self.graphs[item]
            if graph is not None:
                return graph
        
        graph_path = self.graphs_paths[item]
        with open(graph_path, 'rb') as fp:
            graph = pickle.load(graph_path)
        
        if self.cache_graphs:
            self.graphs[item] = graph
        
        return graph
        

    def get_edge_feature_spec(self):
        return self.bond_feature_spec

    def get_node_feature_spec(self):
        return self.atom_feature_spec

    @classmethod
    def from_dataset_spec(cls, *, dataset_spec: DatasetSpec,
                          config: MolecularGraphDatasetConfig=None, 
                          tag=None, **kwargs):
        if config is None:
            config = MolecularGraphDatasetConfig()
        file_name = dataset_spec.file_name
        base_name = file_name.with_suffix('').name
        
        dataset_dir = file_name.parent / f'{base_name}_graph_dataset'
        datafields_path = dataset_dir / 'dataset_fields.pkl'
        pickled_graphs_path = dataset_dir / 'pickled_graphs'
        
        if datafields_path.exists():
            with open(datafields_path, 'rb') as fp:
                db = pickle.load(fp)
                molecules = db['molecules']
                properties = db['properties']
                target_lists = db['target_lists']
                graphs_paths = db['graphs_paths']
        else:
            dataset_dir.mkdir(exist_ok=True)
            pickled_graphs_path.mkdir(exist_ok=True)

            rdkit_dataset = RDKitMolDataset.from_dataset_spec(dataset_spec=dataset_spec)
            molecules = rdkit_dataset.molecules
            properties = rdkit_dataset.properties
            target_lists = rdkit_dataset.target_lists
            graphs_paths = mols2pickled_graphs(molecules, pickled_graphs_path)
            
            with open(datafields_path, 'wb') as fp:
                db = dict()
                db['molecules'] = molecules
                db['properties'] = properties
                db['target_lists'] = target_lists
                db['graphs_paths'] = graphs_paths
                pickle.dump(db, fp)
                
                

                
            

        return cls(molecules=molecules,
                   properties=properties,
                   target_lists=target_lists,
                   config=config,
                   dataset_spec=dataset_spec,
                   graphs_paths=graphs_paths,
                   tag=tag,
                   **kwargs)

