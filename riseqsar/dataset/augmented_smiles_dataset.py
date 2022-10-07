import rdkit.Chem as rdc
from riseqsar.dataset.rdkit_dataset import RDKitMolDataset, shuffle_atoms

class AugmentedSmilesDataset(RDKitMolDataset):
    def __getitem__(self, item):
        rd_mol, *targets = RDKitMolDataset.get_smiles(self, item)
        rd_mol_rnd = shuffle_atoms(rd_mol, self.rng)
        smiles_rnd = rdc.MolToSmiles(rd_mol_rnd, canonical=False)
        return smiles_rnd, *targets
