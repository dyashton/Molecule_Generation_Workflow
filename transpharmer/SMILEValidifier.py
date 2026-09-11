from rdkit import Chem
import os
import sys
import argparse

total_valid = 0
total_invalid = 0
total_smiles = 0

invalid_smiles = set()

def validate_smiles_from_file(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        sys.exit(1)
    with open(input_file, 'r') as infile, open(output_file, 'w') as valid_outfile:
        valid_outfile.write("smiles\n")
        for line in infile:
            smi = line.strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_outfile.write(smi + '\n')
                global total_valid
                total_valid += 1
            else:
                invalid_smiles.add(smi)
                global total_invalid
                total_invalid += 1
            global total_smiles
            total_smiles += 1
    print(f"Valid%: {total_valid / total_smiles * 100 if total_smiles > 0 else 0:.2f}%")
    print(f"Invalid%: {total_invalid / total_smiles * 100 if total_smiles > 0 else 0:.2f}%")
    print(f"Total SMILES processed: {total_smiles}")
    if invalid_smiles:
        print("Invalid SMILES found:")
        for smi in invalid_smiles:
            print(smi)
    else:
        print("No invalid SMILES found.")

def validate_smiles(smiles):
    """Filter a sequence of SMILES to those RDKit can parse.

    Pass a list/tuple (not a single string). For one SMILES use ``valid_smiles``.
    """
    if isinstance(smiles, (str, bytes)):
        raise TypeError(
            "validate_smiles expects a list of SMILES; "
            "for a single string use valid_smiles()"
        )
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out.append(smi)
    return out

def valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

valid_atoms = {'C', 'O', 'N', 'S', 'Cl', 'Br'}
def atom_check(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES")
        return False
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    # print("Atoms in molecule:", atoms)
    atoms_set = set(atoms)

    valid = atoms_set.issubset(valid_atoms)
    if not valid:
        print("Contains invalid atoms:", atoms_set - valid_atoms)
        return False
    halogen_count = atoms.count('Cl') + atoms.count('Br')
    if halogen_count > 2:
        print("More than 2 halogen atoms found.")
        return False
    return True


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, required=True, help='Input SMILES file')
    # parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file')
    # args = parser.parse_args()
    atom_check("CCCCCCCC(=O)OCC(Br)=C(Br)Br")  # Example usage
    # validate_smiles_from_file(args.input, args.output)
