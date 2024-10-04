import os
import gzip
from Bio.PDB import PDBParser, is_aa
from Bio.Data.IUPACData import protein_letters_3to1
import numpy as np
from collections import defaultdict

def process_pdb_file(pdb_file_path):
    parser = PDBParser(QUIET=True)
    if pdb_file_path.endswith('.gz'):
        with gzip.open(pdb_file_path, 'rt') as handle:
            structure = parser.get_structure('X', handle)
    else:
        structure = parser.get_structure('X', pdb_file_path)
    residues = []
    residue_index = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    atoms = list(residue.get_atoms())
                    coords = np.array([atom.get_coord() for atom in atoms])
                    avg_coord = np.mean(coords, axis=0)
                    residue_number = residue.get_id()[1]
                    residue_name = residue.get_resname()
                    try:
                        residue_letter = protein_letters_3to1[residue_name.upper()]
                    except KeyError:
                        continue  # Skip non-standard amino acids
                    residues.append((residue_index, residue_number, residue_letter, avg_coord))
                    residue_index += 1
    return residues

def process_residues(residues):
    near_pairs = defaultdict(int)
    total_pairs = defaultdict(int)
    n = len(residues)
    for i in range(n):
        res_i = residues[i]
        for j in range(i+100, n):  # Residues at least 100 apart in sequence
            res_j = residues[j]
            dist = np.linalg.norm(res_i[3] - res_j[3])
            pair = (res_i[2], res_j[2])  # (residue_letter_i, residue_letter_j)
            total_pairs[pair] += 1
            if dist < 10.0:
                near_pairs[pair] += 1
    return near_pairs, total_pairs

def main():
    pdb_dir = "YEAST_PDBs"  # TODO replace this with the directory
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb.gz") or f.endswith(".pdb")]

    total_near_pairs = defaultdict(int)
    total_pairs = defaultdict(int)
    aa_counts = defaultdict(int)

    for pdb_file in pdb_files:
        pdb_file_path = os.path.join(pdb_dir, pdb_file)
        print(f"Processing {pdb_file}...")
        residues = process_pdb_file(pdb_file_path)
        for res in residues:
            aa_counts[res[2]] += 1  # res[2] is residue_letter
        near_pairs, pairs = process_residues(residues)
        for pair, count in near_pairs.items():
            total_near_pairs[pair] += count
        for pair, count in pairs.items():
            total_pairs[pair] += count

    # Compute amino acid frequencies
    total_aa_count = sum(aa_counts.values())
    aa_freqs = {aa: count / total_aa_count for aa, count in aa_counts.items()}

    # Compute expected counts
    total_pairs_count = sum(total_pairs.values())
    expected_pairs = {}
    for aa1 in aa_freqs:
        for aa2 in aa_freqs:
            expected_count = total_pairs_count * aa_freqs[aa1] * aa_freqs[aa2]
            expected_pairs[(aa1, aa2)] = expected_count

    # Write results to a CSV file
    import csv
    with open('amino_acid_pair_counts.csv', 'w', newline='') as csvfile:
        fieldnames = ['AminoAcid1', 'AminoAcid2', 'ObservedCount', 'ExpectedCount', 'Obs_Exp_Ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pair in expected_pairs:
            observed = total_near_pairs.get(pair, 0)
            expected = expected_pairs[pair]
            ratio = observed / expected if expected > 0 else 0
            writer.writerow({'AminoAcid1': pair[0],
                             'AminoAcid2': pair[1],
                             'ObservedCount': observed,
                             'ExpectedCount': expected,
                             'Obs_Exp_Ratio': ratio})

    print("Analysis complete. Results saved to 'amino_acid_pair_counts.csv'.")

if __name__ == "__main__":
    main()
