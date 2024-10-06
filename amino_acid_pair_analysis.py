import os
import glob
import gzip
import random
import numpy as np
from Bio.PDB import PDBParser
from collections import defaultdict
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Mapping from three-letter to one-letter amino acid codes
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # Add non-standard amino acids if necessary
}


def extract_sequence_and_coords(pdb_file):
    """
    Extracts the amino acid sequence and average coordinates from a PDB file.

    Args:
        pdb_file (str): Path to the gzipped PDB file.

    Returns:
        tuple: (sequence list, coordinates list)
    """
    parser = PDBParser(QUIET=True)
    sequence = []
    coordinates = []

    try:
        with gzip.open(pdb_file, 'rt') as file:
            structure = parser.get_structure('protein', file)
    except Exception as e:
        logging.error(f"Failed to parse {pdb_file}: {e}")
        return sequence, coordinates

    for model in structure:
        for chain in model:
            for residue in chain:
                # Filter out hetero residues (HETATM) and water molecules
                if residue.get_id()[0] != ' ':
                    continue

                amino_acid = residue.get_resname()
                one_letter = THREE_TO_ONE.get(amino_acid, 'X')  # 'X' for unknown
                if one_letter == 'X':
                    logging.warning(f"Unknown amino acid '{amino_acid}' in {pdb_file}, residue {residue.get_id()}")
                sequence.append(one_letter)

                # Extract atom coordinates
                atom_coords = [atom.get_coord() for atom in residue if atom.get_coord().size == 3]
                if atom_coords:
                    avg_coords = np.mean(atom_coords, axis=0)
                    coordinates.append(avg_coords)
                else:
                    # Assign NaN if no coordinates are present
                    coordinates.append(np.array([np.nan, np.nan, np.nan]))

    logging.info(f"Extracted {len(sequence)} residues from {pdb_file}")
    return sequence, coordinates

def randomize_sequence(sequence):
    """
    Returns a shuffled copy of the input sequence.

    Args:
        sequence (list): Original amino acid sequence.

    Returns:
        list: Shuffled amino acid sequence.
    """
    random_sequence = sequence.copy()
    random.shuffle(random_sequence)
    return random_sequence

def calculate_proximity(sequence, coordinates, sequence_gap=100, distance_threshold=10):
    """
    Calculates the frequency of amino acid pairs that are at least `sequence_gap` residues apart
    and within `distance_threshold` Angstroms in 3D space.

    Args:
        sequence (list): Amino acid sequence.
        coordinates (list): List of average coordinates per residue.
        sequence_gap (int): Minimum number of residues separating the pair.
        distance_threshold (float): Maximum distance in Angstroms to consider proximity.

    Returns:
        defaultdict: Counts of amino acid pairs.
    """
    pairs = defaultdict(int)
    total_length = len(sequence)

    for i in range(total_length):
        j_start = i + sequence_gap
        if j_start >= total_length:
            continue

        for j in range(j_start, total_length):
            # Skip if either residue lacks valid coordinates
            if np.isnan(coordinates[i]).any() or np.isnan(coordinates[j]).any():
                continue

            # Calculate Euclidean distance
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            if dist <= distance_threshold:
                aa1 = sequence[i]
                aa2 = sequence[j]
                pair = tuple(sorted((aa1, aa2)))  # Sort to handle unordered pairs (A,B) == (B,A)
                pairs[pair] += 1

        # Optional: Log progress every 10,000 residues
        if (i + 1) % 10000 == 0:
            logging.info(f"Processed {i + 1}/{total_length} residues for proximity")

    return pairs

def analyze_multiple_pdbs(pdb_dir, output_file, randomizations=100, sequence_gap=100, distance_threshold=10):
    """
    Analyzes multiple PDB files to find amino acid pairs that are proximal in 3D space
    but separated by a significant sequence gap, comparing against randomized sequences.

    Args:
        pdb_dir (str): Directory containing gzipped PDB files.
        output_file (str): Path to the output results file.
        randomizations (int): Number of random sequence shuffles per PDB.
        sequence_gap (int): Minimum number of residues separating the pair.
        distance_threshold (float): Maximum distance in Angstroms to consider proximity.
    """
    pdb_files = glob.glob(os.path.join(pdb_dir, '*.pdb.gz'))
    logging.info(f"Found {len(pdb_files)} PDB files in '{pdb_dir}'")

    if not pdb_files:
        logging.error("No PDB files found. Please check the directory path and file extensions.")
        return

    all_observed_pairs = defaultdict(int)
    random_pair_counts = defaultdict(list)

    for idx, pdb_file in enumerate(pdb_files, 1):
        pdb_basename = os.path.basename(pdb_file)
        logging.info(f"Processing file {idx}/{len(pdb_files)}: {pdb_basename}")

        sequence, coordinates = extract_sequence_and_coords(pdb_file)

        if not sequence:
            logging.warning(f"No sequence extracted from {pdb_basename}. Skipping.")
            continue

        if len(sequence) != len(coordinates):
            logging.warning(f"Sequence and coordinates length mismatch in {pdb_basename}. Skipping.")
            continue

        # Calculate observed pairs
        observed_pairs = calculate_proximity(sequence, coordinates, sequence_gap, distance_threshold)
        for pair, count in observed_pairs.items():
            all_observed_pairs[pair] += count

        # Perform randomizations
        for rand in range(1, randomizations + 1):
            random_sequence = randomize_sequence(sequence)
            random_pairs = calculate_proximity(random_sequence, coordinates, sequence_gap, distance_threshold)
            for pair, count in random_pairs.items():
                random_pair_counts[pair].append(count)

            # Optional: Log progress every 10 randomizations
            if rand % 10 == 0:
                logging.debug(f"Completed {rand}/{randomizations} randomizations for {pdb_basename}")

    # Write results to the output file
    try:
        with open(output_file, 'w') as f:
            header = "AA1\tAA2\tObs_Count\tMean_Random_Count\tStdDev_Random_Count\tZ-Score\n"
            f.write(header)
            for pair, observed_count in all_observed_pairs.items():
                random_counts = random_pair_counts.get(pair, [0] * randomizations)
                mean_random_count = np.mean(random_counts)
                std_random_count = np.std(random_counts)

                if std_random_count == 0:
                    z_score = 'NA'
                else:
                    z_score = (observed_count - mean_random_count) / std_random_count

                line = f"{pair[0]}\t{pair[1]}\t{observed_count}\t{mean_random_count:.2f}\t{std_random_count:.2f}\t{z_score}\n"
                f.write(line)
        logging.info(f"Analysis complete. Results saved to '{output_file}'")
    except Exception as e:
        logging.error(f"Failed to write results to '{output_file}': {e}")


if __name__ == "__main__":
    # Update the pdb_dir to your actual data directory
    pdb_dir = "/Users/iridashyti/Documents/Duke/Fall2024/Genome tools and technologies/hw/test"
    output_file = "amino_acid_pair_analysis_v2.txt"

    # Optional: Validate input directory
    if not os.path.isdir(pdb_dir):
        logging.error(f"The specified PDB directory '{pdb_dir}' does not exist.")
    else:
        analyze_multiple_pdbs(
            pdb_dir=pdb_dir,
            output_file=output_file,
            randomizations=100,        # You can reduce this number for quicker testing
            sequence_gap=100,
            distance_threshold=10
        )
