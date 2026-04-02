#!/bin/bash
# Example 1: Pack-only mode with single PDB and single chain
# This example takes a PDB file and repacks it with a provided sequence
python run.py \
    --pack_only 1 \
    --pdb_path "./inputs/1BC8.pdb" \
    --input_sequences "./inputs/test_pack_only.fasta" \
    --out_folder "./outputs/pack_only_test" \
    --number_of_packs_per_design 4 \
    --pack_with_ligand_context 1 \
    --seed 42

# Example 2: Pack-only mode without ligand context (protein-only packing)
python run.py \
    --pack_only 1 \
    --pdb_path "./inputs/1BC8.pdb" \
    --input_sequences "./inputs/test_pack_only.fasta" \
    --out_folder "./outputs/pack_only_no_context" \
    --number_of_packs_per_design 4 \
    --pack_with_ligand_context 0 \
    --seed 42

# Example 3: Pack-only batch mode with multiple PDBs
# First create the JSON mapping files
cat > /tmp/pack_only_pdbs.json << 'EOF'
{
    "./inputs/1BC8.pdb": ""
}
EOF

cat > /tmp/pack_only_sequences.json << 'EOF'
{
    "./inputs/1BC8.pdb": "./inputs/test_pack_only.fasta"
}
EOF

python run.py \
    --pack_only 1 \
    --pdb_path_multi "/tmp/pack_only_pdbs.json" \
    --input_sequences_multi "/tmp/pack_only_sequences.json" \
    --out_folder "./outputs/pack_only_batch" \
    --number_of_packs_per_design 4 \
    --seed 42
