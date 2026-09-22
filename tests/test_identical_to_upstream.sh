#!/usr/bin/env bash
# Verify that with no constraint flags this branch produces byte-identical
# output to the base revision, for every model type.
#
#   bash tests/test_identical_to_upstream.sh [base-revision]
#
# The base revision defaults to "main".  The check runs each model type twice
# at a fixed seed -- once from a checkout of the base revision, once from the
# working tree -- and compares the FASTA files byte for byte.
set -euo pipefail

BASE="${1:-main}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Materialise the base revision's Python sources next to symlinks for the
# large, unchanged directories.
mkdir -p "$WORK/base"
for f in run.py model_utils.py data_utils.py sc_utils.py score.py; do
    git -C "$REPO" show "$BASE:$f" > "$WORK/base/$f"
done
ln -s "$REPO/openfold" "$WORK/base/openfold"

declare -A CKPT=(
    [protein_mpnn]=proteinmpnn_v_48_020.pt
    [soluble_mpnn]=solublempnn_v_48_020.pt
    [ligand_mpnn]=ligandmpnn_v_32_010_25.pt
    [global_label_membrane_mpnn]=global_label_membrane_mpnn_v_48_020.pt
    [per_residue_label_membrane_mpnn]=per_residue_label_membrane_mpnn_v_48_020.pt
)

pass=0
fail=0
for model in "${!CKPT[@]}"; do
    for pdb in 1BC8 2GFB; do
        for side in base tree; do
            dir="$WORK/base"
            [ "$side" = tree ] && dir="$REPO"
            out="$WORK/out_${side}_${model}_${pdb}"
            ( cd "$dir" && PYTHONPATH="$dir" PYTHONSAFEPATH= "$PYTHON" run.py \
                --model_type "$model" \
                --checkpoint_"$model" "$REPO/model_params/${CKPT[$model]}" \
                --pdb_path "$REPO/inputs/$pdb.pdb" \
                --out_folder "$out" \
                --batch_size 3 --number_of_batches 2 --seed 77 --verbose 0 \
                >/dev/null 2>&1 )
        done
        a="$WORK/out_base_${model}_${pdb}/seqs/$pdb.fa"
        b="$WORK/out_tree_${model}_${pdb}/seqs/$pdb.fa"
        if [ -s "$a" ] && cmp -s "$a" "$b"; then
            echo "IDENTICAL  $model $pdb"
            pass=$((pass + 1))
        else
            echo "DIFFERS    $model $pdb"
            diff "$a" "$b" | head -5 || true
            fail=$((fail + 1))
        fi
    done
done

echo
echo "identical: $pass   differing: $fail"
[ "$fail" -eq 0 ]
