"""Fetch and clean the backbone panel used by the constrained-decoding benchmarks.

Sixteen small monomeric domains spanning 20-107 residues and all three broad
fold classes, so a claim rests on a panel rather than on one protein.  Each
entry is reduced to the first model, its largest protein chain, standard
residues only, single altloc.  Chain id is rewritten to A so downstream
commands are uniform.

    python benchmarks/fetch_panel.py
"""

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "structures", "panel")

# (pdb id, approximate length, fold class) -- class is documentation only.
PANEL = [
    ("1L2Y", 20, "alpha"), ("1VII", 36, "alpha"), ("1PIN", 153, "alpha+beta"),
    ("1ENH", 54, "alpha"), ("1PGA", 56, "alpha+beta"), ("1BDD", 60, "alpha"),
    ("1SHG", 62, "beta"), ("1CSP", 67, "beta"), ("1MJC", 69, "beta"),
    ("1UBQ", 76, "alpha+beta"), ("2PTL", 78, "alpha+beta"), ("2CI2", 83, "alpha+beta"),
    ("1LMB", 87, "alpha"), ("1TEN", 90, "beta"), ("1RIS", 97, "alpha+beta"),
    ("1FKB", 107, "alpha+beta"),
]

AA3 = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}


def clean(raw):
    """First model, largest protein chain, standard residues, one altloc."""
    lines, in_model, seen_model = [], True, False
    for line in raw.splitlines(True):
        if line.startswith("MODEL"):
            seen_model = True
            in_model = line.split()[1] == "1" if len(line.split()) > 1 else True
            continue
        if line.startswith("ENDMDL"):
            in_model = False
            continue
        if not in_model or not line.startswith("ATOM"):
            continue
        if line[17:20].strip() not in AA3:
            continue
        if line[16] not in (" ", "A"):
            continue
        lines.append(line)
    by_chain = {}
    for line in lines:
        by_chain.setdefault(line[21], []).append(line)
    if not by_chain:
        raise ValueError("no protein atoms found")
    best = max(by_chain, key=lambda c: len({l[22:27] for l in by_chain[c]}))
    kept = [l[:21] + "A" + l[22:] for l in by_chain[best]]
    n_res = len({l[22:27] for l in kept})
    return kept, n_res, best, seen_model


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for pid, approx, fold in PANEL:
        dest = os.path.join(OUT, f"{pid}.pdb")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            with open(dest) as fh:
                raw = fh.read()
            source = "existing"
        else:
            with urllib.request.urlopen(
                f"https://files.rcsb.org/download/{pid}.pdb", timeout=60
            ) as response:
                raw = response.read().decode()
            source = "downloaded"
        kept, n_res, chain, nmr = clean(raw)
        if source == "downloaded":
            with open(dest, "w") as fh:
                fh.writelines(kept)
                fh.write("END\n")
        rows.append((pid, n_res, approx, fold, chain, nmr))
        print(f"{pid}  {source:9s}  {n_res:4d} res (expected ~{approx})  chain {chain}  "
              f"{'NMR' if nmr else 'X-ray'}  {fold}")
    print(f"\n{len(rows)} structures -> {os.path.relpath(OUT, HERE)}")
    print("length range: %d-%d" % (min(r[1] for r in rows), max(r[1] for r in rows)))


if __name__ == "__main__":
    main()
