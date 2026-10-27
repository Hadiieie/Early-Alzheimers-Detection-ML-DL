from pathlib import Path
import tarfile, re
import pandas as pd

root = Path(__file__).resolve().parents[1]
ids = set((root / "oasis_download_ids.txt").read_text().splitlines())
archives = sorted((root / "OASIS_EXPANSION_RAW").glob("*.tar.gz"))

found = {}

for archive in archives:
    with tarfile.open(archive, "r:gz") as tar:
        for name in tar.getnames():
            m = re.search(r"(OAS1_\d{4}_MR1)", name)
            if not m or m.group(1) not in ids:
                continue

            if "/PROCESSED/MPRAGE/SUBJ_111/" in name and name.endswith((".hdr", ".img")):
                sid = m.group(1)
                base = name[:-4]
                found.setdefault(sid, {}).setdefault(base, set()).add(name[-3:])

rows = []

for sid in sorted(ids):
    pairs = [
        base for base, exts in found.get(sid, {}).items()
        if {"hdr", "img"} <= exts
    ]

    status = "PASS" if len(pairs) == 1 else ("MISSING" if len(pairs) == 0 else "AMBIGUOUS")
    rows.append([sid, status, pairs[0] if len(pairs) == 1 else ""])

df = pd.DataFrame(rows, columns=["Subject", "Status", "Source"])
df.to_csv(root / "oasis_expansion_source_audit.csv", index=False)

print("Wanted:", len(df))
print(df["Status"].value_counts())