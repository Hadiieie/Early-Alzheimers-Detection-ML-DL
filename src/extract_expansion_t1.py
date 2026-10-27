from pathlib import Path
import tarfile
import pandas as pd

root = Path(__file__).resolve().parents[1]
audit = pd.read_csv(root / "oasis_expansion_source_audit.csv")
out = root / "OASIS_EXPANSION_T1"

wanted = {}

for row in audit.itertuples():
    base = row.Source
    wanted[base + ".hdr"] = (row.Subject, "T1.hdr")
    wanted[base + ".img"] = (row.Subject, "T1.img")

found = 0

for archive in sorted((root / "OASIS_EXPANSION_RAW").glob("*.tar.gz")):
    print("Reading:", archive.name)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name not in wanted:
                continue

            subject, filename = wanted[member.name]
            folder = out / subject
            folder.mkdir(parents=True, exist_ok=True)

            source = tar.extractfile(member)

            with open(folder / filename, "wb") as f:
                f.write(source.read())

            found += 1


subjects = list(out.glob("OAS1_*"))

print("\nFiles extracted:", found)
print("Subjects:", len(subjects))

bad = [
    p.name for p in subjects
    if not (p / "T1.hdr").exists()
    or not (p / "T1.img").exists()
]

print("Incomplete subjects:", len(bad))