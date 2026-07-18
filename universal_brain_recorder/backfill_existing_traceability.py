from pathlib import Path

from traceability_common import PROJECT_ROOT, SIDECAR_ROOT, derived_dataset_traceability, rebuild_catalog, standalone_traceability, standard_recording_traceability, write_json


OVERWRITE_INFERRED = True
INCLUDE_EXPERIMENT_DATA = True
INCLUDE_STANDALONE_ROOT_RECORDING = True


def should_write(path):
    if not path.exists():
        return True
    if not OVERWRITE_INFERRED:
        return False
    existing = path.read_text(encoding="utf-8")
    return '"traceability_status": "user_annotated"' not in existing


def main():
    roots = [PROJECT_ROOT / "sessions"]
    if INCLUDE_EXPERIMENT_DATA:
        roots.extend([])
    standard_dirs = set()
    for root in roots:
        if root.exists():
            standard_dirs.update(path.parent for path in root.rglob("eeg_samples.csv"))
    written = []
    for session_dir in sorted(standard_dirs, key=str):
        output_path = session_dir / "traceability.json"
        if should_write(output_path):
            write_json(output_path, standard_recording_traceability(session_dir))
            written.append(output_path)
    for manifest_path in sorted((PROJECT_ROOT / "recordings").glob("*/manifest.json")):
        session_dir = manifest_path.parent
        if session_dir in standard_dirs:
            continue
        csv_files = list(session_dir.glob("*.csv"))
        if len(csv_files) == 0:
            continue
        output_path = session_dir / "traceability.json"
        if should_write(output_path):
            write_json(output_path, derived_dataset_traceability(session_dir))
            written.append(output_path)
    if INCLUDE_STANDALONE_ROOT_RECORDING:
        csv_path = PROJECT_ROOT / "brainaccess_maxi_32ch.csv"
        if csv_path.exists():
            SIDECAR_ROOT.mkdir(parents=True, exist_ok=True)
            output_path = SIDECAR_ROOT / "brainaccess_maxi_32ch.traceability.json"
            if should_write(output_path):
                write_json(output_path, standalone_traceability(csv_path))
                written.append(output_path)
    catalog_csv, catalog_json = rebuild_catalog()
    print("Traceability sidecars written:", len(written))
    for path in written:
        print(path)
    print("Catalog CSV:", catalog_csv)
    print("Catalog JSON:", catalog_json)


if __name__ == "__main__":
    main()
