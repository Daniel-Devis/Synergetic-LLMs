import os
import json
from pathlib import Path

# Directory containing your JSON or JSONL files
data_dir = Path(r"D:/danie/Documents/CSC-40040 19020322 code/test500")
output_file = Path("test case ref.jsonl")

# Helper to load JSON records from a file: handles both JSON array and JSONL
def load_json_records(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding='utf-8')
        stripped = text.lstrip()
        # JSON array
        if stripped.startswith('['):
            records = json.loads(text)
            if not isinstance(records, list):
                raise ValueError("Top-level JSON is not a list")
            return records
        # Single JSON object (not an array)
        if stripped.startswith('{') and '\n' not in stripped:
            return [json.loads(text)]
    except json.JSONDecodeError:
        pass
    # Fallback: parse as JSONL (one object per line)
    records = []
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error in {path.name} at line {idx}: {e}")
    return records

# Extract focal entries from loaded JSON records
def extract_focal_entries(directory: Path) -> list[dict]:
    extracted = []
    files_checked = 0

    for file_path in directory.rglob('*.json'):
        files_checked += 1
        print(f"🔍 Checking file: {file_path.name}")

        records = load_json_records(file_path)
        if not records:
            print("🚫 No JSON records found.")
            continue

        # for record in records:
        #     mapped = record.get('MappedTestCase') if isinstance(record, dict) else None
        #     if not mapped:
        #         continue

        #     focal_class = mapped.get('focal_class')
        #     focal_method = mapped.get('focal_method')
        #     if focal_class and focal_method:
        #         print(f"✅ Found focal_class={focal_class}, method_id={focal_method.get('identifier')}")
        #         extracted.append({
        #             'focal_class': focal_class,
        #             'focal_method': focal_method
        #         })
        for record in records:
            mapped = record.get('MappedTestCase') if isinstance(record, dict) else None
            if not mapped:
                continue

            focal_class = mapped.get('test_class')
            focal_method = mapped.get('test_case')
            if focal_class and focal_method:
                print(f"✅ Found focal_class={focal_class}, method_id={focal_method.get('identifier')}")
                extracted.append({
                    'test_class': focal_class,
                    'test_case': focal_method
                })

    print(f"\n📊 Scanned {files_checked} files, extracted {len(extracted)} entries.")
    return extracted

# Save output as JSONL
def save_to_jsonl(data: list[dict], output_path: Path):
    with output_path.open('w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    if not data_dir.exists():
        print(f"❌ ERROR: Directory does not exist → {data_dir}")
        exit(1)
    print(f"✅ Directory exists. Scanning {data_dir}...")

    results = extract_focal_entries(data_dir)
    save_to_jsonl(results, output_file)
    print(f"\n✅ Done! Saved {len(results)} entries → {output_file}")
