import json
import csv
import re
from codebleu import calc_codebleu

def ensure_str(x):
    """
    Make sure we always return a code‐string:
      - If x is already a str, return it.
      - If it's a dict, try common fields, else json.dumps it.
      - Otherwise str(x).
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ('completion', 'code', 'method'):
            if key in x and isinstance(x[key], str):
                return x[key]
        # fallback: dump the whole dict as a string
        return json.dumps(x)
    return str(x)

def eval1(reference, test_case):
    """
    Compute the CodeBLEU score for a single reference and test case.
    """
    # ensure both args are pure strings
    ref_str = ensure_str(reference)
    tc_str  = ensure_str(test_case)

    result = calc_codebleu(
        [ref_str],
        [tc_str],
        lang="java",
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=None
    )
    return result['codebleu']

def tokenize(code):
    """
    Simple word‐tokenizer for calculating precision/recall on token overlap.
    """
    # code is now guaranteed to be a str
    return re.findall(r'\b\w+\b', code)

def calculate_metrics(reference, test_case):
    """
    Compute token‐based precision, recall, and F1 between reference and test_case.
    """
    ref_str  = ensure_str(reference)
    tc_str   = ensure_str(test_case)

    ref_tokens  = set(tokenize(ref_str))
    test_tokens = set(tokenize(tc_str))

    tp = len(ref_tokens & test_tokens)
    fp = len(test_tokens - ref_tokens)
    fn = len(ref_tokens - test_tokens)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return precision, recall, f1

def split_test_cases_from_string(test_str):
    """
    Split a single string containing multiple @Test methods
    into a list of individual test‐method strings.
    """
    parts = test_str.split("@Test")
    return ["@Test" + p for p in parts if p.strip()]

def normalize_test_cases(tc_field):
    """
    Normalize the `predicted_test_cases` field into a flat list of test‐case strings,
    whether it was originally a list, a dict, or a single big string.
    """
    # if it's a dict, extract string and then split
    if isinstance(tc_field, dict):
        return normalize_test_cases(ensure_str(tc_field))

    if isinstance(tc_field, list):
        # could be list of dicts or strings
        flat = []
        for elt in tc_field:
            elt_str = ensure_str(elt)
            # if it's a big chunk with many @Test, split it further
            if "@Test" in elt_str:
                flat.extend(split_test_cases_from_string(elt_str))
            else:
                flat.append(elt_str)
        return flat

    if isinstance(tc_field, str):
        # one big string: split on @Test
        return split_test_cases_from_string(tc_field)

    # anything else—return empty list so we skip it
    return []

def evaluate_test_cases(reference_file, test_case_file, output_file):
    """
    Main evaluation loop:
      1) Load all predicted test cases from JSON array.
      2) Stream through the reference JSONL file line by line.
      3) For each pair, extract & normalize test cases.
      4) Compute CodeBLEU + token metrics on each sub‐test.
      5) Write results out to a CSV.
    """
    # 1) Load the entire predicted JSON array
    with open(test_case_file, 'r', encoding='utf-8') as f_tst:
        predicted_data = json.load(f_tst)

    # 2) Open reference file (JSONL) and CSV writer
    with open(reference_file, 'r', encoding='utf-8') as f_ref, \
         open(output_file, 'w', encoding='utf-8', newline='') as f_out:

        writer = csv.writer(f_out)
        writer.writerow([
            'Pair Index',
            'Test Case Index',
            'CodeBLEU Score',
            'Precision',
            'Recall',
            'F1 Score'
        ])

        for idx, ref_line in enumerate(f_ref):
            try:
                ref_obj  = json.loads(ref_line)
                pred_obj = predicted_data[idx]
            except json.JSONDecodeError as e:
                print(f"[Line {idx+1}] invalid JSON in reference file: {e}")
                continue
            except IndexError:
                print(f"[Line {idx+1}] no matching prediction (index out of range); stopping.")
                break

            focal = ref_obj.get('test_case')
            preds = pred_obj.get('review_feedback')

            # Check for missing fields
            if focal is None or preds is None:
                missing = []
                if focal is None: missing.append('focal_method')
                if preds  is None: missing.append('predicted_test_cases')
                print(f"[Line {idx+1}] missing field(s): {', '.join(missing)}; skipping.")
                continue

            # Split or normalize into individual test cases
            test_list = normalize_test_cases(preds)
            if not test_list:
                print(f"[Line {idx+1}] extracted zero test cases; skipping.")
                continue

            # Evaluate each test case
            for j, single_test in enumerate(test_list, start=1):
                cb_score = eval1(focal, single_test)
                p, r, f1 = calculate_metrics(focal, single_test)
                writer.writerow([idx+1, j, cb_score, p, r, f1])

if __name__ == "__main__":
    # Example usage: adjust paths as needed
    evaluate_test_cases(
        reference_file  = r'D:/danie/Documents/CSC-40040 19020322 code/3.5Finetune/test/test case ref.jsonl',
        test_case_file  = r'D:/danie/Documents/Disso/data/data/4omini/4ominireviewed_test_cases.json',
        output_file     = r'D:/danie/Documents/Disso/data/data/4omini/4ominireviewed_test_cases.csv'
    )
