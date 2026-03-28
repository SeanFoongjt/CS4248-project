import argparse
import json
from pathlib import Path


FIELDS_TO_FIX = (
    "headline",
    "description",
    "author_description",
    "author_name",
    "article_section",
)


def decode_escape(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


EXACT_MAP = {
    decode_escape(r"\u7ab6\u51b1"): "'s",
    decode_escape(r"\u7ab6\u51b2"): "n't",
    decode_escape(r"\u7ab6\u51b5"): "'ve",
    decode_escape(r"\u7ab6\u51a4"): "'ll",
    decode_escape(r"\u7ab6\u51b3"): "'re",
    decode_escape(r"\u7ab6\u51a6"): "'m",
    decode_escape(r"\u7ab6\u56d8"): "'d",
    decode_escape(r"\u7ab6\u542e\u20ac\u30fb"): "”",
    decode_escape(r"\u7ab6\u6636\u20ac\u30f2"): "”",
    decode_escape(r"\u7ab6\u30fb"): '"',
    decode_escape(r"\u7ab6\u30f2"): "...",
    decode_escape(r"\u7ab6\uff66"): "...",
    decode_escape(r"\u30c6\u3002"): "a",
    decode_escape(r"\u30c6\u30a2"): "ñ",
    decode_escape(r"\u30c6\u300d"): "ã",
    decode_escape(r"\u30c6\u30a6"): "ó",
    decode_escape(r"\u30c6\u30e5"): "í",
    decode_escape(r"\u30c6\u30a5"): "é",
    decode_escape(r"\u30c6\u30b7"): "ü",
    decode_escape(r"\u30c6\u30a9"): "ë",
    decode_escape(r"\u30c6\u30a1"): "ç",
    decode_escape(r"\u30c6\u30c3"): "ï",
    decode_escape(r"\ue05e\u30fb"): "",
    decode_escape(r"\ue05e\u67f1"): "Email",
}

STARTER_MAP = {
    decode_escape(r"\u7ab6\u5f0b"): "T",
    decode_escape(r"\u7ab6\u5f29"): "W",
    decode_escape(r"\u7ab6\u5eec"): "I",
    decode_escape(r"\u7ab6\u7c98"): "S",
    decode_escape(r"\u7ab6\u4efb"): "C",
    decode_escape(r"\u7ab6\u7962"): "I",
    decode_escape(r"\u7ab6\u5c3f"): "A",
    decode_escape(r"\u7ab6\u5efe"): "U",
    decode_escape(r"\u7ab6\u4e43"): "T",
    decode_escape(r"\u7ab6\u8a8d"): "F",
    decode_escape(r"\u7ab6\u598a"): "E",
    decode_escape(r"\u7ab6\u71c3"): "E",
    decode_escape(r"\u7ab6\u676f"): "H",
    decode_escape(r"\u7ab6\u5ee9"): "E",
    decode_escape(r"\u7ab6\u5f38"): "O",
    decode_escape(r"\u7ab6\u5ee3"): "A",
    decode_escape(r"\u7ab6\u5ef4"): "A",
    decode_escape(r"\u7ab6\u57dc"): "I",
    decode_escape(r"\u7ab6\u5fcd"): "E",
    decode_escape(r"\u7ab6\u5ef0"): "L",
    decode_escape(r"\u7ab6\u807e"): "W",
    decode_escape(r"\u7ab6\u8cc2"): "G",
    decode_escape(r"\u7ab6\u8def"): "H",
    decode_escape(r"\u7ab6\u518c"): "'e",
    decode_escape(r"\u7ab6\u5a41"): "K",
}

SUSPECT_SNIPPETS = tuple(
    sorted(
        {decode_escape(r"\u7ab6"), *EXACT_MAP.keys(), *STARTER_MAP.keys()},
        key=len,
        reverse=True,
    )
)


def load_ftfy():
    try:
        from ftfy import fix_text  # type: ignore
    except Exception:
        return None
    return fix_text


def looks_suspicious(text: str) -> bool:
    return any(snippet in text for snippet in SUSPECT_SNIPPETS)


def capitalize_after_prefix(text: str, prefix: str) -> str:
    out = []
    i = 0
    while i < len(text):
        if text.startswith(prefix, i) and i + len(prefix) < len(text):
            nxt = text[i + len(prefix)]
            if "a" <= nxt <= "z":
                out.append(prefix)
                out.append(nxt.upper())
                i += len(prefix) + 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def heuristic_fix(text: str) -> str:
    if not isinstance(text, str) or not looks_suspicious(text):
        return text

    fixed = text
    for bad, good in EXACT_MAP.items():
        fixed = fixed.replace(bad, good)

    for bad, good in STARTER_MAP.items():
        fixed = fixed.replace(" " + bad, ' "' + good)
        fixed = fixed.replace("." + bad, '."' + good)
        fixed = fixed.replace("!" + bad, '!"' + good)
        fixed = fixed.replace("?" + bad, '?"' + good)
        fixed = fixed.replace("," + bad, "," + good)
        fixed = fixed.replace(":" + bad, ':"' + good)
        fixed = fixed.replace("(" + bad, '("' + good)
        fixed = fixed.replace("—" + bad, "—" + good)
        if fixed.startswith(bad):
            fixed = '"' + good + fixed[len(bad) :]

    fixed = capitalize_after_prefix(fixed, '"')
    fixed = capitalize_after_prefix(fixed, "—")
    fixed = fixed.replace('”"', '”')
    fixed = fixed.replace('""', '"')
    fixed = fixed.replace("nn't", "n't")
    fixed = fixed.replace("'vee", "'ve")
    fixed = fixed.replace("'ree", "'re")
    fixed = fixed.replace("'lle", "'ll")
    fixed = fixed.replace("'dde", "'d")
    fixed = fixed.replace(" : ", ": ")
    while "...." in fixed:
        fixed = fixed.replace("....", "...")
    return fixed


def fix_value(text: str, ftfy_fix):
    if not isinstance(text, str):
        return text
    if ftfy_fix is not None:
        fixed = ftfy_fix(text)
        if fixed != text:
            text = fixed
    return heuristic_fix(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean likely mojibake artifacts from dataset metadata fields."
    )
    parser.add_argument(
        "--input",
        default="Sarcasm_Headlines_Dataset_With_Metadata.json",
        help="Input JSONL dataset path.",
    )
    parser.add_argument(
        "--output",
        default="Sarcasm_Headlines_Dataset_With_Metadata_Cleaned.json",
        help="Output JSONL dataset path.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only report suspicious row counts without writing output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    ftfy_fix = load_ftfy()

    total_rows = 0
    changed_rows = 0
    changed_fields = 0
    suspicious_rows_before = 0
    suspicious_rows_after = 0
    samples = []
    output_lines = []

    with input_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            total_rows += 1

            if any(isinstance(obj.get(field), str) and looks_suspicious(obj[field]) for field in FIELDS_TO_FIX):
                suspicious_rows_before += 1

            row_changed = False
            for field in FIELDS_TO_FIX:
                value = obj.get(field)
                if not isinstance(value, str):
                    continue
                fixed = fix_value(value, ftfy_fix)
                if fixed != value:
                    obj[field] = fixed
                    row_changed = True
                    changed_fields += 1
                    if len(samples) < 5:
                        samples.append((field, value[:160], fixed[:160]))

            if row_changed:
                changed_rows += 1

            if any(isinstance(obj.get(field), str) and looks_suspicious(obj[field]) for field in FIELDS_TO_FIX):
                suspicious_rows_after += 1

            output_lines.append(json.dumps(obj, ensure_ascii=False))

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"ftfy_enabled={ftfy_fix is not None}")
    print(f"total_rows={total_rows}")
    print(f"suspicious_rows_before={suspicious_rows_before}")
    print(f"suspicious_rows_after={suspicious_rows_after}")
    print(f"changed_rows={changed_rows}")
    print(f"changed_fields={changed_fields}")

    for idx, (field, before, after) in enumerate(samples, start=1):
        print(f"SAMPLE {idx} field={field}")
        print(f"BEFORE: {before}")
        print(f"AFTER:  {after}")

    if args.report_only:
        return

    with output_path.open("w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
