import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = (
    "headline",
    "description",
    "author_description",
    "author_name",
    "article_section",
)

CJK_SEQ_RE = re.compile(r"[\u4e00-\u9fff]{1,4}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze recurring mojibake patterns in metadata text."
    )
    parser.add_argument(
        "--input",
        default="Sarcasm_Headlines_Dataset_With_Metadata.json",
        help="Input JSONL dataset path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="How many top suspicious sequences to report.",
    )
    parser.add_argument(
        "--samples-per-seq",
        type=int,
        default=5,
        help="How many sample contexts to print per sequence.",
    )
    return parser.parse_args()


def escaped(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def sample_context(text: str, seq: str, window: int = 28) -> str:
    idx = text.find(seq)
    start = max(0, idx - window)
    end = min(len(text), idx + len(seq) + window)
    return escaped(text[start:end])


def main():
    args = parse_args()
    path = Path(args.input)

    counts = Counter()
    samples = defaultdict(list)

    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            for field in FIELDS:
                value = obj.get(field)
                if not isinstance(value, str):
                    continue
                for match in CJK_SEQ_RE.finditer(value):
                    seq = match.group(0)
                    counts[seq] += 1
                    if len(samples[seq]) < args.samples_per_seq:
                        samples[seq].append((field, sample_context(value, seq)))

    top = counts.most_common(args.top_n)
    print(f"input={path}")
    print(f"reported_sequences={len(top)}")

    for seq, count in top:
        print(f"SEQ {escaped(seq)} count={count}")
        for field, context in samples[seq]:
            print(f"  field={field} context={context}")
        print("---")


if __name__ == "__main__":
    main()
