from __future__ import annotations

import argparse
from pathlib import Path


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_lines(lines: list[str], max_chars: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if len(line) <= max_chars:
            wrapped.append(line)
            continue
        start = 0
        while start < len(line):
            wrapped.append(line[start : start + max_chars])
            start += max_chars
    return wrapped


def build_pdf_commands(lines: list[str], *, width: int, height: int) -> list[str]:
    left = 36
    top = height - 36
    line_height = 12
    font_size = 8
    commands: list[str] = []
    y = top
    for line in lines:
        if y < 36:
            commands.append("__NEW_PAGE__")
            y = top
        commands.append(f"BT /F1 {font_size} Tf 1 0 0 1 {left:.2f} {y:.2f} Tm ({pdf_escape(line)}) Tj ET")
        y -= line_height
    return commands


def write_pdf(out_path: Path, pages: list[list[str]], *, width: int, height: int) -> None:
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]
    font_obj_id = 3
    page_object_ids: list[int] = []

    for page_commands in pages:
        content = "\n".join(page_commands).encode("latin-1", "replace")
        objects.append(b"<< /Length " + str(len(content)).encode("latin-1") + b" >>\nstream\n" + content + b"\nendstream")
        content_obj_id = len(objects)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
                f"/Resources << /Font << /F1 {font_obj_id} 0 R >> >> /Contents {content_obj_id} 0 R >>"
            ).encode("latin-1")
        )
        page_object_ids.append(len(objects))

    kids = " ".join(f"{obj_id} 0 R" for obj_id in page_object_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode("latin-1")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        ).encode("latin-1")
    )
    out_path.write_bytes(pdf)


def render_markdown_to_pdf(markdown_path: Path, out_path: Path) -> None:
    width, height = 1100, 850
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    wrapped = wrap_lines(lines, max_chars=150)
    commands = build_pdf_commands(wrapped, width=width, height=height)
    pages: list[list[str]] = [[]]
    for command in commands:
        if command == "__NEW_PAGE__":
            pages.append([])
            continue
        pages[-1].append(command)
    write_pdf(out_path, pages, width=width, height=height)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown report as a simple PDF table/text document.")
    parser.add_argument("markdown_path")
    parser.add_argument("--out", default=None, help="Optional output PDF path")
    args = parser.parse_args()

    markdown_path = Path(args.markdown_path).resolve()
    out_path = Path(args.out).resolve() if args.out else markdown_path.with_suffix(".pdf")
    render_markdown_to_pdf(markdown_path, out_path)


if __name__ == "__main__":
    main()
