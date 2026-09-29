"""Generate a small sample PDF for the corpus. Standard Helvetica, no extra dependencies.

Usage: python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

PAGES = [
    (
        "Week 3 Lecture Notes: Prompting a Grounded Model",
        [
            "A grounded prompt has two parts: a system instruction that fixes the",
            "model's role and forbids outside knowledge, and a user message that",
            "supplies the retrieved context and the question.",
            "",
            "The retrieved passages should be fenced with numbered markers so the",
            "model can refer to them and the interface can attach a citation to",
            "each one. Fences also limit how much a retrieved passage can blur",
            "into the instruction, since a passage cannot close a fence it does",
            "not know the name of.",
            "",
            "Temperature controls randomness. A grounded question-answering bot",
            "should run at a low temperature, around 0.2, because a higher value",
            "invents plausible detail that is not in the source text.",
        ],
    ),
    (
        "Citations and Verification",
        [
            "A citation is only useful if the reader can act on it. That means every",
            "chunk must carry the file it came from and its position within that",
            "file: a page number for a PDF, a section heading for a text document.",
            "",
            "Citations should be derived from the chunks that were actually sent to",
            "the model, not by parsing markers out of the generated prose. A model",
            "that invents a bracket number has not invented a source; it has",
            "invented a reference to a source that was present. Letting the",
            "prose drive the citation list makes that failure visible to the user.",
            "",
            "Showing the retrieved text itself, alongside the answer, is what makes",
            "a grounded system auditable. A reader can check whether the cited",
            "passage really supports the claim in one glance.",
        ],
    ),
]

FONT_SIZE = 11
LEADING = 15
MARGIN_X = 72
MARGIN_Y_TOP = 750


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    parts = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEADING} TL", f"{MARGIN_X} {MARGIN_Y_TOP} Td"]
    for line in lines:
        parts.append(f"({_escape(line)}) Tj T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", "replace")


def build_pdf(pages: list[tuple[str, list[str]]]) -> bytes:
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog_num = add(b"")
    pages_num = add(b"")
    font_num = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_nums: list[int] = []
    for title, body in pages:
        stream = _content_stream([title, ""] + body)
        content_num = add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page_num = add(
            b"<< /Type /Page /Parent "
            + str(pages_num).encode()
            + b" 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 "
            + str(font_num).encode()
            + b" 0 R >> >> /Contents "
            + str(content_num).encode()
            + b" 0 R >>"
        )
        page_nums.append(page_num)

    kids = b" ".join(f"{n} 0 R".encode() for n in page_nums)
    objects[pages_num - 1] = (
        b"<< /Type /Pages /Count " + str(len(page_nums)).encode() + b" /Kids [" + kids + b"] >>"
    )
    objects[catalog_num - 1] = b"<< /Type /Catalog /Pages " + str(pages_num).encode() + b" 0 R >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(index).encode() + b" 0 obj\n" + obj + b"\nendobj\n"

    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root "
        + str(catalog_num).encode()
        + b" 0 R >>\nstartxref\n"
        + str(xref_pos).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def main() -> None:
    target = Path(__file__).resolve().parent.parent / "documents" / "week3_prompting.pdf"
    target.write_bytes(build_pdf(PAGES))
    print(f"wrote {target} ({len(PAGES)} pages, {target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
