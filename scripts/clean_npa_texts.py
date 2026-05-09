# from __future__ import annotations

# import re
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1] / "data" / "npa"
# PATTERN = re.compile(r"\s*\(Дополнение\s+пунктом\s*-.*?\)", flags=re.IGNORECASE)


# def clean_text(text: str) -> str:
#     cleaned = PATTERN.sub("", text)
#     cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
#     return cleaned


# def main() -> None:
#     changed = 0
#     for path in sorted(ROOT.rglob("*.txt")):
#         original = path.read_text(encoding="utf-8")
#         cleaned = clean_text(original)
#         if cleaned != original:
#             path.write_text(cleaned, encoding="utf-8", newline="\n")
#             changed += 1
#             print(path)
#     print(f"changed_files={changed}")


# if __name__ == "__main__":
#     main()
