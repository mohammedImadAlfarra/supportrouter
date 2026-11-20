"""README numbers come from reports/ (Phase 12)."""

import shutil
from pathlib import Path

import pytest

from supportrouter.config import load_config
from supportrouter.readme_results import END, check_readme, main, write_results

CFG = load_config()
README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture
def readme_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "README.md"
    shutil.copy(README, copy)
    return copy


def test_committed_readme_matches_reports() -> None:
    assert check_readme(CFG, README) == []


def test_hand_edited_table_number_fails(readme_copy: Path) -> None:
    text = readme_copy.read_text()
    readme_copy.write_text(text.replace("| 0.9977 |", "| 0.9990 |", 1))
    assert any("results block differs" in p for p in check_readme(CFG, readme_copy))
    assert main(["--check", "--readme", str(readme_copy)], CFG) == 1
    write_results(CFG, readme_copy)  # regenerating restores the honest block
    assert check_readme(CFG, readme_copy) == []


def test_hand_typed_number_in_prose_fails(readme_copy: Path) -> None:
    text = readme_copy.read_text()
    readme_copy.write_text(text + "\nT1 reaches 0.9995 macro-F1.\n")
    assert check_readme(CFG, readme_copy) == [
        "number '0.9995' in README prose is not in the generated results block"
    ]


def test_prose_may_quote_block_numbers_and_code(readme_copy: Path) -> None:
    text = readme_copy.read_text()
    readme_copy.write_text(text + "\nB1 scored 0.9917. Run `python 3.12.4` or\n```\nv9.99\n```\n")
    assert check_readme(CFG, readme_copy) == []


def test_missing_markers_are_refused(readme_copy: Path) -> None:
    readme_copy.write_text(readme_copy.read_text().replace(END, ""))
    with pytest.raises(ValueError, match="markers"):
        check_readme(CFG, readme_copy)
