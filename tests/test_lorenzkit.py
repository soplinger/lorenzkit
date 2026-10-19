import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lorenzkit as lk  # noqa: E402


def pairs_of(values, weights=None):
    weights = weights or [1] * len(values)
    return [(Fraction(v), Fraction(w)) for v, w in zip(values, weights)]


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ----------------------------------------------------------------- parsing

def test_parse_reads_optional_weight_and_group():
    records, errors, warnings = lk.parse_text("ann, 10\nbob, 20, 3, north\n")
    assert errors == [] and warnings == []
    assert records[0].weight == 1 and records[0].group == lk.NO_GROUP
    assert records[1].weight == 3 and records[1].group == "north"


def test_parse_reports_each_bad_line_with_its_number():
    text = "# header\nann, 10\nbad line only\nbob, -5\ncid, 4, 0\ndee, x1\n"
    _, errors, _ = lk.parse_text(text)
    assert [ln for ln, _ in errors] == [3, 4, 5, 6]
    assert "negative" in errors[1][1]
    assert "positive" in errors[2][1]


def test_parse_accepts_fractions_and_warns_on_repeated_labels():
    records, errors, warnings = lk.parse_text("ann, 3/4\nann, 1.5\n")
    assert errors == []
    assert records[0].value == Fraction(3, 4)
    assert warnings and warnings[0][0] == 2


def test_parse_empty_input_is_an_error():
    _, errors, _ = lk.parse_text("# nothing here\n\n")
    assert errors and errors[0][0] == 0


# ---------------------------------------------------------------- measures

def test_gini_two_people_is_half():
    assert lk.gini(pairs_of([0, 1])) == Fraction(1, 2)


def test_gini_equal_incomes_is_zero():
    assert lk.gini(pairs_of([5, 5, 5, 5])) == 0


def test_gini_three_with_two_zeros_is_two_thirds():
    assert lk.gini(pairs_of([0, 0, 1])) == Fraction(2, 3)


def test_gini_weighted_matches_expanded_records():
    weighted = pairs_of([1, 3], [3, 1])
    expanded = pairs_of([1, 1, 1, 3])
    assert lk.gini(weighted) == Fraction(1, 4)
    assert lk.gini(weighted) == lk.gini(expanded)


def test_gini_all_zero_raises():
    with pytest.raises(lk.InputError):
        lk.gini(pairs_of([0, 0]))


def test_theil_and_mld_are_zero_for_equal_values():
    fp = [(1.0, 1.0), (1.0, 1.0)]
    assert lk.theil_t(fp) == pytest.approx(0.0)
    assert lk.theil_l(fp) == pytest.approx(0.0)


def test_theil_l_rejects_zero_values():
    with pytest.raises(lk.InputError):
        lk.theil_l([(0.0, 1.0), (2.0, 1.0)])


def test_atkinson_epsilon_one_matches_geometric_mean_formula():
    # mean 2, geometric mean sqrt(3): A = 1 - sqrt(3)/2
    value = lk.atkinson([(1.0, 1.0), (3.0, 1.0)], 1)
    assert value == pytest.approx(1 - math.sqrt(3) / 2)


def test_atkinson_epsilon_zero_is_always_zero():
    assert lk.atkinson([(1.0, 2.0), (9.0, 1.0)], 0) == pytest.approx(0.0)


def test_top_share_splits_a_boundary_record_by_weight():
    # Top 20% of 5 equal-weight people is the single richest person.
    pairs = pairs_of([1, 1, 1, 1, 6])
    assert lk.top_share(pairs, Fraction(1, 5)) == Fraction(6, 10)


def test_quintile_shares_sum_to_one():
    rng = random.Random(7)
    pairs = pairs_of([rng.randint(0, 500) for _ in range(120)],
                     [rng.randint(1, 4) for _ in range(120)])
    shares = [lk.lowest_share(pairs, Fraction(k, 5)) - lk.lowest_share(pairs, Fraction(k - 1, 5))
              for k in range(1, 6)]
    assert sum(shares) == 1
    assert shares == sorted(shares)


def test_palma_ratio_for_uniform_population_is_a_quarter():
    # Top 10% hold 10% of income and bottom 40% hold 40%: 0.10 / 0.40.
    records = [lk.Record(str(i), Fraction(10), Fraction(1), lk.NO_GROUP, i) for i in range(10)]
    report = lk.build_report(records)
    assert report["palma"] == pytest.approx(0.25)
    assert report["gini"] == 0.0


def test_lorenz_curve_starts_at_origin_and_ends_at_one_one():
    points = lk.lorenz_points(pairs_of([4, 1, 2]))
    assert points[0] == (0.0, 0.0)
    assert points[-1] == pytest.approx((1.0, 1.0))
    assert all(y <= x + 1e-12 for x, y in points)


def test_decomposition_between_plus_within_equals_total():
    rng = random.Random(3)
    records = []
    for i in range(60):
        group = ["north", "south", "east"][i % 3]
        records.append(lk.Record(f"p{i}", Fraction(rng.randint(1, 900)),
                                 Fraction(rng.randint(1, 3)), group, i + 1))
    result = lk.decompose(records)
    assert result["between"] + result["within"] == pytest.approx(result["total"])
    assert len(result["groups"]) == 3


def test_bootstrap_is_reproducible_with_a_seed():
    fp = [(float(v), 1.0) for v in range(1, 40)]
    assert lk.bootstrap_gini(fp, 50, seed=9) == lk.bootstrap_gini(fp, 50, seed=9)
    assert lk.bootstrap_gini(fp, 50, seed=9) != lk.bootstrap_gini(fp, 50, seed=10)


# --------------------------------------------------------------------- CLI

def test_report_json_and_gate_exit_codes(tmp_path, capsys):
    path = write(tmp_path, "incomes.txt", "a, 0\nb, 0\nc, 0\nd, 100\n")
    assert lk.main(["report", path, "--json"]) == lk.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["gini"] == pytest.approx(0.75)
    assert lk.main(["report", path, "--max-gini", "0.5"]) == lk.EXIT_GATE
    assert "GATE FAILED" in capsys.readouterr().out


def test_lint_reports_line_numbers_and_fails(tmp_path, capsys):
    path = write(tmp_path, "bad.txt", "ann, 3\nbob, oops\n")
    assert lk.main(["lint", path]) == lk.EXIT_INPUT
    assert f"{path}:2: error" in capsys.readouterr().out


def test_report_on_bad_file_exits_two(tmp_path, capsys):
    path = write(tmp_path, "bad.txt", "ann, -1\n")
    assert lk.main(["report", path]) == lk.EXIT_INPUT
    assert "negative" in capsys.readouterr().err


def test_compare_prints_difference_and_bootstrap_interval(tmp_path, capsys):
    a = write(tmp_path, "a.txt", "\n".join(f"p{i}, {i + 1}" for i in range(20)))
    b = write(tmp_path, "b.txt", "\n".join(f"q{i}, {(i + 1) * (i + 1)}" for i in range(20)))
    assert lk.main(["compare", a, b, "--bootstrap", "100", "--json"]) == lk.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["difference_b_minus_a"] > 0
    assert len(data["difference_ci95"]) == 2


def test_lorenz_on_all_zero_file_exits_two(tmp_path, capsys):
    path = write(tmp_path, "zeros.txt", "a, 0\nb, 0\n")
    assert lk.main(["lorenz", path]) == lk.EXIT_INPUT
    assert "every value is zero" in capsys.readouterr().err


def test_lorenz_and_decompose_commands_run(tmp_path, capsys):
    path = write(tmp_path, "g.txt", "a, 1, 1, x\nb, 2, 1, x\nc, 9, 2, y\n")
    assert lk.main(["lorenz", path]) == lk.EXIT_OK
    assert "Lorenz curve" in capsys.readouterr().out
    assert lk.main(["decompose", path]) == lk.EXIT_OK
    out = capsys.readouterr().out
    assert "between groups" in out and "within groups" in out
