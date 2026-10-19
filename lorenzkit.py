#!/usr/bin/env python3
"""lorenzkit: offline inequality toolkit.

Reads a plain-text file of people (or households, firms, regions) with a
value such as income or wealth, an optional weight and an optional group,
then reports the usual inequality measures:

  * Gini coefficient, computed exactly with fractions, plus a seeded
    bootstrap confidence interval
  * Theil T, Theil L (mean log deviation) and Atkinson indices
  * Palma ratio, top and bottom shares and quintile shares
  * Lorenz curve table and an ASCII plot
  * Theil between-group / within-group decomposition
  * Bootstrap comparison of two files
  * Line-numbered input linting, JSON output and CI exit codes

Input format, one record per line:

    label, value[, weight[, group]]

Blank lines and lines starting with '#' are ignored. Values are
non-negative exact decimals (or fractions such as 3/4). Weight defaults to 1
and must be positive. Group defaults to "(none)".

Exit codes: 0 ok, 1 a --max-gini / --max-palma gate failed,
2 input or usage error.
"""

import argparse
import json
import math
import random
import sys
from collections import OrderedDict, namedtuple
from fractions import Fraction

Record = namedtuple("Record", "label value weight group line")

EXIT_OK = 0
EXIT_GATE = 1
EXIT_INPUT = 2
NO_GROUP = "(none)"


class InputError(Exception):
    """Raised when data cannot support the requested measure."""


# ---------------------------------------------------------------- parsing

def parse_text(text):
    """Return (records, errors, warnings). Errors are (line, message)."""
    records, errors, warnings = [], [], []
    first_seen = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split(",")]
        if not 2 <= len(fields) <= 4:
            errors.append((lineno, "expected 'label, value[, weight[, group]]'"))
            continue
        label = fields[0]
        if not label:
            errors.append((lineno, "label is empty"))
            continue
        try:
            value = Fraction(fields[1])
        except (ValueError, ZeroDivisionError):
            errors.append((lineno, f"value {fields[1]!r} is not a number"))
            continue
        if value < 0:
            errors.append((lineno, f"value {fields[1]!r} is negative"))
            continue
        weight_text = fields[2] if len(fields) >= 3 and fields[2] else "1"
        try:
            weight = Fraction(weight_text)
        except (ValueError, ZeroDivisionError):
            errors.append((lineno, f"weight {weight_text!r} is not a number"))
            continue
        if weight <= 0:
            errors.append((lineno, f"weight {weight_text!r} must be positive"))
            continue
        group = fields[3] if len(fields) == 4 and fields[3] else NO_GROUP
        if label in first_seen:
            warnings.append((lineno, f"label {label!r} repeats line {first_seen[label]}"))
        else:
            first_seen[label] = lineno
        records.append(Record(label, value, weight, group, lineno))
    if not records and not errors:
        errors.append((0, "no data rows found"))
    return records, errors, warnings


def load_file(path):
    """Read and parse a file, raising InputError with a clear message."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise InputError(f"{path}: cannot read file ({exc.strerror})")
    records, errors, warnings = parse_text(text)
    if errors:
        raise InputError("\n".join(f"{path}:{ln}: error: {msg}" for ln, msg in errors))
    return records, warnings


# ---------------------------------------------------------- core measures
# Every measure takes a list of (value, weight) pairs. Exact Fraction
# inputs give exact results; float inputs are used for bootstrap speed.

def _totals(pairs):
    total_w = sum(w for _, w in pairs)
    total_s = sum(x * w for x, w in pairs)
    return total_w, total_s


def gini(pairs):
    """Weighted Gini: D / (W * S), D = sum over i<j of w_i w_j (x_j - x_i)."""
    pairs = sorted(pairs, key=lambda p: p[0])
    total_w, total_s = _totals(pairs)
    if total_s == 0:
        raise InputError("Gini is undefined when every value is zero")
    cross_a = 0
    cross_b = 0
    weight_before = 0
    for x, w in pairs:
        cross_a += w * x * weight_before
        cross_b += w * x * (total_w - weight_before - w)
        weight_before += w
    return (cross_a - cross_b) / (total_w * total_s)


def lowest_share(pairs, q):
    """Exact share of total value held by the lowest fraction q of weight."""
    pairs = sorted(pairs, key=lambda p: p[0])
    total_w, total_s = _totals(pairs)
    remaining = q * total_w
    taken_value = 0
    for x, w in pairs:
        if remaining <= 0:
            break
        take = min(w, remaining)
        taken_value += take * x
        remaining -= take
    return taken_value / total_s


def top_share(pairs, p):
    """Exact share of total value held by the highest fraction p of weight."""
    return 1 - lowest_share(pairs, 1 - p)


def theil_t(pairs):
    """Theil T index. Zero values contribute nothing (x ln x -> 0)."""
    total_w, total_s = _totals(pairs)
    mu = total_s / total_w
    acc = 0.0
    for x, w in pairs:
        if x > 0:
            acc += w * (x / mu) * math.log(x / mu)
    return acc / total_w


def theil_l(pairs):
    """Theil L (mean log deviation). Requires every value to be positive."""
    if any(x <= 0 for x, _ in pairs):
        raise InputError("Theil L (mean log deviation) needs every value above zero")
    total_w, total_s = _totals(pairs)
    mu = total_s / total_w
    return sum(w * math.log(mu / x) for x, w in pairs) / total_w


def atkinson(pairs, epsilon):
    """Atkinson index for inequality aversion epsilon >= 0."""
    if epsilon < 0:
        raise InputError("Atkinson epsilon must be zero or positive")
    total_w, total_s = _totals(pairs)
    mu = total_s / total_w
    if epsilon == 1:
        if any(x == 0 for x, _ in pairs):
            return 1.0
        geo = math.exp(sum(w * math.log(x) for x, w in pairs) / total_w)
        return 1 - geo / mu
    if epsilon > 1 and any(x == 0 for x, _ in pairs):
        return 1.0
    mean_power = sum(w * x ** (1 - epsilon) for x, w in pairs) / total_w
    equally_distributed = mean_power ** (1 / (1 - epsilon))
    return 1 - equally_distributed / mu


def lorenz_points(pairs):
    """Piecewise-linear Lorenz curve as (cumulative weight, cumulative value)."""
    pairs = sorted(pairs, key=lambda p: p[0])
    total_w, total_s = _totals(pairs)
    points = [(0.0, 0.0)]
    cum_w = 0.0
    cum_s = 0.0
    for x, w in pairs:
        cum_w += w
        cum_s += x * w
        points.append((cum_w / total_w, cum_s / total_s))
    return points


def _interp(points, p):
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if p <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (p - x0) / (x1 - x0)
    return points[-1][1]


def _percentile(sorted_values, q):
    index = q * (len(sorted_values) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    frac = index - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def _resample(pairs, rng):
    count = len(pairs)
    return [pairs[rng.randrange(count)] for _ in range(count)]


def bootstrap_gini(pairs, draws, seed):
    """Sorted Gini values from resampling records with replacement."""
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        try:
            values.append(gini(_resample(pairs, rng)))
        except InputError:
            continue
    values.sort()
    return values


def bootstrap_gini_difference(pairs_b, pairs_a, draws, seed):
    """Sorted values of Gini(B) - Gini(A) using independent resamples."""
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        try:
            values.append(gini(_resample(pairs_b, rng)) - gini(_resample(pairs_a, rng)))
        except InputError:
            continue
    values.sort()
    return values


def decompose(records):
    """Exact Theil T decomposition into between-group and within-group parts."""
    pairs = [(float(r.value), float(r.weight)) for r in records]
    total_w, total_s = _totals(pairs)
    mu = total_s / total_w
    groups = OrderedDict()
    for r in records:
        groups.setdefault(r.group, []).append((float(r.value), float(r.weight)))
    between = 0.0
    within = 0.0
    rows = []
    for name, members in groups.items():
        g_w, g_s = _totals(members)
        share = g_s / total_s if total_s else 0.0
        t_g = theil_t(members)
        if g_s > 0:
            between += share * math.log((g_s / g_w) / mu)
        within += share * t_g
        rows.append({"group": name, "records": len(members),
                     "share_of_total": share, "theil_t": t_g,
                     "mean": g_s / g_w})
    return {"total": theil_t(pairs), "between": between, "within": within,
            "groups": rows}


def build_report(records, epsilon=0.5, draws=0, seed=1):
    """Collect every headline measure for one file as a plain dict."""
    pairs = [(r.value, r.weight) for r in records]
    fpairs = [(float(x), float(w)) for x, w in pairs]
    total_w, total_s = _totals(pairs)
    report = {
        "records": len(records),
        "total_weight": float(total_w),
        "total_value": float(total_s),
        "mean": float(total_s / total_w),
        "gini": float(gini(pairs)),
        "theil_t": theil_t(fpairs),
        "theil_l": theil_l(fpairs) if all(x > 0 for x, _ in fpairs) else None,
        "atkinson": {"epsilon": epsilon, "value": atkinson(fpairs, epsilon)},
    }
    top10 = top_share(pairs, Fraction(1, 10))
    bottom40 = lowest_share(pairs, Fraction(2, 5))
    report["shares"] = {
        "top_1_percent": float(top_share(pairs, Fraction(1, 100))),
        "top_10_percent": float(top10),
        "bottom_50_percent": float(lowest_share(pairs, Fraction(1, 2))),
        "quintiles": [float(lowest_share(pairs, Fraction(k, 5))
                            - lowest_share(pairs, Fraction(k - 1, 5)))
                      for k in range(1, 6)],
    }
    report["palma"] = float(top10 / bottom40) if bottom40 > 0 else None
    report["gini_ci95"] = None
    if draws:
        values = bootstrap_gini(fpairs, draws, seed)
        if values:
            report["gini_ci95"] = [_percentile(values, 0.025), _percentile(values, 0.975)]
            report["bootstrap_draws"] = len(values)
    return report


# -------------------------------------------------------------- rendering

def ascii_lorenz(points, width=41, height=17):
    """Return text lines plotting the Lorenz curve ('*') against equality ('.')."""
    grid = [[" "] * width for _ in range(height)]
    for col in range(width):
        p = col / (width - 1)
        row_eq = round(p * (height - 1))
        grid[height - 1 - row_eq][col] = "."
        y = min(1.0, max(0.0, _interp(points, p)))
        row = round(y * (height - 1))
        grid[height - 1 - row][col] = "*"
    lines = []
    for i, row in enumerate(grid):
        label = 1 - i / (height - 1)
        lines.append(f"{label:3.1f} |" + "".join(row) + "|")
    lines.append("    +" + "-" * width + "+")
    lines.append("     0" + " " * (width - 2) + "1  cumulative population share")
    lines.append("     '*' Lorenz curve   '.' line of equality")
    return lines


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(report, source):
    lines = [f"lorenzkit report: {source}",
             f"  records          {report['records']}",
             f"  total weight     {report['total_weight']:g}",
             f"  total value      {report['total_value']:g}",
             f"  weighted mean    {report['mean']:.4f}",
             "",
             f"  Gini             {_fmt(report['gini'])}"]
    if report["gini_ci95"]:
        lo, hi = report["gini_ci95"]
        lines.append(f"  Gini 95% CI      [{lo:.4f}, {hi:.4f}]  "
                     f"({report['bootstrap_draws']} bootstrap draws)")
    lines.append(f"  Theil T          {_fmt(report['theil_t'])}")
    lines.append(f"  Theil L (MLD)    {_fmt(report['theil_l'])}")
    atk = report["atkinson"]
    lines.append(f"  Atkinson e={atk['epsilon']:g}  {_fmt(atk['value'])}")
    lines.append(f"  Palma ratio      {_fmt(report['palma'], 2)}  (top 10% / bottom 40%)")
    shares = report["shares"]
    lines.append("")
    lines.append(f"  top 1% share     {shares['top_1_percent'] * 100:.2f}%")
    lines.append(f"  top 10% share    {shares['top_10_percent'] * 100:.2f}%")
    lines.append(f"  bottom 50% share {shares['bottom_50_percent'] * 100:.2f}%")
    quint = " ".join(f"{q * 100:.1f}%" for q in shares["quintiles"])
    lines.append(f"  quintiles S1..S5 {quint}")
    return lines


def _lorenz_table(pairs):
    lines = ["  pop share   income share   gap to equality"]
    for k in range(1, 11):
        q = Fraction(k, 10)
        inc = lowest_share(pairs, q)
        lines.append(f"  {k * 10:>7}%   {float(inc) * 100:>10.2f}%   "
                     f"{(float(q) - float(inc)) * 100:>12.2f}%")
    return lines


# --------------------------------------------------------------- commands

def cmd_report(args):
    records, warnings = load_file(args.file)
    report = build_report(records, args.epsilon, args.bootstrap, args.seed)
    failures = []
    if args.max_gini is not None and report["gini"] > args.max_gini:
        failures.append(f"Gini {report['gini']:.4f} is above --max-gini {args.max_gini}")
    if args.max_palma is not None and report["palma"] is not None \
            and report["palma"] > args.max_palma:
        failures.append(f"Palma {report['palma']:.2f} is above --max-palma {args.max_palma}")
    if args.json:
        report["warnings"] = [f"{args.file}:{ln}: warning: {msg}" for ln, msg in warnings]
        report["gate_failures"] = failures
        print(json.dumps(report, indent=2))
    else:
        for ln, msg in warnings:
            print(f"{args.file}:{ln}: warning: {msg}", file=sys.stderr)
        print("\n".join(render_report(report, args.file)))
        for message in failures:
            print(f"GATE FAILED: {message}")
    return EXIT_GATE if failures else EXIT_OK


def cmd_lorenz(args):
    records, _ = load_file(args.file)
    pairs = [(r.value, r.weight) for r in records]
    if all(r.value == 0 for r in records):
        raise InputError("Lorenz curve is undefined when every value is zero")
    print(f"Lorenz curve: {args.file}")
    print("\n".join(_lorenz_table(pairs)))
    print()
    print("\n".join(ascii_lorenz(lorenz_points(pairs))))
    return EXIT_OK


def cmd_decompose(args):
    records, _ = load_file(args.file)
    result = decompose(records)
    if args.json:
        print(json.dumps(result, indent=2))
        return EXIT_OK
    total = result["total"]
    lines = [f"Theil T decomposition: {args.file}",
             f"  total            {total:.4f}",
             f"  between groups   {result['between']:.4f}  "
             f"({_pct(result['between'], total)})",
             f"  within groups    {result['within']:.4f}  "
             f"({_pct(result['within'], total)})",
             "",
             "  group                 records  share    mean        theil T"]
    for row in result["groups"]:
        lines.append(f"  {row['group']:<22}{row['records']:>7}  "
                     f"{row['share_of_total'] * 100:6.2f}%  {row['mean']:>10.4f}  "
                     f"{row['theil_t']:>10.4f}")
    print("\n".join(lines))
    return EXIT_OK


def _pct(part, whole):
    return "n/a" if whole == 0 else f"{part / whole * 100:.1f}% of total"


def cmd_compare(args):
    records_a, _ = load_file(args.file_a)
    records_b, _ = load_file(args.file_b)
    pairs_a = [(float(r.value), float(r.weight)) for r in records_a]
    pairs_b = [(float(r.value), float(r.weight)) for r in records_b]
    gini_a = float(gini([(r.value, r.weight) for r in records_a]))
    gini_b = float(gini([(r.value, r.weight) for r in records_b]))
    diff = gini_b - gini_a
    result = {"a": {"file": args.file_a, "gini": gini_a},
              "b": {"file": args.file_b, "gini": gini_b},
              "difference_b_minus_a": diff}
    if args.bootstrap:
        values = bootstrap_gini_difference(pairs_b, pairs_a, args.bootstrap, args.seed)
        if values:
            lo, hi = _percentile(values, 0.025), _percentile(values, 0.975)
            result["difference_ci95"] = [lo, hi]
            result["clear_difference"] = lo > 0 or hi < 0
    if args.json:
        print(json.dumps(result, indent=2))
        return EXIT_OK
    lines = ["Gini comparison",
             f"  A {args.file_a:<30} {gini_a:.4f}",
             f"  B {args.file_b:<30} {gini_b:.4f}",
             f"  difference (B - A)             {diff:+.4f}"]
    if "difference_ci95" in result:
        lo, hi = result["difference_ci95"]
        verdict = ("clear difference at 95%" if result["clear_difference"]
                   else "no clear difference at 95%")
        lines.append(f"  95% CI                         [{lo:+.4f}, {hi:+.4f}]  {verdict}")
    print("\n".join(lines))
    return EXIT_OK


def cmd_lint(args):
    try:
        with open(args.file, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        print(f"{args.file}: cannot read file ({exc.strerror})", file=sys.stderr)
        return EXIT_INPUT
    records, errors, warnings = parse_text(text)
    for ln, msg in errors:
        print(f"{args.file}:{ln}: error: {msg}")
    for ln, msg in warnings:
        print(f"{args.file}:{ln}: warning: {msg}")
    if not errors and not warnings:
        print(f"{args.file}: ok, {len(records)} records")
    failed = bool(errors) or (args.strict and bool(warnings))
    return EXIT_INPUT if failed else EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lorenzkit",
        description="Offline inequality toolkit: Gini, Theil, Atkinson, Palma, "
                    "Lorenz curves and group decomposition from plain text.")
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("report", help="headline inequality measures")
    rep.add_argument("file")
    rep.add_argument("--epsilon", type=float, default=0.5,
                     help="Atkinson inequality aversion (default 0.5)")
    rep.add_argument("--bootstrap", type=int, default=0, metavar="N",
                     help="bootstrap draws for a 95%% Gini interval (default 0 = off)")
    rep.add_argument("--seed", type=int, default=1, help="bootstrap seed (default 1)")
    rep.add_argument("--max-gini", type=float, help="CI gate: fail above this Gini")
    rep.add_argument("--max-palma", type=float, help="CI gate: fail above this Palma ratio")
    rep.add_argument("--json", action="store_true", help="print JSON")
    rep.set_defaults(func=cmd_report)

    lor = sub.add_parser("lorenz", help="Lorenz table and ASCII plot")
    lor.add_argument("file")
    lor.set_defaults(func=cmd_lorenz)

    dec = sub.add_parser("decompose", help="Theil between/within group split")
    dec.add_argument("file")
    dec.add_argument("--json", action="store_true", help="print JSON")
    dec.set_defaults(func=cmd_decompose)

    cmp_ = sub.add_parser("compare", help="Gini difference between two files")
    cmp_.add_argument("file_a")
    cmp_.add_argument("file_b")
    cmp_.add_argument("--bootstrap", type=int, default=0, metavar="N",
                      help="bootstrap draws for a 95%% CI on the difference")
    cmp_.add_argument("--seed", type=int, default=1, help="bootstrap seed (default 1)")
    cmp_.add_argument("--json", action="store_true", help="print JSON")
    cmp_.set_defaults(func=cmd_compare)

    lint = sub.add_parser("lint", help="check a data file with line-numbered errors")
    lint.add_argument("file")
    lint.add_argument("--strict", action="store_true", help="treat warnings as errors")
    lint.set_defaults(func=cmd_lint)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "bootstrap", 0) < 0:
        parser.error("--bootstrap must be zero or more")
    try:
        return args.func(args)
    except InputError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
