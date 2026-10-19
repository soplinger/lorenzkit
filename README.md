# Lorenzkit

**Offline inequality toolkit: Gini, Theil, Atkinson, Palma, Lorenz curves and group decomposition from one plain-text file.**

Lorenzkit answers "how unequal is this?" for any list of incomes, wealth,
pay, rainfall, or anything else with a non-negative value. You describe the
records in plain text, and it prints exact, reproducible measures, a Lorenz
curve, group breakdowns and CI gates. It uses only the Python 3 standard
library, makes no network calls, and keeps everything in files you own.

## Features

- **Gini coefficient, exact** — computed with fractions, so results are
  reproducible to the last digit; weighted records behave like expanded rows.
- **Seeded bootstrap intervals** — a 95% confidence interval for the Gini,
  identical on every run for the same `--seed`.
- **Theil T and Theil L (mean log deviation)** — with a clear error when L is
  asked of data containing zeros.
- **Atkinson index** — for any inequality aversion `--epsilon` (default 0.5).
- **Palma ratio** — top 10% share divided by bottom 40% share.
- **Top and bottom shares** — top 1%, top 10%, bottom 50%, and quintile shares
  S1–S5, with a record that straddles a cut-off split by its weight.
- **Lorenz table and ASCII plot** — deciles of population against income, with
  a picture of the curve against the line of equality.
- **Between-group / within-group decomposition** — exact Theil T split by a
  `group` column, so you can see whether gaps sit between neighbourhoods or
  inside them.
- **Two-file comparison** — Gini difference (B − A) with an optional bootstrap
  CI that says whether the difference is clear at 95%.
- **Line-numbered linting** — every bad line is reported with its number;
  repeated labels are warnings, and `--strict` makes warnings fail.
- **CI gates** — `--max-gini` and `--max-palma` exit 1 when a threshold is
  crossed; input errors exit 2.
- **JSON output** — `--json` on `report`, `decompose` and `compare` for scripts
  and dashboards.

## Input format

One record per line:

```
label, value[, weight[, group]]
```

- `label` — any text (required).
- `value` — non-negative exact number, such as `42000`, `1.5` or `3/4`.
- `weight` — optional, positive; the number of people the record stands for.
  Default `1`.
- `group` — optional; default `(none)`.
- Lines starting with `#` and blank lines are ignored.

See [`examples/town-incomes.txt`](examples/town-incomes.txt) for a worked file.

## Quick start

```sh
python3 lorenzkit.py report examples/town-incomes.txt
```

This prints record count, total weight, weighted mean, Gini, Theil T and L,
Atkinson, Palma, top and bottom shares and quintiles.

Add a confidence interval and a gate for CI:

```sh
python3 lorenzkit.py report examples/town-incomes.txt --bootstrap 1000 --max-gini 0.6
```

Look at the shape of the distribution:

```sh
python3 lorenzkit.py lorenz examples/town-incomes.txt
```

See where the inequality sits:

```sh
python3 lorenzkit.py decompose examples/town-incomes.txt
```

Compare two years or two towns:

```sh
python3 lorenzkit.py compare last-year.txt this-year.txt --bootstrap 1000
```

Check a file before using it:

```sh
python3 lorenzkit.py lint examples/town-incomes.txt --strict
```

## Commands

| Command | Purpose |
| --- | --- |
| `report FILE` | Headline measures; `--bootstrap N`, `--seed`, `--epsilon`, `--max-gini`, `--max-palma`, `--json` |
| `lorenz FILE` | Decile table and ASCII Lorenz plot |
| `decompose FILE` | Theil between/within split by group; `--json` |
| `compare A B` | Gini difference with optional bootstrap CI; `--bootstrap`, `--seed`, `--json` |
| `lint FILE` | Line-numbered errors and warnings; `--strict` |

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | A `--max-gini` or `--max-palma` gate failed |
| 2 | Input error (bad line, unreadable file, undefined measure) or usage error |

Use them directly in CI:

```sh
python3 lorenzkit.py report data/incomes.txt --max-gini 0.45 --max-palma 1.5 || exit 1
```

## How the measures are defined

- **Gini** = Σᵢ<ⱼ wᵢwⱼ(xⱼ − xᵢ) ÷ (W · S), with values sorted ascending,
  W the total weight and S the total value. Equal values give 0; one person
  holding everything gives (n−1)/n.
- **Theil T** = (1/W) Σ wᵢ (xᵢ/μ) ln(xᵢ/μ). Zero values contribute nothing.
- **Theil L** = (1/W) Σ wᵢ ln(μ/xᵢ). Needs every value above zero.
- **Atkinson** with ε = 1 uses the geometric mean; other ε use the power mean.
- **Palma** = top 10% income share ÷ bottom 40% income share.
- **Decomposition**: Theil T = between-group part + within-group part,
  where the within part is each group's Theil T weighted by its income share.

## Running the tests

The tests use pytest, which is the only dependency outside the standard library.

```sh
python3 -m pip install pytest
python3 -m pytest -q tests
```

The suite covers parsing and line-numbered errors, exact Gini values
(including weighted records), Theil and Atkinson identities, quintile sums,
Palma on a uniform population, the decomposition identity, bootstrap
reproducibility, and the CLI exit codes and JSON output.

## Project layout

```
lorenzkit.py            the whole tool (single file, standard library only)
examples/town-incomes.txt   sample input
tests/test_lorenzkit.py     pytest suite
```

## Notes and limits

- Inputs are treated as non-negative. Negative values (such as net worth with
  debts) are rejected rather than guessed at.
- Gini is undefined when every value is zero; `report` says so instead of
  printing a number.
- The bootstrap resamples records, so it assumes they are independent samples.
  Treat its interval as a rough guide, not a survey-design calculation.
