"""Command line entry point.

Four subcommands, in the order someone new to the tool would want them:

``demo``
    Run the whole instrument against a simulated model whose artifacts are known
    exactly. Needs no model, no dataset, no network, and finishes in seconds. It
    exists so that a reader can watch the audit recover a planted effect and
    refuse an absent one before deciding whether to trust anything it says about
    a real model.

``pathology``
    Inspect a benchmark for defects without loading a model at all.

``convert``
    Turn a published benchmark file into the canonical format.

``audit``
    The real thing: decompose a model's score on a corpus.

``compare``
    Difference two saved audits, refusing when their manifests disagree about
    anything that should have been held fixed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from evalassay import __version__
from evalassay.audit import AuditConfig, run_audit
from evalassay.corpus.canonical import read_jsonl, subsample, write_jsonl
from evalassay.corpus.loaders import (
    load_arc_jsonl,
    load_arc_parquet,
    load_hellaswag_jsonl,
    load_mmlu_csv,
    load_mmlu_directory,
    load_mmlu_parquet,
    load_truthfulqa_mc_jsonl,
)
from evalassay.corpus.synthetic import CorpusSpec, generate
from evalassay.pathology.runner import run_all as run_pathology
from evalassay.report.compare import compare, render_comparison
from evalassay.report.render import RULE, render
from evalassay.report.serialise import from_json, to_json
from evalassay.score.oracle import OracleScorer, OracleSpec
from evalassay.stats.decision import GateConfig
from evalassay.types import ItemSet, Verdict

LOADERS: Final = {
    "arc": load_arc_jsonl,
    "arc-parquet": load_arc_parquet,
    "hellaswag": load_hellaswag_jsonl,
    "mmlu-csv": load_mmlu_csv,
    "mmlu-dir": load_mmlu_directory,
    "mmlu-parquet": load_mmlu_parquet,
    "truthfulqa": load_truthfulqa_mc_jsonl,
    "canonical": read_jsonl,
}
"""Published formats this tool can read, keyed by the name used on the command line."""

DEMO_ITEMS: Final = 500


def _load(fmt: str, path: Path) -> ItemSet:
    """Load a corpus in the named format.

    Args:
        fmt: A key of :data:`LOADERS`.
        path: File or directory to read.

    Returns:
        The corpus.
    """
    return LOADERS[fmt](path)


def _prepare(corpus: ItemSet, items: int | None, seed: int) -> ItemSet:
    """Optionally subsample a corpus before auditing it.

    Args:
        corpus: The full corpus.
        items: How many items to keep, or ``None`` for all of them.
        seed: Seed for the selection.

    Returns:
        The corpus to audit.
    """
    if items is None or items >= len(corpus):
        return corpus
    return subsample(corpus, items, seed, stratify_by_subject=True)


def _gate(args: argparse.Namespace) -> GateConfig:
    """Build the default-deny thresholds from parsed arguments.

    Args:
        args: Parsed arguments.

    Returns:
        The thresholds.
    """
    return GateConfig(
        alpha=args.alpha,
        power=args.power,
        bootstrap_draws=args.bootstrap,
        min_effect=args.min_effect,
    )


def _emit(report_text: str, json_text: str | None, json_path: Path | None) -> None:
    """Print the text report and optionally write the JSON alongside it.

    Args:
        report_text: The rendered report.
        json_text: The serialised report, if wanted.
        json_path: Where to write it.
    """
    sys.stdout.write(report_text)
    if json_text is not None and json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_text, encoding="utf-8")
        sys.stdout.write(f"\nmachine-readable report written to {json_path}\n")


def cmd_demo(args: argparse.Namespace) -> int:
    """Audit a simulated model whose artifacts are known exactly.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    planted = {
        "memorisation": args.memorisation,
        "distractor reliance": args.distractor,
        "positional preference": args.position,
    }
    corpus = generate(
        CorpusSpec(
            n_items=DEMO_ITEMS,
            n_choices=4,
            seed=args.seed,
            # Positional preference only buys accuracy when the benchmark's own
            # key is skewed, so the demo plants the skew alongside it.
            position_bias=0.5 if args.position else 0.0,
            biased_position=1,
        )
    )
    oracle = OracleSpec(
        skill=args.skill,
        memorisation=args.memorisation,
        distractor_reliance=args.distractor,
        position_preference=args.position,
        favoured_position=1,
        choices_only_skill=args.choices_only,
        seed=args.seed,
    )

    sys.stdout.write("Auditing a simulated model with these artifacts planted in it:\n")
    for name, value in planted.items():
        sys.stdout.write(f"  {name:<24}{value:.2f}\n")
    sys.stdout.write(f"  {'genuine skill':<24}{args.skill:.2f}\n")
    sys.stdout.write(f"\n{RULE}\n\n")

    report = run_audit(
        corpus,
        OracleScorer(oracle, corpus),
        AuditConfig(seed=args.audit_seed, gate=_gate(args)),
    )
    _emit(render(report), to_json(report) if args.json else None, args.json)

    sys.stdout.write(
        "\nAn artifact that was planted should be charged; one that was not should\n"
        "read 'not established'. That is the whole claim, and you have just run it.\n"
    )
    return 0


def cmd_pathology(args: argparse.Namespace) -> int:
    """Inspect a benchmark for defects, with no model.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    corpus = _prepare(_load(args.format, args.corpus), args.items, args.seed)
    result = run_pathology(corpus, args.seed, _gate(args))

    sys.stdout.write(f"Benchmark defects in {corpus.name} ({len(corpus)} items)\n")
    sys.stdout.write(f"{RULE}\n")
    if result.duplicates_removed:
        sys.stdout.write(
            f"  {result.duplicates_removed} exact repeats withheld from the detectors\n"
            "  that assume items are independent draws.\n"
        )
    for finding in result.findings:
        if finding.verdict is Verdict.ESTABLISHED:
            sys.stdout.write(
                f"  {finding.detector:<20}{finding.estimate.point:>9.4f}  "
                f"[{finding.estimate.ci_low:.4f}, {finding.estimate.ci_high:.4f}]\n"
                f"  {'':<20}{'':>9}  {finding.detail}\n"
            )
        else:
            sys.stdout.write(
                f"  {finding.detector:<20}{'-':>9}  "
                f"not established; an effect of {finding.mde:.4f} would have shown\n"
            )
    for name in result.skipped:
        sys.stdout.write(f"  {name:<20}{'-':>9}  not measured on a corpus this small\n")
    sys.stdout.write("\n")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert a published benchmark file into the canonical format.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    corpus = _load(args.format, args.corpus)
    write_jsonl(corpus, args.output)
    sys.stdout.write(f"wrote {len(corpus)} items to {args.output}\n")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Difference two saved audits of the same corpus.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    baseline = from_json(args.baseline.read_text(encoding="utf-8"))
    variant = from_json(args.variant.read_text(encoding="utf-8"))
    sys.stdout.write(render_comparison(compare(baseline, variant)))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Decompose a model's score on a corpus.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    from evalassay.score.local import LocalScorer  # noqa: PLC0415 - optional dependency

    corpus = _prepare(_load(args.format, args.corpus), args.items, args.seed)
    scorer = LocalScorer(args.model, style=args.style, length_normalise=not args.unnormalised)
    report = run_audit(corpus, scorer, AuditConfig(seed=args.seed, gate=_gate(args)))
    _emit(render(report), to_json(report) if args.json else None, args.json)
    return 0


def _add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the pre-registered threshold options to a subparser.

    Args:
        parser: The subparser.
    """
    group = parser.add_argument_group("default-deny thresholds")
    group.add_argument("--alpha", type=float, default=0.01, help="family-wise significance level")
    group.add_argument("--power", type=float, default=0.80, help="target power, for the MDE")
    group.add_argument("--bootstrap", type=int, default=10_000, help="bootstrap resamples")
    group.add_argument(
        "--min-effect", type=float, default=0.005, help="smallest effect worth reporting"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="assay",
        description="Measure how much of a reported benchmark score is capability.",
    )
    parser.add_argument("--version", action="version", version=f"evalassay {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="audit a simulated model with known artifacts; needs no model or data"
    )
    demo.add_argument("--skill", type=float, default=0.40)
    demo.add_argument("--memorisation", type=float, default=0.25)
    demo.add_argument("--distractor", type=float, default=0.20)
    demo.add_argument("--position", type=float, default=0.0)
    demo.add_argument("--choices-only", type=float, default=0.0)
    demo.add_argument("--seed", type=int, default=11, help="seed for the corpus and the oracle")
    demo.add_argument("--audit-seed", type=int, default=7, help="seed for the audit itself")
    demo.add_argument("--json", type=Path, default=None, help="also write JSON here")
    _add_gate_arguments(demo)
    demo.set_defaults(func=cmd_demo)

    pathology = subparsers.add_parser(
        "pathology", help="inspect a benchmark for defects, with no model"
    )
    pathology.add_argument("corpus", type=Path)
    pathology.add_argument("--format", choices=sorted(LOADERS), default="canonical")
    pathology.add_argument("--items", type=int, default=None, help="subsample to this many items")
    pathology.add_argument("--seed", type=int, default=7)
    _add_gate_arguments(pathology)
    pathology.set_defaults(func=cmd_pathology)

    convert = subparsers.add_parser("convert", help="convert a benchmark to the canonical format")
    convert.add_argument("corpus", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--format", choices=sorted(LOADERS), default="arc")
    convert.set_defaults(func=cmd_convert)

    comparison = subparsers.add_parser(
        "compare",
        help="difference two saved audits of the same corpus, refusing if they disagree",
    )
    comparison.add_argument("baseline", type=Path, help="JSON report to subtract from")
    comparison.add_argument("variant", type=Path, help="JSON report to compare against it")
    comparison.set_defaults(func=cmd_compare)

    audit = subparsers.add_parser("audit", help="decompose a model's score on a corpus")
    audit.add_argument("corpus", type=Path)
    audit.add_argument("--model", required=True, help="model identifier or local path")
    audit.add_argument("--format", choices=sorted(LOADERS), default="canonical")
    audit.add_argument("--items", type=int, default=None, help="subsample to this many items")
    audit.add_argument("--seed", type=int, default=7)
    audit.add_argument(
        "--style",
        choices=["cloze", "labelled"],
        default="cloze",
        help=(
            "how the question is put to the model. 'cloze' scores each option as a "
            "continuation and never shows the option list, so option position cannot "
            "affect the result; 'labelled' presents the options as a list and scores "
            "the label, which is how most leaderboard numbers are produced and the "
            "only setting in which positional preference is measurable"
        ),
    )
    audit.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help=(
            "weight precision. 'bfloat16' halves the memory a model occupies, which "
            "can be the difference between a large model fitting alongside other work "
            "and not fitting at all. Recorded in the scorer identity, so runs at "
            "different precisions can never be silently compared"
        ),
    )
    audit.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "torch thread count. Defaults to leaving four processors free so the "
            "machine stays usable while an audit runs"
        ),
    )
    audit.add_argument(
        "--unnormalised",
        action="store_true",
        help="sum option log-likelihood instead of averaging it, which prefers short options",
    )
    audit.add_argument("--json", type=Path, default=None, help="also write JSON here")
    _add_gate_arguments(audit)
    audit.set_defaults(func=cmd_audit)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Arguments, defaulting to the process arguments.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"assay: {exc}\n")
    return exit_code


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
