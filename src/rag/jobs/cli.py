"""Offline jobs CLI: indexing, reindexing, promotion, and evaluation.

Deliberately separate from the API process. Nothing here is reachable from a
request handler.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.backends import BackendContext
from rag.eval.report import format_report
from rag.jobs.reindex import activate_version, list_versions, reindex
from rag.jobs.scheduled_eval import scheduled_eval
from rag.observability.logging import configure_logging
from rag.schemas import AclTags
from rag.security.auth import parse_groups
from rag.settings import Settings, get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="RAG offline jobs")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest and index a source directory")
    ingest_p.add_argument("--source", type=Path, default=Path("data/raw"))
    ingest_p.add_argument("--index-version", default=None)
    ingest_p.add_argument("--tenant", default=None, help="ACL tenant for ingested docs")
    ingest_p.add_argument("--groups", default=None, help="Comma-separated ACL groups")
    ingest_p.add_argument(
        "--no-activate",
        action="store_true",
        help="Build without flipping the live alias",
    )

    reindex_p = sub.add_parser("reindex", help="Rebuild an index version")
    reindex_p.add_argument("--source", type=Path, default=Path("data/raw"))
    reindex_p.add_argument("--index-version", required=True)
    reindex_p.add_argument("--tenant", default=None)
    reindex_p.add_argument("--groups", default=None)
    reindex_p.add_argument("--no-activate", action="store_true")

    activate_p = sub.add_parser("activate", help="Promote or roll back to a built version")
    activate_p.add_argument("--index-version", required=True)

    sub.add_parser("versions", help="List known index versions")

    eval_p = sub.add_parser("eval", help="Run golden-set evaluation")
    eval_p.add_argument("--dataset", type=Path, default=Path("data/golden/golden.jsonl"))
    eval_p.add_argument(
        "--index-source",
        type=Path,
        default=None,
        help=(
            "Index this source first, in-process. Required with the in-memory "
            "backends, whose index does not outlive the process."
        ),
    )
    eval_p.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Exit non-zero if Recall@5 falls below this (for CI gating)",
    )

    agent_p = sub.add_parser(
        "eval-agent",
        help="Run hermetic agent evaluation (EchoChatLLM; gates steps/cost)",
    )
    agent_p.add_argument("--dataset", type=Path, default=Path("tests/fixtures/golden_agent.jsonl"))
    agent_p.add_argument(
        "--index-source",
        type=Path,
        default=Path("tests/fixtures/corpus"),
        help="Index this source first (required for in-memory backends)",
    )
    agent_p.add_argument("--max-avg-steps", type=float, default=4.0)
    agent_p.add_argument("--max-avg-cost", type=float, default=0.05)
    agent_p.add_argument("--min-success-rate", type=float, default=1.0)
    agent_p.add_argument(
        "--min-trajectory-rate",
        type=float,
        default=1.0,
        help=(
            "Exit non-zero if fewer than this fraction of examples took the shape "
            "they declare (retrieval_rounds, expect_self_correction)"
        ),
    )

    return parser


def _acl(args: argparse.Namespace, settings: Settings) -> AclTags | None:
    """ACL stamped onto every document from this run."""
    if not (args.tenant or args.groups):
        return None
    return AclTags(
        tenant=args.tenant or settings.default_tenant,
        allow_groups=parse_groups(args.groups, default=frozenset({"public"})),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir, json_output=settings.log_json)

    if args.command in {"ingest", "reindex"}:
        version = getattr(args, "index_version", None) or settings.index_version
        context = BackendContext.from_settings(settings, index_version=version)
        count = reindex(
            args.source,
            index_version=version,
            activate=not args.no_activate,
            context=context,
            acl=_acl(args, settings),
        )
        print(f"Indexed {count} chunks into version {version}")
        if count == 0:
            print(f"warning: no documents found under {args.source}", file=sys.stderr)
        return 0

    if args.command == "activate":
        activate_version(args.index_version)
        print(f"Activated index version {args.index_version}")
        return 0

    if args.command == "versions":
        versions = list_versions()
        print("\n".join(versions) if versions else "No index versions recorded yet")
        return 0

    if args.command == "eval":
        if not args.dataset.exists():
            print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
            return 2
        context = BackendContext.from_settings(settings)
        if args.index_source is not None:
            reindex(
                args.index_source,
                index_version=settings.index_version,
                activate=True,
                context=context,
                acl=AclTags(
                    tenant=settings.default_tenant,
                    allow_groups=frozenset({"public"}),
                ),
            )
        report = scheduled_eval(args.dataset, context=context)
        print(format_report(report))
        if args.min_recall is not None and report.recall_at_5 < args.min_recall:
            print(
                f"error: Recall@5 {report.recall_at_5:.3f} below threshold {args.min_recall:.3f}",
                file=sys.stderr,
            )
            return 1
        return 0

    if args.command == "eval-agent":
        from rag.jobs.agent_eval import format_report as format_agent_report
        from rag.jobs.agent_eval import run_agent_eval

        if not args.dataset.exists():
            print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
            return 2
        context = BackendContext.from_settings(settings)
        try:
            agent_report = run_agent_eval(
                args.dataset,
                context=context,
                index_source=args.index_source,
                max_avg_steps=args.max_avg_steps,
                max_avg_cost=args.max_avg_cost,
                min_success_rate=args.min_success_rate,
                min_trajectory_rate=args.min_trajectory_rate,
            )
        except SystemExit as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_agent_report(agent_report))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
