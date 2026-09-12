# Data

Raw and derived data stay here — never under `src/`.

| Path | Contents |
|------|----------|
| `raw/` | Source documents as received |
| `interim/` | Parsed / cleaned intermediates |
| `processed/` | Chunked artefacts ready for indexing |
| `golden/` | Curated evaluation sets used by `rag eval` |

Everything under these directories is gitignored except this README, `.gitkeep`
files, and `golden/*.example.jsonl`. That is the point: a corpus, its
intermediates, and its indexes are operational data, not source code, and they
must not be reviewable-by-accident in a pull request.

## Getting started

Drop documents into `raw/` and index them:

```bash
rag ingest --source data/raw --tenant acme --groups engineering
```

Supported out of the box: `.txt`, `.md`, `.html`. PDFs need the `pdf` extra.

## Evaluation sets

Copy the example and curate your own:

```bash
cp data/golden/golden.example.jsonl data/golden/golden.jsonl
rag eval --dataset data/golden/golden.jsonl --min-recall 0.8
```

A small committed set lives in `tests/fixtures/golden.jsonl` so CI can run without
this directory. Guidance on building a set worth having:
[../docs/evaluation.md](../docs/evaluation.md).

## In containers

`./data` is mounted **read-only** into the indexing worker. The API container never
mounts it at all — it has no reason to read raw documents.
