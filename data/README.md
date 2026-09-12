# Data

Raw and derived data stay here — never under `src/`.

| Path | Contents |
|------|----------|
| `raw/` | Source documents as received |
| `interim/` | Parsed / cleaned intermediates |
| `processed/` | Chunked artefacts ready for indexing |
| `golden/` | Human-curated eval sets (runtime); test fixtures live in `tests/fixtures/` |

Everything under these dirs is gitignored except this README and `.gitkeep` files.
