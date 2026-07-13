# Provenance Rules Placeholder

Every future workflow must connect numeric claims to one of these source types:

- a cell or row in a local manual input file under `.data/manual`
- an unmodified API response under `.data/api_raw`
- a cached retrieval artifact under `.cache`
- a deterministic calculation trace in a run directory under `.runs`
- a cited local reference under `references/local`

When provenance cannot be supplied, the output must state that the claim is
unsupported instead of supplying a number.
