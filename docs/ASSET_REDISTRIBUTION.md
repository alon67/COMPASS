# Third-party model assets: what may be redistributed, and what may not

**The repository's MIT licence covers the COMPASS source only.** It does not,
and cannot, relicense the published neuron models this project imports. The
2026-08-09 post-response-3 audit (P1-4) flagged the mismatch directly: the
release inventory listed `data/models/**/*.mod`, `*.hoc`, `*.asc` and `*.swc`
among the artifacts a release distributes, while several of those files carry
"ModelDB terms" or "free for academic use" — terms that permit *use* but say
nothing about a third party redistributing them inside a data archive.

## The rule

| tier | what it means | what the release does |
|---|---|---|
| **Redistributable** | the upstream licence explicitly permits redistribution | included in the release archive, with the upstream licence text alongside |
| **Fetch-at-runtime** | the upstream terms permit academic use but do not grant redistribution rights, or are silent | **excluded from the release archive**; fetched by `tools/fetch_models.py`, whose SHA-256 record in `data/provenance/downloads.json` is what the release actually distributes |
| **Restricted** | redistribution is refused or the terms forbid derivative distribution | as fetch-at-runtime, plus an explicit note in the manuscript's data-availability statement |

A recorded SHA-256 plus a resolvable URL is a *stronger* reproducibility
guarantee than a copied file, not a weaker one: it fixes the exact bytes and it
identifies where they came from. What it does not survive is the upstream host
disappearing, and that risk is stated rather than engineered around by copying
files this project has no right to copy.

## Per-asset status

Every implementation, its upstream, and the tier it falls in. `status` is what
the release does today; `rights` is what has been established, not assumed.

| implementation | upstream | stated terms | tier | rights established |
|---|---|---|---|---|
| `hh_point` | NEURON built-in `hh` mechanism | BSD-like (NEURON distribution) | redistributable | yes — NEURON's own licence |
| `hh_cable` | NEURON built-in `hh` mechanism | BSD-like (NEURON distribution) | redistributable | yes — NEURON's own licence |
| `pospischil_rs` | ModelDB 123623 (Pospischil et al. 2008) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `pospischil_fs` | ModelDB 123623 (Pospischil et al. 2008) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `mainen_l5pc` | ModelDB 2488 (Mainen & Sejnowski 1996) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `migliore_ca1` | ModelDB 2796 (Migliore et al. 1999) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `khaliq_purkinje` | ModelDB 48332 (Khaliq et al. 2003) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `miyasho_purkinje` | ModelDB 17664 (Miyasho et al. 2001) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `bbp_l5pc` | ModelDB 139653 (Hay et al. 2011) + BluePyOpt config | ModelDB terms; BluePyOpt LGPL-3.0 | fetch-at-runtime | partial — the BluePyOpt configuration is LGPL-3.0 and redistributable with its licence; the Hay mechanisms are not established |
| `tomko_ca1` | ModelDB 267306-adjacent (Tomko et al. 2021) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `tzilivaki_fsbc` | ModelDB (Tzilivaki et al. 2019) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `dougalis_vlpag_da` | ModelDB 226019 (Dougalis et al. 2017) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `moubarak_snc_da` | ModelDB 267306 (Moubarak et al. 2022) | ModelDB terms; free for academic use | fetch-at-runtime | not established |
| `ortner_snc_da` | ModelDB 201748 (Ortner et al. 2017) | ModelDB terms; free for academic use | fetch-at-runtime | not established |

Twelve of fourteen are **fetch-at-runtime**. That is the honest reading of
"free for academic use": it is a use grant to the reader, not a distribution
grant to this project.

## What this means for the release archive

1. `configs/release_inventory.json` keeps `model_assets` as an inventory class,
   because the *hashes* are release content and a reader must be able to verify
   that the files they fetched are the files the results were computed from.
2. The deposited archive (Zenodo or equivalent) contains
   `data/provenance/downloads.json` — URL, SHA-256, byte count and fetch
   timestamp per asset — and **not** the asset bytes for any implementation in
   the fetch-at-runtime tier.
3. `tools/fetch_models.py` reconstructs `data/models/` from that record and
   fails if a hash does not match, so a reviewer's tree is verifiably the same
   tree.
4. The manuscript's data-availability statement says exactly this, naming the
   accession for each model rather than implying the archive contains them.

## Open action

Redistribution rights have not been *requested* from any upstream author. Doing
so for the twelve fetch-at-runtime models would let the archive be
self-contained, which is the better outcome for long-term reproducibility. Until
a permission is obtained and recorded here, the tier stands as
fetch-at-runtime — the conservative reading, chosen deliberately over assuming
that "free for academic use" extends to third-party redistribution.
