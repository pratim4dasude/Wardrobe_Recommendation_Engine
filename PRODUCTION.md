# Production Notes : Taking This to Thousands of Wardrobes

This note covers how I would evolve the current prototype (CLIP + OpenAI text embeddings, hybrid retrieval over pgvector, LLM outfit composition with code-side validation) into a system serving thousands of users' closets. It is organised around the three questions in the brief: the embedding store and retrieval approach, evaluation of recommendation quality, and cold-start for a brand-new closet.

---

## 1. Embedding store and retrieval at scale

### Why pgvector, and when I'd reconsider

I would stay on **PostgreSQL + pgvector** for the first phase of production, for three reasons:

1. **Closets are small.** A wardrobe is typically 50–500 items. Every retrieval in this product is scoped to a single user (`WHERE user_id = ...`), so each query is an exact cosine scan over a few hundred vectors — sub-millisecond, no ANN index required. Approximate indexes (HNSW/IVF) solve a problem this access pattern doesn't have.
2. **Vectors and metadata belong together.** Outfit retrieval is heavily filtered (role = bottomwear, occasion tags, colour overlap). Keeping vectors, structured metadata, and chat state in one transactional store means filters are plain SQL `WHERE` clauses, ingestion is atomic (item row + vector written together), and there is no sync problem between a vector DB and a metadata DB.
3. **Operational simplicity.** One database to back up, migrate, and monitor. At a startup, that matters more than peak QPS.

**Schema for multi-tenancy:** add `user_id` to the items table, index `(user_id, role)`, and enforce row-level security so a query can never cross closets. Vector columns carry a `model_version` so embeddings from different models never get compared.

**When I'd move to a dedicated vector DB (Qdrant / Milvus / managed):** the moment we need *cross-user* search — e.g. "find items like this across the whole catalogue" for shopping recommendations, or trend analysis over millions of items. That's a global ANN problem with payload filtering at scale, which is exactly what Qdrant's HNSW + filtered search is built for. I'd run it alongside Postgres (Postgres stays the source of truth; the vector DB is a derived index rebuilt from it), rather than migrating wholesale.

### Embedding model

The prototype uses generic `clip-vit-base-patch32`. In production I would swap to a **fashion-specialised encoder** — FashionCLIP or Marqo-FashionSigLIP — which separate garment attributes (sleeve length, neckline, wash, fabric) far better than base CLIP, directly improving both similarity and pairing candidates. Because `model_version` is stored per vector, the migration is a background re-embedding job: write new vectors under the new version, flip a config flag per user once their closet is fully re-embedded, then garbage-collect old vectors. No downtime, easy rollback.

### Retrieval approach

Keep the **hybrid scoring** from the prototype, with two changes:

- **Dense retrieval first, as a filtered pgvector query** (`role`, `user_id`, optional occasion tags), returning a candidate pool of ~50.
- **Rerank in process** with the metadata-overlap bonus. I would replace the custom BM25 with **Postgres full-text search (`tsvector`)** — same exact-keyword benefit ("denim", "blazer"), one less custom component to maintain.

### Ingestion pipeline

Uploads go onto a queue (SQS/Redis); workers run caption → tag extraction → embed → write, idempotent on image hash. Captioning with a VLM is the slow, costly step, so it's done once at upload time, never at query time. Query-time LLM calls (intent parse, composition, stylist note) are the latency budget; I'd cap them at two calls per request by merging the intro/note generations, and cache intent parses for repeated phrasings.

---

## 2. Evaluating recommendation quality

Recommendation quality here decomposes into three measurable layers, plus online signals.

### Layer 1 — Retrieval quality (deterministic, offline)

Build a **golden set**: ~200 (query garment, occasion) → hand-labelled relevant candidates, sourced from a public outfit-compatibility dataset (e.g. Polyvore Outfits) plus a small internally-labelled set on our own item taxonomy. Track **precision@k and recall@k** per role (does retrieval surface bottoms that *could* pair with this top?). This runs in CI, so any change to the embedding model, scoring weights, or filters shows up as a metric diff before it ships.

### Layer 2 — Outfit validity (deterministic, free)

The code-side validator already enforces hard rules (complete outfit, real item IDs, role constraints). Log its rejection rate in production: **validator rejection rate** is a direct health metric for the composer — if a prompt or model change makes the LLM hallucinate IDs or break role rules more often, this number moves first.

### Layer 3 — Compatibility quality (LLM-as-judge, calibrated)

Style compatibility is subjective, so use a rubric-based **LLM judge** (colour harmony, formality match, occasion fit, scored 1–5) over sampled production outputs. Critically, **calibrate it**: have humans label a few hundred outfits, measure judge–human agreement, and only trust the judge for relative comparisons (model A vs model B) rather than absolute scores. This makes offline A/B of prompt and retrieval changes cheap.

### Online signals

Once real users exist, behavioural metrics dominate: **outfit acceptance/save rate**, item click-through, explicit thumbs up/down on outfit cards, and "regenerate" rate (a strong negative signal). Ship changes behind an A/B framework keyed on these. Logged accept/reject pairs also become training data for a future learned reranker (Layer 1 then gets a personalised stage).

---

## 3. Cold-start for a brand-new closet

The architecture is **content-based by design**, which is the main cold-start mitigation: embeddings and tags are computed from the images themselves at upload, so similarity search and rule-based pairing work from the very first item — no interaction history required. The cold-start plan is therefore about making the first session good, not about making the system function at all.

**At onboarding:**

- **Caption-and-tag on upload**, with a quick "confirm or fix" step shown to the user. This both improves metadata quality and is itself labelled data.
- A 30-second **style quiz** (preferred vibe, typical occasions, colours avoided) seeds the same intent fields the chat assistant extracts, so the first recommendation is already conditioned on preferences.

**Aggregate priors (privacy-safe):** pairing knowledge learned across *all* users — co-occurrence statistics of (category, colour, style) combinations that get accepted — applies to a new user immediately, because it operates on item attributes, not user identity. A new user's white shirt benefits from every other user who paired white shirts with navy trousers.

**The sparse-closet problem:** the harder case is a closet with 5 items, where no valid outfit may exist. Handle it explicitly:

- Relax role constraints gracefully (offer partial outfits with a note) rather than failing.
- Turn the gap into product value: **"you have tops for office but no formal bottoms"** — a wardrobe-gap suggestion, which is also a natural commerce hook later.

**Personalisation ramp:** start fully content-based → blend in aggregate priors → after ~20–30 interactions, enable a lightweight per-user reranker trained on accept/reject signals. Each stage degrades gracefully to the previous one, so there is never a point where a new user gets a worse experience than the prototype gives today.

---

## What I'd do next (beyond the brief)

In priority order: an offline eval harness wired to the golden set (makes every later change measurable); swap to a fashion-specific embedding model; replace the CLI with a thin FastAPI layer; add footwear/accessories to the taxonomy; cost/latency dashboards on the LLM calls, which will dominate unit economics.
