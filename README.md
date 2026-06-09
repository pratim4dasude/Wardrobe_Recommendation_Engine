# Wardrobe Recommendation Engine

An AI personal stylist that you interact with in plain English. You describe an occasion, point
at a specific garment, or share an outside photo, and the system builds complete, wearable
outfits using only the clothes that actually exist in your wardrobe.

The core design principle is that the language model is never allowed to invent clothing. The
system first *retrieves* real items from the closet using a hybrid search engine, then asks the
model to assemble outfits strictly from those retrieved items, and finally *validates* every
outfit in code before it is shown. In other words, this is a Retrieval-Augmented Generation
(RAG) pipeline applied to fashion.

---

## What it can do

| Request | Behaviour |
|---|---|
| *"I have office tomorrow, what should I wear?"* | Asks for the vibe if unclear, then builds three full outfits from the wardrobe |
| *"Style item_003 for a casual day out"* | Builds outfits around that exact garment, plus alternatives from similar items already owned |
| *"What can I pair with C:\...\11.jpg for a party?"* | Captions an outside photo, treats it as a temporary garment, and pairs it with wardrobe items |
| *"Find similar items to that"* | Uses CLIP image similarity to surface the closest matches already owned |

Every recommendation includes a reason, styling notes, a confidence level, and a final stylist
note describing the preferred choice.

---

## How it works

For each message, the assistant runs a short pipeline:

1. **Understand** — A language model converts the raw message into clean structured intent
   (occasion, vibe, item id, image path, whether a follow-up is needed). A rule-based layer
   repairs edge cases and prevents stale conversation context from leaking into a fresh request.
2. **Remember** — Per-session memory merges the new information with prior turns, so a follow-up
   such as *"smart clean look"* attaches correctly to an earlier *"office"* request.
3. **Retrieve** — For the chosen source garment, the engine pulls compatible candidates per role
   (top, bottom, outerwear) using a hybrid of semantic embeddings, a metadata bonus, and BM25
   keyword scoring.
4. **Compose** — A language model assembles three outfits (minimal, layered, alternative)
   strictly from the retrieved candidate identifiers, following hard rules such as pairing an
   upper-body source with a bottom.
5. **Validate** — Incomplete or rule-breaking outfits are removed in code rather than trusted to
   the model.
6. **Present** — Two further model calls produce a natural intro and a final stylist note.

---

## Architecture

```
                          +----------------------------------+
                          |          USER MESSAGE            |
                          |   text  |  item_id  |  image     |
                          +----------------+-----------------+
                                           |
                                           v
                          +----------------------------------+
                          |       QUERY UNDERSTANDING        |
                          |   LLM intent parse + rule repair |
                          +----------------+-----------------+
                                           |
                                           v
                          +----------------------------------+
                          |          CHAT MEMORY             |
                          |  session context: occasion,      |
                          |  vibe, item_id, image, history   |
                          +----------------+-----------------+
                                           |
                                           v
                          +----------------------------------+
                          |          INTENT ROUTER           |
                          +----+-------------+-----------+---+
                               |             |           |
                   text only   |   item_id   |  outside image
                               v             v           v
                    +----------------+  +----------+  +----------------------+
                    | Semantic pool  |  |  Load    |  |  Caption + embed     |
                    | (text search)  |  |  source  |  |  GPT-4.1-mini + CLIP |
                    +-------+--------+  +----+-----+  +----------+-----------+
                            |               |                    |
                            |               v                    v
                            |        +--------------------------------------+
                            |        |     HYBRID CANDIDATE RETRIEVAL       |
                            |        |  +-------------------------------+   |
                            |        |  | text-embedding cosine (vibe)  |   |
                            |        |  | metadata bonus (color/style)  |   |
                            |        |  | BM25 keyword rerank (exact)   |   |
                            |        |  +-------------------------------+   |
                            |        +------------------+-------------------+
                            |                           |
                            +-----------+---------------+
                                        v
                          +----------------------------------+
                          |       OUTFIT COMPOSER (LLM)      |
                          |  grounded strictly in real IDs   |
                          +----------------+-----------------+
                                           |
                                           v
                          +----------------------------------+
                          |       OUTFIT VALIDATION          |
                          |  complete? real IDs? roles ok?   |
                          +----------------+-----------------+
                                           |
                          +----------------+----------------+
                          |                                 |
                          v                                 v
              +-------------------------+      +------------------------------+
              | SIMILAR ITEMS (CLIP)    |      |     FINAL RESPONSE           |
              | -> alternative outfits  | ---> |  intro + outfit cards +      |
              +-------------------------+      |  stylist note                |
                                               +--------------+---------------+
                                                              |
                                                              v
                                                        back to USER
```

---

## The retrieval engine

A single signal is not enough for clothing, so three are combined before any candidate reaches
the language model:

| Signal | What it captures | Implementation |
|---|---|---|
| Semantic similarity | meaning and vibe, e.g. *"smart clean office look"* | OpenAI `text-embedding-3-small` (1536-d) + cosine |
| Metadata bonus | shared color, style, category, and occasion tags | rule-based overlap score |
| BM25 keyword | exact terms embeddings blur, e.g. *denim, flannel, blazer* | custom BM25, no external library |

These merge into a combined keyword score used to rank candidates per outfit role. Image
similarity (the *find similar* and *outside-image* flows) is handled separately by CLIP visual
embeddings (512-d), compared by cosine against the wardrobe.

---

## Technology

| Tool | Role in the pipeline |
|---|---|
| Python 3.11+ | Core language |
| OpenAI `gpt-4.1-mini` | Query understanding, image captioning, outfit composition, stylist notes |
| OpenAI `text-embedding-3-small` | Semantic text embeddings for retrieval (1536-d) |
| CLIP `clip-vit-base-patch32` (Hugging Face Transformers) | Visual embeddings for image similarity (512-d) |
| PyTorch | Runs CLIP and tensor cosine similarity |
| Pillow | Image loading and decoding before CLIP |
| python-dotenv | Loads the OpenAI API key from `.env` |
| Custom BM25 | Keyword reranking, dependency-free |
| Local JSON store | Wardrobe data, embeddings, and per-session chat memory |

---

## Data pipeline

Run once to turn raw closet photos into a searchable wardrobe:

```
images  -->  inventory.json  -->  metadata.json  -->  text embeddings  -->  hybrid embeddings
            (list of files)     (caption + tags)    (semantic vectors)   (+ CLIP visual vectors)
```

- Captioning and metadata: each image is captioned, then converted into structured
  category, color, style, and search-text fields.
- Text embeddings: the search text is embedded into a 1536-d semantic vector.
- Visual and hybrid: the CLIP 512-d vector is added, producing the single hybrid embeddings
  file that the live engine reads.

The prepared embedding files are already included, so the chat assistant can run without
rebuilding anything.

---

## Getting started

1. Prerequisites: Python 3.11+ and an OpenAI API key.

2. Install dependencies:

```
python -m venv venv
venv\Scripts\activate          # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install openai python-dotenv torch transformers pillow
```

3. Add the API key by creating a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

4. Run the chat assistant from the project root:

```
python -m scripts.fashion_chat_demo
```

Type a request such as `I have office tomorrow, what should I wear?`, and type `exit` to quit.
The first run downloads the CLIP model (about 600 MB) from Hugging Face.

---

## Example interaction

```
You: i have office tomorrow what should i wear
Assistant: What kind of vibe are you looking for? formal, casual, classy, bold, or smart clean?

You: smart clean look
Assistant:
  Outfit #1: Classic Smart Casual  (confidence: high)
    - upperwear  item_093  crisp white button-up shirt
    - bottomwear item_064  slim-fit navy dress pants
  Outfit #2: Layered Smart Casual  (+ light blue blazer item_065)
  Outfit #3: Alternative Earthy Smart Casual  (beige tonal look)

  Stylist note: I would pick the Classic Smart Casual first - timeless, polished, effortless.
```

Every item shown (`item_093`, `item_064`, and so on) is a real garment retrieved from the
wardrobe, not an invention.

---

## Current status and roadmap

This is an actively evolving project. Honest status:

- Wardrobe scope: currently tops, bottoms, and outerwear. Footwear and accessories are not yet
  in the dataset, so outfits stop at shirt and trousers with an optional layer.
- Serving layer: the engine runs as a local command-line application. The REST API, database,
  and web interface are planned and not yet implemented.
- Scale: retrieval loads and scores embeddings in-process, which suits closets of around a
  hundred items. A vector index such as FAISS or pgvector is the planned upgrade for larger
  wardrobes.

Planned next: add footwear and accessories, an offline evaluation harness for recommendation
quality, structured-output JSON for the model calls, and the API and web serving layer.
