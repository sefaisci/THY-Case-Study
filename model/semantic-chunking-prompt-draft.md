# Semantic Chunking Prompt Contract

## Purpose

Document the executable prompt and validated structured-output contract implemented in `model/semantic_chunking/prompts.py` and `model/semantic_chunking/schemas.py`.

## Inputs

- `document_id`
- `document_name`
- `document_type`
- `page_number` or `slide_number`
- base64 data URL created locally for the current page image

## Prompt Intent

Ask the selected semantic model to inspect only the current page image and return a strict, flat `SemanticPageResult` through OpenAI Structured Outputs. Every page or slide is analyzed independently; no previous-page memory is added to the prompt.

## Output Expectations

Each validated chunk includes:

- `chunk_key` and `title`
- one authoritative, variable-length `text` value that is both embedded and stored
- `keywords` and `relationships`
- a bounded citation-ready `source_excerpt`
- confidence in the range `[0, 1]`

Each page result also includes `page_classification`:

- `content` requires one or more validated chunks.
- `blank` requires zero chunks and an explicit summary or warning explaining that the complete image contains no meaningful legible information.

Chunks form one flat list. Recursive nodes, nested semantic chunks, continuation notes, and memory fields are rejected by the schema.

## Constraints

- Do not include the entire document in every prompt.
- Do not add previous-page summaries, continuation notes, chunk keys, titles, images, or full text.
- Divide the visible content by meaning rather than a fixed token or character length.
- Inspect the complete image in reading order and cover meaningful headings, prose, lists, tables, equations, code, charts, diagrams, captions, and visible labels without inventing content.
- Never persist base64 data in Qdrant or notebook outputs.
- Reject duplicate keys, invalid page identity, empty chunk text, and invalid confidence.
- Retry transient or invalid responses at most twice; persistent failure is not upserted.
- Embed and upsert records in bounded batches so large documents do not retain every vector in memory.
- After awaited upserts, retrieve every deterministic point ID from Qdrant and verify matching `user_id`, `document_id`, and `chunk_id` provenance.
- Fail the document and remove partial owner/document points when analysis, embedding, upsert, or persistence verification is incomplete.
