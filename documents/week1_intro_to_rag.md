# Week 1 — Introduction to Retrieval-Augmented Generation

## What RAG Is

Retrieval-augmented generation (RAG) is a technique that grounds a language model by supplying
it with relevant source text at answer time instead of relying on weights learned during
training. The model is asked to answer only from text supplied in the prompt.

RAG exists because language models have two structural weaknesses. First, they can generate
fluent text that is factually wrong, because a plausible-sounding continuation is not the same
as a correct one. Second, they have no access to private or recent material at inference time.
Grounding in retrieved text addresses both, provided the retrieval is good.

## The Two Stages

A RAG system has two stages that run at different times.

The **offline indexing stage** runs once. Documents are loaded, split into chunks, and each
chunk is converted into a vector embedding. The embeddings are written to a vector store
along with metadata identifying the source file and the position within it.

The **online query stage** runs per question. The user's question is embedded with the same
embedding model used at index time, the vector store returns the most similar chunks, and those
chunks are placed into the prompt for the language model.

The single most common cause of a broken RAG system is using different embedding models at the
two stages. Embeddings from different models are not comparable, so similarity scores are
meaningless. Any system must record which embedding model built the index and refuse to serve
if the running configuration disagrees.

## Chunking

A chunk is a contiguous span of text that becomes one retrievable unit. Chunks must be small
enough that they describe one idea precisely, and large enough that they carry enough context
to be interpretable. A common default is around five hundred tokens with fifty tokens of
overlap between adjacent chunks.

Overlap exists so that an idea spanning a chunk boundary is still fully present in at least one
chunk. Without overlap, a passage cut in half between two chunks can be retrieved by neither.

Chunks should not span a page or section boundary. When a chunk crosses such a boundary, the
citation attached to it becomes ambiguous, and the reader cannot be told where to look.

## Embeddings

An embedding is a fixed-length list of numbers representing the meaning of a piece of text.
Two texts about the same topic produce vectors that are close together; unrelated texts produce
vectors that are far apart.

Similarity between two vectors is usually measured as cosine similarity, which compares the
angle between them. Cosine similarity ranges from minus one to one, where one means the
vectors point in the same direction and zero means they are unrelated.

The critical property for this course is that cosine similarity is *directional*, not
semantic, at the embedding level: it reflects topical similarity. It does not know whether a
statement is true. Grounding quality therefore depends on both retrieval and on how the
retrieved text is presented to the model.

## Refusing to Answer

A grounded system should be able to say that it does not know. The usual mechanism is a
similarity threshold: if the best-matching chunk scores below the threshold, the question is
treated as outside the corpus and the system declines to answer instead of generating.

The threshold should be calibrated against real questions rather than chosen by feel. Run a set
of questions that the corpus covers and a set that it does not, record the best score for each,
and choose a value that separates them with margin. If the two score populations overlap, the
problem is retrieval quality, not the threshold, and the chunking should be fixed instead.
