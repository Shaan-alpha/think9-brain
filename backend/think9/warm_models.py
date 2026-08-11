"""Download the ONNX weights into the cache directory. Run at build time.

    uv run python -m think9.warm_models

Fetching a model at request time costs the download's memory on top of whatever is
already resident. On a 512 MB instance the embedder plus an in-flight reranker download
exceeded the limit and the container was killed mid-question, which the caller saw as a
502. Doing it here moves that cost into the build, where failure is visible as a failed
build rather than as a broken answer.

Deliberately does not import the app or touch the database — a build has no database.
"""

import sys

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from think9.config import EMBEDDING_MODEL, RERANKER_MODEL, fastembed_cache_dir


def main() -> int:
    cache = fastembed_cache_dir()
    print(f"warming models into {cache or 'the default cache'}")

    embedder = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=cache)
    vector = next(iter(embedder.embed(["warm"])))
    print(f"  {EMBEDDING_MODEL}: ready, dim={len(vector)}")

    encoder = TextCrossEncoder(model_name=RERANKER_MODEL, cache_dir=cache)
    scores = list(encoder.rerank("warm", ["a passage"]))
    print(f"  {RERANKER_MODEL}: ready, scored={len(scores)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
