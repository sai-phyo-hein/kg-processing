"""S3 chunk retrieval tools for multi-agent reasoning system.

NEW MODULE.

Chunks are stored in S3 at:
    s3://{SOURCE_DOC_BUCKET}/{community_id}/chunk_{chunk_id:03d}.txt
    s3://{SOURCE_DOC_BUCKET}/{community_id}/manifest.json

The community_id and chunk_id both live as properties on Neo4j relationship
edges (stored by neo4j_graph_builder.py: edge_props["chunk_id"] = chunk_id,
edge_props["community_id"] = community_id). They are NOT the Qdrant point IDs.

So the retrieval path for "get the source text behind a graph triple" is:
    Neo4j edge → r.community_id + r.chunk_id → S3 key → raw chunk text

This gives the synthesizer the actual source sentences, not just the entity
and predicate names that were extracted from them. It's the richest signal
available and should be preferred for questions asking "what", "why", "how"
or "what does the source say about X".

surround_chunks uses manifest.json to find adjacent chunks within the same
document (same community_id prefix), giving true positional context without
needing chunk_index/source_doc_id fields in Qdrant.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


def _get_s3_client():
    """Return a boto3 S3 client using env credentials."""
    if not HAS_BOTO3:
        raise RuntimeError("boto3 is not installed. Run: pip install boto3")

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION")

    if not (access_key and secret_key):
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment"
        )

    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def _get_bucket() -> str:
    return os.getenv("SOURCE_DOC_BUCKET", "document")


def _s3_key(community_id: str, chunk_id: int) -> str:
    """Build the S3 object key for a chunk file.

    Matches the upload path in semantic_chunker.upload_chunks_to_s3:
        key = f"{community_id}/{fp.name}"
    where fp.name is "chunk_001.txt", "chunk_002.txt", etc.
    """
    return f"{community_id}/chunk_{chunk_id:03d}.txt"


def _manifest_key(community_id: str) -> str:
    return f"{community_id}/manifest.json"


def fetch_s3_chunks(
    chunk_refs: List[Dict[str, Any]],
    max_chars_per_chunk: int = 4000,
) -> List[Dict[str, Any]]:
    """Fetch raw source text for a list of (community_id, chunk_id) pairs from S3.

    Args:
        chunk_refs: List of {"community_id": str, "chunk_id": int} dicts.
            Typically built from Neo4j edge properties (r.community_id, r.chunk_id).
        max_chars_per_chunk: Character cap per chunk to avoid oversized context.

    Returns:
        List of {"community_id", "chunk_id", "s3_key", "text"} dicts.
        On individual fetch failure, "text" is an error string and "error" is set.
    """
    if not chunk_refs:
        return []

    try:
        s3 = _get_s3_client()
    except RuntimeError as e:
        return [{"error": str(e), "chunk_refs": chunk_refs}]

    bucket = _get_bucket()
    results = []

    # Deduplicate by (community_id, chunk_id) — multiple triples in a
    # session often reference the same chunk, no need to fetch it twice.
    seen: set = set()
    for ref in chunk_refs:
        community_id = ref.get("community_id", "")
        chunk_id = ref.get("chunk_id")
        if not community_id or chunk_id is None:
            continue

        key_tuple = (community_id, int(chunk_id))
        if key_tuple in seen:
            continue
        seen.add(key_tuple)

        s3_key = _s3_key(community_id, int(chunk_id))
        try:
            obj = s3.get_object(Bucket=bucket, Key=s3_key)
            text = obj["Body"].read().decode("utf-8")
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk] + "\n\n[truncated]"
            results.append({
                "community_id": community_id,
                "chunk_id": chunk_id,
                "s3_key": s3_key,
                "text": text,
            })
        except ClientError as e:
            code = e.response["Error"]["Code"]
            results.append({
                "community_id": community_id,
                "chunk_id": chunk_id,
                "s3_key": s3_key,
                "error": f"S3 {code}: {e}",
                "text": f"[fetch failed: {code}]",
            })
        except Exception as e:
            results.append({
                "community_id": community_id,
                "chunk_id": chunk_id,
                "s3_key": s3_key,
                "error": str(e),
                "text": f"[fetch failed: {e}]",
            })

    return results


def fetch_surrounding_s3_chunks(
    community_id: str,
    anchor_chunk_id: int,
    window: int = 2,
    max_chars_per_chunk: int = 3000,
) -> List[Dict[str, Any]]:
    """Fetch a chunk plus its positional neighbors from S3.

    Uses manifest.json to discover the full list of chunk IDs in the document,
    then fetches anchor ± window chunks in order. This gives true positional
    context since chunk files are numbered sequentially by the chunker.

    Args:
        community_id: S3 key prefix (== Neo4j/Qdrant community_id)
        anchor_chunk_id: Integer chunk_id of the anchor chunk
        window: How many chunks before/after anchor to include
        max_chars_per_chunk: Character cap per individual chunk

    Returns:
        List of chunk dicts in document order, each with
        {"community_id", "chunk_id", "s3_key", "text"}.
    """
    try:
        s3 = _get_s3_client()
    except RuntimeError as e:
        return [{"error": str(e)}]

    bucket = _get_bucket()

    # Read manifest to find total chunk count and validate anchor exists
    manifest_key = _manifest_key(community_id)
    try:
        import json
        obj = s3.get_object(Bucket=bucket, Key=manifest_key)
        manifest = json.loads(obj["Body"].read().decode("utf-8"))
        total_chunks = manifest.get("total_chunks", 0)
        # Chunk IDs in the manifest are 1-based integers
        valid_chunk_ids = {c["chunk_id"] for c in manifest.get("chunks", [])}
    except ClientError as e:
        code = e.response["Error"]["Code"]
        print(f"Warning: could not read manifest {manifest_key}: {code} — "
              f"falling back to sequential probe")
        # Fallback: probe sequentially around the anchor
        total_chunks = anchor_chunk_id + window + 1
        valid_chunk_ids = set(range(1, total_chunks + 1))
    except Exception as e:
        print(f"Warning: manifest read failed: {e}")
        total_chunks = anchor_chunk_id + window + 1
        valid_chunk_ids = set(range(1, total_chunks + 1))

    # Determine the range of chunk IDs to fetch
    start_id = max(1, anchor_chunk_id - window)
    end_id = min(
        anchor_chunk_id + window,
        max(valid_chunk_ids) if valid_chunk_ids else anchor_chunk_id + window,
    )

    chunk_refs = [
        {"community_id": community_id, "chunk_id": cid}
        for cid in range(start_id, end_id + 1)
        if cid in valid_chunk_ids or not valid_chunk_ids
    ]

    results = fetch_s3_chunks(chunk_refs, max_chars_per_chunk=max_chars_per_chunk)
    # Sort by chunk_id so the synthesizer reads them in document order
    results.sort(key=lambda r: r.get("chunk_id", 0))
    return results


def get_chunk_refs_from_neo4j_results(
    neo4j_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract (community_id, chunk_id) pairs from Neo4j worker output rows.

    Neo4j graph_hop/direct/community_explore results are slimmed dicts shaped:
        {"subject": {...}, "predicate": "...", "object": {...}, "community_id": "..."}

    The community_id is on the row directly (from r.community_id AS community_id
    in the Cypher). The chunk_id is a relationship property also stored on the
    edge but not currently returned by the slim format.

    This helper is used by the fetch_s3_chunks worker variant that re-queries
    Neo4j for chunk_ids given entity results, or for callers that already have
    the full edge props available.

    Args:
        neo4j_results: Slimmed result dicts from a worker markdown file

    Returns:
        Deduplicated list of {"community_id", "chunk_id"} dicts
    """
    seen = set()
    refs = []
    for row in neo4j_results:
        cid = row.get("community_id")
        chunk_id = row.get("chunk_id")  # present if Cypher returned r.chunk_id
        if cid and chunk_id is not None:
            key = (cid, int(chunk_id))
            if key not in seen:
                seen.add(key)
                refs.append({"community_id": cid, "chunk_id": int(chunk_id)})
    return refs