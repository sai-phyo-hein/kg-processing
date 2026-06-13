"""Compare an entity between Qdrant ci_entity_registry and Neo4j by canonical_id."""

import os
import json
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

CANONICAL_ID = "922dedf4-a8a0-5c3c-9e68-5f0b7bc3b06a"

# ── Qdrant lookup ────────────────────────────────────────────────────────────

qdrant_url = os.getenv("QDRANT_URL").rstrip("/")
qdrant_key = os.getenv("QDRANT_API_KEY")
headers = {"api-key": qdrant_key, "Content-Type": "application/json"}
collection = "ci_entity_registry"

print(f"🔍 Looking up: {CANONICAL_ID}\n")

# Try as Qdrant point ID first
print("═══ Qdrant (by point ID) ═══")
resp = requests.post(
    f"{qdrant_url}/collections/{collection}/points",
    headers=headers,
    json={"ids": [CANONICAL_ID], "with_payload": True, "with_vector": False},
)
qdrant_by_pointid = None
if resp.status_code == 200 and resp.json().get("result"):
    qdrant_by_pointid = resp.json()["result"]
    for pt in qdrant_by_pointid:
        payload = pt.get("payload", {})
        print(json.dumps(payload, indent=2, ensure_ascii=False))
else:
    print(f"  Not found as point ID ({resp.status_code})")

# Try as canonical_id in payload filter
print(f"\n═══ Qdrant (by canonical_id filter) ═══")
resp = requests.post(
    f"{qdrant_url}/collections/{collection}/points/scroll",
    headers=headers,
    json={
        "limit": 10,
        "with_payload": True,
        "with_vector": False,
        "filter": {
            "must": [
                {"key": "canonical_id", "match": {"value": CANONICAL_ID}}
            ]
        },
    },
)
qdrant_by_filter = None
if resp.status_code == 200:
    points = resp.json()["result"].get("points", [])
    qdrant_by_filter = points
    if points:
        for pt in points:
            print(f"  point.id = {pt['id']}")
            print(json.dumps(pt.get("payload", {}), indent=2, ensure_ascii=False))
    else:
        print("  No points match canonical_id filter")
else:
    print(f"  Error: {resp.status_code} {resp.text}")

# ── Neo4j lookup ─────────────────────────────────────────────────────────────

print(f"\n═══ Neo4j ═══")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)

neo4j_node = None
with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
    # Match by canonical_id property
    result = session.run(
        "MATCH (e {canonical_id: $cid}) RETURN e, labels(e) AS labels",
        cid=CANONICAL_ID,
    )
    record = result.single()
    if record:
        neo4j_node = dict(record["e"])
        labels = record["labels"]
        print(f"  Labels: {labels}")
        print(json.dumps(neo4j_node, indent=2, ensure_ascii=False, default=str))
    else:
        print("  Not found")

driver.close()

# ── Comparison ───────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("📊 COMPARISON")
print(f"{'='*60}")

qdrant_payload = None
if qdrant_by_pointid:
    qdrant_payload = qdrant_by_pointid[0].get("payload", {})
elif qdrant_by_filter:
    qdrant_payload = qdrant_by_filter[0].get("payload", {})

if not qdrant_payload and not neo4j_node:
    print("  ❌ Not found in either system")
elif qdrant_payload and not neo4j_node:
    print("  ⚠️  Found in Qdrant only")
    print(f"     Qdrant keys: {sorted(qdrant_payload.keys())}")
elif neo4j_node and not qdrant_payload:
    print("  ⚠️  Found in Neo4j only")
    print(f"     Neo4j keys: {sorted(neo4j_node.keys())}")
else:
    # Both found — compare keys
    q_keys = set(qdrant_payload.keys())
    n_keys = set(neo4j_node.keys())

    common = q_keys & n_keys
    only_q = q_keys - n_keys
    only_n = n_keys - q_keys

    print(f"\n  Common keys ({len(common)}):")
    for k in sorted(common):
        qv = qdrant_payload.get(k)
        nv = neo4j_node.get(k)
        match = "✅" if qv == nv else "❌"
        print(f"    {match} {k}:")
        print(f"       Qdrant: {json.dumps(qv, ensure_ascii=False, default=str)}")
        print(f"       Neo4j:  {json.dumps(nv, ensure_ascii=False, default=str)}")

    if only_q:
        print(f"\n  Only in Qdrant ({len(only_q)}): {sorted(only_q)}")
        for k in sorted(only_q):
            print(f"    {k}: {json.dumps(qdrant_payload[k], ensure_ascii=False, default=str)}")

    if only_n:
        print(f"\n  Only in Neo4j ({len(only_n)}): {sorted(only_n)}")
        for k in sorted(only_n):
            print(f"    {k}: {json.dumps(neo4j_node[k], ensure_ascii=False, default=str)}")

print(f"\n✅ Done")
