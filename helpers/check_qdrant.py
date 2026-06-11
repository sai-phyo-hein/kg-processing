"""Check Qdrant connectivity and response times."""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("QDRANT_URL").rstrip("/")
api_key = os.getenv("QDRANT_API_KEY")
headers = {"api-key": api_key, "Content-Type": "application/json"}

# 1. Basic connectivity
print("=== 1. Connectivity ===")
t = time.time()
resp = requests.get(f"{url}/", headers=headers)
print(f"  GET /  →  {resp.status_code}  ({(time.time()-t)*1000:.0f}ms)")

# 2. Collection info
print("\n=== 2. Collection info ===")
t = time.time()
resp = requests.get(f"{url}/collections/ci_evidence_registry", headers=headers)
info = resp.json()
print(f"  GET /collections/ci_evidence_registry  →  {resp.status_code}  ({(time.time()-t)*1000:.0f}ms)")
print(f"  points_count:          {info['result']['points_count']}")
print(f"  indexed_vectors_count: {info['result']['indexed_vectors_count']}")
print(f"  optimizer_status:      {info['result']['optimizer_status']}")

# 3. Simple scroll (no vector, no filter)
print("\n=== 3. Scroll (1 point, no vector) ===")
t = time.time()
resp = requests.post(
    f"{url}/collections/ci_evidence_registry/points/scroll",
    headers=headers,
    json={"limit": 1, "with_payload": False, "with_vector": False},
)
print(f"  POST /scroll  →  {resp.status_code}  ({(time.time()-t)*1000:.0f}ms)")

# 4. Vector search (the operation that timed out)
print("\n=== 4. Vector search (dummy zero vector) ===")
dummy = [0.0] * 3072
t = time.time()
resp = requests.post(
    f"{url}/collections/ci_evidence_registry/points/query",
    headers=headers,
    json={
        "query": dummy,
        "using": "evidence_quote_en",
        "limit": 1,
        "with_payload": False,
        "score_threshold": 1.0,
    },
)
print(f"  POST /query  →  {resp.status_code}  ({(time.time()-t)*1000:.0f}ms)")
if resp.status_code != 200:
    print(f"  Error: {resp.text}")