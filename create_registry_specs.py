"""Create registry specification files."""

import json
from pathlib import Path

# Create registry_info directory
registry_dir = Path("/home/saiphyohein/claude-code-learn/registry_info")
registry_dir.mkdir(exist_ok=True)

# Entity registry specification
entity_registry_spec = {
    "collection_name": "entity_registry",
    "description": "Entity registry for subject and object entities",
    "sparse_vector_name": "entity_vector",
    "vector_config": {
        "datatype": "float32",
        "distance": "Cosine",
        "hnsw_config": {
            "ef_construct": 256,
            "m": 24,
            "payload_m": 24
        },
        "on_disk": False,
        "size": 1536
    },
    "vector_name": "entity"
}

# Predicate registry specification
predicate_registry_spec = {
    "collection_name": "predicate_registry",
    "description": "Predicate registry for relationship predicates",
    "sparse_vector_name": "predicate_vector",
    "vector_config": {
        "datatype": "float32",
        "distance": "Cosine",
        "hnsw_config": {
            "ef_construct": 256,
            "m": 24,
            "payload_m": 24
        },
        "on_disk": False,
        "size": 1536
    },
    "vector_name": "predicate"
}

# Ontology registry specification
ontology_registry_spec = {
    "collection_name": "ontology_registry",
    "description": "Ontology registry for entity types",
    "sparse_vector_name": "ontology_vector",
    "vector_config": {
        "datatype": "float32",
        "distance": "Cosine",
        "hnsw_config": {
            "ef_construct": 256,
            "m": 24,
            "payload_m": 24
        },
        "on_disk": False,
        "size": 1536
    },
    "vector_name": "ontology"
}

# Write specification files
with open(registry_dir / "entity_registry.json", "w") as f:
    json.dump(entity_registry_spec, f, indent=2)

with open(registry_dir / "predicate_registry.json", "w") as f:
    json.dump(predicate_registry_spec, f, indent=2)

with open(registry_dir / "ontology_registry.json", "w") as f:
    json.dump(ontology_registry_spec, f, indent=2)

print("Registry specification files created successfully!")
print(f"Files created in: {registry_dir}")
