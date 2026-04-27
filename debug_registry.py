"""Debug registry specs loading."""

from pathlib import Path
import json

# Check if registry info directory exists
registry_dir = Path('/home/saiphyohein/claude-code-learn/registry_info')
print(f'Registry directory exists: {registry_dir.exists()}')
print(f'Registry directory: {registry_dir}')

# Check if spec files exist
for collection in ['entity_registry', 'predicate_registry', 'ontology_registry']:
    spec_file = registry_dir / f'{collection}.json'
    print(f'{collection}.json exists: {spec_file.exists()}')
    if spec_file.exists():
        with open(spec_file) as f:
            spec = json.load(f)
            print(f'  Vector name: {spec.get("vector_name")}')
