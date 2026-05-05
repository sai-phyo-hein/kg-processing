"""Test query relationships from Neo4j."""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)

print("🔍 Querying Neo4j for sample relationships...\n")

with driver.session() as session:
    # Get total counts
    result = session.run("MATCH (n) RETURN count(n) as node_count")
    node_count = result.single()["node_count"]

    result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
    rel_count = result.single()["rel_count"]

    print(f"📊 Total nodes: {node_count}")
    print(f"📊 Total relationships: {rel_count}\n")

    # Get sample relationships
    query = """
    MATCH (source:Entity)-[r]->(target:Entity)
    RETURN source.name as source_name, type(r) as relationship, target.name as target_name, r
    LIMIT 5
    """

    result = session.run(query)

    print("🔗 Sample relationships:")
    for i, record in enumerate(result, 1):
        source_name = record["source_name"]
        relationship = record["relationship"]
        target_name = record["target_name"]
        props = dict(record["r"])

        print(f"\n{i}. {source_name} --[{relationship}]--> {target_name}")
        if props:
            print(f"   Properties: {props}")

    # Get relationship types
    result = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type")
    rel_types = [record["rel_type"] for record in result]

    print(f"\n🏷️  Relationship types ({len(rel_types)}):")
    for rel_type in sorted(rel_types):
        print(f"   - {rel_type}")

driver.close()
print("\n✅ Query completed!")