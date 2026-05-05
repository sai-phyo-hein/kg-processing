"""Schema parser and validation module for knowledge graph extraction."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class SchemaParser:
    """Parse and validate against the RECAP schema definition."""

    def __init__(self, schema_path: Optional[str] = None):
        """Initialize the schema parser.

        Args:
            schema_path: Path to schema.md file (default: utils/schema.md)
        """
        if schema_path is None:
            # Default to schema.md in utils directory
            schema_path = str(Path(__file__).parent / "schema.md")

        self.schema_path = Path(schema_path)
        self.node_types: Dict[str, Dict[str, Any]] = {}
        self.relations: List[Dict[str, str]] = []
        self.enums: Dict[str, Set[str]] = {}
        self._parse_schema()

    def _parse_schema(self):
        """Parse the schema.md file."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse node types
        self._parse_node_types(content)

        # Parse relations
        self._parse_relations(content)

        # Parse enums
        self._parse_enums(content)

    def _parse_node_types(self, content: str):
        """Parse node type definitions from schema."""
        # Match node type sections
        node_pattern = r"### (\w+)\n\| Field \| Type \| Required \|\n((?:\|.*?\|\n)+)"
        matches = re.findall(node_pattern, content)

        for node_name, table_content in matches:
            fields = {}
            lines = table_content.strip().split("\n")[1:]  # Skip header row

            for line in lines:
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    field_name = parts[1]
                    field_type = parts[2]
                    required = parts[3].lower() == "yes"

                    fields[field_name] = {
                        "type": field_type,
                        "required": required,
                    }

            self.node_types[node_name] = fields

    def _parse_relations(self, content: str):
        """Parse relation definitions from schema."""
        # Match relation tables
        relation_pattern = r"\| Subject \| Relation \| Object \|\n((?:\|.*?\|\n)+)"
        matches = re.findall(relation_pattern, content)

        for table_content in matches:
            lines = table_content.strip().split("\n")[1:]  # Skip header row

            for line in lines:
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    self.relations.append({
                        "subject": parts[1],
                        "relation": parts[2],
                        "object": parts[3],
                    })

    def _parse_enums(self, content: str):
        """Parse enum definitions from schema."""
        # Match enum blocks
        enum_pattern = r"### `(\w+)`\n```\n((?:[^\n]+\n)+)```"
        matches = re.findall(enum_pattern, content)

        for enum_name, enum_content in matches:
            values = set()
            for line in enum_content.strip().split("\n"):
                if line.strip():
                    values.add(line.strip())

            self.enums[enum_name] = values

    def validate_node_type(self, node_type: str) -> bool:
        """Check if a node type is valid according to schema.

        Args:
            node_type: Node type to validate

        Returns:
            True if node type is valid, False otherwise
        """
        return node_type in self.node_types

    def validate_node_fields(
        self,
        node_type: str,
        node_data: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """Validate node fields against schema definition.

        Args:
            node_type: Type of node
            node_data: Node data dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if node_type not in self.node_types:
            return False, [f"Unknown node type: {node_type}"]

        schema_fields = self.node_types[node_type]
        errors = []

        # Check required fields
        for field_name, field_def in schema_fields.items():
            if field_def["required"] and field_name not in node_data:
                errors.append(f"Missing required field: {field_name}")

        # Check for unknown fields (optional - can be disabled if needed)
        for field_name in node_data:
            if field_name not in schema_fields:
                errors.append(f"Unknown field: {field_name}")

        # Validate enum fields
        for field_name, field_def in schema_fields.items():
            if field_name in node_data:
                field_type = field_def["type"]
                if field_type.startswith("enum → "):
                    enum_name = field_type.replace("enum → ", "")
                    if enum_name in self.enums:
                        value = node_data[field_name]
                        if value not in self.enums[enum_name]:
                            errors.append(
                                f"Invalid value for {field_name}: {value}. "
                                f"Must be one of: {', '.join(self.enums[enum_name])}"
                            )

        return len(errors) == 0, errors

    def validate_relation(
        self,
        subject_type: str,
        relation: str,
        object_type: str,
    ) -> bool:
        """Check if a relation is valid according to schema.

        Args:
            subject_type: Type of subject node
            relation: Relation name
            object_type: Type of object node

        Returns:
            True if relation is valid, False otherwise
        """
        for rel in self.relations:
            if (
                rel["subject"] == subject_type
                and rel["relation"] == relation
                and rel["object"] == object_type
            ):
                return True

        return False

    def get_valid_node_types(self) -> Set[str]:
        """Get all valid node types from schema.

        Returns:
            Set of valid node type names
        """
        return set(self.node_types.keys())

    def get_valid_relations(self) -> List[Dict[str, str]]:
        """Get all valid relations from schema.

        Returns:
            List of relation dictionaries
        """
        return self.relations.copy()

    def get_enum_values(self, enum_name: str) -> Optional[Set[str]]:
        """Get valid values for an enum.

        Args:
            enum_name: Name of the enum

        Returns:
            Set of valid enum values, or None if enum doesn't exist
        """
        return self.enums.get(enum_name)

    def validate_ingestion_payload(
        self,
        payload: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """Validate ingestion payload against schema format.

        Args:
            payload: Ingestion payload dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate required top-level fields
        required_fields = ["tambon", "village", "domain"]
        for field in required_fields:
            if field not in payload:
                errors.append(f"Missing required field: {field}")

        # Validate social_capital if present
        if "social_capital" in payload:
            sc_data = payload["social_capital"]
            if "name" not in sc_data or "level" not in sc_data:
                errors.append("social_capital must have 'name' and 'level' fields")

            # Validate level is a valid enum value
            if "level" in sc_data:
                valid_levels = self.enums.get("SocialCapitalLevel", set())
                if sc_data["level"] not in valid_levels:
                    errors.append(
                        f"Invalid social_capital level: {sc_data['level']}. "
                        f"Must be one of: {', '.join(valid_levels)}"
                    )

        # Validate activity if present
        if "activity" in payload:
            act_data = payload["activity"]
            required_act_fields = ["name", "type", "scope_level", "is_routine", "is_innovation", "status"]
            for field in required_act_fields:
                if field not in act_data:
                    errors.append(f"activity must have '{field}' field")

            # Validate scope_level is a valid enum value
            if "scope_level" in act_data:
                valid_scopes = self.enums.get("ScopeLevel", set())
                if act_data["scope_level"] not in valid_scopes:
                    errors.append(
                        f"Invalid activity scope_level: {act_data['scope_level']}. "
                        f"Must be one of: {', '.join(valid_scopes)}"
                    )

        # Validate impacts if present
        if "impacts" in payload:
            if not isinstance(payload["impacts"], list):
                errors.append("impacts must be a list")
            else:
                for i, impact in enumerate(payload["impacts"]):
                    if "name" not in impact or "type" not in impact or "evidence_strength" not in impact:
                        errors.append(
                            f"impact[{i}] must have 'name', 'type', and 'evidence_strength' fields"
                        )

                    # Validate evidence_strength is a valid enum value
                    if "evidence_strength" in impact:
                        valid_strengths = self.enums.get("EvidenceStrength", set())
                        if impact["evidence_strength"] not in valid_strengths:
                            errors.append(
                                f"Invalid impact evidence_strength: {impact['evidence_strength']}. "
                                f"Must be one of: {', '.join(valid_strengths)}"
                            )

        # Validate domain is a valid enum value
        if "domain" in payload:
            valid_domains = self.enums.get("DomainCode", set())
            if payload["domain"] not in valid_domains:
                errors.append(
                    f"Invalid domain: {payload['domain']}. "
                    f"Must be one of: {', '.join(valid_domains)}"
                )

        return len(errors) == 0, errors


# Global schema parser instance
_schema_parser: Optional[SchemaParser] = None


def get_schema_parser(schema_path: Optional[str] = None) -> SchemaParser:
    """Get or create the global schema parser instance.

    Args:
        schema_path: Optional path to schema.md file

    Returns:
        SchemaParser instance
    """
    global _schema_parser
    if _schema_parser is None or schema_path is not None:
        _schema_parser = SchemaParser(schema_path)
    return _schema_parser


def reset_schema_parser():
    """Reset the global schema parser instance."""
    global _schema_parser
    _schema_parser = None