"""DuckDB utilities for metadata storage and retrieval."""

import duckdb
from pathlib import Path
from typing import Dict, Any, List, Optional


class MetadataDB:
    """DuckDB interface for metadata storage."""

    def __init__(self, db_path: str = "metadata.db"):
        """Initialize MetadataDB connection.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        """Create metadata table if it doesn't exist."""
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                unique_id VARCHAR PRIMARY KEY,
                document_title VARCHAR,
                document_file_name VARCHAR,
                document_file_type VARCHAR,
                document_total_pages INTEGER,
                document_content_type VARCHAR,
                location_village VARCHAR,
                location_moo VARCHAR,
                location_country VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.close()

    def insert_metadata(self, metadata: Dict[str, Any]) -> None:
        """Insert or update metadata record.

        Args:
            metadata: Metadata dictionary with required fields
        """
        conn = duckdb.connect(self.db_path)

        # Prepare values with defaults for missing fields
        values = {
            'unique_id': metadata.get('unique_id', ''),
            'document_title': metadata.get('document_title', ''),
            'document_file_name': metadata.get('document_file_name', ''),
            'document_file_type': metadata.get('document_file_type', ''),
            'document_total_pages': metadata.get('document_total_pages', 0),
            'document_content_type': metadata.get('document_content_type', ''),
            'location_village': metadata.get('location_village', ''),
            'location_moo': metadata.get('location_moo', ''),
            'location_country': metadata.get('location_country', ''),
        }

        # Insert or replace record
        conn.execute("""
            INSERT OR REPLACE INTO metadata (
                unique_id, document_title, document_file_name, document_file_type,
                document_total_pages, document_content_type, location_village,
                location_moo, location_country, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            values['unique_id'], values['document_title'], values['document_file_name'],
            values['document_file_type'], values['document_total_pages'],
            values['document_content_type'], values['location_village'],
            values['location_moo'], values['location_country']
        ])

        conn.close()
        print(f"💾 Metadata saved to database: {values['unique_id']}")

    def get_metadata(self, unique_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata by unique ID.

        Args:
            unique_id: Unique identifier for the metadata record

        Returns:
            Metadata dictionary or None if not found
        """
        conn = duckdb.connect(self.db_path)
        result = conn.execute(
            "SELECT * FROM metadata WHERE unique_id = ?", [unique_id]
        ).fetchone()
        conn.close()

        if result:
            columns = [
                'unique_id', 'document_title', 'document_file_name', 'document_file_type',
                'document_total_pages', 'document_content_type', 'location_village',
                'location_moo', 'location_country', 'created_at', 'updated_at'
            ]
            return dict(zip(columns, result))
        return None

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Retrieve all metadata records.

        Returns:
            List of metadata dictionaries
        """
        conn = duckdb.connect(self.db_path)
        results = conn.execute("SELECT * FROM metadata ORDER BY updated_at DESC").fetchall()
        conn.close()

        columns = [
            'unique_id', 'document_title', 'document_file_name', 'document_file_type',
            'document_total_pages', 'document_content_type', 'location_village',
            'location_moo', 'location_country', 'created_at', 'updated_at'
        ]
        return [dict(zip(columns, row)) for row in results]

    def search_by_location(self, moo: str = None, village: str = None) -> List[Dict[str, Any]]:
        """Search metadata by location.

        Args:
            moo: หมู่ number to search for
            village: Village name to search for

        Returns:
            List of matching metadata dictionaries
        """
        conn = duckdb.connect(self.db_path)

        query = "SELECT * FROM metadata WHERE 1=1"
        params = []

        if moo:
            query += " AND location_moo LIKE ?"
            params.append(f"%{moo}%")

        if village:
            query += " AND location_village LIKE ?"
            params.append(f"%{village}%")

        query += " ORDER BY updated_at DESC"

        results = conn.execute(query, params).fetchall()
        conn.close()

        columns = [
            'unique_id', 'document_title', 'document_file_name', 'document_file_type',
            'document_total_pages', 'document_content_type', 'location_village',
            'location_moo', 'location_country', 'created_at', 'updated_at'
        ]
        return [dict(zip(columns, row)) for row in results]

    def delete_metadata(self, unique_id: str) -> bool:
        """Delete metadata by unique ID.

        Args:
            unique_id: Unique identifier for the metadata record

        Returns:
            True if deleted, False if not found
        """
        conn = duckdb.connect(self.db_path)
        result = conn.execute("DELETE FROM metadata WHERE unique_id = ?", [unique_id])
        conn.close()
        return result.fetchone()[0] > 0


def save_metadata(metadata: Dict[str, Any], db_path: str = "metadata.db") -> None:
    """Save metadata to DuckDB.

    Args:
        metadata: Metadata dictionary
        db_path: Path to DuckDB database file
    """
    db = MetadataDB(db_path)
    db.insert_metadata(metadata)


def load_metadata(unique_id: str, db_path: str = "metadata.db") -> Optional[Dict[str, Any]]:
    """Load metadata from DuckDB.

    Args:
        unique_id: Unique identifier
        db_path: Path to DuckDB database file

    Returns:
        Metadata dictionary or None
    """
    db = MetadataDB(db_path)
    return db.get_metadata(unique_id)
