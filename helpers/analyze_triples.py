#!/usr/bin/env python3
"""
Analyze triples JSON file - summarize and calculate statistics.

Usage:
    python analyze_triples.py <path_to_triples_json>
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def analyze_triples_file(file_path: str) -> Dict[str, Any]:
    """
    Analyze a triples JSON file and return statistics.
    
    Args:
        file_path: Path to the triples JSON file
        
    Returns:
        Dictionary containing analysis results
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_chunks = data.get('total_chunks', 0)
    total_triples = data.get('total_triples', 0)
    chunks = data.get('chunks', [])
    
    # Calculate triples per chunk
    triples_per_chunk = []
    chunk_details = []
    
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id')
        num_triples = len(chunk.get('triples', []))
        triples_per_chunk.append(num_triples)
        chunk_details.append({
            'chunk_id': chunk_id,
            'num_triples': num_triples
        })
    
    # Calculate statistics
    avg_triples = sum(triples_per_chunk) / len(triples_per_chunk) if triples_per_chunk else 0
    min_triples = min(triples_per_chunk) if triples_per_chunk else 0
    max_triples = max(triples_per_chunk) if triples_per_chunk else 0
    
    # Find chunks with min and max triples
    chunks_with_min = [c['chunk_id'] for c in chunk_details if c['num_triples'] == min_triples]
    chunks_with_max = [c['chunk_id'] for c in chunk_details if c['num_triples'] == max_triples]
    
    # Distribution analysis
    distribution = {}
    for count in triples_per_chunk:
        distribution[count] = distribution.get(count, 0) + 1
    
    return {
        'file_path': file_path,
        'source_file': data.get('source_file', 'N/A'),
        'llm_provider': data.get('llm_provider', 'N/A'),
        'llm_model': data.get('llm_model', 'N/A'),
        'total_chunks': total_chunks,
        'total_triples': total_triples,
        'avg_triples_per_chunk': avg_triples,
        'min_triples_per_chunk': min_triples,
        'max_triples_per_chunk': max_triples,
        'chunks_with_min_triples': chunks_with_min,
        'chunks_with_max_triples': chunks_with_max,
        'distribution': distribution,
        'chunk_details': chunk_details
    }


def print_summary(analysis: Dict[str, Any], show_details: bool = False) -> None:
    """
    Print a formatted summary of the analysis.
    
    Args:
        analysis: Analysis results dictionary
        show_details: Whether to show per-chunk details
    """
    print("=" * 80)
    print("TRIPLES FILE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\n📁 File: {Path(analysis['file_path']).name}")
    print(f"📄 Source: {Path(analysis['source_file']).name}")
    print(f"🤖 LLM: {analysis['llm_provider']} / {analysis['llm_model']}")
    
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total Chunks:           {analysis['total_chunks']}")
    print(f"Total Triples:          {analysis['total_triples']}")
    print(f"Avg Triples/Chunk:      {analysis['avg_triples_per_chunk']:.2f}")
    print(f"Min Triples/Chunk:      {analysis['min_triples_per_chunk']}")
    print(f"Max Triples/Chunk:      {analysis['max_triples_per_chunk']}")
    
    print("\n" + "-" * 80)
    print("DISTRIBUTION OF TRIPLES PER CHUNK")
    print("-" * 80)
    print(f"{'Triples Count':<20} {'Number of Chunks':<20} {'Percentage':<20}")
    print("-" * 80)
    
    sorted_dist = sorted(analysis['distribution'].items())
    for triples_count, num_chunks in sorted_dist:
        percentage = (num_chunks / analysis['total_chunks']) * 100
        print(f"{triples_count:<20} {num_chunks:<20} {percentage:.1f}%")
    
    print("\n" + "-" * 80)
    print("EXTREME CASES")
    print("-" * 80)
    print(f"Chunks with minimum triples ({analysis['min_triples_per_chunk']}): {analysis['chunks_with_min_triples'][:10]}")
    if len(analysis['chunks_with_min_triples']) > 10:
        print(f"  ... and {len(analysis['chunks_with_min_triples']) - 10} more")
    
    print(f"Chunks with maximum triples ({analysis['max_triples_per_chunk']}): {analysis['chunks_with_max_triples'][:10]}")
    if len(analysis['chunks_with_max_triples']) > 10:
        print(f"  ... and {len(analysis['chunks_with_max_triples']) - 10} more")
    
    if show_details:
        print("\n" + "-" * 80)
        print("PER-CHUNK DETAILS")
        print("-" * 80)
        print(f"{'Chunk ID':<15} {'Number of Triples':<20}")
        print("-" * 80)
        for chunk in analysis['chunk_details']:
            print(f"{chunk['chunk_id']:<15} {chunk['num_triples']:<20}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_triples.py <path_to_triples_json> [--details]")
        print("\nExample:")
        print("  python analyze_triples.py output/file_triples.json")
        print("  python analyze_triples.py output/file_triples.json --details")
        sys.exit(1)
    
    file_path = sys.argv[1]
    show_details = '--details' in sys.argv
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        analysis = analyze_triples_file(file_path)
        print_summary(analysis, show_details=show_details)
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
