#!/usr/bin/env python3
"""
Analyze how much of the original chunk content remains in evidence quotes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def extract_chunk_content(chunks_file: str) -> str:
    """Extract all content from chunks file."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Combine all chunk content
    all_content = []
    for chunk in data.get('chunks', []):
        content = chunk.get('content', '')
        if content:
            all_content.append(content)
    
    return ' '.join(all_content)


def extract_evidence_quotes(triples_file: str) -> List[str]:
    """Extract all evidence quotes from triples file."""
    with open(triples_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    evidence_quotes = []
    for chunk in data.get('chunks', []):
        for triple in chunk.get('triples', []):
            properties = triple.get('properties', {})
            evidence_quote = properties.get('evidence_quote', '')
            if evidence_quote:
                evidence_quotes.append(evidence_quote)
    
    return evidence_quotes


def calculate_coverage(chunk_content: str, evidence_quotes: List[str]) -> Dict:
    """Calculate how much of chunk content is covered by evidence quotes."""
    
    # Total characters in chunks
    total_chunk_chars = len(chunk_content)
    
    # Total characters in evidence quotes
    all_evidence_text = ' '.join(evidence_quotes)
    total_evidence_chars = len(all_evidence_text)
    
    # Calculate unique content coverage
    # Count how many characters from chunks appear in evidence quotes
    matched_chars = 0
    chunk_words = set(chunk_content.split())
    evidence_words = set(all_evidence_text.split())
    
    # Word-level matching
    matched_words = chunk_words.intersection(evidence_words)
    total_chunk_words = len(chunk_words)
    total_evidence_words = len(evidence_words)
    matched_word_count = len(matched_words)
    
    # Character-level exact substring matching
    # Find what percentage of chunk content appears verbatim in evidence
    chunk_sentences = [s.strip() for s in chunk_content.replace('\n', ' ').split('.') if s.strip()]
    evidence_text_combined = all_evidence_text
    
    matched_sentence_chars = 0
    total_sentence_chars = 0
    
    for sentence in chunk_sentences:
        sentence_len = len(sentence)
        total_sentence_chars += sentence_len
        
        # Check if sentence (or significant part) appears in evidence
        if sentence in evidence_text_combined:
            matched_sentence_chars += sentence_len
        else:
            # Check for partial matches (80% or more of sentence)
            words_in_sentence = sentence.split()
            if len(words_in_sentence) > 3:
                for i in range(len(words_in_sentence) - 2):
                    phrase = ' '.join(words_in_sentence[i:i+3])
                    if phrase in evidence_text_combined:
                        matched_sentence_chars += len(phrase)
                        break
    
    return {
        'total_chunk_chars': total_chunk_chars,
        'total_evidence_chars': total_evidence_chars,
        'total_chunk_words': total_chunk_words,
        'total_evidence_words': total_evidence_words,
        'matched_words': matched_word_count,
        'matched_sentence_chars': matched_sentence_chars,
        'total_sentence_chars': total_sentence_chars,
        'evidence_quote_count': len(evidence_quotes),
        'unique_evidence_quotes': len(set(evidence_quotes)),
        # Percentages
        'word_coverage_pct': (matched_word_count / total_chunk_words * 100) if total_chunk_words > 0 else 0,
        'sentence_coverage_pct': (matched_sentence_chars / total_sentence_chars * 100) if total_sentence_chars > 0 else 0,
        'evidence_to_chunk_ratio_pct': (total_evidence_chars / total_chunk_chars * 100) if total_chunk_chars > 0 else 0,
    }


def print_analysis(stats: Dict, chunks_file: str, triples_file: str):
    """Print the analysis results."""
    print("=" * 80)
    print("Evidence Coverage Analysis")
    print("=" * 80)
    print(f"\nInput Files:")
    print(f"  Chunks: {Path(chunks_file).name}")
    print(f"  Triples: {Path(triples_file).name}")
    print()
    
    print("Content Statistics:")
    print(f"  Total chunk characters: {stats['total_chunk_chars']:,}")
    print(f"  Total evidence characters: {stats['total_evidence_chars']:,}")
    print(f"  Total chunk words: {stats['total_chunk_words']:,}")
    print(f"  Total evidence words: {stats['total_evidence_words']:,}")
    print(f"  Evidence quotes count: {stats['evidence_quote_count']:,}")
    print(f"  Unique evidence quotes: {stats['unique_evidence_quotes']:,}")
    print()
    
    print("Coverage Metrics:")
    print(f"  Word coverage: {stats['word_coverage_pct']:.2f}%")
    print(f"    ({stats['matched_words']:,} out of {stats['total_chunk_words']:,} unique words)")
    print()
    print(f"  Sentence coverage: {stats['sentence_coverage_pct']:.2f}%")
    print(f"    ({stats['matched_sentence_chars']:,} out of {stats['total_sentence_chars']:,} sentence characters)")
    print()
    print(f"  Evidence to chunk ratio: {stats['evidence_to_chunk_ratio_pct']:.2f}%")
    print(f"    (total evidence chars / total chunk chars)")
    print()
    
    print("Interpretation:")
    if stats['word_coverage_pct'] > 70:
        print("  ✓ High coverage: Most chunk content is represented in evidence quotes")
    elif stats['word_coverage_pct'] > 40:
        print("  ~ Moderate coverage: Fair amount of chunk content in evidence quotes")
    else:
        print("  ✗ Low coverage: Only small portion of chunk content in evidence quotes")
    
    print("=" * 80)


def main():
    if len(sys.argv) != 3:
        print("Usage: uv run analyze_evidence_coverage.py <chunks_file> <triples_file>")
        print("\nExample:")
        print("  uv run helpers/analyze_evidence_coverage.py \\")
        print(r"    'output/ปฏิบัติการที่-1_หมู่-1  เสร็จแล้ว_chunks.json' \\")
        print(r"    'output/ปฏิบัติการที่-1_หมู่-1  เสร็จแล้ว_triples.json'")
        sys.exit(1)
    
    chunks_file = sys.argv[1]
    triples_file = sys.argv[2]
    
    # Check if files exist
    if not Path(chunks_file).exists():
        print(f"Error: Chunks file not found: {chunks_file}")
        sys.exit(1)
    
    if not Path(triples_file).exists():
        print(f"Error: Triples file not found: {triples_file}")
        sys.exit(1)
    
    print("Loading data...")
    chunk_content = extract_chunk_content(chunks_file)
    evidence_quotes = extract_evidence_quotes(triples_file)
    
    print("Analyzing coverage...")
    stats = calculate_coverage(chunk_content, evidence_quotes)
    
    print_analysis(stats, chunks_file, triples_file)
    
    # Save detailed report
    output_file = Path(triples_file).parent / f"{Path(triples_file).stem}_coverage_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
