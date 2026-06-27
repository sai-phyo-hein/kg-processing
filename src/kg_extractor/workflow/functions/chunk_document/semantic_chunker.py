"""Semantic chunking module for markdown documents based on LLM line-range analysis.

The chunker sends numbered source lines to an LLM, which returns explicit line ranges
(e.g. chunk1: 1~14, chunk2: 15~31). A post-processing step then slices the original
source text using those ranges and writes each chunk to its own file.
"""

import json
import os
import re
import tiktoken
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.model_setup import CHUNKING_PROVIDER, CHUNKING_MODEL
from kg_extractor.utils.prompts import create_chunking_prompt
from kg_extractor.utils.llm_response_parser import parse_chunk_ranges_response

# Optional S3 upload of chunk files (boto3). Soft-imported so chunking still
# runs in environments without boto3 — the upload is a best-effort side-effect.
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    HAS_BOTO3 = True
except ImportError:  # pragma: no cover
    HAS_BOTO3 = False


class SemanticChunker:
    """Chunk markdown documents using LLM-assisted line-range approach.

    Process:
    1. Read source → split into lines
    2. LLM Analysis: send numbered lines, receive chunk ranges (start ~ end)
    3. Post-process: extract content from original source using those ranges
    4. Output: each chunk saved as a separate file

    Features:
    - LLM returns only line-number ranges (minimal token output)
    - Content extracted programmatically from original source (no rewrites)
    - Individual chunk files in a directory with a manifest.json
    - Sliding window for large documents with context overlap
    - Graceful fallback to rule-based chunking
    """

    def __init__(
        self,
        chunk_granularity: float = 0.1,
        llm_provider: str = CHUNKING_PROVIDER,
        llm_model: str = CHUNKING_MODEL,
        max_retries: int = 3,
        enable_fallback: bool = True,
        start_chunk_id: int = 1,
    ):
        self.chunk_granularity = chunk_granularity
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback
        # First chunk_id to assign. Defaults to 1, but when a community already
        # has chunks in S3 we resume numbering so ids stay unique across
        # separately processed files (no chunk_NNN.txt overwrite). See
        # ``chunk_markdown_file`` / ``get_max_chunk_id_from_s3``.
        self.start_chunk_id = start_chunk_id
        self.context_limit = 50_000  # Token limit per section
        self.CONTEXT_OVERLAP = 50    # Lines to overlap between sections
        self.SAFETY_MARGIN = 500

    # ── Token counting ──────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        try:
            if "gpt-4" in self.llm_model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.llm_model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Warning: Failed to count tokens: {e}")
            return len(text) // 4

    # ── Source preparation ──────────────────────────────────────────────

    @staticmethod
    def _extract_content_between_flags(content: str) -> str:
        match = re.search(r'<start>(.*?)</end>', content, re.DOTALL)
        return match.group(1) if match else content

    MAX_LINE_WIDTH = 120  # characters

    def _prepare_source_lines(self, file_path: str) -> List[str]:
        """Read file, extract content between flags, wrap long lines, return list of lines."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        extracted = self._extract_content_between_flags(content)
        if not extracted:
            extracted = content

        raw_lines = extracted.split("\n")

        # Wrap lines that exceed MAX_LINE_WIDTH using textwrap
        import textwrap
        wrapped: List[str] = []
        for line in raw_lines:
            stripped = line.rstrip("\n")
            if len(stripped) <= self.MAX_LINE_WIDTH:
                wrapped.append(stripped)
            else:
                # Preserve leading indent style for subsequent wrapped lines
                indent = " " * (len(stripped) - len(stripped.lstrip()))
                wrapped.extend(textwrap.wrap(
                    stripped,
                    width=self.MAX_LINE_WIDTH,
                    subsequent_indent=indent,
                    break_long_words=False,
                    break_on_hyphens=False,
                ))
        # Preserve trailing newlines on all but the last line
        return [
            line + "\n" if i < len(wrapped) - 1 else line
            for i, line in enumerate(wrapped)
        ]

    # ── Numbered content builder ────────────────────────────────────────

    @staticmethod
    def _build_numbered_content(lines: List[str], start_number: int = 1) -> str:
        """Build numbered-line text for the LLM prompt.

        Each line becomes ``[NNNN] <text>`` (no trailing newline in the output
        string — lines are joined with ``\\n``).
        """
        numbered = []
        for i, line in enumerate(lines):
            num = start_number + i
            text = line.rstrip("\n")
            numbered.append(f"[{num:04d}] {text}")
        return "\n".join(numbered)

    # ── Section sizing ──────────────────────────────────────────────────

    def _calculate_section_size(self, lines: List[str], context_overlap: int) -> int:
        """Return number of lines to include in the current section."""
        section_lines = len(lines)
        max_section_lines = 1000

        if section_lines > max_section_lines:
            section_lines = max_section_lines

        while section_lines > 100:
            test_content = "".join(lines[:section_lines])
            token_count = self._count_tokens(test_content)
            if token_count + 1000 <= self.context_limit:
                return min(section_lines + context_overlap, len(lines))
            section_lines = int(section_lines * 0.8)

        return max(section_lines, 100)

    # ── LLM interaction ─────────────────────────────────────────────────

    def _get_llm_response_with_retry(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                return self._get_llm_response(prompt)
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(
                        f"Failed after {self.max_retries} attempts: {e}"
                    )
        return ""

    def _get_llm_response(self, prompt: str) -> str:
        try:
            from kg_extractor.utils.model_setup import get_llm_response
            return get_llm_response(
                prompt, self.llm_provider, self.llm_model, temperature=0.3
            )
        except Exception as e:
            print(f"Warning: Failed to get LLM response: {e}")
            return '{"chunks": []}'

    # ── LLM prompt builder ──────────────────────────────────────────────

    def _create_chunking_prompt(
        self,
        numbered_content: str,
        file_path: str,
        section_number: int,
        section_line_count: int,
    ) -> str:
        base_prompt = create_chunking_prompt(
            content=numbered_content,
            file_path=file_path,
            chunk_granularity=self.chunk_granularity,
        )
        return (
            f"{base_prompt}\n\n"
            f"**SECTION CONTEXT:**\n"
            f"- This is section {section_number} of a larger document\n"
            f"- Section contains {section_line_count} lines\n"
            f"- Return ONLY line ranges as JSON — no content, no explanation\n\n"
            f"Respond with ONLY the JSON object:"
        )

    # ── Main public API ─────────────────────────────────────────────────

    def chunk_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Chunk a markdown file using LLM line-range approach.

        1. Read source → prepare lines
        2. LLM returns explicit line ranges for each chunk
        3. Post-process: extract content from source using those ranges

        Returns:
            List of chunk dicts with chunk_id, content, start_line, end_line.
        """
        # Step 1 — prepare source lines
        print("📋 Step 1: Preparing source lines...")
        source_lines = self._prepare_source_lines(file_path)
        print(f"  📝 {len(source_lines)} source lines")

        if not source_lines or all(l.strip() == "" for l in source_lines):
            raise ValueError("No content found in document")

        # Step 2 — get chunk ranges from LLM
        print("🧠 Step 2: Getting chunk ranges from LLM...")
        chunk_ranges = self._get_chunk_ranges(source_lines, file_path)

        if not chunk_ranges:
            print("⚠️  LLM returned no chunk ranges, using fallback...")
            if self.enable_fallback:
                return self._fallback_chunking(source_lines, file_path)
            raise RuntimeError("Failed to get chunk ranges from LLM")

        # Step 3 — post-process: extract content from source using ranges
        print("📦 Step 3: Post-processing — extracting chunks from source...")
        chunks = self._post_process_chunks(source_lines, chunk_ranges)

        print(f"✅ Successfully created {len(chunks)} chunks")
        return chunks

    # ── Step 2: get chunk ranges from LLM (sliding window) ──────────────

    def _get_chunk_ranges(
        self, source_lines: List[str], file_path: str
    ) -> List[Dict[str, int]]:
        """Send numbered lines to LLM and collect chunk ranges across sections.

        Returns:
            List of ``{"chunk_id": N, "start": M, "end": K}`` where start and
            end are **0-based** indices into *source_lines* (inclusive).
        """
        all_ranges: List[Dict[str, int]] = []
        current_line = 0
        total_lines = len(source_lines)
        section_number = 1
        chunk_id_counter = self.start_chunk_id
        context_overlap = self.CONTEXT_OVERLAP
        MIN_LINES_PER_CHUNK = 3

        while current_line < total_lines:
            section_size = self._calculate_section_size(
                source_lines[current_line:], context_overlap
            )
            section_end = min(current_line + section_size, total_lines)

            print(
                f"  📖 Section {section_number} "
                f"(lines {current_line + 1}–{section_end})..."
            )

            section_lines = source_lines[current_line:section_end]
            numbered_content = self._build_numbered_content(
                section_lines, start_number=1
            )

            prompt = self._create_chunking_prompt(
                numbered_content, file_path, section_number, len(section_lines)
            )

            try:
                response = self._get_llm_response_with_retry(prompt)
                print(f"  🤖 LLM response for section {section_number}:\n{response}\n")
                # [{"id": 1, "start": 1, "end": 10}, ...]  (1-based within section)
                raw_ranges = parse_chunk_ranges_response(response, len(section_lines))

                if not raw_ranges:
                    print(
                        f"  ⚠️  No ranges for section {section_number}, "
                        "treating as one chunk"
                    )
                    raw_ranges = [
                        {"id": 1, "start": 1, "end": len(section_lines)}
                    ]

                # Merge ranges that are too small
                if len(raw_ranges) > 1:
                    filtered = [raw_ranges[0]]
                    for r in raw_ranges[1:]:
                        if r["start"] - filtered[-1]["start"] >= MIN_LINES_PER_CHUNK:
                            filtered.append(r)
                        else:
                            filtered[-1]["end"] = r["end"]
                    if len(filtered) < len(raw_ranges):
                        print(
                            f"  ℹ️  Merged small ranges: "
                            f"{len(raw_ranges)} → {len(filtered)}"
                        )
                    raw_ranges = filtered

                # Convert 1-based-section-relative → 0-based-absolute
                for r in raw_ranges:
                    all_ranges.append(
                        {
                            "chunk_id": chunk_id_counter,
                            "start": current_line + (r["start"] - 1),
                            "end": current_line + r["end"] - 1,  # inclusive
                        }
                    )
                    chunk_id_counter += 1

                print(f"  ✅ Section {section_number}: {len(raw_ranges)} chunks")

            except Exception as e:
                print(f"  ⚠️  Section {section_number} failed: {e}")
                if self.enable_fallback:
                    all_ranges.append(
                        {
                            "chunk_id": chunk_id_counter,
                            "start": current_line,
                            "end": section_end - 1,
                        }
                    )
                    chunk_id_counter += 1
                else:
                    raise RuntimeError(f"Failed on section {section_number}: {e}")

            advance = max(section_size - context_overlap, 1)
            current_line += advance
            section_number += 1

        return all_ranges

    # ── Step 3: post-process — extract content from source ──────────────

    def _post_process_chunks(
        self, source_lines: List[str], chunk_ranges: List[Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        """Extract chunk content from original source lines using LLM ranges.

        Args:
            source_lines: Original lines from the source document
            chunk_ranges: ``[{"chunk_id": N, "start": M, "end": K}]``
                          (0-based, inclusive)

        Returns:
            List of chunk dicts ready for saving.
        """
        chunks: List[Dict[str, Any]] = []

        for r in chunk_ranges:
            start = r["start"]
            end = r["end"] + 1  # inclusive → exclusive
            content = "".join(source_lines[start:end])
            content = self._normalize_content(content.strip())

            if content:
                chunks.append(
                    {
                        "chunk_id": r["chunk_id"],
                        "content": content,
                        "start_line": start + 1,  # 1-based for display
                        "end_line": end,           # 1-based for display
                    }
                )

        return chunks

    # ── Saving ──────────────────────────────────────────────────────────

    def save_chunks(self, chunks: List[Dict[str, Any]], output_dir: str) -> str:
        """Save chunks as individual files in a directory.

        Creates::

            {output_dir}/
                manifest.json      — metadata + chunk index
                chunk_001.txt      — individual chunk content
                chunk_002.txt
                ...

        Args:
            chunks: List of chunk dicts (with content, start_line, end_line)
            output_dir: Directory to create chunk files in

        Returns:
            Path to ``manifest.json``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        chunk_sizes = [len(c.get("content", "")) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

        manifest: Dict[str, Any] = {
            "total_chunks": len(chunks),
            "processing_metadata": {
                "method": "line_range_sliding_window",
                "chunk_granularity": self.chunk_granularity,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "context_limit": self.context_limit,
                "context_overlap": self.CONTEXT_OVERLAP,
                "start_chunk_id": self.start_chunk_id,
            },
            "chunk_statistics": {
                "total_characters": sum(chunk_sizes),
                "average_chunk_size": round(avg_size, 2),
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            },
            "chunks": [],
        }

        for chunk in chunks:
            chunk_filename = f"chunk_{chunk['chunk_id']:03d}.txt"
            chunk_path = out / chunk_filename
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk["content"])

            manifest["chunks"].append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                    "file": chunk_filename,
                    "size": len(chunk.get("content", "")),
                }
            )

        manifest_path = out / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(chunks)} chunk files to: {out}")
        return str(manifest_path)

    # ── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_content(content: str) -> str:
        return content

    def _fallback_chunking(
        self, source_lines: List[str], file_path: str
    ) -> List[Dict[str, Any]]:
        """Fallback rule-based chunking by markdown headers."""
        print("📋 Using rule-based fallback chunking...")

        chunks: List[Dict[str, Any]] = []
        current_lines: List[str] = []
        chunk_id = self.start_chunk_id
        chunk_start = 0

        for i, line in enumerate(source_lines):
            if line.lstrip().startswith("#") and current_lines:
                content = self._normalize_content(
                    "".join(current_lines).strip()
                )
                if content:
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "content": content,
                            "start_line": chunk_start + 1,
                            "end_line": i,
                        }
                    )
                    chunk_id += 1
                current_lines = [line]
                chunk_start = i
            else:
                current_lines.append(line)

        if current_lines:
            content = self._normalize_content(
                "".join(current_lines).strip()
            )
            if content:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "content": content,
                        "start_line": chunk_start + 1,
                        "end_line": len(source_lines),
                    }
                )

        print(f"✅ Created {len(chunks)} chunks via fallback")
        return chunks


# ── Public entry point ──────────────────────────────────────────────────


def chunk_markdown_file(
    file_path: str,
    chunk_granularity: float = 0.1,
    llm_provider: str = CHUNKING_PROVIDER,
    llm_model: str = CHUNKING_MODEL,
    output_dir: str = None,
    max_retries: int = 3,
    enable_fallback: bool = True,
    community_id: str = None,
) -> str:
    """Chunk a markdown file into separate chunk files.

    Args:
        community_id: Optional community/document unique_id. When set:

            * Chunk numbering *resumes* from the largest existing chunk_id in
              ``s3://<bucket>/<community_id>/`` so that separately processed
              files sharing a community never overwrite earlier chunks
              (``chunk_001.txt`` etc.). Falls back to 1 when S3 is unreachable
              or the community is new.
            * Every chunk file (chunk_001.txt, chunk_002.txt, ... plus
              manifest.json) is uploaded to ``s3://<bucket>/<community_id>/``
              after saving. Upload is best-effort and never aborts chunking.

    Returns:
        Path to ``manifest.json`` inside the chunks directory.
    """
    if output_dir is None:
        output_dir = str(
            Path(__file__).parent.parent.parent.parent.parent.parent / "output"
        )

    # Resume chunk numbering from S3 so separately processed files under the
    # same community never collide/overwrite. A fresh or unreachable community
    # yields 0, so numbering starts at 1 as before.
    start_chunk_id = 1
    if community_id:
        max_existing = get_max_chunk_id_from_s3(community_id)
        start_chunk_id = max_existing + 1
        if start_chunk_id > 1:
            print(
                f"🔢 Resuming chunk numbering at {start_chunk_id} for "
                f"community {community_id}"
            )

    chunker = SemanticChunker(
        chunk_granularity=chunk_granularity,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_retries=max_retries,
        enable_fallback=enable_fallback,
        start_chunk_id=start_chunk_id,
    )

    try:
        chunks = chunker.chunk_markdown(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to chunk document: {e}")

    # Directory: output/{stem}_chunks/  (strip _analysis suffix from markdown stem)
    input_path = Path(file_path)
    stem = input_path.stem
    if stem.endswith("_analysis"):
        stem = stem[: -len("_analysis")]
    chunks_dir = Path(output_dir) / f"{stem}_chunks"

    try:
        manifest_path = chunker.save_chunks(chunks, str(chunks_dir))
    except Exception as e:
        raise RuntimeError(f"Failed to save chunks: {e}")

    # Best-effort S3 upload of the chunk files, keyed by community_id.
    if community_id:
        try:
            upload_chunks_to_s3(str(chunks_dir), community_id)
        except Exception as e:
            print(f"⚠️  S3 chunk upload failed: {e}")

    return manifest_path


def upload_chunks_to_s3(
    chunks_dir: str,
    community_id: str,
    bucket: str = None,
) -> int:
    """Upload chunk files to ``s3://<bucket>/<community_id>/``.

    Uploads every file in ``chunks_dir`` (chunk_001.txt, chunk_002.txt, ... and
    manifest.json) under the ``<community_id>/`` key prefix, preserving the
    original filenames.

    AWS credentials are read from the environment (``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, ``AWS_REGION``) — already loaded at module import
    via ``load_dotenv()``.

    Args:
        chunks_dir: Local directory holding the saved chunk files.
        community_id: S3 key prefix (the community/document unique_id).
        bucket: S3 bucket name. Defaults to env ``SOURCE_DOC_BUCKET`` or ``"document"``.

    Returns:
        Number of files successfully uploaded.
    """
    if not HAS_BOTO3:
        print("⚠️  boto3 not installed — skipping S3 chunk upload")
        return 0

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION")
    if not (access_key and secret_key):
        print(
            "⚠️  AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY not set — "
            "skipping S3 chunk upload"
        )
        return 0

    if bucket is None:
        bucket = os.getenv("SOURCE_DOC_BUCKET", "document")

    src_dir = Path(chunks_dir)
    files = sorted(p for p in src_dir.iterdir() if p.is_file())
    if not files:
        print(f"⚠️  No chunk files found in {src_dir} — skipping S3 upload")
        return 0

    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    uploaded = 0
    for fp in files:
        key = f"{community_id}/{fp.name}"
        try:
            s3.upload_file(str(fp), bucket, key)
            uploaded += 1
            print(f"☁️  Uploaded {fp.name} → s3://{bucket}/{key}")
        except (ClientError, NoCredentialsError) as e:
            print(f"⚠️  Failed to upload {fp.name} to S3: {e}")

    print(
        f"☁️  Uploaded {uploaded}/{len(files)} chunk files "
        f"to s3://{bucket}/{community_id}/"
    )
    return uploaded


def get_max_chunk_id_from_s3(
    community_id: str,
    bucket: str = None,
) -> int:
    """Return the largest existing ``chunk_id`` under ``s3://<bucket>/<community_id>/``.

    Lists the actual chunk files (``chunk_001.txt``, ``chunk_002.txt``, ...) under
    the ``<community_id>/`` key prefix and returns the highest numeric id found.
    Used by :func:`chunk_markdown_file` to resume chunk numbering across
    separately processed files so ids never restart at 1 and overwrite earlier
    chunks.

    The chunk *files* are the source of truth rather than ``manifest.json``,
    because each processed file overwrites the manifest for its community — the
    files themselves persist as long as ids keep increasing.

    AWS credentials are read from the environment, mirroring
    :func:`upload_chunks_to_s3`.

    Args:
        community_id: S3 key prefix (the community/document unique_id).
        bucket: S3 bucket name. Defaults to env ``SOURCE_DOC_BUCKET`` or ``"document"``.

    Returns:
        The largest existing chunk_id, or ``0`` when the community has no chunks
        yet (or S3 cannot be queried — boto3 missing or credentials unset), in
        which case the caller starts numbering at 1.
    """
    if not HAS_BOTO3:
        return 0

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION")
    if not (access_key and secret_key):
        return 0

    if bucket is None:
        bucket = os.getenv("SOURCE_DOC_BUCKET", "document")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    pattern = re.compile(r"chunk_(\d+)\.txt$")
    prefix = f"{community_id}/"
    max_id = 0
    continuation_token = None

    try:
        while True:
            list_kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token
            resp = s3.list_objects_v2(**list_kwargs)
            for obj in resp.get("Contents", []):
                match = pattern.search(obj["Key"])
                if match:
                    max_id = max(max_id, int(match.group(1)))
            if resp.get("IsTruncated"):
                continuation_token = resp.get("NextContinuationToken")
            else:
                break
    except (ClientError, NoCredentialsError) as e:
        print(f"⚠️  Could not list S3 chunks for {community_id}: {e}")
        return 0

    print(
        f"☁️  Largest existing chunk_id in s3://{bucket}/{prefix} is {max_id}"
    )
    return max_id
