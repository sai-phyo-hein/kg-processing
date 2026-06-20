"""Thai->English translation for extracted triples.

This runs as its own LangGraph node, immediately after triple extraction and
before refinement.  Splitting it out of ``TripleExtractor`` keeps each pipeline
stage responsible for a single model/responsibility and makes translation
independently debuggable.

  COST  — translatable field values are first de-duplicated by (role, value)
          across all qualifying chunks, then batched
          (``translation_batch_size``), so each distinct value (e.g. a predicate
          repeated across many triples/chunks) is translated exactly once and
          the system prompt is amortised across many values per call.

  HANG  — there is a single ThreadPoolExecutor over translation batches; no
          nested pool.  Each batch worker makes its own translation call
          (thread-safe: a fresh message list per call + a shared semaphore).

Translation runs only on chunks that contain non-English text.  Each chunk's
triple fields are classified per field with lingua
(https://github.com/pemistahl/lingua-py); a chunk is translated iff at least
one field is detected as non-English.  Output is written back *in place* to
the same triples file — translation only adds ``_en`` fields to existing
triples.
"""

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.model_setup import (
    TRANSLATION_MODEL,
    get_reasoning_llm,
)
from kg_extractor.utils.prompts import (
    TRANSLATION_SYSTEM_PROMPT,
    create_translation_user_message,
)

# ---------------------------------------------------------------------------
# Per-chunk language detection (lingua)
# ---------------------------------------------------------------------------
#
# A chunk is translated only if at least one of its triple fields is detected
# as non-English.  Detection is done with lingua
# (https://github.com/pemistahl/lingua-py), which stays accurate on the short
# text (entity names) that triples are made of.
#
# Design notes:
#   * Candidate languages are restricted to English + Thai.  lingua's accuracy
#     drops sharply as the candidate set grows: with all ~75 languages a bare
#     English name like "John Smith" misdetects as IRISH and "Bangkok" as
#     INDONESIAN.  These documents are Thai<->English, so two languages give
#     clean per-field separation (Thai script is unambiguous to lingua's
#     rule engine).  Add more `Language` values in _get_language_detector if
#     the document profile ever widens.
#   * Detection is per FIELD, not on the concatenated chunk text.  A single
#     Thai entity name buried among English fields is caught per field but is
#     drowned out (and missed) when everything is concatenated first.
#   * lingua is imported lazily (first detection), so extract-only runs never
#     require it.  A LanguageDetector is thread-safe and shares its language
#     models with every other instance, so the one cached here is safe to read
#     concurrently from the chunk-worker threads.

_detector: Any = None
_detector_lock = threading.Lock()


def _get_language_detector():
    """Build and cache the shared lingua LanguageDetector (lazy, thread-safe)."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                from lingua import Language, LanguageDetectorBuilder

                _detector = (
                    LanguageDetectorBuilder
                    .from_languages(Language.ENGLISH, Language.THAI)
                    .build()
                )
    return _detector


def _detect_non_english_languages(triples: List[Dict[str, Any]]) -> List[str]:
    """Non-English language names found among a chunk's translatable fields.

    Classifies each translatable field (subject/object names, predicate,
    relationship-attribute values, evidence quote) on its own and returns the
    name of the first non-English language found, or an empty list when every
    field is English.  An empty result means the chunk needs no translation.

    With the English+Thai detector the only possible non-English language is
    Thai, so the first hit is returned immediately (early exit).
    """
    from lingua import Language

    detector = _get_language_detector()

    for triple in triples:
        subj = triple.get("subject", {}) or {}
        obj = triple.get("object", {}) or {}
        props = triple.get("properties", {}) or {}
        rel_attrs = triple.get("relationship_attributes") or {}

        fields = [subj.get("name"), triple.get("predicate"), obj.get("name")]
        fields += [str(v) for v in rel_attrs.values() if v]
        fields.append(props.get("evidence_quote"))

        for field in fields:
            if not field or not str(field).strip():
                continue
            lang = detector.detect_language_of(str(field))
            if lang is not None and lang != Language.ENGLISH:
                return [lang.name]

    return []


def _chunk_translation_complete(triples: List[Dict[str, Any]]) -> bool:
    """True when every identity field in ``triples`` already has a usable ``_en``.

    A chunk counts as complete only when each subject/object name and predicate
    that carries a source value also has a non-empty ``_en`` that does not merely
    echo the source.  Used as the full-run idempotency guard: checking the whole
    chunk (not just the first triple) means a partially translated chunk — first
    triple fine, later triples dropped because a dedup item failed — is repaired
    rather than silently skipped.  Auxiliary fields (evidence_quote, rel_attrs)
    are intentionally excluded so their absence does not force re-translation.
    """
    for triple in triples:
        subj = triple.get("subject", {}) or {}
        obj = triple.get("object", {}) or {}
        for th, en in (
            (subj.get("name"), subj.get("name_en")),
            (obj.get("name"), obj.get("name_en")),
            (triple.get("predicate"), triple.get("predicate_en")),
        ):
            if not th or not str(th).strip():
                continue
            if not en or not str(en).strip():
                return False
            if _sanitize(str(en)) == _sanitize(str(th)):
                return False
    return True


def _strip_en_fields(triples: List[Dict[str, Any]]) -> None:
    """Remove every ``*_en`` translation field from a chunk's triples.

    Used in single-chunk (re-)translation mode so a forced re-translate
    reflects only the current run, not ``_en`` values left over from an earlier
    translation.  Only the fields ``_translate_chunk_triples`` writes are
    touched; the original Thai/source fields are left intact.
    """
    for triple in triples:
        triple.pop("predicate_en", None)
        triple.pop("relationship_attributes_en", None)
        subj = triple.get("subject")
        if isinstance(subj, dict):
            subj.pop("name_en", None)
        obj = triple.get("object")
        if isinstance(obj, dict):
            obj.pop("name_en", None)
        props = triple.get("properties")
        if isinstance(props, dict):
            props.pop("evidence_quote_en", None)


def _sanitize(text: str) -> str:
    """Replace literal newlines/tabs and trim so JSON/keys stay stable."""
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def _apply_en(triple: Dict[str, Any], role: str, en_value: Any) -> None:
    """Write one translated ``*_en`` value back onto a triple by role.

    Mirrors the write-back previously done inline in
    ``_translate_chunk_triples``: subject/object -> ``name_en``,
    predicate -> ``predicate_en``, rel_attrs -> ``relationship_attributes_en``,
    evidence_quote -> ``properties.evidence_quote_en``.
    """
    if en_value is None or (isinstance(en_value, str) and not en_value.strip()):
        return
    if role == "subject":
        triple.setdefault("subject", {})["name_en"] = en_value
    elif role == "object":
        triple.setdefault("object", {})["name_en"] = en_value
    elif role == "predicate":
        triple["predicate_en"] = en_value
    elif role == "rel_attrs":
        triple["relationship_attributes_en"] = en_value
    elif role == "evidence_quote":
        triple.setdefault("properties", {})["evidence_quote_en"] = en_value


class TripleTranslator:
    """Translate Thai fields in extracted triples to English using a cheap model.

    Thread safety
    -------------
    ``_get_translation_response`` builds a fresh ``[SystemMessage, HumanMessage]``
    list per call and acquires a shared semaphore around the API invocation, so
    it is safe to call concurrently from the cross-chunk thread pool.  No two
    workers ever touch the same chunk's triples.
    """

    def __init__(
        self,
        translation_model: str = TRANSLATION_MODEL,
        translation_batch_size: int = 10,
        max_workers: int = 20,
        max_translation_retries: int = 2,
    ):
        """Initialize the triple translator.

        Args:
            translation_model: Cheap model used for the Thai->English _en fields.
            translation_batch_size: Number of triples sent per translation call.
                This is the single biggest cost lever: at 1 (the old behaviour)
                the system prompt is re-sent for every triple; at 10 it is sent
                once per 10 triples, cutting translation spend ~10x.
            max_workers: Maximum number of chunks translated concurrently.
            max_translation_retries: Extra passes re-translating items that came
                back missing or echoing their Thai source.  Dedup collapses every
                triple sharing a value into one item, so one bad item poisons all
                of them — a small retry budget here repairs a large share of the
                output for almost no extra cost.
        """
        self.translation_model = translation_model
        self.translation_batch_size = max(1, translation_batch_size)
        self.max_workers = max(1, max_workers)
        self.max_translation_retries = max(0, max_translation_retries)

        # Cheap model for Thai->English translation.
        # max_tokens raised to 32000 to fit a *batch* of triples in one response:
        #   ~270 output tokens/triple worst case x translation_batch_size (10)
        #   = ~2,700 tokens -> 32000 leaves comfortable headroom for long Thai
        #   evidence quotes.  If you raise translation_batch_size, raise this too.
        self._translation_llm = get_reasoning_llm(
            model=self.translation_model,
            temperature=0.1,
            max_tokens=32000,
        )

        # Rate-limit guard for translation calls.  Caps in-flight translation
        # requests within the model's RPM limit.
        translation_max_concurrent = 50
        self._translation_semaphore = threading.Semaphore(translation_max_concurrent)

    # ------------------------------------------------------------------ #
    # Cross-chunk orchestration
    # ------------------------------------------------------------------ #

    def translate_chunks(
        self,
        chunks: List[Dict[str, Any]],
        chunk_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Translate fields across chunks in place.

        Translates only the chunks that contain non-English text.  Two skips
        keep the full run cheap and idempotent:

          * **Idempotency** — chunks already fully translated (every identity
            field has a usable ``_en``) are skipped.  The whole chunk is checked,
            not just the first triple, so partially translated chunks are still
            repaired.
          * **Language gate** — chunks whose translatable fields lingua detects
            as English-only (see ``_detect_non_english_languages``).

        The qualifying chunks are then translated together in one
        value-deduplicated pass (:meth:`_translate_chunks_dedup`): every
        distinct ``(role, value)`` pair (e.g. a predicate that recurs across
        many triples/chunks) is sent to the model exactly once, and the English
        is remapped back onto every triple that used it.  Batches of unique
        values run concurrently across worker threads.

        Args:
            chunks: List of chunk dicts as saved by the extraction node:
                ``{"chunk_id", "triples": [...]}``.
            chunk_id: Optional single chunk_id to (re-)translate.  When set,
                only that chunk is processed — its prior ``_en`` fields are
                stripped first (so this is a forced re-translate, never a no-op
                under the idempotency guard), the language gate still applies,
                and every other chunk is left untouched.  Raises ``ValueError``
                if no chunk carries that id.

        Returns:
            The same ``chunks`` list, mutated in place where translated.
        """
        if not chunks:
            return chunks

        # Single-chunk (re-)translation mode: target only the requested chunk
        # and force a clean re-translate (strip prior _en fields so the result
        # reflects this run, not an earlier one).  The language gate still
        # applies — an English-only chunk has nothing to translate.
        if chunk_id is not None:
            target = next(
                (c for c in chunks if c.get("chunk_id") == chunk_id), None
            )
            if target is None:
                raise ValueError(
                    f"chunk_id {chunk_id} not found among {len(chunks)} chunks"
                )
            triples = target.get("triples", [])
            if not triples:
                print(f"  chunk {chunk_id}: no triples, nothing to translate")
                return chunks
            _strip_en_fields(triples)
            non_english = _detect_non_english_languages(triples)
            if not non_english:
                print(f"  chunk {chunk_id}: English-only, nothing to translate")
                return chunks
            print(f"  chunk {chunk_id}: non-English {non_english} -> translate")
            self._translate_chunks_dedup([target])
            print(f"  [1/1] chunk {chunk_id} translated")
            return chunks

        # Full mode: collect every chunk that actually needs translation.
        to_translate: List[Dict[str, Any]] = []
        skipped_english = 0
        skipped_complete = 0
        for chunk in chunks:
            triples = chunk.get("triples", [])
            if not triples:
                continue

            # Language gate: translate only if the chunk contains any
            # non-English text (detected per field with lingua).
            non_english = _detect_non_english_languages(triples)
            if not non_english:
                skipped_english += 1
                continue

            # Idempotency guard: skip chunks already fully translated.  The whole
            # chunk is checked — not just the first triple — so a chunk whose
            # first triple is fine but whose later triples lost an _en (a failed
            # dedup item poisons every triple sharing that value) is repaired
            # instead of silently skipped.
            if _chunk_translation_complete(triples):
                skipped_complete += 1
                continue

            print(f"  chunk {chunk.get('chunk_id')}: non-English {non_english} -> translate")
            to_translate.append(chunk)

        print(f"🌐 Language gate: {len(to_translate)} chunk(s) need translation, "
              f"{skipped_english} skipped as English-only, "
              f"{skipped_complete} already complete.")

        if not to_translate:
            return chunks

        # Translate all qualifying chunks in one value-deduplicated pass:
        # predicates/entity names recur across chunks, so translate each
        # distinct (role, value) once and remap onto every triple that used it.
        self._translate_chunks_dedup(to_translate)
        print(f"✅ Translated {len(to_translate)} chunk(s) (dedup across all)")
        return chunks

    # ------------------------------------------------------------------ #
    # Translation
    # ------------------------------------------------------------------ #

    def _get_translation_response(self, user_message: str) -> str:
        """Thread-safe translation call using the cheap model.

        Acquires the shared semaphore around the API call so the total number of
        in-flight translation requests stays within the model's RPM limit.
        Retries up to 5 times with exponential back-off on rate-limit (429)
        errors.  The sleep happens *outside* the semaphore so a retrying thread
        never holds a permit while waiting.

        Args:
            user_message: JSON array of translation items (a batch) from
                create_translation_user_message.

        Returns:
            Raw LLM response text (expected to be a JSON array).
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=TRANSLATION_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        max_attempts = 5
        wait = 5.0  # seconds before first retry

        for attempt in range(1, max_attempts + 1):
            try:
                with self._translation_semaphore:
                    response = self._translation_llm.invoke(messages)
                return response.content
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = (
                    "429" in err
                    or "rate limit" in err
                    or "rate_limit" in err
                    or "too many" in err
                )

                if is_rate_limit and attempt < max_attempts:
                    print(f"Warning: Translation rate-limited (attempt {attempt}/{max_attempts}), "
                          f"retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    wait = min(wait * 2, 60)  # exponential back-off, cap at 60s
                else:
                    if not is_rate_limit:
                        print(f"Warning: Failed to get translation response: {e}")
                    else:
                        print(f"Warning: Translation rate-limited after {max_attempts} attempts, skipping.")
                    return "[]"

        return "[]"

    @staticmethod
    def _parse_translation_response(raw_response: str) -> List[Dict[str, Any]]:
        """Parse a (batched) translation response into a list of item dicts.

        Handles the two malformations the cheap model is prone to:
          - markdown code fences around the JSON
          - unquoted ALL_CAPS_SNAKE_CASE predicate values (e.g. HAS_DETAILS)

        Returns an empty list on failure so the caller degrades gracefully
        (those triples simply keep their Thai-only fields).
        """
        try:
            text = raw_response.strip()
            # Strip markdown fences if present
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
            # Quote any unquoted ALL_CAPS_SNAKE_CASE value after a colon
            text = re.sub(
                r':\s*([A-Z][A-Z0-9_]{2,})(\s*[,}\]])',
                r': "\1"\2',
                text,
            )

            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except Exception as e:
            print(f"Warning: Failed to parse translation batch: {e}")
            print(f"  Raw response (first 300 chars): {repr(raw_response[:300])}")
        return []

    def _translate_items(
        self, items: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Translate single-field items, batched and concurrent.

        Each item is ``{"id": int, "<role>_th": value}`` (one ``_th`` field), as
        built by :meth:`_translate_chunks_dedup`.  Items are sharded into
        ``translation_batch_size`` batches — one model call per batch — and the
        batches run concurrently across ``max_workers`` threads.

        Returns a mapping ``item_id -> parsed response entry``.  Entries whose
        ``id`` is missing or non-int are dropped; a batch that fails to parse or
        errors yields nothing for its items (the caller retries those).
        """
        if not items:
            return {}

        def translate_batch(batch: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
            user_message = create_translation_user_message(batch)
            raw_response = self._get_translation_response(user_message)
            result: Dict[int, Dict[str, Any]] = {}
            for entry in self._parse_translation_response(raw_response):
                if "id" not in entry:
                    continue
                try:
                    result[int(entry["id"])] = entry
                except (TypeError, ValueError):
                    continue
            return result

        batches = [
            items[i:i + self.translation_batch_size]
            for i in range(0, len(items), self.translation_batch_size)
        ]
        total_batches = len(batches)
        completed = 0
        translations: Dict[int, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(translate_batch, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    translations.update(future.result())
                except Exception as e:
                    print(f"Warning: translation batch {idx} failed: {e}")
                finally:
                    completed += 1
                    print(f"  [{completed}/{total_batches}] translation batch done")

        return translations

    @staticmethod
    def _is_translation_failed(
        item: Dict[str, Any], entry: Optional[Dict[str, Any]]
    ) -> bool:
        """True if a translation item is still untranslated.

        An item is failed when its entry is missing, its ``_en`` value is empty,
        or (for scalar roles) the model echoed the Thai source verbatim.  rel_attrs
        is a dict, so only the missing/empty check applies to it.
        """
        role = next((k[:-3] for k in item if k.endswith("_th")), None)
        if role is None:
            return False
        if not entry:
            return True
        en = entry.get(f"{role}_en")
        if en is None or (isinstance(en, str) and not en.strip()):
            return True
        if role != "rel_attrs":
            th = item.get(f"{role}_th")
            if isinstance(th, str) and _sanitize(str(en)) == _sanitize(th):
                return True
        return False

    def _translate_chunks_dedup(self, chunks: List[Dict[str, Any]]) -> None:
        """Translate every translatable field in ``chunks`` with value dedup.

        Collects every distinct ``(role, value)`` pair across the given chunks,
        translates each one exactly once, then writes the English back onto
        every triple that referenced it.  ``rel_attrs`` are deduped at the
        whole-dict level.  Mutates each chunk's triples in place.

        Roles (role -> item ``*_th`` key / response ``*_en`` key):

            subject, predicate, object, evidence_quote  (scalars)
            rel_attrs                                    (dict)

        Dedup is keyed on ``(role, value)`` rather than bare value because the
        translation is role-specific (predicate_th -> ALL_CAPS_SNAKE, names ->
        preserve proper nouns): the same Thai string must be translated
        differently per role, but identical values within one role collapse to
        a single call.

        Reuses :func:`create_translation_user_message` and
        :meth:`_parse_translation_response` unchanged — the translation prompt
        already supports items carrying a single ``*_th`` field.
        """
        # (role, normalized_value) -> item id; item id -> [(triple, role), ...]
        dedup: Dict[tuple, int] = {}
        locations: Dict[int, List[tuple]] = {}
        items: List[Dict[str, Any]] = []

        def register(role: str, raw_value: Any, triple: Dict[str, Any]) -> None:
            if raw_value is None:
                return
            if role == "rel_attrs":
                if not isinstance(raw_value, dict) or not raw_value:
                    return
                value = {k: _sanitize(str(v)) for k, v in raw_value.items()}
                key = json.dumps(value, sort_keys=True, ensure_ascii=False)
            else:
                value = _sanitize(str(raw_value))
                if not value:
                    return
                key = value
            dedup_key = (role, key)
            item_id = dedup.get(dedup_key)
            if item_id is None:
                item_id = len(items)
                dedup[dedup_key] = item_id
                items.append({"id": item_id, f"{role}_th": value})
                locations[item_id] = []
            locations[item_id].append((triple, role))

        for chunk in chunks:
            for triple in chunk.get("triples", []):
                subj = triple.get("subject", {}) or {}
                obj = triple.get("object", {}) or {}
                props = triple.get("properties", {}) or {}
                register("subject", subj.get("name"), triple)
                register("predicate", triple.get("predicate"), triple)
                register("object", obj.get("name"), triple)
                register("evidence_quote", props.get("evidence_quote"), triple)
                rel_attrs = triple.get("relationship_attributes")
                if rel_attrs:
                    register("rel_attrs", rel_attrs, triple)

        if not items:
            return

        instances = sum(len(locs) for locs in locations.values())
        unique = len(items)
        print(f"  dedup: {instances} field instances -> {unique} unique values "
              f"(-{instances - unique} translation items)")

        # Translate the unique items in batches, concurrently across batches.
        translations = self._translate_items(items)

        # Retry stragglers: items that came back missing or echoing their Thai
        # source.  Dedup collapses every triple sharing a value into one item,
        # so one bad item poisons all of them — retranslating the handful of
        # failures repairs a large share of the output at near-zero extra cost.
        for round_idx in range(1, self.max_translation_retries + 1):
            failed = [
                it for it in items
                if self._is_translation_failed(it, translations.get(it["id"]))
            ]
            if not failed:
                break
            print(
                f"  retry {round_idx}/{self.max_translation_retries}: "
                f"{len(failed)} item(s) still missing/echoed — retranslating"
            )
            translations.update(self._translate_items(failed))

        # Remap each translation back onto every triple that used it.
        for item_id, locs in locations.items():
            entry = translations.get(item_id)
            if not entry:
                continue
            for triple, role in locs:
                _apply_en(triple, role, entry.get(f"{role}_en"))


def translate_triples_from_file(
    input_path: str,
    translation_model: str = TRANSLATION_MODEL,
    translation_batch_size: int = 10,
    max_workers: int = 20,
    max_translation_retries: int = 2,
    chunk_id: int | None = None,
) -> str:
    """Translate the non-English chunks' triples in a JSON file, in place.

    Loads the triples file written by the extraction node, translates every
    chunk that contains non-English text (detected per chunk with lingua), and
    writes the result back to the SAME ``input_path`` (translation only adds
    ``_en`` fields).

    Args:
        input_path: Path to the triples JSON file (read and overwritten).
        translation_model: Cheap model used for the Thai->English _en fields.
        translation_batch_size: Triples per translation call (cost lever).
        max_workers: Maximum concurrent chunks to translate.
        max_translation_retries: Extra passes retranslating items that came back
            missing or echoing their Thai source (dedup amplifies each miss).
        chunk_id: Optional single chunk_id to (re-)translate in place; all other
            chunks are preserved.  When omitted, every non-English chunk is
            translated.

    Returns:
        The same ``input_path`` (unchanged — in-place contract).
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    translator = TripleTranslator(
        translation_model=translation_model,
        translation_batch_size=translation_batch_size,
        max_workers=max_workers,
        max_translation_retries=max_translation_retries,
    )

    translator.translate_chunks(data.get("chunks", []), chunk_id=chunk_id)

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return input_path
