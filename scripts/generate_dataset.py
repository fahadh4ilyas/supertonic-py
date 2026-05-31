#!/usr/bin/env python3
"""Generate a multilingual JSONL dataset for AudioEncoder training.

Pulls paragraphs from Wikipedia dumps or Flores-200 via HuggingFace datasets,
extracts texts in the 31 languages supported by Supertonic-3, validates them
against the model's unicode_indexer, and writes a JSONL file.

Requires: pip install datasets

Usage:
    python scripts/generate_dataset.py --output data_train.jsonl --samples 10000 --model_dir ~/.cache/supertonic3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supertonic.config import SUPPORTED_LANGUAGES
from supertonic.loader import get_cache_dir


# ──────────────────────────────────────────────────────────────────────────────
# Text validation (filters unsupported characters)
# ──────────────────────────────────────────────────────────────────────────────

_validator_cache: dict[str, "UnicodeProcessor | None"] = {}


def get_validator(model_dir: str | Path) -> "UnicodeProcessor | None":
    """Load UnicodeProcessor for text validation. Cached per model_dir."""
    key = str(model_dir)
    if key not in _validator_cache:
        indexer_path = Path(model_dir) / "unicode_indexer.json"
        if indexer_path.exists():
            from supertonic.core import UnicodeProcessor
            _validator_cache[key] = UnicodeProcessor(str(indexer_path))
        else:
            print(f"Warning: unicode_indexer.json not found at {indexer_path} — skipping validation")
            _validator_cache[key] = None
    return _validator_cache[key]


def is_text_valid(text: str, lang: str, validator: "UnicodeProcessor") -> bool:
    """Check if text contains only characters supported by the model."""
    if validator is None:
        return True
    try:
        is_valid, _ = validator.validate_text(text)
        return is_valid
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Language mapping: Supertonic lang codes → Wikipedia/flores codes
# ──────────────────────────────────────────────────────────────────────────────

LANG_MAP = {
    "en": "en",
    "ko": "ko",
    "ja": "ja",
    "ar": "ar",
    "bg": "bg",
    "cs": "cs",
    "da": "da",
    "de": "de",
    "el": "el",
    "es": "es",
    "et": "et",
    "fi": "fi",
    "fr": "fr",
    "hi": "hi",
    "hr": "hr",
    "hu": "hu",
    "id": "id",
    "it": "it",
    "lt": "lt",
    "lv": "lv",
    "nl": "nl",
    "pl": "pl",
    "pt": "pt",
    "ro": "ro",
    "ru": "ru",
    "sk": "sk",
    "sl": "sl",
    "sv": "sv",
    "tr": "tr",
    "uk": "uk",
    "vi": "vi",
}

# Languages with Latin/Cyrillic scripts that are well-represented in Wikipedia
# We target 70% of samples from these (more reliable), 30% from the rest
HIGH_RESOURCE = {"en", "de", "fr", "es", "pt", "ru", "it", "pl", "nl", "ja", "ko"}
LOW_RESOURCE = set(SUPPORTED_LANGUAGES) - HIGH_RESOURCE


# ──────────────────────────────────────────────────────────────────────────────
# Wikipedia source
# ──────────────────────────────────────────────────────────────────────────────


def generate_from_wikipedia(
    output_path: Path,
    num_samples: int = 5000,
    min_chars: int = 200,
    max_chars: int = 500,
    seed: int = 42,
    token: str | None = None,
    validator: "UnicodeProcessor | None" = None,
) -> None:
    """Generate dataset from Wikipedia via wikimedia/wikipedia.

    Uses the HuggingFace ``wikimedia/wikipedia`` dataset (Parquet-based,
    no deprecated loading scripts). Millions of articles per language.
    """
    from datasets import load_dataset

    random.seed(seed)
    lang_items = list(LANG_MAP.items())
    random.shuffle(lang_items)

    samples_written = 0
    all_paragraphs: list[dict] = []

    print(f"Target: {num_samples} samples across {len(SUPPORTED_LANGUAGES)} languages")

    for lang_st, lang_wiki in lang_items:
        if samples_written >= num_samples:
            break

        try:
            print(f"  Loading Wikipedia ({lang_st})...", end=" ", flush=True)
            dataset = load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{lang_wiki}",
                split="train",
                token=token,
            )
            print(f"OK ({len(dataset)} articles)")
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        # Extract paragraphs
        lang_samples = 0
        target_per_lang = max(20, num_samples // len(lang_items))

        # Randomly sample articles instead of iterating sequentially
        n_articles = len(dataset)
        indices = list(range(n_articles))
        random.shuffle(indices)
        indices = indices[: min(n_articles, target_per_lang * 10)]  # search up to 10× needed

        # Collect all valid paragraphs from sampled articles
        lang_paragraphs = []
        for idx in indices:
            article = dataset[idx]
            text = article.get("text", "")
            if not text:
                continue
            for para in text.split("\n"):
                para = para.strip()
                if not (min_chars <= len(para) <= max_chars):
                    continue
                if validator is not None and not is_text_valid(para, lang_st, validator):
                    continue
                lang_paragraphs.append(para)
                if len(lang_paragraphs) >= target_per_lang * 2:
                    break
            if len(lang_paragraphs) >= target_per_lang * 2:
                break

        # Pair paragraphs: text_encoder and text_tts come from different articles
        random.shuffle(lang_paragraphs)
        for i in range(0, len(lang_paragraphs) - 1, 2):
            enc = lang_paragraphs[i]
            tts = lang_paragraphs[i + 1]
            # Use first 2 sentences of tts paragraph as the short text
            tts_sentences = tts.split(". ")
            tts_short = ". ".join(tts_sentences[:2]).strip()
            if tts_short and not tts_short.endswith("."):
                tts_short += "."
            all_paragraphs.append({
                "text_encoder": enc,
                "text_tts": tts_short or tts,
                "lang": lang_st,
            })
            lang_samples += 1
            samples_written += 1
            if samples_written >= num_samples:
                break

        if samples_written >= num_samples:
            break

        print(f"    → {lang_samples} paragraphs")

    # Shuffle and write
    random.shuffle(all_paragraphs)
    all_paragraphs = all_paragraphs[:num_samples]

    write_jsonl(output_path, all_paragraphs)
    print(f"\nWrote {len(all_paragraphs)} samples to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Flores-200 source (lightweight, fast to download)
# ──────────────────────────────────────────────────────────────────────────────


def generate_from_flores(
    output_path: Path,
    num_samples: int = 5000,
    min_chars: int = 200,
    max_chars: int = 500,
    seed: int = 42,
    token: str | None = None,
    validator: "UnicodeProcessor | None" = None,
) -> None:
    """Generate dataset from Flores-200 (parallel corpus, 200+ languages).

    Flores-200 has ~1000 parallel sentences per language. Since individual
    sentences are short (50-150 chars), we concatenate consecutive sentences
    to reach the target 200-500 char length.

    Requires HuggingFace token for gated access to facebook/flores.
    """
    from datasets import load_dataset

    random.seed(seed)

    # Flores-200 language codes → Supertonic language codes
    flores_langs = {
        "en": "eng_Latn", "ko": "kor_Hang", "ja": "jpn_Jpan",
        "ar": "arb_Arab", "bg": "bul_Cyrl", "cs": "ces_Latn",
        "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek",
        "es": "spa_Latn", "et": "est_Latn", "fi": "fin_Latn",
        "fr": "fra_Latn", "hi": "hin_Deva", "hr": "hrv_Latn",
        "hu": "hun_Latn", "id": "ind_Latn", "it": "ita_Latn",
        "lt": "lit_Latn", "lv": "lvs_Latn", "nl": "nld_Latn",
        "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn",
        "ru": "rus_Cyrl", "sk": "slk_Latn", "sl": "slv_Latn",
        "sv": "swe_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl",
        "vi": "vie_Latn",
    }

    all_samples: list[dict] = []
    total_langs = len(flores_langs)
    print(f"Target: {num_samples} samples across {total_langs} languages")

    for lang_st, lang_flores in flores_langs.items():
        try:
            print(f"  Loading Flores-200 ({lang_st})...", end=" ", flush=True)
            dataset = load_dataset(
                "facebook/flores",
                lang_flores,
                split="devtest",
                token=token,
            )
            sentences = [item["sentence"] for item in dataset]
            print(f"OK ({len(sentences)} sentences)")
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        random.shuffle(sentences)

        buffer = ""
        buffer_start = 0
        lang_count = 0
        target_per_lang = max(10, num_samples // total_langs)

        for i, sent in enumerate(sentences):
            if not sent:
                continue
            candidate = f"{buffer} {sent}".strip() if buffer else sent
            if len(candidate) >= min_chars:
                encoder_text = candidate if len(candidate) <= max_chars else buffer
                if min_chars <= len(encoder_text) <= max_chars:
                    # Validate against model's character set
                    if validator is not None and not is_text_valid(encoder_text, lang_st, validator):
                        buffer = sent
                        buffer_start = i
                        continue
                    group = sentences[buffer_start:i]
                    tts_text = ". ".join([s for s in group[:2] if s]).strip()
                    if tts_text and (validator is None or is_text_valid(tts_text, lang_st, validator)):
                        all_samples.append({
                            "text_encoder": encoder_text,
                            "text_tts": tts_text + ("." if tts_text[-1] != "." else ""),
                            "lang": lang_st,
                        })
                        lang_count += 1
                buffer = sent
                buffer_start = i
            else:
                buffer = candidate

            if lang_count >= target_per_lang:
                break

        print(f"    → {lang_count} samples")

    random.shuffle(all_samples)
    all_samples = all_samples[:num_samples]

    write_jsonl(output_path, all_samples)
    print(f"\nWrote {len(all_samples)} samples to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Output helper
# ──────────────────────────────────────────────────────────────────────────────


def write_jsonl(path: Path, samples: list[dict]) -> None:
    """Write samples as JSONL with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Stats helper
# ──────────────────────────────────────────────────────────────────────────────


def print_stats(path: Path) -> None:
    """Print dataset statistics."""
    from collections import Counter

    lang_counts = Counter()
    char_lengths = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            lang_counts[sample["lang"]] += 1
            char_lengths.append(len(sample["text_encoder"]))

    if not char_lengths:
        print(f"  (empty dataset)")
        return
    
    print(f"\nDataset: {path}")
    print(f"  Total samples: {sum(lang_counts.values())}")
    print(f"  Languages: {len(lang_counts)}")
    print(f"  Char length: min={min(char_lengths)}, avg={sum(char_lengths)/len(char_lengths):.0f}, max={max(char_lengths)}")
    print(f"  Top languages: {lang_counts.most_common(10)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate multilingual dataset for AudioEncoder training"
    )
    parser.add_argument("--output", type=Path, default=Path("data_train.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--source", type=str, default="flores",
                        choices=["wikipedia", "flores"],
                        help="Data source: 'wikipedia' (richer, slower) or 'flores' (Flores-200, faster)")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Total number of samples to generate")
    parser.add_argument("--min_chars", type=int, default=200,
                        help="Minimum characters per text_encoder entry")
    parser.add_argument("--max_chars", type=int, default=500,
                        help="Maximum characters per text_encoder entry")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace API token for authenticated access (for gated datasets)")
    parser.add_argument("--model_dir", type=Path,
                        default=get_cache_dir("supertonic-3"),
                        help="Model directory for unicode_indexer.json (used to validate texts). "
                             "Defaults to ~/.cache/supertonic3.")
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip text validation (faster, but may include unsupported characters)")
    parser.add_argument("--stats_only", action="store_true",
                        help="Print stats for an existing file instead of generating")
    args = parser.parse_args()

    # Load validator (unicode character checker)
    validator = None if args.no_validate else get_validator(args.model_dir)

    if args.stats_only:
        if not args.output.exists():
            print(f"File not found: {args.output}")
            sys.exit(1)
        print_stats(args.output)
        return

    if args.source == "wikipedia":
        generate_from_wikipedia(
            args.output, args.samples, args.min_chars, args.max_chars, args.seed,
            token=args.token, validator=validator,
        )
    elif args.source == "flores":
        generate_from_flores(
            args.output, args.samples, args.min_chars, args.max_chars, args.seed,
            token=args.token, validator=validator,
        )

    if args.output.exists():
        print_stats(args.output)


if __name__ == "__main__":
    main()
