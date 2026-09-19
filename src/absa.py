"""
CMASSIP Aspect-Based Sentiment Analysis

English:
    DeBERTa end-to-end ABSA transformer + contextual evidence linker.

Hindi / Tamil:
    Language-aware aspect/opinion extraction with explicit
    negation handling and local aspect-opinion linking.

Important:
    Stance detection is NOT implemented here.
    Stance remains model-backed through BART-MNLI in stance.py.
"""

from functools import lru_cache
import re
import unicodedata


MODEL_ID = "yangheng/deberta-v3-base-end2end-absa"


# ============================================================
# MODEL
# ============================================================

@lru_cache(maxsize=1)
def _load_pipeline():
    from transformers import pipeline

    return pipeline(
        task="token-classification",
        model=MODEL_ID,
        aggregation_strategy="simple",
    )


# ============================================================
# LABEL NORMALISATION
# ============================================================

def _normalise_label(label: str) -> str:
    x = str(label).lower()

    if "positive" in x or x.endswith("pos") or x == "pos":
        return "Positive"

    if "negative" in x or x.endswith("neg") or x == "neg":
        return "Negative"

    if "neutral" in x or x.endswith("neu") or x == "neu":
        return "Neutral"

    return "Neutral"


# ============================================================
# ENGLISH LEXICON
# ============================================================

_POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "beautiful",
    "perfect",
    "best",
    "wonderful",
    "fantastic",
    "superb",
    "love",
    "loved",
    "like",
    "liked",
    "helpful",
    "fast",
    "easy",
    "smooth",
    "durable",
    "clear",
    "comfortable",
    "strong",
    "reliable",
    "affordable",
    "bright",
    "impressive",
    "recommend",
    "recommended",
}


_NEGATIVE_WORDS = {
    "bad",
    "poor",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "hate",
    "hated",
    "slow",
    "difficult",
    "ugly",
    "broken",
    "disappointing",
    "disappointed",
    "drains",
    "draining",
    "expensive",
    "weak",
    "problem",
    "problems",
    "annoying",
    "useless",
    "rude",
    "late",
    "poorly",
    "laggy",
    "dim",
    "unreliable",
    "noisy",
}


_OPINION_WORDS = _POSITIVE_WORDS | _NEGATIVE_WORDS


_MODIFIERS = {
    "very",
    "really",
    "extremely",
    "quite",
    "so",
    "too",
    "fairly",
    "highly",
    "incredibly",
    "absolutely",
    "slightly",
    "somewhat",
    "a",
    "bit",
    "not",
}


_CLAUSE_BREAKS = re.compile(
    r"(?:[,;]|\b(?:but|although|though|however|while|whereas|yet|and)\b)",
    re.I,
)


_SENTENCE_BREAKS = re.compile(
    r"(?<=[.!?])\s+"
)


# ============================================================
# HINDI ASPECTS
# ============================================================

_HINDI_ASPECTS = {
    "मोबाइल": "मोबाइल",
    "मोबाइल फोन": "मोबाइल",
    "फोन": "फोन",
    "कैमरा": "कैमरा",
    "डिस्प्ले": "डिस्प्ले",
    "स्क्रीन": "स्क्रीन",
    "बैटरी": "बैटरी",
    "कीमत": "कीमत",
    "मूल्य": "मूल्य",
    "डिजाइन": "डिजाइन",
    "डिज़ाइन": "डिज़ाइन",
    "परफॉर्मेंस": "परफॉर्मेंस",
    "प्रदर्शन": "प्रदर्शन",
    "स्पीड": "स्पीड",
    "आवाज": "आवाज",
    "आवाज़": "आवाज़",
    "साउंड": "साउंड",
    "सॉफ्टवेयर": "सॉफ्टवेयर",
    "फीचर": "फीचर",
    "फीचर्स": "फीचर्स",
    "सेवा": "सेवा",
    "सर्विस": "सर्विस",
    "स्टोरेज": "स्टोरेज",
    "कीबोर्ड": "कीबोर्ड",
    "माउस": "माउस",
}


# ============================================================
# HINDI OPINIONS
# ============================================================

_HINDI_POSITIVE = {
    "अच्छा",
    "अच्छी",
    "अच्छे",
    "बहुत अच्छा",
    "बहुत अच्छी",
    "बहुत अच्छे",
    "शानदार",
    "बेहतरीन",
    "उत्कृष्ट",
    "कमाल",
    "बढ़िया",
    "बढिया",
    "सुंदर",
    "साफ",
    "स्पष्ट",
    "तेज़",
    "तेज",
    "आसान",
    "मजबूत",
    "विश्वसनीय",
    "पसंद",
    "पसंद है",
    "सिफारिश",
    "सिफारिश करता",
    "सिफारिश करती",
    "सिफारिश करूंगा",
    "सिफारिश करूंगी",
    "समर्थन",
    "समर्थन करता",
    "समर्थन करती",
}


_HINDI_NEGATIVE = {
    "खराब",
    "बुरा",
    "बुरी",
    "बहुत खराब",
    "घटिया",
    "बेकार",
    "धीमा",
    "धीमी",
    "महंगा",
    "महंगी",
    "कमजोर",
    "कमज़ोर",
    "परेशान",
    "समस्या",
    "समस्याएं",
    "समस्याएँ",
    "गर्म",
    "लैग",
    "अटकता",
    "अटकती",
    "खत्म",
    "जल्दी खत्म",
    "जल्दी खत्म हो जाती है",
    "जल्दी खत्म हो जाता है",
    "खत्म हो जाती है",
    "खत्म हो जाता है",
    "निराश",
    "निराशाजनक",
    "अविश्वसनीय",
    "विरोध",
    "समर्थन नहीं",
    "सिफारिश नहीं",
}


# ============================================================
# TAMIL ASPECTS
# ============================================================

_TAMIL_ASPECTS = {
    "மொபைல்": "மொபைல்",
    "மொபைல் போன்": "மொபைல்",
    "போன்": "போன்",
    "கேமரா": "கேமரா",
    "டிஸ்ப்ளே": "டிஸ்ப்ளே",
    "திரை": "திரை",
    "பேட்டரி": "பேட்டரி",
    "விலை": "விலை",
    "வடிவமைப்பு": "வடிவமைப்பு",
    "செயல்திறன்": "செயல்திறன்",
    "வேகம்": "வேகம்",
    "ஒலி": "ஒலி",
    "சவுண்ட்": "சவுண்ட்",
    "சாப்ட்வேர்": "சாப்ட்வேர்",
    "அம்சம்": "அம்சம்",
    "அம்சங்கள்": "அம்சங்கள்",
    "சேவை": "சேவை",
    "சேமிப்பு": "சேமிப்பு",
}


# ============================================================
# TAMIL OPINIONS
# ============================================================

_TAMIL_POSITIVE = {
    "நல்ல",
    "நன்றாக",
    "மிகவும் நல்ல",
    "சிறந்த",
    "அருமையான",
    "அற்புதமான",
    "அழகான",
    "தெளிவான",
    "வேகமான",
    "எளிதான",
    "வலுவான",
    "நம்பகமான",
    "பிடிக்கும்",
    "பரிந்துரைக்கிறேன்",
    "பரிந்துரை",
    "ஆதரிக்கிறேன்",
    "ஆதரிக்கிறோம்",
}


_TAMIL_NEGATIVE = {
    "மோசமான",
    "மோசம்",
    "மிகவும் மோசமான",
    "மோசமாக",
    "மெதுவான",
    "விலை உயர்ந்த",
    "பலவீனமான",
    "பிரச்சனை",
    "பிரச்சினை",
    "எரிச்சலூட்டும்",
    "வேலை செய்யவில்லை",
    "விரைவாக காலியாகிறது",
    "விரைவாக தீர்ந்துவிடுகிறது",
    "காலியாகிறது",
    "நம்பகமற்ற",
    "ஆதரிக்கவில்லை",
    "பரிந்துரைக்கவில்லை",
    "எதிர்க்கிறேன்",
}


# ============================================================
# SCRIPT DETECTION
# ============================================================

def _script_language(text):
    """
    Detect Hindi/Tamil based on Unicode script.
    """

    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"

    return "other"


# ============================================================
# UNICODE HELPERS
# ============================================================

def _contains_real_letter(text):
    for ch in text:
        if ch.isalpha():
            return True

    return False


def _clean_span(text):
    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip(
        " .,!?:;\"'“”‘’"
    )


def _valid_multilingual_aspect(word):
    """
    Reject one-character/subword/combining-mark garbage.
    """

    word = _clean_span(word)

    if not word:
        return False

    compact = re.sub(
        r"\s+",
        "",
        word,
    )

    if len(compact) < 2:
        return False

    if not _contains_real_letter(compact):
        return False

    categories = [
        unicodedata.category(ch)
        for ch in compact
    ]

    if all(
        c in {"Mn", "Mc"}
        for c in categories
    ):
        return False

    if compact in {
        "##",
        "▁",
        "Ġ",
    }:
        return False

    return True


# ============================================================
# HINDI NEGATION
# ============================================================

def _hindi_negative_recommendation(clause):
    patterns = [
        r"सिफारिश\s+नहीं\s+करता",
        r"सिफारिश\s+नहीं\s+करती",
        r"सिफारिश\s+नहीं\s+करता\s+हूँ",
        r"सिफारिश\s+नहीं\s+करती\s+हूँ",
        r"सिफारिश\s+नहीं\s+करूंगा",
        r"सिफारिश\s+नहीं\s+करूंगी",
    ]

    return any(
        re.search(pattern, clause)
        for pattern in patterns
    )


def _hindi_positive_recommendation(clause):
    patterns = [
        r"सिफारिश\s+करता",
        r"सिफारिश\s+करती",
        r"सिफारिश\s+करता\s+हूँ",
        r"सिफारिश\s+करती\s+हूँ",
        r"सिफारिश\s+करूंगा",
        r"सिफारिश\s+करूंगी",
    ]

    return any(
        re.search(pattern, clause)
        for pattern in patterns
    )


def _hindi_opinion_sentiment(clause, opinion):
    """
    Determine Hindi opinion polarity with negation priority.
    """

    low = clause.lower()
    opinion_low = opinion.lower()

    # --------------------------------------------------------
    # Strong explicit negative phrases FIRST
    # --------------------------------------------------------

    if _hindi_negative_recommendation(low):
        if (
            "सिफारिश" in opinion_low
            or "recommend" in opinion_low
        ):
            return "Negative"

    negative_phrases = [
        "समर्थन नहीं करता",
        "समर्थन नहीं करती",
        "पसंद नहीं",
        "अच्छा नहीं",
        "अच्छी नहीं",
        "अच्छे नहीं",
        "बहुत अच्छा नहीं",
        "बहुत अच्छी नहीं",
        "विश्वसनीय नहीं",
        "भरोसेमंद नहीं",
    ]

    if any(
        phrase in low
        for phrase in negative_phrases
    ):
        return "Negative"

    # --------------------------------------------------------
    # Explicit positive recommendation
    # --------------------------------------------------------

    if _hindi_positive_recommendation(low):
        if (
            "सिफारिश" in opinion_low
            or "recommend" in opinion_low
        ):
            return "Positive"

    positive_phrases = [
        "समर्थन करता",
        "समर्थन करती",
        "पसंद है",
        "बहुत अच्छा",
        "बहुत अच्छी",
        "बहुत अच्छे",
    ]

    if any(
        phrase in low
        for phrase in positive_phrases
    ):
        if not re.search(
            r"नहीं",
            low,
        ):
            return "Positive"

    # --------------------------------------------------------
    # Generic negation
    # --------------------------------------------------------

    has_negation = bool(
        re.search(
            r"(?:^|\s)(नहीं|नही|मत)(?:\s|$)",
            low,
        )
    )

    if opinion_low in _HINDI_POSITIVE:
        return (
            "Negative"
            if has_negation
            else "Positive"
        )

    if opinion_low in _HINDI_NEGATIVE:
        return (
            "Positive"
            if has_negation
            else "Negative"
        )

    return None


# ============================================================
# TAMIL NEGATION
# ============================================================

def _tamil_opinion_sentiment(clause, opinion):
    """
    Determine Tamil opinion polarity with explicit negation.
    """

    low = clause.lower()
    opinion_low = opinion.lower()

    negative_phrases = [
        "பரிந்துரைக்கவில்லை",
        "ஆதரிக்கவில்லை",
        "பிடிக்கவில்லை",
        "வேலை செய்யவில்லை",
        "நல்லதல்ல",
        "சிறந்ததல்ல",
        "நம்பகமானதல்ல",
    ]

    if any(
        phrase in low
        for phrase in negative_phrases
    ):
        return "Negative"

    positive_phrases = [
        "பரிந்துரைக்கிறேன்",
        "ஆதரிக்கிறேன்",
        "பிடிக்கும்",
    ]

    if any(
        phrase in low
        for phrase in positive_phrases
    ):
        if not any(
            x in low
            for x in [
                "இல்லை",
                "வில்லை",
                "அல்ல",
            ]
        ):
            return "Positive"

    if opinion_low in _TAMIL_POSITIVE:
        return "Positive"

    if opinion_low in _TAMIL_NEGATIVE:
        return "Negative"

    return None


# ============================================================
# GENERIC ENGLISH CONTEXT
# ============================================================

def _sentence_and_clause(text, pos):
    sent_start = 0
    sent_end = len(text)

    for match in _SENTENCE_BREAKS.finditer(text):

        if match.start() < pos:
            sent_start = match.end()
        else:
            sent_end = match.start()
            break

    sentence = text[
        sent_start:sent_end
    ]

    relative_pos = pos - sent_start

    clause_start = 0
    clause_end = len(sentence)

    for match in _CLAUSE_BREAKS.finditer(sentence):

        if match.start() < relative_pos:
            clause_start = match.end()
        else:
            clause_end = match.start()
            break

    clause = sentence[
        clause_start:clause_end
    ].strip()

    return sentence, clause


def _opinion_candidates(clause):

    candidates = []

    words = list(
        re.finditer(
            r"\b[A-Za-z][A-Za-z'-]*\b",
            clause,
        )
    )

    for i, match in enumerate(words):

        word = match.group(0)

        if word.lower() not in _OPINION_WORDS:
            continue

        left = i

        while (
            left > 0
            and words[left - 1]
            .group(0)
            .lower()
            in _MODIFIERS
        ):
            left -= 1

        right = i + 1

        while (
            right < len(words)
            and right <= i + 2
        ):

            nxt = (
                words[right]
                .group(0)
                .lower()
            )

            if nxt in {
                "very",
                "really",
                "quickly",
                "slowly",
                "easily",
                "badly",
                "well",
            }:
                right += 1
            else:
                break

        phrase = clause[
            words[left].start():
            words[right - 1].end()
        ]

        candidates.append(
            (
                match.start(),
                phrase,
            )
        )

    return candidates


def _opinion_near(text, start, end):

    sentence, clause = _sentence_and_clause(
        text,
        start,
    )

    clause_abs = text.find(
        clause,
        max(
            0,
            start - len(sentence),
        ),
    )

    if clause_abs < 0:
        clause_abs = max(
            0,
            start - 1,
        )

    aspect_local = max(
        0,
        start - clause_abs,
    )

    candidates = _opinion_candidates(
        clause
    )

    if not candidates:
        return "contextual opinion"

    scored = []

    for pos, phrase in candidates:

        distance = abs(
            pos - aspect_local
        )

        score = float(distance)

        scored.append(
            (
                score,
                phrase,
            )
        )

    return min(
        scored,
        key=lambda x: x[0],
    )[1]


def _opinion_polarity(opinion):

    tokens = re.findall(
        r"[A-Za-z][A-Za-z'-]*",
        opinion.lower(),
    )

    has_positive = any(
        token in _POSITIVE_WORDS
        for token in tokens
    )

    has_negative = any(
        token in _NEGATIVE_WORDS
        for token in tokens
    )

    if has_positive and not has_negative:
        return "Positive"

    if has_negative and not has_positive:
        return "Negative"

    return None


# ============================================================
# FIND LANGUAGE OPINION PHRASES
# ============================================================

def _find_phrases(clause, lexicon):
    """
    Find all known phrases in a clause.
    Longer phrases are preferred.
    """

    matches = []

    ordered = sorted(
        lexicon,
        key=len,
        reverse=True,
    )

    for phrase in ordered:

        start = clause.find(
            phrase
        )

        if start >= 0:
            matches.append(
                (
                    start,
                    phrase,
                )
            )

    return matches


# ============================================================
# HINDI FALLBACK
# ============================================================

def _hindi_fallback(text):

    rows = []

    # --------------------------------------------------------
    # Sentence-level processing
    # --------------------------------------------------------

    sentences = re.split(
        r"(?<=[।.!?])\s*",
        text,
    )

    last_aspects = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # ----------------------------------------------------
        # Clause split
        # ----------------------------------------------------

        clauses = re.split(
            r"(?:,|;|लेकिन|मगर|पर|हालांकि|फिर भी)",
            sentence,
            flags=re.IGNORECASE,
        )

        sentence_aspects = []

        for clause in clauses:

            clause = clause.strip()

            if not clause:
                continue

            # -----------------------------------------------
            # Find aspects
            # -----------------------------------------------

            found_aspects = []

            for key, display in sorted(
                _HINDI_ASPECTS.items(),
                key=lambda x: len(x[0]),
                reverse=True,
            ):

                if re.search(
                    re.escape(key),
                    clause,
                ):
                    found_aspects.append(
                        (
                            key,
                            display,
                        )
                    )

            # ------------------------------------------------
            # If no aspect occurs in this sentence but
            # the previous sentence had one, carry it forward.
            # ------------------------------------------------

            if not found_aspects and last_aspects:

                has_opinion = (
                    bool(
                        _find_phrases(
                            clause,
                            _HINDI_POSITIVE,
                        )
                    )
                    or
                    bool(
                        _find_phrases(
                            clause,
                            _HINDI_NEGATIVE,
                        )
                    )
                )

                if has_opinion:
                    found_aspects = last_aspects.copy()

            if not found_aspects:
                continue

            for item in found_aspects:
                if item not in sentence_aspects:
                    sentence_aspects.append(item)

            # ------------------------------------------------
            # Find positive / negative phrases
            # ------------------------------------------------

            positive = _find_phrases(
                clause,
                _HINDI_POSITIVE,
            )

            negative = _find_phrases(
                clause,
                _HINDI_NEGATIVE,
            )

            candidates = []

            for position, phrase in positive:
                candidates.append(
                    (
                        position,
                        phrase,
                    )
                )

            for position, phrase in negative:
                candidates.append(
                    (
                        position,
                        phrase,
                    )
                )

            if not candidates:
                continue

            # ------------------------------------------------
            # Associate closest opinion with each aspect
            # ------------------------------------------------

            for aspect_key, aspect_display in found_aspects:

                match = re.search(
                    re.escape(aspect_key),
                    clause,
                )

                if match:

                    aspect_position = (
                        match.start()
                    )

                else:

                    # Aspect came from previous sentence.
                    aspect_position = 0

                best = min(
                    candidates,
                    key=lambda x: abs(
                        x[0]
                        - aspect_position
                    ),
                )

                opinion = best[1]

                sentiment = (
                    _hindi_opinion_sentiment(
                        clause,
                        opinion,
                    )
                )

                if sentiment is None:

                    if opinion in _HINDI_POSITIVE:
                        sentiment = "Positive"

                    elif opinion in _HINDI_NEGATIVE:
                        sentiment = "Negative"

                    else:
                        continue

                # --------------------------------------------
                # Preserve the full recommendation negation
                # in the evidence.
                # --------------------------------------------

                if (
                    "सिफारिश" in opinion
                    and _hindi_negative_recommendation(
                        clause
                    )
                ):
                    opinion = "सिफारिश नहीं करता"

                    if "करती" in clause:
                        opinion = "सिफारिश नहीं करती"

                    sentiment = "Negative"

                rows.append(
                    {
                        "aspect": aspect_display,
                        "opinion": opinion,
                        "sentiment": sentiment,
                        "confidence": 0.88,
                        "source": "Hindi multilingual ABSA",
                    }
                )

        if sentence_aspects:
            last_aspects = sentence_aspects

    return _deduplicate(rows)


# ============================================================
# TAMIL FALLBACK
# ============================================================

def _tamil_fallback(text):

    rows = []

    sentences = re.split(
        r"(?<=[.!?])\s*",
        text,
    )

    last_aspects = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        clauses = re.split(
            r"(?:,|;|ஆனால்|என்றாலும்)",
            sentence,
            flags=re.IGNORECASE,
        )

        sentence_aspects = []

        for clause in clauses:

            clause = clause.strip()

            if not clause:
                continue

            # ------------------------------------------------
            # Find aspects
            # ------------------------------------------------

            found_aspects = []

            for key, display in sorted(
                _TAMIL_ASPECTS.items(),
                key=lambda x: len(x[0]),
                reverse=True,
            ):

                if re.search(
                    re.escape(key),
                    clause,
                ):
                    found_aspects.append(
                        (
                            key,
                            display,
                        )
                    )

            # ------------------------------------------------
            # Carry previous aspect forward
            # ------------------------------------------------

            if not found_aspects and last_aspects:

                has_opinion = (
                    bool(
                        _find_phrases(
                            clause,
                            _TAMIL_POSITIVE,
                        )
                    )
                    or
                    bool(
                        _find_phrases(
                            clause,
                            _TAMIL_NEGATIVE,
                        )
                    )
                )

                if has_opinion:
                    found_aspects = last_aspects.copy()

            if not found_aspects:
                continue

            for item in found_aspects:
                if item not in sentence_aspects:
                    sentence_aspects.append(item)

            # ------------------------------------------------
            # Find opinions
            # ------------------------------------------------

            positive = _find_phrases(
                clause,
                _TAMIL_POSITIVE,
            )

            negative = _find_phrases(
                clause,
                _TAMIL_NEGATIVE,
            )

            candidates = []

            for position, phrase in positive:
                candidates.append(
                    (
                        position,
                        phrase,
                    )
                )

            for position, phrase in negative:
                candidates.append(
                    (
                        position,
                        phrase,
                    )
                )

            if not candidates:
                continue

            # ------------------------------------------------
            # Link opinions to aspects
            # ------------------------------------------------

            for aspect_key, aspect_display in found_aspects:

                match = re.search(
                    re.escape(aspect_key),
                    clause,
                )

                if match:
                    aspect_position = (
                        match.start()
                    )
                else:
                    aspect_position = 0

                best = min(
                    candidates,
                    key=lambda x: abs(
                        x[0]
                        - aspect_position
                    ),
                )

                opinion = best[1]

                sentiment = (
                    _tamil_opinion_sentiment(
                        clause,
                        opinion,
                    )
                )

                if sentiment is None:

                    if opinion in _TAMIL_POSITIVE:
                        sentiment = "Positive"

                    elif opinion in _TAMIL_NEGATIVE:
                        sentiment = "Negative"

                    else:
                        continue

                rows.append(
                    {
                        "aspect": aspect_display,
                        "opinion": opinion,
                        "sentiment": sentiment,
                        "confidence": 0.88,
                        "source": "Tamil multilingual ABSA",
                    }
                )

        if sentence_aspects:
            last_aspects = sentence_aspects

    return _deduplicate(rows)


# ============================================================
# DEDUPLICATION
# ============================================================

def _deduplicate(rows):

    seen = set()
    output = []

    for row in rows:

        key = (
            row["aspect"],
            row["opinion"],
            row["sentiment"],
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(row)

    return output


# ============================================================
# ENGLISH TRANSFORMER ANALYSIS
# ============================================================

def _transformer_analysis(
    text,
    multilingual=False,
):

    pipe = _load_pipeline()

    entities = pipe(text)

    rows = []

    for entity in entities:

        word = str(
            entity.get(
                "word",
                "",
            )
        ).strip()

        word = _clean_span(word)

        if not word:
            continue

        if (
            multilingual
            and not _valid_multilingual_aspect(word)
        ):
            continue

        start = entity.get("start")
        end = entity.get("end")

        if start is None or end is None:

            start = text.lower().find(
                word.lower()
            )

            end = (
                start + len(word)
                if start >= 0
                else 0
            )

        if start < 0:
            continue

        opinion = _opinion_near(
            text,
            int(start),
            int(end),
        )

        model_sentiment = _normalise_label(
            entity.get(
                "entity_group",
                entity.get(
                    "entity",
                    "Neutral",
                ),
            )
        )

        lexical_sentiment = (
            _opinion_polarity(
                opinion
            )
        )

        if (
            model_sentiment == "Neutral"
            and lexical_sentiment
        ):
            sentiment = lexical_sentiment
        else:
            sentiment = model_sentiment

        rows.append(
            {
                "aspect": word,
                "opinion": opinion,
                "sentiment": sentiment,
                "confidence": float(
                    entity.get(
                        "score",
                        0.0,
                    )
                ),
                "source":
                    "DeBERTa end-to-end ABSA "
                    "+ contextual evidence linker",
            }
        )

    return rows


# ============================================================
# PUBLIC ABSA
# ============================================================

def analyze(
    text,
    fallback_aspects=None,
):

    lang = _script_language(text)

    # ========================================================
    # HINDI
    # ========================================================

    if lang == "hi":

        # IMPORTANT:
        # Do NOT use the English ABSA transformer first.
        # The Hindi language layer is more reliable for the
        # supported project-domain vocabulary and negation.

        rows = _hindi_fallback(text)

        if rows:
            return rows, True

        return [], False

    # ========================================================
    # TAMIL
    # ========================================================

    if lang == "ta":

        # Same approach for Tamil.
        rows = _tamil_fallback(text)

        if rows:
            return rows, True

        return [], False

    # ========================================================
    # ENGLISH
    # ========================================================

    try:

        rows = _transformer_analysis(
            text,
            multilingual=False,
        )

    except Exception:

        rows = []

    # --------------------------------------------------------
    # Add missed English domain aspects
    # --------------------------------------------------------

    existing = {
        str(row["aspect"]).lower()
        for row in rows
    }

    if fallback_aspects:

        for aspect in fallback_aspects:

            if aspect.lower() in existing:
                continue

            start = text.lower().find(
                aspect.lower()
            )

            if start < 0:
                continue

            opinion = _opinion_near(
                text,
                start,
                start + len(aspect),
            )

            lexical = _opinion_polarity(
                opinion
            )

            if lexical:

                sentiment = lexical
                confidence = 0.82

            else:

                try:

                    from .nlp import sentiment_fallback

                    sentiment, confidence = (
                        sentiment_fallback(
                            text[
                                max(
                                    0,
                                    start - 60,
                                ):
                                min(
                                    len(text),
                                    start
                                    + len(aspect)
                                    + 80,
                                )
                            ]
                        )
                    )

                except Exception:

                    sentiment = "Neutral"
                    confidence = 0.50

            rows.append(
                {
                    "aspect": aspect,
                    "opinion": opinion,
                    "sentiment": sentiment,
                    "confidence": float(
                        confidence
                    ),
                    "source":
                        "contextual evidence completion",
                }
            )

    return _deduplicate(rows), True


# ============================================================
# PIPELINE COMPATIBILITY WRAPPER
# ============================================================

def analyze_absa(
    text,
    target=None,
):

    rows, model_used = analyze(text)

    lang = _script_language(text)

    if rows:

        if lang == "hi":

            engine = "Hindi multilingual ABSA"

        elif lang == "ta":

            engine = "Tamil multilingual ABSA"

        else:

            engine = "transformer"

        return {
            "engine": engine,
            "supported": True,
            "evidence": rows,
            "message": None,
        }

    return {
        "engine": "reference",
        "supported": False,
        "evidence": [],
        "message":
            "No reliable aspect-opinion pairs were extracted.",
    }