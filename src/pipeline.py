import re
import pandas as pd

from .nlp import clean, language, sentiment_fallback
from .absa import analyze_absa
from .stance import stance
from .model_store import predict_sentiment


def affect(text, sentiment):
    """
    Temporary reference estimator for valence/arousal.

    This is NOT a trained V-A model.
    """

    words = len(re.findall(r"\w+", text))

    base = {
        "Positive": 0.72,
        "Negative": 0.28,
        "Neutral": 0.50,
        "Mixed": 0.50,
    }.get(sentiment, 0.50)

    arousal = min(
        1,
        max(
            0,
            0.35
            + min(words, 80) / 160
            + (0.18 if "!" in text else 0),
        ),
    )

    return base, arousal


def _overall_from_absa(rows, fallback_sent):
    """
    Calculate overall sentiment from ABSA evidence.

    This gives aspect-level evidence priority over the
    document-level fallback classifier.
    """

    if not rows:
        return fallback_sent, 0.50

    labels = [
        r.get("sentiment", "Neutral")
        for r in rows
    ]

    confidences = [
        float(r.get("confidence", 0.50))
        for r in rows
    ]

    confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.50
    )

    if all(label == "Positive" for label in labels):
        return "Positive", confidence

    if all(label == "Negative" for label in labels):
        return "Negative", confidence

    if all(label == "Neutral" for label in labels):
        return "Neutral", confidence

    return "Mixed", confidence


def _explicit_stance(text, target):
    """
    Detect strong explicit stance expressions.

    Explicit statements are preferred over zero-shot
    stance inference when they clearly contain a stance cue.
    """

    if not target:
        return None

    low = text.lower()

    against_patterns = [
        r"\bdo not recommend\b",
        r"\bdon't recommend\b",
        r"\bdont recommend\b",
        r"\bwould not recommend\b",
        r"\bwouldn't recommend\b",
        r"\bdo not support\b",
        r"\bdon't support\b",
        r"\bdont support\b",
        r"\bwould not support\b",
        r"\bwouldn't support\b",
        r"\bdo not endorse\b",
        r"\bdon't endorse\b",
        r"\bdont endorse\b",
        r"\bwould not endorse\b",
        r"\bwouldn't endorse\b",
        r"\bnever recommend\b",
        r"\bnever support\b",
        r"\bnever endorse\b",
        r"\bregret buying\b",
        r"\bregret purchasing\b",
    ]

    support_patterns = [
        r"\bstrongly recommend\b",
        r"\brecommend this\b",
        r"\brecommend the\b",
        r"\bhighly recommend\b",
        r"\bdefinitely recommend\b",
        r"\bstrongly support\b",
        r"\bfully support\b",
        r"\bsupport this\b",
        r"\bsupport the\b",
        r"\bdefinitely support\b",
        r"\bhighly recommend\b",
    ]

    for pattern in against_patterns:
        if re.search(pattern, low):
            return {
                "target": target,
                "label": "Against",
                "confidence": 0.95,
                "method": (
                    "explicit stance cue + "
                    "target-aware consistency rule"
                ),
            }

    for pattern in support_patterns:
        if re.search(pattern, low):
            return {
                "target": target,
                "label": "Support",
                "confidence": 0.95,
                "method": (
                    "explicit stance cue + "
                    "target-aware consistency rule"
                ),
            }

    return None


def analyze_text(text, target=None, model=None):

    # ---------------------------------------------------------
    # 1. Clean and detect language
    # ---------------------------------------------------------

    text = clean(str(text or ""))
    lang = language(text)

    # ---------------------------------------------------------
    # 2. Document-level fallback sentiment
    # ---------------------------------------------------------

    fallback_sent, fallback_conf = sentiment_fallback(text)

    # ---------------------------------------------------------
    # 3. ABSA
    # ---------------------------------------------------------

    try:

        absa_result = analyze_absa(
            text,
            target=target,
        )

    except Exception as exc:

        absa_result = {
            "engine": "error-fallback",
            "supported": False,
            "evidence": [],
            "message": (
                "Aspect-level analysis failed: "
                f"{type(exc).__name__}"
            ),
        }

    aspects = absa_result.get(
        "evidence",
        [],
    )

    absa_mode = absa_result.get(
        "engine",
        "reference",
    )

    absa_message = absa_result.get(
        "message"
    )

    # ---------------------------------------------------------
    # 4. Trained document sentiment model
    # ---------------------------------------------------------

    trained_sent, trained_conf = predict_sentiment(
        model,
        text,
    )

    # ---------------------------------------------------------
    # 5. Overall sentiment consistency
    # ---------------------------------------------------------

    if aspects:

        sent, conf = _overall_from_absa(
            aspects,
            fallback_sent,
        )

    elif trained_sent is not None:

        sent = trained_sent.title()

        conf = (
            trained_conf
            if trained_conf is not None
            else fallback_conf
        )

    else:

        sent = fallback_sent
        conf = fallback_conf

    # ---------------------------------------------------------
    # 6. Stance
    # ---------------------------------------------------------

    explicit_stance = _explicit_stance(
        text,
        target,
    )

    if explicit_stance is not None:

        st = explicit_stance

    else:

        try:
            st = stance(
                text,
                target,
            )

        except Exception as exc:

            st = {
                "target": target,
                "label": "Unknown",
                "confidence": 0.0,
                "method": (
                    "stance model unavailable: "
                    f"{type(exc).__name__}"
                ),
            }

    # ---------------------------------------------------------
    # 7. Valence / arousal
    # ---------------------------------------------------------

    affect_sentiment = (
        sent
        if sent != "Mixed"
        else fallback_sent
    )

    valence, arousal = affect(
        text,
        affect_sentiment,
    )

    # ---------------------------------------------------------
    # 8. Evidence strings
    # ---------------------------------------------------------

    evidence = []

    for row in aspects:

        aspect = row.get(
            "aspect",
            "",
        )

        opinion = row.get(
            "opinion",
            "",
        )

        sentiment = row.get(
            "sentiment",
            "Neutral",
        )

        evidence.append(
            f"{aspect} → {opinion} → {sentiment}"
        )

    # ---------------------------------------------------------
    # 9. Model note
    # ---------------------------------------------------------

    if model is not None:

        model_note = (
            "Kaggle-trained reference sentiment model "
            "is active. "
        )

    else:

        model_note = (
            "Kaggle-trained reference model not found; "
            "using development fallback. "
        )

    model_note += (
        "Overall sentiment follows explicit ABSA evidence "
        "when reliable evidence is available. "
        "ABSA is unavailable for unsupported languages "
        "rather than generating unreliable character-level "
        "aspects. "
        "Valence/arousal is currently a reference estimator "
        "until a labelled V-A corpus is trained."
    )

    # ---------------------------------------------------------
    # 10. Final result
    # ---------------------------------------------------------

    return {
        "text": text,
        "language": lang,

        "overall_sentiment": sent,
        "overall_confidence": conf,

        "aspects": aspects,

        "stance": st,

        "valence": valence,
        "arousal": arousal,

        "evidence": evidence,

        "absa_mode": absa_mode,
        "absa_message": absa_message,

        "model_note": model_note,
    }


def analyze_dataframe(
    df,
    text_col,
    target_col,
    model=None,
):

    rows = []

    for _, row in df.iterrows():

        text_value = str(
            row[text_col]
        )

        target_value = (
            None
            if not target_col
            else str(row[target_col])
        )

        output = analyze_text(
            text_value,
            target_value,
            model,
        )

        rows.append(
            {
                "text": output["text"],

                "language": output["language"],

                "sentiment": (
                    output["overall_sentiment"]
                ),

                "confidence": (
                    output["overall_confidence"]
                ),

                "valence": output["valence"],

                "arousal": output["arousal"],

                "stance": (
                    output["stance"].get(
                        "label",
                        "Unknown",
                    )
                ),

                "target": (
                    output["stance"].get(
                        "target",
                        target_value,
                    )
                ),

                "aspects": "; ".join(
                    item.get("aspect", "")
                    for item in output["aspects"]
                ),

                "aspect_opinions": " | ".join(
                    (
                        f"{item.get('aspect', '')}"
                        f"={item.get('opinion', '')}"
                    )
                    for item in output["aspects"]
                ),

                "evidence": " | ".join(
                    output["evidence"]
                ),
            }
        )

    return pd.DataFrame(rows)