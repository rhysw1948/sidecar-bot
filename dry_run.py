"""
Dry-run of the template bot on at most 2 bot-testing-area questions.
Publishing is OFF in two ways:
  1. publish_reports_to_metaculus=False (the template's own switch).
  2. Every MetaculusClient method that sends a forecast or comment is
     replaced with one that raises an error, so nothing can be posted
     even if (1) were somehow ignored.
Usage: .venv/bin/python dry_run.py [max_questions]
"""
import asyncio
import json
import logging
import sys
import time

import dotenv

dotenv.load_dotenv("/root/Metaculus/.env")

from main import FallTemplateBot2026  # noqa: E402  (template bot class)
from forecasting_tools import GeneralLlm, MetaculusClient  # noqa: E402

TOURNAMENT_ID = 32977  # bot-testing-area
MAX_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
# Note: inclusionai/ling-3.0-tiny (cheapest) was tried first on 2026-11-12 and
# failed with upstream 429 rate limits (single provider), so gpt-oss-20b is the default.
MODEL = sys.argv[2] if len(sys.argv) > 2 else "openrouter/openai/gpt-oss-20b"

# --- Safety belt: block every posting method on the Metaculus client ---
POST_METHODS = [
    "post_question_comment",
    "post_binary_question_prediction",
    "post_numeric_question_prediction",
    "post_multiple_choice_question_prediction",
    "_post_question_prediction",
    "_post_bulk_forecasts_and_comments",
]


def _make_blocker(name):
    def _blocked(*args, **kwargs):
        raise RuntimeError(f"DRY RUN: blocked call to MetaculusClient.{name}")

    return _blocked


for _name in POST_METHODS:
    if hasattr(MetaculusClient, _name):
        setattr(MetaculusClient, _name, _make_blocker(_name))
    else:
        print(f"note: MetaculusClient has no method {_name}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    llm = GeneralLlm(model=MODEL, temperature=0.3, timeout=120, allowed_tries=2)
    bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=True,
        llms={
            "default": llm,
            "summarizer": llm,
            "researcher": llm,  # plain LLM, no web search
            "parser": llm,
        },
    )
    assert bot.publish_reports_to_metaculus is False
    print(f"publish_reports_to_metaculus = {bot.publish_reports_to_metaculus}")

    questions = MetaculusClient().get_all_open_questions_from_tournament(TOURNAMENT_ID)
    print(f"Open questions in tournament {TOURNAMENT_ID}: {len(questions)}")
    # Prefer one binary plus one other type, to test more than one code path.
    binary = [q for q in questions if type(q).__name__ == "BinaryQuestion"]
    other = [q for q in questions if type(q).__name__ != "BinaryQuestion"]
    chosen = (binary[:1] + other[:1] + binary[1:] + other[1:])[:MAX_QUESTIONS]

    results = []
    for q in chosen:
        t0 = time.time()
        [report] = asyncio.run(bot.forecast_questions([q], return_exceptions=True))
        secs = time.time() - t0
        row = {
            "url": q.page_url,
            "type": type(q).__name__,
            "title": q.question_text,
            "seconds": round(secs, 1),
        }
        if isinstance(report, BaseException):
            row["error"] = f"{type(report).__name__}: {str(report)[:500]}"
        else:
            pred = report.prediction
            row["prediction"] = (
                pred if isinstance(pred, (int, float)) else str(pred)[:600]
            )
            row["price_estimate_usd"] = report.price_estimate
            row["minor_errors"] = report.errors
            row["explanation_chars"] = len(report.explanation)
        results.append(row)
        print("RESULT " + json.dumps(row, default=str))


if __name__ == "__main__":
    main()
