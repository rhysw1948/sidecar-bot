"""
LIVE posting test: forecast exactly ONE binary question on the
bot-testing-area (project 32977) and publish the forecast plus its
private reasoning comment.

Safety checks (each one stops the script with an error):
  A. More than 1 post ID given -> refuse before any API call.
  B. Every Metaculus HTTP request goes through one guard that:
       - allows only 3 requests in total:
           GET  /api/posts/<post>/           (once, to read the question)
           POST /api/questions/forecast/     (once, only for that question)
           POST /api/comments/create/        (once, only on that post, private)
         anything else (other URL, other question, public comment, a retry,
         a 4th request) is refused and nothing is sent;
       - waits at least 4.5 s between Metaculus requests;
       - stops the whole process at once on HTTP 429;
       - respects a shared total cap (15) if MC_COUNTER_DIR is set.
  C. After the GET: refuse unless default_project.id == 32977 and the
     question type is binary. This happens before any LLM call or POST.
  D. Numeric / multiple-choice / bulk posting methods are blocked, and
     acting_user must be None.
Usage: .venv/bin/python -u live_test.py [post_id]   (default 43327)
"""
import asyncio
import json
import logging
import os
import sys
import time

ALLOWED_PROJECT = 32977  # bot-testing-area
POST_IDS = [int(a) for a in sys.argv[1:]] or [43327]

# --- Check A: exactly one question, before anything else (no imports of the bot yet) ---
if len(POST_IDS) != 1:
    sys.exit(f"REFUSED: asked to forecast {len(POST_IDS)} questions; only 1 allowed.")
POST_ID = POST_IDS[0]

import dotenv  # noqa: E402
import requests  # noqa: E402

dotenv.load_dotenv("/root/Metaculus/.env")

# --- Check B: guard on every Metaculus HTTP request ---
GAP_SECONDS = 4.5
TOTAL_CAP = 15
COUNTER_DIR = os.getenv("MC_COUNTER_DIR")  # shared counter with helper scripts
_state = {"last": 0.0, "count": 0, "question_id": None, "done": set()}
REQUEST_LOG = []  # (method, path, status)


def _shared_count() -> int:
    if not COUNTER_DIR:
        return 0
    p = os.path.join(COUNTER_DIR, "mc_count.txt")
    return int(open(p).read()) if os.path.exists(p) else 0


def _shared_record() -> None:
    if not COUNTER_DIR:
        return
    n = _shared_count()
    open(os.path.join(COUNTER_DIR, "mc_count.txt"), "w").write(str(n + 1))
    open(os.path.join(COUNTER_DIR, "mc_last.txt"), "w").write(str(time.time()))


def _shared_last() -> float:
    if not COUNTER_DIR:
        return 0.0
    p = os.path.join(COUNTER_DIR, "mc_last.txt")
    return float(open(p).read()) if os.path.exists(p) else 0.0


def _check_allowed(method: str, url: str, body) -> str:
    path = url.split("metaculus.com", 1)[1].split("?")[0]
    if method == "GET" and path == f"/api/posts/{POST_ID}/":
        key = "get_post"
    elif method == "POST" and path == "/api/questions/forecast/":
        key = "forecast"
        if not (isinstance(body, list) and len(body) == 1
                and body[0].get("question") == _state["question_id"]
                and _state["question_id"] is not None):
            raise RuntimeError("REFUSED: forecast body is not exactly 1 forecast on the checked question")
    elif method == "POST" and path == "/api/comments/create/":
        key = "comment"
        if not (isinstance(body, dict) and body.get("on_post") == POST_ID
                and body.get("is_private") is True):
            raise RuntimeError("REFUSED: comment is not a private comment on the checked post")
    else:
        raise RuntimeError(f"REFUSED: request not on the allow-list: {method} {path}")
    if key in _state["done"]:
        raise RuntimeError(f"REFUSED: second '{key}' request (no retries/repeats allowed)")
    if key != "get_post" and _state["question_id"] is None:
        raise RuntimeError("REFUSED: POST before the project check passed")
    return key


_orig_request = requests.sessions.Session.request


def _guarded_request(self, method, url, *args, **kwargs):
    if "metaculus.com" not in str(url):
        return _orig_request(self, method, url, *args, **kwargs)
    method = method.upper()
    key = _check_allowed(method, str(url), kwargs.get("json"))
    if _state["count"] >= 3:
        raise RuntimeError("REFUSED: more than 3 Metaculus requests in this run")
    if _shared_count() >= TOTAL_CAP:
        raise RuntimeError(f"REFUSED: shared cap of {TOTAL_CAP} Metaculus requests reached")
    wait = GAP_SECONDS - (time.time() - max(_state["last"], _shared_last()))
    if wait > 0:
        time.sleep(wait)
    _state["done"].add(key)  # mark before sending, so a retry is refused
    resp = _orig_request(self, method, url, *args, **kwargs)
    _state["last"] = time.time()
    _state["count"] += 1
    _shared_record()
    path = str(url).split("metaculus.com", 1)[1]
    REQUEST_LOG.append((method, path, resp.status_code))
    print(f"METACULUS {method} {path} -> {resp.status_code}", flush=True)
    if resp.status_code >= 400:
        print(f"METACULUS ERROR BODY: {resp.text[:800]}", flush=True)
    if resp.status_code == 429:
        print("STOP: HTTP 429 from Metaculus", flush=True)
        os._exit(3)
    return resp


requests.sessions.Session.request = _guarded_request

from main import FallTemplateBot2026  # noqa: E402
from forecasting_tools import GeneralLlm, MetaculusClient  # noqa: E402

# --- Check D: block non-binary and bulk posting methods ---
for _name in [
    "post_numeric_question_prediction",
    "post_multiple_choice_question_prediction",
    "_post_bulk_forecasts_and_comments",
]:
    def _blocked(*a, _n=_name, **k):
        raise RuntimeError(f"REFUSED: blocked call to MetaculusClient.{_n}")
    setattr(MetaculusClient, _name, _blocked)

MODEL = "openrouter/openai/gpt-oss-20b"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    client = MetaculusClient()
    if client.acting_user is not None:
        sys.exit("REFUSED: acting_user is set")

    # One GET (through the guard), then Check C before any LLM call or POST.
    resp = requests.get(
        f"{client.base_url}/posts/{POST_ID}/",
        **client._get_auth_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    post_json = resp.json()
    project_id = (post_json.get("projects", {}).get("default_project") or {}).get("id")
    if project_id != ALLOWED_PROJECT:
        sys.exit(f"REFUSED: post {POST_ID} is in project {project_id}, not {ALLOWED_PROJECT}")
    if post_json.get("question", {}).get("type") != "binary":
        sys.exit(f"REFUSED: post {POST_ID} is not a single binary question")
    questions = client._post_json_to_questions_while_handling_groups(post_json, "exclude")
    if len(questions) != 1 or type(questions[0]).__name__ != "BinaryQuestion":
        sys.exit("REFUSED: did not get exactly 1 binary question")
    q = questions[0]
    if q.id_of_post != POST_ID:
        sys.exit("REFUSED: post id mismatch")
    _state["question_id"] = q.id_of_question
    print(f"CHECKS PASSED: post {POST_ID}, question {q.id_of_question}, "
          f"project {project_id}, binary", flush=True)

    llm = GeneralLlm(model=MODEL, temperature=0.3, timeout=120, allowed_tries=2)
    bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,  # LIVE
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=True,
        metaculus_client=client,
        llms={"default": llm, "summarizer": llm, "researcher": llm, "parser": llm},
    )
    t0 = time.time()
    [report] = asyncio.run(bot.forecast_questions([q], return_exceptions=True))
    row = {"post": POST_ID, "question": q.id_of_question, "seconds": round(time.time() - t0, 1),
           "metaculus_requests": REQUEST_LOG}
    if isinstance(report, BaseException):
        row["error"] = f"{type(report).__name__}: {str(report)[:800]}"
    else:
        row["prediction"] = report.prediction
        row["price_estimate_usd"] = report.price_estimate
        row["minor_errors"] = report.errors
        row["explanation_chars"] = len(report.explanation)
    print("RESULT " + json.dumps(row, default=str), flush=True)
    if "error" in row:
        sys.exit(1)


if __name__ == "__main__":
    main()
