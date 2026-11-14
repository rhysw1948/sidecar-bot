import argparse
import asyncio
import contextvars
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    MonetaryCostManager,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL SETTINGS (Sidecar-bot) -- change the models here, in this one place.
# POSTED_MODEL: the forecast that is POSTED to Metaculus (Build 12: Gemini 3.7
#   Flash). Used for ALL roles of the posted arm: default (forecasting),
#   summarizer, researcher and parser.
# LAB_MODEL: the lab arms (lab-plain, shadow-news, shadow-sceptic). They are
#   never posted; they only go into the encrypted forecast log.
# Research in the posted and lab-plain arms is a plain LLM call, NO web search.
# Backup posted model if Gemini 3.7 Flash has problems:
#   "openrouter/google/gemini-3.8-flash" (about +30% cost, tested 13 Nov).
# =============================================================================
POSTED_MODEL = "openrouter/google/gemini-3.7-flash"
LAB_MODEL = "openrouter/openai/gpt-oss-120b"
MODEL_NAME = POSTED_MODEL  # old name, kept so older test scripts still work
POSTED_ALLOWED_TRIES = 4  # Build 12: was 2. Each try also has its own 429 backoff (see OPENROUTER CALLS).
LAB_ALLOWED_TRIES = 2


def build_llms(model: str | None = None, allowed_tries: int | None = None) -> dict:
    llm = GeneralLlm(
        model=model or POSTED_MODEL,
        temperature=0.3,
        timeout=120,
        allowed_tries=allowed_tries or POSTED_ALLOWED_TRIES,
    )
    RESEARCHER = llm  # plain LLM, no web search
    return {
        "default": llm,
        "summarizer": llm,
        "researcher": RESEARCHER,
        "parser": llm,
    }


def build_lab_llms() -> dict:
    return build_llms(LAB_MODEL, LAB_ALLOWED_TRIES)
# =============================================================================

# =============================================================================
# FORECAST LOG (Sidecar-bot). One JSON line per question per run, written to
# forecast_log.jsonl. It keeps the raw individual forecasts BEFORE any future
# adjustment, so resolved questions can be replayed later. No reasoning text.
# The GitHub workflows encrypt this file and upload only the encrypted copy.
# Logging must never break a run: every step is wrapped in try/except.
# =============================================================================
PROMPT_VERSION = "v2-gemini37flash"  # prompts unchanged since v1-build6; the posted model changed
FORECAST_LOG_PATH = os.environ.get("FORECAST_LOG_PATH", "forecast_log.jsonl")
RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds")
RUN_MODE = "unknown"  # set in __main__


def _iso(value) -> str | None:
    if value is None:
        return None
    try:
        return value.astimezone(timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return str(value)


def _prediction_to_json(pred):
    """Plain JSON form of one prediction: a probability (binary), option
    probabilities (multiple choice), or declared percentiles (numeric,
    discrete, date; date values are Unix timestamps)."""
    if isinstance(pred, (int, float)):
        return float(pred)
    if isinstance(pred, PredictedOptionList):
        return {o.option_name: o.probability for o in pred.predicted_options}
    if isinstance(pred, NumericDistribution):
        return [[p.percentile, p.value] for p in pred.declared_percentiles]
    if hasattr(pred, "model_dump"):
        return pred.model_dump(mode="json")
    return str(pred)


def _final_to_json(pred):
    """The aggregated forecast in the form that is sent. For numeric/date
    questions this is the CDF (x values and cumulative probabilities)."""
    if isinstance(pred, NumericDistribution):
        cdf = pred.get_cdf()
        return {
            "cdf_x": [p.value for p in cdf],
            "cdf": [p.percentile for p in cdf],
        }
    return _prediction_to_json(pred)


def _write_forecast_log_line(record: dict) -> None:
    with open(FORECAST_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
# =============================================================================

# =============================================================================
# OPENROUTER CALLS (Sidecar-bot, Build 12). Every LLM call of every arm goes
# through _sidecar_acompletion (it wraps forecasting_tools' litellm call).
# 1. Real cost: OpenRouter returns usage.cost on each answer; it is added to
#    the arm's bucket (logged per question) and to the run total.
# 2. Rate limit: tested 14 Nov, OpenRouter refuses more than 20 requests per
#    minute PER MODEL for this (new) account with HTTP 429 (seen for Gemini
#    3.7 Flash; 25 parallel gpt-oss-120b calls were all accepted). Calls to
#    POSTED_MODEL are throttled to POSTED_MODEL_MAX_CALLS_PER_MINUTE, and any
#    429 is waited out and retried (RATE_LIMIT_EXTRA_TRIES times, growing
#    waits) BEFORE the normal allowed_tries retry is used up.
# 3. Cost guard (lab arms only, posting never stops): if this run's real
#    spend passes LAB_RUN_SPEND_CAP_USD, or the OpenRouter key's remaining
#    credit (checked when the lab phase starts, then reduced by this run's
#    spend) drops below LAB_MIN_KEY_REMAINING_USD, the lab arms are skipped
#    for the rest of the run.
# 4. Lab model: inside a lab arm, get_llm() returns LAB_MODEL instead of
#    POSTED_MODEL (a context variable, so the posted arm never sees it).
# =============================================================================
POSTED_MODEL_MAX_CALLS_PER_MINUTE = 18  # OpenRouter limit is 20; keep a margin
RATE_LIMIT_EXTRA_TRIES = 4
RATE_LIMIT_BASE_WAIT_S = 15.0  # waits 15, 30, 45, 60 s (+ up to a third of 15 s, random)
LAB_RUN_SPEND_CAP_USD = 8.0
LAB_MIN_KEY_REMAINING_USD = 2.0

_ARM: contextvars.ContextVar = contextvars.ContextVar("sidecar_arm", default="posted")
_CALL_BUCKET: contextvars.ContextVar = contextvars.ContextVar("sidecar_call_bucket", default=None)
_LAB_LLMS: contextvars.ContextVar = contextvars.ContextVar("sidecar_lab_llms", default=None)
_RUN_COST = {"total": 0.0, "by_arm": {}}
_LAB_GUARD = {"reason": None, "key_remaining": None, "total_at_key_check": 0.0, "logged": False}


class LabCostGuardStop(RuntimeError):
    pass


def _new_bucket() -> dict:
    return {"cost": 0.0, "calls": 0, "cost_missing_calls": 0, "rate_limit_429s": 0}


def _count(arm: str, field: str, value=1) -> None:
    for b in (_CALL_BUCKET.get(), _RUN_COST["by_arm"].setdefault(arm, _new_bucket())):
        if b is not None:
            b[field] += value


def _is_rate_limit_error(error: BaseException) -> bool:
    if type(error).__name__ == "RateLimitError":
        return True
    if getattr(error, "status_code", None) == 429:
        return True
    text = str(error).lower()
    return "429" in text and "rate limit" in text


def _model_key(model) -> str:
    return str(model or "").strip().lower().removeprefix("openrouter/")


def _lab_guard_trip(reason: str) -> None:
    if _LAB_GUARD["reason"] is None:
        _LAB_GUARD["reason"] = reason
        logger.warning(f"LAB COST GUARD: {reason}. Lab arms skipped for the rest of this run; posting is not affected.")


def _lab_guard_check() -> None:
    total = _RUN_COST["total"]
    if total > LAB_RUN_SPEND_CAP_USD:
        _lab_guard_trip(f"this run's OpenRouter spend ${total:.2f} passed the ${LAB_RUN_SPEND_CAP_USD:.2f} cap")
    if _LAB_GUARD["key_remaining"] is not None:
        left = _LAB_GUARD["key_remaining"] - (total - _LAB_GUARD["total_at_key_check"])
        if left < LAB_MIN_KEY_REMAINING_USD:
            _lab_guard_trip(f"OpenRouter credit left is about ${left:.2f}, under ${LAB_MIN_KEY_REMAINING_USD:.2f}")


def _fetch_openrouter_remaining() -> float | None:
    """Smallest of: the key's limit_remaining (GET /api/v1/key) and the
    account's total_credits - total_usage (GET /api/v1/credits). None if both
    fail. The key itself is never logged."""
    import requests

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    values = []
    headers = {"Authorization": f"Bearer {key}"}
    try:
        d = requests.get("https://openrouter.ai/api/v1/key", headers=headers, timeout=20).json()["data"]
        if d.get("limit_remaining") is not None:
            values.append(float(d["limit_remaining"]))
    except Exception:
        pass
    try:
        d = requests.get("https://openrouter.ai/api/v1/credits", headers=headers, timeout=20).json()["data"]
        values.append(float(d["total_credits"]) - float(d["total_usage"]))
    except Exception:
        pass
    return min(values) if values else None


class _RollingLimiter:
    """At most `n` calls start in any 60 s window (per event loop)."""

    def __init__(self, n: int, window: float = 60.0):
        self.n, self.window = n, window
        self.times: list[float] = []
        self._lock = None
        self._loop = None

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:  # new event loop (e.g. a second asyncio.run)
            self._loop, self._lock = loop, asyncio.Lock()
        async with self._lock:
            while True:
                now = time.monotonic()
                self.times = [t for t in self.times if now - t < self.window]
                if len(self.times) < self.n:
                    self.times.append(now)
                    return
                await asyncio.sleep(self.times[0] + self.window - now + 0.05)


_POSTED_LIMITER = _RollingLimiter(POSTED_MODEL_MAX_CALLS_PER_MINUTE)


def _install_openrouter_wrapper() -> None:
    import random

    import forecasting_tools.ai_models.general_llm as general_llm_module

    if getattr(general_llm_module.acompletion, "_sidecar_wrapper", False):
        return
    original = general_llm_module.acompletion

    async def _sidecar_acompletion(*args, **kwargs):
        arm = _ARM.get()
        is_lab = arm != "posted"
        throttled = _model_key(kwargs.get("model")) == _model_key(POSTED_MODEL)
        rate_limit_tries = 0
        while True:
            if is_lab and _LAB_GUARD["reason"]:
                raise LabCostGuardStop(f"lab arm stopped by cost guard: {_LAB_GUARD['reason']}")
            if throttled:
                await _POSTED_LIMITER.acquire()
            try:
                response = await original(*args, **kwargs)
            except Exception as e:
                if not _is_rate_limit_error(e):
                    raise
                _count(arm, "rate_limit_429s")
                if rate_limit_tries >= RATE_LIMIT_EXTRA_TRIES:
                    raise
                rate_limit_tries += 1
                wait = RATE_LIMIT_BASE_WAIT_S * rate_limit_tries + random.uniform(0, RATE_LIMIT_BASE_WAIT_S / 3)
                logger.warning(
                    f"OpenRouter 429 rate limit ({arm} arm, {kwargs.get('model')}); "
                    f"waiting {wait:.0f} s, then retry {rate_limit_tries} of {RATE_LIMIT_EXTRA_TRIES}."
                )
                await asyncio.sleep(wait)
                continue
            try:
                cost = getattr(getattr(response, "usage", None), "cost", None)
                _count(arm, "calls")
                if isinstance(cost, (int, float)):
                    _count(arm, "cost", float(cost))
                    _RUN_COST["total"] += float(cost)
                else:
                    _count(arm, "cost_missing_calls")
                _lab_guard_check()
            except Exception:
                pass  # cost tracking must never break a call
            return response

    _sidecar_acompletion._sidecar_wrapper = True
    general_llm_module.acompletion = _sidecar_acompletion


_install_openrouter_wrapper()


def _bucket_fields(bucket: dict | None) -> dict:
    b = bucket or _new_bucket()
    return {
        "openrouter_cost_usd": round(b["cost"], 6),
        "openrouter_calls": b["calls"],
        "openrouter_cost_missing_calls": b["cost_missing_calls"],
        "rate_limit_429s": b["rate_limit_429s"],
    }


def run_cost_summary() -> str:
    parts = []
    for arm, b in _RUN_COST["by_arm"].items():
        parts.append(f"{arm} ${b['cost']:.4f} ({b['calls']} calls, {b['rate_limit_429s']} x 429)")
    guard = f" Lab cost guard: {_LAB_GUARD['reason']}." if _LAB_GUARD["reason"] else ""
    return f"OpenRouter real cost this run: total ${_RUN_COST['total']:.4f}. " + "; ".join(parts) + "." + guard
# =============================================================================

# =============================================================================
# SHADOW NEWS ARM (Sidecar-bot, Build 10). LOG ONLY, NEVER POSTED.
# After ALL normal forecasts of a run have been posted, each question gets a
# second "with-news" forecast: one LLM call writes a news search query, one
# AskNews "latest news" search (1 credit), then the same model and prompts
# (5 predictions) with the news text as research. The result goes ONLY into
# forecast_log.jsonl (arm "shadow-news"). Nothing is posted, no comment.
# Any shadow failure logs a warning and carries on; it never changes the
# normal forecast, the posting or the exit code.
# Switch off with env SHADOW_NEWS=off (this also stops the sceptic arm,
# which needs its news). Needs env ASKNEWS_API_KEY.
# =============================================================================
SHADOW_TAG = "shadow-news-v1"
SHADOW_CREDIT_CAP_PER_RUN = 80  # counted from the API's usage.credits field
SHADOW_ASKNEWS_GAP_SECONDS = 2.5  # free plan: 1 request per 2 s
SHADOW_ARTICLES = 6  # "latest news" search, same size as AskNews test 1
# Build 12: 2 -> 4. The lab arms run on gpt-oss (25 parallel calls were all
# accepted, 14 Nov) and the 14 Nov dry run took about 2.3 min per question at
# 2 at a time, too slow for 60 questions in the budget. AskNews calls stay one
# at a time, 2.5 s apart (lock below).
SHADOW_PARALLEL_QUESTIONS = 4
SHADOW_PER_QUESTION_TIMEOUT_S = 300  # per arm, per question
# Build 12: 1800 -> 3600. Three lab arms now (lab-plain added). Build 11 took
# about 21 s per question for news + sceptic; lab-plain adds about 12 s, so
# about 33 min for 60 questions, which would not fit in 30 min.
SHADOW_PHASE_BUDGET_S = 3600  # no new lab arm starts after this many seconds

# LAB-PLAIN ARM (Build 12). LOG ONLY, NEVER POSTED. The old posted setup
# (plain LLM research, no news, same prompts, 5 predictions) on LAB_MODEL, so
# LAB_MODEL vs POSTED_MODEL can be compared on the same questions.
# Switch off with env LAB_PLAIN=off. The shadow POST guard covers it.
LAB_PLAIN_TAG = "lab-plain-v1"

_SHADOW_RUN = {
    "credits_used": 0,
    "disabled_reason": None,  # set once (no key, credits exhausted...) -> skip rest of run
    "warned": set(),
    "last_asknews_call": 0.0,
}
_SHADOW_ACTIVE = {"depth": 0}  # > 0 while shadow code runs: Metaculus writes are refused


class ShadowPostBlocked(RuntimeError):
    pass


def _install_shadow_post_guard() -> None:
    """Wrap requests.Session.request (which requests.get/post also use, and
    which MetaculusClient uses for every call). While the shadow runs, any
    non-GET request to metaculus.com raises ShadowPostBlocked BEFORE sending."""
    import requests

    if getattr(requests.sessions.Session.request, "_sidecar_shadow_guard", False):
        return
    original = requests.sessions.Session.request

    def guarded(self, method, url, *args, **kwargs):
        if (
            _SHADOW_ACTIVE["depth"] > 0
            and "metaculus.com" in str(url).lower()
            and str(method).upper() not in ("GET", "HEAD", "OPTIONS")
        ):
            raise ShadowPostBlocked(
                f"SHADOW GUARD: refused {str(method).upper()} to Metaculus inside the shadow arm"
            )
        return original(self, method, url, *args, **kwargs)

    guarded._sidecar_shadow_guard = True
    requests.sessions.Session.request = guarded


_install_shadow_post_guard()


def _shadow_warn_once(key: str, message: str) -> None:
    if key in _SHADOW_RUN["warned"]:
        return
    _SHADOW_RUN["warned"].add(key)
    logger.warning(f"Lab arms: {message}")


def _is_credit_exhausted_error(error: BaseException) -> bool:
    """[INFERRED, not tested at 0 credits] AskNews out-of-credit / bad-key
    answers: HTTP 401/402/403, or a message about credits/balance/payment."""
    code = getattr(error, "code", None)
    if isinstance(code, int) and code // 1000 in (401, 402, 403):
        return True
    text = f"{getattr(error, 'detail', '')} {error}".lower()
    return any(w in text for w in ("credit", "balance", "insufficient", "payment", "quota", "wallet"))
# =============================================================================

# =============================================================================
# SHADOW SCEPTIC ARM (Sidecar-bot, Build 11). LOG ONLY, NEVER POSTED.
# Runs right after the shadow news arm for the same question and reuses the
# SAME news text (no extra AskNews search, 0 extra credits). Same model and
# 5 predictions; the only difference is a scepticism block added to the
# forecast prompt, just before the final-answer instructions. The block text
# lives in prompts/sceptic_v1.txt. If the text changes, bump SCEPTIC_TAG.
# If the news arm has no news text, this arm is skipped and the reason logged.
# Switch off with env SHADOW_SCEPTIC=off. The shadow POST guard covers it.
# =============================================================================
SCEPTIC_TAG = "shadow-sceptic-v1"
SCEPTIC_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "sceptic_v1.txt")
_FINAL_ANSWER_MARKER = "The last thing you write is"
# Set only inside the sceptic arm's own asyncio tasks (context variables are
# copied per task), so the posted and news arms never see the block.
_SCEPTIC_SECTION: contextvars.ContextVar = contextvars.ContextVar("sidecar_sceptic_section", default=None)


def _load_sceptic_block(path: str | None = None) -> dict:
    """Read the three parts of the block file: 'all', 'numeric', 'mc'."""
    with open(path or SCEPTIC_PROMPT_FILE, encoding="utf-8") as f:
        text = f.read()

    def between(start: str, end: str) -> str:
        i = text.index(start) + len(start)
        j = text.index(end, i)
        part = text[i:j].strip()
        if not part:
            raise ValueError(f"empty part after {start!r}")
        return part

    return {
        "all": between("--- start block (all question types) ---", "--- end block ---"),
        "numeric": between("--- extra lines for numeric, discrete and date questions only ---", "--- end ---"),
        "mc": between("--- extra lines for multiple-choice questions only ---", "--- end ---"),
    }


def _sceptic_section_for(question, block: dict) -> str:
    text = block["all"]
    if isinstance(question, (NumericQuestion, DateQuestion)):  # DiscreteQuestion is a NumericQuestion
        text += "\n" + block["numeric"]
    elif isinstance(question, MultipleChoiceQuestion):
        text += "\n" + block["mc"]
    return text


def _insert_sceptic_section(prompt: str, section: str) -> str:
    """Put the block as its own section just before the LAST final-answer
    instruction. Everything else in the prompt stays exactly as it was."""
    i = prompt.rfind(_FINAL_ANSWER_MARKER)
    if i < 0:
        raise ValueError("sceptic arm: final-answer instruction not found in prompt")
    return prompt[:i] + section.strip() + "\n\n" + prompt[i:]
# =============================================================================


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Sidecar-bot: the template said OpenRouter has large rate limits on new
    accounts. That is WRONG for this account: tested 13-14 Nov, OpenRouter
    refuses more than 20 requests per minute per model (HTTP 429, "new
    accounts are limited to 20 requests per minute for this model"). See
    OPENROUTER CALLS near the top of this file for the throttle and retries.
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def get_llm(self, purpose: str = "default", guarantee_type=None):
        """Build 12: inside a lab arm (context variable set), return the
        LAB_MODEL llm for this purpose; everywhere else the normal (posted) llm."""
        lab = _LAB_LLMS.get()
        if lab is None:
            return super().get_llm(purpose, guarantee_type)
        if purpose not in lab:
            raise ValueError(f"lab arm: unknown llm purpose {purpose!r}")
        llm = lab[purpose]
        return llm.model if guarantee_type == "string_name" else llm

    ##################################### SIDECAR-BOT FORECAST LOG #####################################

    def _fl_state(self, question: MetaculusQuestion) -> dict:
        """Per-question scratch data for the forecast log (keyed by object id,
        because several questions can run at the same time)."""
        if not hasattr(self, "_fl_states"):
            self._fl_states = {}
        return self._fl_states.setdefault(
            id(question),
            {"individual": [], "prediction_errors": [], "range_check_retries": 0},
        )

    async def _research_and_make_predictions(self, question: MetaculusQuestion):
        result = await super()._research_and_make_predictions(question)
        try:
            state = self._fl_state(question)
            state["individual"].extend(
                _prediction_to_json(p.prediction_value) for p in result.predictions
            )
            state["prediction_errors"].extend(str(e)[:500] for e in result.errors)
        except Exception as e:
            logger.warning(f"Forecast log: could not record predictions ({type(e).__name__}: {str(e)[:200]})")
        return result

    async def _run_individual_question(self, question: MetaculusQuestion):
        report = None
        error: BaseException | None = None
        cost = None
        bucket = _new_bucket()  # real OpenRouter cost of this question (posted arm)
        arm_token = _ARM.set("posted")
        bucket_token = _CALL_BUCKET.set(bucket)
        try:
            with MonetaryCostManager() as cost_manager:
                try:
                    report = await super()._run_individual_question(question)
                finally:
                    cost = cost_manager.current_usage
            return report
        except BaseException as e:
            error = e
            raise
        finally:
            for var, token in ((_CALL_BUCKET, bucket_token), (_ARM, arm_token)):
                try:
                    var.reset(token)
                except Exception:
                    pass
            self._fl_write_record(question, report, error, cost, bucket)
            try:  # remember the question for the shadow arm (runs after ALL posting)
                if not hasattr(self, "_shadow_queue"):
                    self._shadow_queue = []
                self._shadow_queue.append((question, report is not None and error is None))
            except Exception:
                pass

    def _fl_write_record(self, question, report, error, cost, bucket=None) -> None:
        try:
            state = self._fl_states.pop(id(question), None) if hasattr(self, "_fl_states") else None
            state = state or {"individual": [], "prediction_errors": [], "range_check_retries": 0}
            errors = list(state["prediction_errors"])
            if report is not None:
                errors += [str(e)[:500] for e in (report.errors or []) if str(e)[:500] not in errors]
            if error is not None:
                errors.append(f"{type(error).__name__}: {str(error)[:500]}")
            record = {
                "arm": "posted",
                "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "run_time_utc": RUN_STARTED_UTC,
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "mode": RUN_MODE,
                "model": POSTED_MODEL,
                "prompt_version": PROMPT_VERSION,
                "post_id": question.id_of_post,
                "question_id": question.id_of_question,
                "question_type": type(question).__name__,
                "group_option": question.group_question_option,
                "title": question.question_text,
                "page_url": question.page_url,
                "open_time": _iso(question.open_time),
                "scheduled_close_time": _iso(question.close_time),
                "scheduled_resolution_time": _iso(getattr(question, "scheduled_resolution_time", None)),
                **_bucket_fields(bucket),
                "individual_predictions": state["individual"],
                "n_individual": len(state["individual"]),
                "final_forecast": _final_to_json(report.prediction) if report is not None else None,
                "publish_enabled": bool(self.publish_reports_to_metaculus),
                "succeeded": report is not None and error is None,
                "range_check_retries": state["range_check_retries"],
                "errors": errors,
                "cost_usd_estimate": cost,
            }
            _write_forecast_log_line(record)
        except Exception as e:
            logger.warning(f"Forecast log: could not write a line ({type(e).__name__}: {str(e)[:200]}). Carrying on.")

    ##################################### SIDECAR-BOT SHADOW NEWS ARM #####################################

    async def forecast_questions(self, questions, return_exceptions: bool = False):
        # Normal forecasting AND posting for every question happens first,
        # inside super(). The shadow arm only starts after that has returned.
        self._shadow_queue = []
        reports = await super().forecast_questions(questions, return_exceptions)
        queue, self._shadow_queue = self._shadow_queue, []
        try:
            await self._run_shadow_phase(queue)
        except Exception as e:  # never let the shadow change the run
            _shadow_warn_once("phase", f"shadow phase failed ({type(e).__name__}: {str(e)[:200]}). Carrying on.")
        try:
            if queue:
                logger.info(run_cost_summary())
        except Exception:
            pass
        return reports

    async def _run_lab_task(self, arm: str, fn, bucket: dict, timeout: float = SHADOW_PER_QUESTION_TIMEOUT_S):
        """Run one lab arm for one question in its OWN asyncio task, with the
        lab model and the arm's cost bucket set only inside that task."""
        if not hasattr(self, "_lab_llms_cache"):
            self._lab_llms_cache = build_lab_llms()
        lab_llms = self._lab_llms_cache

        async def inner():
            _ARM.set(arm)
            _CALL_BUCKET.set(bucket)
            _LAB_LLMS.set(lab_llms)
            return await fn()

        return await asyncio.wait_for(asyncio.create_task(inner()), timeout=timeout)

    async def _run_shadow_phase(self, queue: list) -> None:
        if not queue:
            return
        try:  # Build 12 cost guard: remaining OpenRouter credit, once, at the start of the lab phase
            remaining = await asyncio.to_thread(_fetch_openrouter_remaining)
            _LAB_GUARD["key_remaining"] = remaining
            _LAB_GUARD["total_at_key_check"] = _RUN_COST["total"]
            if remaining is None:
                logger.warning("LAB COST GUARD: could not read OpenRouter credit; only the per-run spend cap applies.")
            else:
                logger.info(f"LAB COST GUARD: OpenRouter credit left at lab start about ${remaining:.2f}.")
            _lab_guard_check()
        except Exception as e:
            logger.warning(f"LAB COST GUARD: credit check failed ({type(e).__name__}). Only the per-run spend cap applies.")
        if os.environ.get("SHADOW_NEWS", "on").strip().lower() in ("off", "0", "false", "no"):
            _SHADOW_RUN["disabled_reason"] = _SHADOW_RUN["disabled_reason"] or "switched off (SHADOW_NEWS=off)"
        elif not os.environ.get("ASKNEWS_API_KEY"):
            _SHADOW_RUN["disabled_reason"] = _SHADOW_RUN["disabled_reason"] or "no ASKNEWS_API_KEY"
        # Created here (not at class level) so they belong to this event loop.
        self._shadow_asknews_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(SHADOW_PARALLEL_QUESTIONS)
        deadline = asyncio.get_running_loop().time() + SHADOW_PHASE_BUDGET_S
        _SHADOW_ACTIVE["depth"] += 1
        old_level = logger.level
        logger.setLevel(logging.WARNING)  # keep shadow reasoning out of the public Actions log
        try:
            async def one(question, posted_ok):
                async with semaphore:
                    await self._shadow_one_question_safely(question, posted_ok, deadline)

            await asyncio.gather(*(one(q, ok) for q, ok in queue), return_exceptions=True)
        finally:
            logger.setLevel(old_level)
            _SHADOW_ACTIVE["depth"] -= 1
        logger.info(
            f"Lab arms (lab-plain, shadow-news, shadow-sceptic): done for {len(queue)} question(s). AskNews credits used this run: {_SHADOW_RUN['credits_used']}."
        )

    async def _lab_plain_one_question_safely(self, question, posted_ok: bool, deadline: float) -> None:
        """Build 12 lab-plain arm: plain LLM research + 5 predictions on LAB_MODEL. Never posted."""
        result = {"individual": [], "final": None, "errors": [], "retries": 0, "cost": None}
        bucket = _new_bucket()
        skipped = None
        lq = None
        try:
            if os.environ.get("LAB_PLAIN", "on").strip().lower() in ("off", "0", "false", "no"):
                skipped = "skipped: switched off (LAB_PLAIN=off)"
            elif not posted_ok:
                skipped = "skipped: the posted arm failed for this question"
            elif _LAB_GUARD["reason"]:
                skipped = f"skipped: lab cost guard ({_LAB_GUARD['reason']})"
            elif asyncio.get_running_loop().time() > deadline:
                skipped = "skipped: shadow phase time budget used up"
                _shadow_warn_once("lab-plain-deadline", "lab-plain: time budget used up; remaining lab-plain forecasts skipped.")
            else:
                lq = question.model_copy(deep=True)  # own copy: own retry count, never touches posted state

                async def run():
                    research = await self.run_research(lq)
                    await self._shadow_predict(lq, research, result)

                with MonetaryCostManager() as cm:
                    try:
                        await self._run_lab_task("lab-plain", run, bucket)
                    finally:
                        result["cost"] = cm.current_usage
        except Exception as e:
            result["errors"].append(f"{type(e).__name__}: {str(e)[:500]}")
            _shadow_warn_once(
                f"lab-plain-err-{id(question)}",
                f"lab-plain failed for {question.page_url} ({type(e).__name__}: {str(e)[:200]}). Carrying on.",
            )
        finally:
            try:
                if lq is not None and hasattr(self, "_fl_states"):
                    st = self._fl_states.pop(id(lq), None)
                    if st:
                        result["retries"] = st.get("range_check_retries", 0)
                if skipped:
                    result["errors"].append(skipped)
                self._shadow_write_record(
                    question, {}, result, succeeded=(not skipped and result["final"] is not None),
                    arm="lab-plain", tag=LAB_PLAIN_TAG, bucket=bucket,
                )
            except Exception as e:
                _shadow_warn_once("lab-plain-log", f"lab-plain: could not write a log line ({type(e).__name__}: {str(e)[:200]}).")

    async def _shadow_one_question_safely(self, question, posted_ok: bool, deadline: float) -> None:
        # Build 12: lab-plain first (own try/except inside; it never raises).
        await self._lab_plain_one_question_safely(question, posted_ok, deadline)
        info = {
            "search_query": None, "article_count": None, "newest_article_utc": None,
            "oldest_article_utc": None, "asknews_credits": 0,
        }
        result = {"individual": [], "final": None, "errors": [], "retries": 0, "cost": None, "news": None}
        bucket = _new_bucket()
        skipped = None
        shadow_q = None
        try:
            if not posted_ok:
                skipped = "skipped: the posted arm failed for this question"
            elif _LAB_GUARD["reason"]:
                skipped = f"skipped: lab cost guard ({_LAB_GUARD['reason']})"
            elif _SHADOW_RUN["disabled_reason"]:
                skipped = f"skipped: {_SHADOW_RUN['disabled_reason']}"
                _shadow_warn_once("disabled", f"{_SHADOW_RUN['disabled_reason']}; shadow skipped for the rest of this run.")
            elif asyncio.get_running_loop().time() > deadline:
                skipped = "skipped: shadow phase time budget used up"
                _shadow_warn_once("deadline", "time budget used up; remaining shadows skipped.")
            else:
                # A copy, so retry counts etc. never touch the posted question's state.
                shadow_q = question.model_copy(deep=True)
                with MonetaryCostManager() as cm:
                    try:
                        await self._run_lab_task(
                            "shadow-news", lambda: self._shadow_forecast(shadow_q, info, result), bucket
                        )
                    finally:
                        result["cost"] = cm.current_usage
        except Exception as e:
            result["errors"].append(f"{type(e).__name__}: {str(e)[:500]}")
            _shadow_warn_once(
                f"err-{id(question)}",
                f"failed for {question.page_url} ({type(e).__name__}: {str(e)[:200]}). Carrying on.",
            )
        finally:
            try:
                if shadow_q is not None and hasattr(self, "_fl_states"):
                    st = self._fl_states.pop(id(shadow_q), None)
                    if st:
                        result["retries"] = st.get("range_check_retries", 0)
                if skipped:
                    result["errors"].append(skipped)
                self._shadow_write_record(
                    question, info, result, succeeded=(not skipped and result["final"] is not None), bucket=bucket
                )
            except Exception as e:
                _shadow_warn_once("log", f"could not write a shadow log line ({type(e).__name__}: {str(e)[:200]}).")
        # Build 11: the sceptic arm, same news text, after the news arm's line is written.
        await self._sceptic_one_question_safely(question, info, result.get("news"), deadline)

    async def _sceptic_one_question_safely(self, question, news_info: dict, news: str | None, deadline: float) -> None:
        info = dict(news_info)  # same search query and article info as shadow-news
        info["asknews_credits"] = 0  # no extra search
        info["asknews_credits_reported"] = None
        result = {"individual": [], "final": None, "errors": [], "retries": 0, "cost": None}
        bucket = _new_bucket()
        skipped = None
        sq = None
        try:
            if os.environ.get("SHADOW_SCEPTIC", "on").strip().lower() in ("off", "0", "false", "no"):
                skipped = "skipped: switched off (SHADOW_SCEPTIC=off)"
            elif _LAB_GUARD["reason"]:
                skipped = f"skipped: lab cost guard ({_LAB_GUARD['reason']})"
            elif news is None:
                skipped = "skipped: shadow-news has no news text for this question"
            elif asyncio.get_running_loop().time() > deadline:
                skipped = "skipped: shadow phase time budget used up"
                _shadow_warn_once("sceptic-deadline", "shadow-sceptic: time budget used up; remaining sceptic forecasts skipped.")
            else:
                if not hasattr(self, "_sceptic_block"):
                    self._sceptic_block = _load_sceptic_block()
                section = _sceptic_section_for(question, self._sceptic_block)
                sq = question.model_copy(deep=True)  # own copy: own retry count, never touches posted state

                async def run():
                    _SCEPTIC_SECTION.set(section)  # only this task and the tasks it starts
                    await self._shadow_predict(sq, news, result)

                with MonetaryCostManager() as cm:
                    try:
                        await self._run_lab_task("shadow-sceptic", run, bucket)
                    finally:
                        result["cost"] = cm.current_usage
        except Exception as e:
            result["errors"].append(f"{type(e).__name__}: {str(e)[:500]}")
            _shadow_warn_once(
                f"sceptic-err-{id(question)}",
                f"shadow-sceptic failed for {question.page_url} ({type(e).__name__}: {str(e)[:200]}). Carrying on.",
            )
        finally:
            try:
                if sq is not None and hasattr(self, "_fl_states"):
                    st = self._fl_states.pop(id(sq), None)
                    if st:
                        result["retries"] = st.get("range_check_retries", 0)
                if skipped:
                    result["errors"].append(skipped)
                self._shadow_write_record(
                    question, info, result, succeeded=(not skipped and result["final"] is not None),
                    arm="shadow-sceptic", tag=SCEPTIC_TAG, bucket=bucket,
                )
            except Exception as e:
                _shadow_warn_once("sceptic-log", f"shadow-sceptic: could not write a log line ({type(e).__name__}: {str(e)[:200]}).")

    def _apply_sceptic_section(self, question, prompt: str) -> str:
        """No-op for the posted and news arms (prompt returned unchanged)."""
        section = _SCEPTIC_SECTION.get()
        if not section:
            return prompt
        return _insert_sceptic_section(prompt, section)

    async def _shadow_forecast(self, question, info: dict, result: dict) -> None:
        query = await self._shadow_write_query(question)
        info["search_query"] = query
        news = await self._shadow_search_news(query, info)
        if news is None:
            raise RuntimeError("no news (search skipped or failed)")
        result["news"] = news  # kept in memory for the sceptic arm; never logged
        await self._shadow_predict(question, news, result)

    async def _shadow_predict(self, question, news: str, result: dict) -> None:
        forecast_fn = self._shadow_forecast_function(question)
        outcomes = await asyncio.gather(
            *(forecast_fn(question, news) for _ in range(self.predictions_per_research_report)),
            return_exceptions=True,
        )
        preds = [o.prediction_value for o in outcomes if not isinstance(o, BaseException)]
        result["errors"] += [f"{type(o).__name__}: {str(o)[:500]}" for o in outcomes if isinstance(o, BaseException)]
        result["individual"] = [_prediction_to_json(p) for p in preds]
        # Same rule as the normal arm: at least half of the predictions must succeed.
        needed = self.predictions_per_research_report * self.required_successful_predictions
        if not preds or len(preds) < needed:
            raise RuntimeError(f"only {len(preds)} of {self.predictions_per_research_report} shadow predictions succeeded")
        aggregated = await self._aggregate_predictions(preds, question)
        result["final"] = _final_to_json(aggregated)

    def _shadow_forecast_function(self, question):
        # Same order as forecasting_tools' _make_prediction. Conditional questions are not supported.
        if isinstance(question, BinaryQuestion):
            return self._run_forecast_on_binary
        if isinstance(question, MultipleChoiceQuestion):
            return self._run_forecast_on_multiple_choice
        if isinstance(question, NumericQuestion):
            return self._run_forecast_on_numeric
        if isinstance(question, DateQuestion):
            return self._run_forecast_on_date
        raise ValueError(f"shadow arm does not support {type(question).__name__}")

    async def _shadow_write_query(self, question) -> str:
        prompt = clean_indents(
            f"""
            Write ONE short news search query (3 to 10 words) that would find recent news articles
            most useful for forecasting the question below. Use the key names, places, organisations
            and topic words. No quotes, no operators, no explanation. Reply with the query only.

            Question: {question.question_text}

            Resolution criteria: {str(question.resolution_criteria)[:1500]}

            The question closes on {self._format_time(question.close_time)}.
            """
        )
        raw = await self.get_llm("default", "llm").invoke(prompt)
        lines = [ln.strip() for ln in str(raw).strip().splitlines() if ln.strip()]
        query = (lines[-1] if lines else "").strip().strip('"').strip("'").strip()
        if query.lower().startswith("query:"):
            query = query[6:].strip()
        if not query:
            query = question.question_text  # fall back to the title
        return query[:200]

    async def _shadow_search_news(self, query: str, info: dict) -> str | None:
        """One AskNews 'latest news' search (never the archive search).
        Only one AskNews call at a time, at least 2.5 s apart."""
        from asknews_sdk import AsyncAskNewsSDK

        async with self._shadow_asknews_lock:
            if _SHADOW_RUN["disabled_reason"]:
                return None
            if _SHADOW_RUN["credits_used"] + 1 > SHADOW_CREDIT_CAP_PER_RUN:
                _SHADOW_RUN["disabled_reason"] = f"per-run AskNews cap of {SHADOW_CREDIT_CAP_PER_RUN} credits reached"
                _shadow_warn_once("cap", f"{_SHADOW_RUN['disabled_reason']}; shadow skipped for the rest of this run.")
                return None
            wait = SHADOW_ASKNEWS_GAP_SECONDS - (time.time() - _SHADOW_RUN["last_asknews_call"])
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async with AsyncAskNewsSDK(
                    api_key=os.environ["ASKNEWS_API_KEY"], scopes={"news"}, timeout=60, retries=1
                ) as ask:
                    resp = await ask.news.search_news(
                        query=query,
                        n_articles=SHADOW_ARTICLES,
                        return_type="both",
                        strategy="latest news",
                    )
            except Exception as e:
                if _is_credit_exhausted_error(e):
                    _SHADOW_RUN["disabled_reason"] = f"AskNews refused ({str(e)[:150]}), probably out of credits"
                    _shadow_warn_once("credits", f"{_SHADOW_RUN['disabled_reason']}; shadow skipped for the rest of this run.")
                raise
            finally:
                _SHADOW_RUN["last_asknews_call"] = time.time()
            reported = resp.usage.credits if getattr(resp, "usage", None) is not None else None
            charged = reported if isinstance(reported, int) else 1  # count 1 if not reported
            _SHADOW_RUN["credits_used"] += charged
            info["asknews_credits"] = charged
            info["asknews_credits_reported"] = reported
        articles = list(resp.as_dicts or [])
        info["article_count"] = len(articles)
        dates = [a.pub_date for a in articles if getattr(a, "pub_date", None) is not None]
        if dates:
            info["newest_article_utc"] = _iso(max(dates))
            info["oldest_article_utc"] = _iso(min(dates))
        if not articles:
            return "No recent news articles were found for this question."
        text = "Here are recent news articles (AskNews, latest news):\n\n"
        for a in sorted(articles, key=lambda x: x.pub_date, reverse=True):
            text += (
                f"**{a.eng_title}**\n{a.summary}\n"
                f"Publish date: {a.pub_date.strftime('%Y-%m-%d %H:%M UTC')}\nSource: {a.source_id}\n\n"
            )
        return text

    def _shadow_write_record(
        self, question, info: dict, result: dict, succeeded: bool, arm: str = "shadow-news", tag: str = SHADOW_TAG,
        bucket: dict | None = None,
    ) -> None:
        record = {
            "arm": arm,
            "shadow_tag": tag,
            "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_time_utc": RUN_STARTED_UTC,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "mode": RUN_MODE,
            "model": LAB_MODEL,
            **_bucket_fields(bucket),
            "prompt_version": PROMPT_VERSION,
            "post_id": question.id_of_post,
            "question_id": question.id_of_question,
            "question_type": type(question).__name__,
            "group_option": question.group_question_option,
            "title": question.question_text,
            "page_url": question.page_url,
            "open_time": _iso(question.open_time),
            "scheduled_close_time": _iso(question.close_time),
            "scheduled_resolution_time": _iso(getattr(question, "scheduled_resolution_time", None)),
            "individual_predictions": result["individual"],
            "n_individual": len(result["individual"]),
            "final_forecast": result["final"],
            "publish_enabled": False,  # the shadow is never posted
            "succeeded": succeeded,
            "range_check_retries": result["retries"],
            "errors": result["errors"],
            "search_query": info.get("search_query"),
            "article_count": info.get("article_count"),
            "newest_article_utc": info.get("newest_article_utc"),
            "oldest_article_utc": info.get("oldest_article_utc"),
            "asknews_credits": info.get("asknews_credits", 0),
            "asknews_credits_reported": info.get("asknews_credits_reported"),
            "asknews_credits_run_total": _SHADOW_RUN["credits_used"],
            "shadow_llm_cost_usd": result["cost"],
        }
        _write_forecast_log_line(record)

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = self._get_research_prompt(question, researcher)

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}

            {FallTemplateBot2026._get_timing_message(question)}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {self._get_timing_message(question)}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        prompt = self._apply_sceptic_section(question, prompt)  # Build 11: no-op outside the sceptic arm
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {self._get_timing_message(question)}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = self._apply_sceptic_section(question, prompt)  # Build 11: no-op outside the sceptic arm
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {self._get_timing_message(question)}

            {lower_bound_message}
            {upper_bound_message}
            {self._get_expected_range_message(question)}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        # Sidecar-bot: if this one forecast fails the range check, retry it
        # once (new LLM call) before letting it be dropped.
        prompt = self._apply_sceptic_section(question, prompt)  # Build 11: no-op outside the sceptic arm
        try:
            return await self._numeric_prompt_to_forecast_once(question, prompt)
        except Exception as e:
            if not self._is_range_check_error(e):
                raise
            logger.warning(
                f"Range check failed for {question.page_url}; retrying this forecast once. ({str(e)[:200]})"
            )
            self._fl_count_retry(question)
            return await self._numeric_prompt_to_forecast_once(question, prompt)

    async def _numeric_prompt_to_forecast_once(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {self._get_timing_message(question)}

            {lower_bound_message}
            {upper_bound_message}
            {self._get_expected_range_message(question)}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        # Sidecar-bot: same single retry as numeric questions.
        prompt = self._apply_sceptic_section(question, prompt)  # Build 11: no-op outside the sceptic arm
        try:
            return await self._date_prompt_to_forecast_once(question, prompt)
        except Exception as e:
            if not self._is_range_check_error(e):
                raise
            logger.warning(
                f"Range check failed for {question.page_url}; retrying this forecast once. ({str(e)[:200]})"
            )
            self._fl_count_retry(question)
            return await self._date_prompt_to_forecast_once(question, prompt)

    async def _date_prompt_to_forecast_once(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### SIDECAR-BOT PROMPT HELPERS #####################################

    @staticmethod
    def _format_time(value: datetime | None) -> str:
        if value is None:
            return "not stated"
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def _get_timing_message(question: MetaculusQuestion) -> str:
        """Sidecar-bot: tell the model when the question opened and closes,
        and that it is still open (Build 5 found models assuming it had closed)."""
        opened = FallTemplateBot2026._format_time(question.open_time)
        closes = FallTemplateBot2026._format_time(question.close_time)
        return (
            f"This question opened on {opened} and is scheduled to close on {closes}. "
            "This question is still open; it has not resolved yet."
        )

    @staticmethod
    def _format_number(value: float) -> str:
        # Plain digits, never scientific notation (e.g. 4,000,000 or 0.1).
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,.6f}".rstrip("0").rstrip(".")

    def _get_expected_range_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        """Sidecar-bot: state the expected answer range and units from the
        question's bounds, so the model answers on the right scale."""
        if isinstance(question, DateQuestion):
            low = question.lower_bound.date().isoformat()
            high = question.upper_bound.date().isoformat()
            return (
                f"Answers are expected between {low} and {high}. "
                "Give your dates in this range, as YYYY-MM-DD."
            )
        low_num = (
            question.nominal_lower_bound
            if question.nominal_lower_bound is not None
            else question.lower_bound
        )
        high_num = (
            question.nominal_upper_bound
            if question.nominal_upper_bound is not None
            else question.upper_bound
        )
        unit = f" {question.unit_of_measure}" if question.unit_of_measure else ""
        return (
            f"Answers are expected between {self._format_number(low_num)} and "
            f"{self._format_number(high_num)}{unit}. Give your numbers on this scale."
        )

    def _fl_count_retry(self, question: MetaculusQuestion) -> None:
        try:
            self._fl_state(question)["range_check_retries"] += 1
        except Exception:
            pass

    @staticmethod
    def _is_range_check_error(error: Exception) -> bool:
        """True if the error is forecasting_tools' 'too far from bounds' check."""
        return isinstance(error, ValueError) and (
            "No declared percentiles are within the range of the question"
            in str(error)
        )

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


def run_outcome_exit_code(forecast_reports: list) -> int:
    """Sidecar-bot: the run fails (exit code 1) only if MORE than half of the
    questions errored. A few skipped questions do not fail the whole run."""
    total = len(forecast_reports)
    failed = sum(1 for r in forecast_reports if isinstance(r, BaseException))
    if total == 0:
        print("Run outcome: no questions to forecast. Exit code 0.")
        return 0
    if failed * 2 > total:
        print(
            f"RUN FAILED: {failed} of {total} questions errored (more than half). Exit code 1."
        )
        return 1
    if failed:
        print(
            f"Run outcome: {failed} of {total} questions errored (not more than half). Exit code 0."
        )
    else:
        print(f"Run outcome: all {total} questions succeeded. Exit code 0.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "minibench", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal[
        "tournament", "minibench", "metaculus_cup", "test_questions"
    ] = args.mode
    RUN_MODE = run_mode  # recorded in forecast_log.jsonl

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Configure the bot. Models are set in MODEL SETTINGS near the top of
    # this file (MODEL_NAME / build_llms).
    template_bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=build_llms(),
    )

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "minibench": "https://www.metaculus.com/tournament/minibench/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "minibench":
        # Sidecar-bot: MiniBench only (no Fall tournament, no Metaculus Cup).
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )

    # Sidecar-bot: raise_errors=False so the summary always prints.
    template_bot.log_report_summary(forecast_reports, raise_errors=False)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
    sys.exit(run_outcome_exit_code(forecast_reports))
