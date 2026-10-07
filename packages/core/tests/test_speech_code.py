import numpy as np
from interview_core.avatar import MOUTH_SHAPES, from_polly, from_text
from interview_core.codeexec import grade_submission, load_challenges, pick_challenge, runner
from interview_core.delivery import compute
from interview_core.speech.asr import Word, to_wav_bytes
from interview_core.speech.tts import VisemeMark


def _ch(cid):
    return next(c for c in load_challenges() if c.id == cid)


def test_delivery_metrics_with_timestamps():
    words = [
        Word("So", 0.0, 0.2),
        Word("um", 0.3, 0.5),
        Word("I", 0.6, 0.7),
        Word("built", 0.8, 1.1),
        Word("it", 2.5, 2.7),
        Word("uh", 2.8, 3.0),
        Word("quickly.", 3.1, 3.6),
    ]
    d = compute(words)
    assert d.words == 5 and [f["word"] for f in d.fillers] == ["um", "uh"]
    assert d.pauses == [{"start": 1.1, "duration": 1.4}] and d.longest_pause_s == 1.4
    assert d.words_per_minute and d.words_per_minute > 0
    assert not compute([Word("hi", 0, 0.2)]).measured


def test_visemes():
    assert from_polly([VisemeMark(0, "sil"), VisemeMark(50, "p"), VisemeMark(90, "a")]) == [
        (0, "rest"),
        (50, "mbp"),
        (90, "aa"),
    ]
    tl = from_text("Hello world", 1000)
    assert tl[0][0] == 0 and all(s in MOUTH_SHAPES for _, s in tl) and tl[-1][0] < 1000


def test_wav_bytes_header():
    assert to_wav_bytes(np.zeros(160, np.float32), 16000)[:4] == b"RIFF"


def test_challenges_public_view_hides_hidden_tests():
    for c in load_challenges():
        pub = c.public()
        assert all(e["expected"] in [t.expected for t in c.tests if not t.hidden] for e in pub["examples"])
        assert pub["hidden_tests"] == sum(t.hidden for t in c.tests)
    assert pick_challenge("data", 2, "s").id.startswith(("sql", "py"))
    assert pick_challenge("design", 2, "s") is None


def test_sql_graded_locally():
    good = (
        "SELECT c.name, SUM(o.amount) AS total FROM customers c JOIN orders o ON o.customer_id = c.id "
        "GROUP BY c.id ORDER BY total DESC LIMIT 3;"
    )
    r = grade_submission(_ch("sql-top-customers"), "sql", good, None)
    assert (r.passed, r.total, r.executor) == (1, 1, "sqlite-local")
    bad = grade_submission(_ch("sql-top-customers"), "sql", "SELECT name, 0 FROM customers", None)
    assert bad.passed == 0 and bad.outcomes[0].status == "wrong answer"
    multi = grade_submission(_ch("sql-duplicates"), "sql", "DROP TABLE users; SELECT 1", None)
    assert multi.passed == 0 and "exactly one SELECT" in multi.outcomes[0].status


def test_sql_runaway_query_is_stopped():
    q = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT count(*) FROM n"
    r = grade_submission(_ch("sql-duplicates"), "sql", q, None)
    assert r.passed == 0 and r.outcomes[0].status.startswith("error")


def test_general_code_requires_sandbox_and_hides_hidden_outputs(monkeypatch):
    ch = _ch("py-word-freq")
    assert "not configured" in grade_submission(ch, "python", "print(1)", None).error

    class FakeJudge:
        def run(self, source, language_id, stdin):
            word = sorted(stdin.split(), key=lambda w: (-stdin.split().count(w), w))[0]
            return {"status": {"description": "Accepted"}, "stdout": word + "\n", "time": "0.01"}

    r = grade_submission(ch, "python", "solution", FakeJudge())
    assert r.passed == r.total == 3 and r.executor == "judge0"
    assert all(o.stdout is None for o in r.outcomes if o.hidden)
    assert grade_submission(ch, "ruby", "x", FakeJudge()).error


def test_language_ids_match_judge0_ce():
    assert runner.LANGUAGE_IDS == {"python": 71, "javascript": 63, "java": 62, "cpp": 54, "sql": 82}
