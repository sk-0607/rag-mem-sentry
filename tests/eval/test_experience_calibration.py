from eval.experience_calibration import calibrate


class FakeStore:
    def __init__(self):
        self.seen = []

    def retrieve(self, task, k=3):
        return []

    def write(self, record, gate=None):
        self.seen.append(record)


def runner(question, store=None):
    store.seen.append(question)
    return {"answer": f"answer to {question}", "path_trace": ["retrieve", "generate"],
            "experiences": ["e1"]}


def test_every_question_is_run():
    store = FakeStore()
    calibrate(["q1", "q2", "q3"], runner, store)
    assert store.seen == ["q1", "q2", "q3"]


def test_one_row_per_question():
    assert len(calibrate(["q1", "q2"], runner, FakeStore())) == 2


def test_row_carries_the_question_and_answer():
    row = calibrate(["q1"], runner, FakeStore())[0]
    assert row["question"] == "q1"
    assert row["answer"] == "answer to q1"


def test_row_records_the_retrieved_experience_ids():
    assert calibrate(["q1"], runner, FakeStore())[0]["experiences"] == ["e1"]


def test_rows_are_indexed_in_run_order():
    rows = calibrate(["q1", "q2"], runner, FakeStore())
    assert [r["i"] for r in rows] == [0, 1]


def test_no_questions_produces_no_rows():
    assert calibrate([], runner, FakeStore()) == []
