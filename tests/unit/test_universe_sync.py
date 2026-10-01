from datetime import date

from src.ingestion.universe_sync import load_current_swing_symbols


class FakeResult:
    def scalars(self):
        return self

    def all(self):
        return ["NSE:INFY-EQ", "NSE:TCS-EQ", "NSE:RELIANCE-EQ", ""]


class FakeDb:
    def __init__(self):
        self.sql = None
        self.params = None

    def execute(self, statement, params):
        self.sql = str(statement)
        self.params = params
        return FakeResult()


def test_load_current_swing_symbols_uses_point_in_time_membership():
    db = FakeDb()

    result = load_current_swing_symbols(db, date(2026, 10, 1))

    assert result == [
        "NSE:INFY-EQ",
        "NSE:TCS-EQ",
        "NSE:RELIANCE-EQ",
    ]
    assert "universe_membership_history" in db.sql
    assert "effective_from <= :as_of" in db.sql
    assert "effective_to IS NULL OR effective_to > :as_of" in db.sql
    assert db.params == {"as_of": date(2026, 10, 1)}
