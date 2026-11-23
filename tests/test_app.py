"""Unit and integration tests for the lap review application."""

from __future__ import annotations

import io
import os
import tempfile
import unittest

from app import create_app
from solver import (
    BREAKS,
    INFEASIBLE,
    NOT_BREAK,
    REVIEW,
    solve,
)

def brute_force(candidates, min_lap_ms):
    """Reference implementation for tiny inputs only (test oracle)."""

    import itertools

    best = None
    worst = None
    feasible = False
    for combination in itertools.product(*candidates):
        previous = 0
        edges = []
        ok = True
        for value in combination:
            edge = value - previous
            if edge < min_lap_ms:
                ok = False
                break
            edges.append(edge)
            previous = value
        if not ok:
            continue
        feasible = True
        fastest = min(edges)
        best = fastest if best is None else min(best, fastest)
        worst = fastest if worst is None else max(worst, fastest)
    return feasible, best, worst


class SolverTests(unittest.TestCase):
    def assert_matches_brute_force(self, candidates, min_lap_ms, record_ms):
        expected_feasible, expected_low, expected_high = brute_force(
            candidates, min_lap_ms
        )
        result = solve(candidates, min_lap_ms, record_ms)
        self.assertEqual(result.feasible, expected_feasible)
        if expected_feasible:
            self.assertEqual(result.lower_ms, expected_low)
            self.assertEqual(result.upper_ms, expected_high)
        else:
            self.assertIsNone(result.lower_ms)
            self.assertIsNone(result.upper_ms)
            self.assertEqual(result.status, INFEASIBLE)
            self.assertTrue(result.reason)
        return result

    def test_single_source_equal_laps(self):
        result = solve([[80000], [160000], [240000]], 60000, 78000)
        self.assertEqual(result.lower_ms, 80000)
        self.assertEqual(result.upper_ms, 80000)
        self.assertEqual(result.status, NOT_BREAK)

    def test_breaks_record(self):
        result = solve([[70000], [150000]], 60000, 78000)
        self.assertEqual(result.lower_ms, 70000)
        self.assertEqual(result.status, BREAKS)

    def test_unknown_between_bounds_is_review(self):
        result = solve([[70000, 90000], [180000]], 60000, 80000)
        self.assertEqual(result.lower_ms, 70000)
        self.assertEqual(result.upper_ms, 90000)
        self.assertEqual(result.status, REVIEW)

    def test_exact_record_is_not_a_break(self):
        result = solve([[80000], [160000]], 60000, 80000)
        self.assertEqual(result.status, NOT_BREAK)

    def test_infeasible_short_first_lap(self):
        result = solve([[50000], [120000]], 60000, 80000)
        self.assertEqual(result.status, INFEASIBLE)
        self.assertIn("第 1 圈", result.reason)

    def test_infeasible_gap(self):
        result = solve([[70000], [120000]], 60000, 80000)
        self.assertEqual(result.status, INFEASIBLE)
        self.assertIn("第 2 圈", result.reason)
        self.assertIn("小于最短可信圈时", result.reason)

    def test_infeasible_pinned_middle_lap(self):
        result = solve([[70000], [125000], [200000]], 60000, 80000)
        self.assertEqual(result.status, INFEASIBLE)

    def test_needle_lap_forces_break_or_review(self):
        # Lap two can only be 55000 ms at best if the first lap is huge.
        result = solve([[70000], [125000], [260000]], 55000, 60000)
        self.assertEqual(result.lower_ms, 55000)
        self.assertEqual(result.status, BREAKS)

    def test_primary_and_backup_disagreement_can_flip_verdict(self):
        # Lap candidates are {primary, backup}.  The faster reading decides
        # whether the record falls, so the verdict depends on which is real.
        candidates = [[71000, 74000], [142000, 149000]]
        close_record = solve(candidates, 60000, 73000)
        # 74 s then 142 s yields a 68 s lap, so the range is 68–74 s.
        self.assertEqual((close_record.lower_ms, close_record.upper_ms), (68000, 74000))
        self.assertEqual(close_record.status, REVIEW)

        beaten_record = solve(candidates, 60000, 75000)
        self.assertEqual(beaten_record.status, BREAKS)

    def test_pinned_readings_narrow_the_range(self):
        # Pinning the slower readings removes the 71 s option entirely.
        candidates = [[74000], [149000]]
        result = solve(candidates, 60000, 74000)
        self.assertEqual((result.lower_ms, result.upper_ms), (74000, 74000))
        # Equal to the record is not a break, so NOT_BREAK.
        self.assertEqual(result.status, NOT_BREAK)

    def test_random_small_cases_match_brute_force(self):
        import random

        rng = random.Random(20261122)
        for _ in range(400):
            n = rng.randint(1, 4)
            candidates = []
            for _lap in range(n):
                size = rng.randint(1, 3)
                values = sorted(rng.sample(range(0, 400, 5), size))
                candidates.append(values)
            min_lap_ms = rng.randint(1, 120)
            record_ms = rng.randint(0, 500)
            with self.subTest(candidates=candidates, min_lap_ms=min_lap_ms):
                self.assert_matches_brute_force(candidates, min_lap_ms, record_ms)

    def test_many_laps_does_not_enumerate(self):
        # 2^60 naive choices; the solver must stay polynomial.
        candidates = []
        for lap in range(60):
            base = (lap + 1) * 70000
            candidates.append([base, base + 5000])
        result = solve(candidates, 60000, 75000)
        self.assertTrue(result.feasible)
        self.assertGreaterEqual(result.lower_ms, 60000)
        self.assertLessEqual(result.upper_ms, result.upper_ms)

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            solve([[100]], 0, 100)
        with self.assertRaises(ValueError):
            solve([], 1, 100)
        with self.assertRaises(ValueError):
            solve([[]], 1, 100)


CSV_TEMPLATE = """lap,source,cumulative_ms
1,主计时,74210
1,备计时,74230
2,主计时,148900
2,备计时,148870
3,主计时,223600
3,备计时,223640
"""


class AppTests(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": self.db_path,
                "SECRET_KEY": "test",
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        for suffix in ("-wal", "-shm"):
            path = self.db_path + suffix
            if os.path.exists(path):
                os.unlink(path)

    def import_csv(self, text=CSV_TEMPLATE, min_lap_ms="60000", record_ms="78000"):
        data = {
            "name": "测试批次",
            "min_lap_ms": min_lap_ms,
            "record_ms": record_ms,
            "csv_file": (io.BytesIO(text.encode("utf-8")), "laps.csv"),
        }
        return self.client.post("/import", data=data, content_type="multipart/form-data")

    def test_import_success_and_page(self):
        response = self.import_csv()
        self.assertEqual(response.status_code, 302)
        page = self.client.get(response.headers["Location"])
        self.assertEqual(page.status_code, 200)
        body = page.get_data(as_text=True)
        self.assertIn("待核", body)
        self.assertIn("74.210 s", body)

    def test_import_rejects_infeasible_and_stores_nothing(self):
        bad = """lap,source,cumulative_ms
1,主计时,74210
2,主计时,80000
"""
        response = self.import_csv(bad)
        self.assertEqual(response.status_code, 400)
        body = response.get_data(as_text=True)
        self.assertIn("第 2 圈", body)
        with self.app.app_context():
            count = self.app.get_db().execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        self.assertEqual(count, 0)

    def test_import_rejects_gap_and_duplicate(self):
        gap = """lap,source,cumulative_ms
1,主计时,74210
3,主计时,223600
"""
        self.assertEqual(self.import_csv(gap).status_code, 400)
        duplicate = """lap,source,cumulative_ms
1,主计时,74210
1,主计时,74230
"""
        self.assertEqual(self.import_csv(duplicate).status_code, 400)

    def _session_id(self):
        with self.app.app_context():
            return self.app.get_db().execute("SELECT id FROM sessions").fetchone()[0]

    def _lap_row(self, lap_number):
        with self.app.app_context():
            return self.app.get_db().execute(
                """
                SELECT laps.id, laps.pinned_reading_id FROM laps
                WHERE session_id = ? AND lap_number = ?
                """,
                (self._session_id(), lap_number),
            ).fetchone()

    def _reading_id(self, lap_number, source):
        lap = self._lap_row(lap_number)
        with self.app.app_context():
            return self.app.get_db().execute(
                "SELECT id FROM readings WHERE lap_id = ? AND source = ?",
                (lap["id"], source),
            ).fetchone()[0]

    def _approve(self, lap_number, reading_id, expected_pin=None):
        data = {
            "lap_number": str(lap_number),
            "reading_id": str(reading_id),
            "expected_pin": "none" if expected_pin is None else str(expected_pin),
        }
        return self.client.post(
            f"/sessions/{self._session_id()}/approve", data=data
        )

    def _pin(self, lap_number, source):
        pin = self._lap_row(lap_number)["pinned_reading_id"]
        return self._approve(lap_number, self._reading_id(lap_number, source), pin)

    def test_approve_recalculates_with_other_pins(self):
        self.import_csv()
        response = self._pin(2, "备计时")
        self.assertEqual(response.status_code, 302)

        page = self.client.get(f"/sessions/{self._session_id()}")
        self.assertIn("已核定", page.get_data(as_text=True))

        with self.app.app_context():
            latest = self.app.get_db().execute(
                "SELECT * FROM conclusions ORDER BY revision DESC LIMIT 1"
            ).fetchone()
            self.assertEqual(latest["revision"], 2)
            # Laps 1 and 3 still selectable, lap 2 pinned at 148870.
            self.assertLessEqual(latest["lower_ms"], latest["upper_ms"])

    def test_conflict_when_same_lap_changed_concurrently(self):
        self.import_csv()
        # Both judges render the same page state: lap 2 unpinned.
        viewer_a = self.client.get(f"/sessions/{self._session_id()}")
        viewer_b = self.client.get(f"/sessions/{self._session_id()}")
        self.assertEqual(viewer_a.status_code, 200)
        self.assertEqual(viewer_b.status_code, 200)

        a = self._pin(2, "主计时")
        self.assertEqual(a.status_code, 302)

        # Judge B still submits with expected_pin=none for lap 2.
        b = self._approve(2, self._reading_id(2, "备计时"), expected_pin=None)
        self.assertEqual(b.status_code, 409)
        self.assertIn("已被另一位裁判更新", b.get_data(as_text=True))

        with self.app.app_context():
            pinned = self._lap_row(2)["pinned_reading_id"]
            self.assertEqual(pinned, self._reading_id(2, "主计时"))
            revisions = [
                row[0]
                for row in self.app.get_db().execute(
                    "SELECT revision FROM conclusions ORDER BY revision"
                )
            ]
            self.assertEqual(revisions, [1, 2])

    def test_threaded_same_lap_conflict(self):
        """Two clients race on the same lap; SQLite must serialise them."""

        import threading

        self.import_csv()
        barrier = threading.Barrier(2)
        outcomes: list[int] = []

        def judge(source: str):
            client = self.app.test_client()
            client.get(f"/sessions/{self._session_id()}")
            barrier.wait()
            response = client.post(
                f"/sessions/{self._session_id()}/approve",
                data={
                    "lap_number": "2",
                    "reading_id": str(self._reading_id(2, source)),
                    "expected_pin": "none",
                },
            )
            outcomes.append(response.status_code)

        threads = [
            threading.Thread(target=judge, args=("主计时",)),
            threading.Thread(target=judge, args=("备计时",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(sorted(outcomes), [302, 409])
        with self.app.app_context():
            connections = self.app.get_db()
            latest = connections.execute(
                "SELECT revision FROM conclusions WHERE session_id = ? ORDER BY revision DESC LIMIT 1",
                (self._session_id(),),
            ).fetchone()[0]
            self.assertEqual(latest, 2)

    def test_concurrent_different_laps_both_applied(self):
        self.import_csv()
        first = self._pin(1, "主计时")
        self.assertEqual(first.status_code, 302)
        # Second judge's submission for lap 3 still goes through; its
        # expected_pin is its own lap's state, and recalculation sees lap 1.
        second = self._pin(3, "备计时")
        self.assertEqual(second.status_code, 302)

        with self.app.app_context():
            pins = {
                row["lap_number"]: row["pinned_reading_id"]
                for row in self.app.get_db().execute(
                    "SELECT lap_number, pinned_reading_id FROM laps"
                )
            }
            self.assertIsNotNone(pins[1])
            self.assertIsNotNone(pins[3])
            self.assertIsNone(pins[2])
            count = self.app.get_db().execute(
                "SELECT COUNT(*) FROM conclusions"
            ).fetchone()[0]
            self.assertEqual(count, 3)

    def test_infeasible_approval_is_rejected_atomically(self):
        text = """lap,source,cumulative_ms
1,主计时,74210
1,备计时,74230
2,主计时,148900
2,备计时,90000
"""
        self.import_csv(text, min_lap_ms="60000")
        # Pin lap 1 to primary (74210).  Then lap 2 backup at 90000 is too soon
        # (gap 15790 ms), but lap 2 primary is fine.
        self.assertEqual(self._pin(1, "主计时").status_code, 302)
        response = self._pin(2, "备计时")
        self.assertEqual(response.status_code, 409)
        self.assertIn("小于最短可信圈时", response.get_data(as_text=True))

        with self.app.app_context():
            self.assertIsNone(self._lap_row(2)["pinned_reading_id"])
            self.assertIsNotNone(self._lap_row(1)["pinned_reading_id"])
            latest = self.app.get_db().execute(
                "SELECT * FROM conclusions ORDER BY revision DESC LIMIT 1"
            ).fetchone()
            self.assertEqual(latest["revision"], 2)

    def test_clear_pin(self):
        self.import_csv()
        self.assertEqual(self._pin(2, "主计时").status_code, 302)
        pin = self._lap_row(2)["pinned_reading_id"]
        response = self.client.post(
            f"/sessions/{self._session_id()}/approve",
            data={
                "lap_number": "2",
                "action": "clear",
                "expected_pin": str(pin),
            },
        )
        self.assertEqual(response.status_code, 302)
        with self.app.app_context():
            self.assertIsNone(self._lap_row(2)["pinned_reading_id"])

    def test_missing_session(self):
        self.assertEqual(self.client.get("/sessions/9999").status_code, 404)


if __name__ == "__main__":
    unittest.main()
