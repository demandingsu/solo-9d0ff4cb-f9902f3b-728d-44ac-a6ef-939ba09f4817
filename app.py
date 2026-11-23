"""Lap-record review application."""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from flask import (
    Flask,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

from solver import (
    BREAKS,
    INFEASIBLE,
    NOT_BREAK,
    REVIEW,
    STATUS_LABELS,
    solve,
)

MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_CSV_BYTES + 512 * 1024  # multipart overhead
MAX_READINGS = 20_000

HEADER_ALIASES = {
    "lap": {"lap", "lap_number", "lapno", "圈号", "圈"},
    "source": {"source", "timer", "计时来源", "来源"},
    "cumulative_ms": {
        "cumulative_ms",
        "cumulative",
        "elapsed_ms",
        "发车后累计毫秒数",
        "累计毫秒",
        "累计毫秒数",
    },
}

SOURCE_ALIASES = {
    "主": "主计时",
    "主表": "主计时",
    "主计时": "主计时",
    "primary": "主计时",
    "main": "主计时",
    "备": "备计时",
    "备表": "备计时",
    "备计时": "备计时",
    "backup": "备计时",
    "secondary": "备计时",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    min_lap_ms INTEGER NOT NULL CHECK (min_lap_ms > 0),
    record_ms INTEGER NOT NULL CHECK (record_ms >= 0),
    revision INTEGER NOT NULL DEFAULT 1,
    laps_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS laps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    lap_number INTEGER NOT NULL,
    pinned_reading_id INTEGER REFERENCES readings(id),
    UNIQUE (session_id, lap_number)
);

CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lap_id INTEGER NOT NULL REFERENCES laps(id),
    table_version INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL,
    cumulative_ms INTEGER NOT NULL CHECK (cumulative_ms >= 0),
    UNIQUE (lap_id, table_version, source)
);

CREATE TABLE IF NOT EXISTS conclusions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    revision INTEGER NOT NULL,
    lower_ms INTEGER,
    upper_ms INTEGER,
    status TEXT NOT NULL,
    min_lap_ms INTEGER NOT NULL,
    record_ms INTEGER NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    lap_number INTEGER NOT NULL,
    reading_id INTEGER,
    source TEXT,
    cumulative_ms INTEGER,
    revision INTEGER NOT NULL,
    lower_ms INTEGER,
    upper_ms INTEGER,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    revision INTEGER NOT NULL,
    laps_version INTEGER NOT NULL,
    min_lap_ms INTEGER NOT NULL,
    record_ms INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, revision)
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    previous_laps_version INTEGER NOT NULL,
    laps_version INTEGER NOT NULL,
    previous_revision INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    kept_pins INTEGER NOT NULL,
    unpinned_laps INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS readings_lap_version
    ON readings (lap_id, table_version);
CREATE INDEX IF NOT EXISTS conclusions_session_revision
    ON conclusions (session_id, revision);
CREATE INDEX IF NOT EXISTS approvals_session_revision
    ON approvals (session_id, revision);
"""


class CsvError(ValueError):
    """Raised when an uploaded CSV cannot be accepted."""


class ReviewConflict(Exception):
    """Raised when a review submission must not change stored state."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(path: str, *, initialise: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    if initialise:
        # WAL is persistent; setting it on every request would take an extra
        # write lock for no benefit.
        connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def _transaction(connection: sqlite3.Connection):
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}


def _migrate(connection: sqlite3.Connection) -> None:
    """Bring pre-correction databases up to the versioned-reading schema."""

    if "laps_version" not in _table_columns(connection, "sessions"):
        connection.execute(
            "ALTER TABLE sessions ADD COLUMN laps_version INTEGER NOT NULL DEFAULT 1"
        )

    if "table_version" not in _table_columns(connection, "readings"):
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            connection.executescript(
                """
                BEGIN;
                CREATE TABLE readings_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lap_id INTEGER NOT NULL REFERENCES laps(id),
                    table_version INTEGER NOT NULL DEFAULT 1,
                    source TEXT NOT NULL,
                    cumulative_ms INTEGER NOT NULL CHECK (cumulative_ms >= 0),
                    UNIQUE (lap_id, table_version, source)
                );
                INSERT INTO readings_new (id, lap_id, table_version, source, cumulative_ms)
                    SELECT id, lap_id, 1, source, cumulative_ms FROM readings;
                DROP TABLE readings;
                ALTER TABLE readings_new RENAME TO readings;
                COMMIT;
                """
            )
        finally:
            connection.execute("PRAGMA foreign_keys = ON")


def _normalise_source(raw: str) -> str:
    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        raise CsvError("计时来源不能为空")
    return SOURCE_ALIASES.get(cleaned.casefold(), cleaned)


def _find_header_mapping(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise CsvError("CSV 缺少表头")
    cleaned = {name: (name or "").strip().casefold() for name in fieldnames}
    mapping: dict[str, str] = {}
    for canonical, aliases in HEADER_ALIASES.items():
        matches = [name for name, lowered in cleaned.items() if lowered in aliases]
        if len(matches) > 1:
            raise CsvError(f"CSV 表头重复：{canonical} 对应多个列")
        if matches:
            mapping[canonical] = matches[0]
    missing = [key for key in HEADER_ALIASES if key not in mapping]
    if missing:
        raise CsvError(
            "CSV 缺少必需列：" + "、".join(missing)
            + "（支持 lap/圈号、source/计时来源、cumulative_ms/发车后累计毫秒数）"
        )
    return mapping


def _parse_rows(text: str):
    if len(text) > MAX_CSV_BYTES:
        raise CsvError("CSV 文件过大")
    if not text.strip():
        raise CsvError("CSV 文件为空")

    last_error: Exception | None = None
    reader = None
    mapping = None
    for delimiter in (",", ";", "\t"):
        try:
            candidate = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            candidate_mapping = _find_header_mapping(candidate.fieldnames)
        except CsvError as exc:
            last_error = exc
            continue
        reader = candidate
        mapping = candidate_mapping
        break

    if reader is None or mapping is None:
        if isinstance(last_error, CsvError):
            raise last_error
        raise CsvError("无法识别 CSV 表头")

    laps: dict[int, dict[str, int]] = {}
    seen_pairs: set[tuple[int, str]] = set()
    total = 0

    for row_number, row in enumerate(reader, start=2):
        if row.get(None):
            raise CsvError(f"第 {row_number} 行多出无法识别的列")
        values = [
            (row.get(mapping[key]) or "").strip()
            for key in ("lap", "source", "cumulative_ms")
        ]
        if not any(values):
            continue

        lap_text, source_text, cumulative_text = values
        try:
            lap = int(lap_text)
        except ValueError:
            raise CsvError(f"第 {row_number} 行圈号不是整数：{lap_text!r}") from None
        if lap < 1:
            raise CsvError(f"第 {row_number} 行圈号必须从 1 开始")
        if lap > 500:
            raise CsvError(f"第 {row_number} 行圈号超出允许范围（1-500）")

        try:
            cumulative = int(cumulative_text)
        except ValueError:
            raise CsvError(
                f"第 {row_number} 行累计毫秒数不是整数：{cumulative_text!r}"
            ) from None
        if cumulative < 0:
            raise CsvError(f"第 {row_number} 行累计毫秒数不能为负")

        source = _normalise_source(source_text)
        pair = (lap, source)
        if pair in seen_pairs:
            raise CsvError(f"第 {row_number} 行重复：第 {lap} 圈来源 {source!r} 已有读数")
        seen_pairs.add(pair)

        laps.setdefault(lap, {})[source] = cumulative
        total += 1
        if total > MAX_READINGS:
            raise CsvError(f"读数超过 {MAX_READINGS} 条上限")

    if not laps:
        raise CsvError("CSV 没有有效数据行")

    missing_laps = [lap for lap in range(1, max(laps) + 1) if lap not in laps]
    if missing_laps:
        preview = "、".join(str(lap) for lap in missing_laps[:10])
        suffix = "…" if len(missing_laps) > 10 else ""
        raise CsvError(f"圈号必须从 1 连续编号，缺少第 {preview}{suffix} 圈")

    return [
        (lap, sorted(laps[lap].items(), key=lambda item: item[0]))
        for lap in sorted(laps)
    ]


def parse_csv(raw: bytes):
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CsvError("CSV 必须使用 UTF-8 编码") from None
    return _parse_rows(text)


def _readings_by_lap(
    connection: sqlite3.Connection,
    session_id: int,
):
    rows = connection.execute(
        """
        SELECT laps.lap_number, laps.pinned_reading_id,
               readings.id AS reading_id, readings.table_version,
               readings.source, readings.cumulative_ms
        FROM laps
        JOIN readings ON readings.lap_id = laps.id
        WHERE laps.session_id = ?
        ORDER BY laps.lap_number, readings.table_version, readings.source
        """,
        (session_id,),
    ).fetchall()

    laps: list[dict] = []
    by_number: dict[int, dict] = {}
    for row in rows:
        lap = by_number.get(row["lap_number"])
        if lap is None:
            lap = {
                "lap_number": row["lap_number"],
                "pinned_reading_id": row["pinned_reading_id"],
                "readings": [],
            }
            by_number[row["lap_number"]] = lap
            laps.append(lap)
        lap["readings"].append(
            {
                "id": row["reading_id"],
                "table_version": row["table_version"],
                "source": row["source"],
                "cumulative_ms": row["cumulative_ms"],
            }
        )
    return sorted(laps, key=lambda item: item["lap_number"])


def _build_visible_laps(
    all_laps: list[dict],
    laps_version: int,
    pins: dict[int, int | None],
) -> list[dict]:
    """Project stored data onto one reading-table version.

    Rows are never deleted, so an audit view can ask for any older version.
    ``pins`` holds each lap's current pin; when that pin belongs to another
    table version it is shown as absent for the requested version.
    """

    visible: list[dict] = []
    for lap in all_laps:
        readings = [r for r in lap["readings"] if r["table_version"] == laps_version]
        pin_id = pins.get(lap["lap_number"])
        pinned = any(r["id"] == pin_id for r in readings)
        visible.append(
            {
                "lap_number": lap["lap_number"],
                "pinned_reading_id": pin_id if pinned else None,
                "readings": readings,
            }
        )
    return visible


def _pin_entry(
    all_laps: list[dict], lap_number: int, reading_id: int | None
) -> dict | None:
    """Describe a pin for the snapshot, keeping its status under its own table."""

    if reading_id is None:
        return None
    lap = next((item for item in all_laps if item["lap_number"] == lap_number), None)
    reading = None
    if lap is not None:
        reading = next((r for r in lap["readings"] if r["id"] == reading_id), None)
    if reading is None:  # Defensive; foreign keys and app logic prevent this.
        return {"reading_id": reading_id, "table_version": None,
                "source": None, "cumulative_ms": None}
    return {
        "reading_id": reading_id,
        "table_version": reading["table_version"],
        "source": reading["source"],
        "cumulative_ms": reading["cumulative_ms"],
    }


def _state_from_snapshot(snapshot: dict) -> dict:
    pinned = {
        entry["lap_number"]: entry.get("pin")
        for entry in snapshot.get("laps", [])
    }
    return {
        "revision": snapshot.get("revision"),
        "laps_version": snapshot.get("laps_version"),
        "min_lap_ms": snapshot.get("min_lap_ms"),
        "record_ms": snapshot.get("record_ms"),
        "created_at": snapshot.get("created_at"),
        "pinned": pinned,
        "laps": snapshot.get("laps", []),
        "conclusion": snapshot.get("conclusion"),
    }


def _snapshot_json(snapshot: dict) -> str:
    return json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))


def _state_from_revision(
    connection: sqlite3.Connection,
    session_id: int,
    revision: int,
) -> dict | None:
    row = connection.execute(
        "SELECT snapshot_json FROM revisions WHERE session_id = ? AND revision = ?",
        (session_id, revision),
    ).fetchone()
    if row is None:
        return None
    return _state_from_snapshot(json.loads(row["snapshot_json"]))


def _table_version_at(connection: sqlite3.Connection, session: sqlite3.Row, revision: int):
    """Which reading-table version was active when ``revision`` was created."""

    active = session["laps_version"]
    for row in connection.execute(
        """
        SELECT previous_laps_version, laps_version, previous_revision, revision
        FROM corrections WHERE session_id = ? ORDER BY revision
        """,
        (session["id"],),
    ):
        if revision <= row["previous_revision"]:
            return row["previous_laps_version"]
        if revision <= row["revision"]:
            return row["laps_version"]
        active = row["laps_version"]
    return active


def _synthesize_state(
    connection: sqlite3.Connection,
    session: sqlite3.Row,
    revision: int,
) -> dict | None:
    """Rebuild a revision view for databases created before snapshots existed.

    Revisions written by the current code always have a snapshot.  For data
    imported by older versions we can still replay the conclusion and the
    approval history against the reading-table version that was active then.
    """

    conclusion = connection.execute(
        "SELECT * FROM conclusions WHERE session_id = ? AND revision = ?",
        (session["id"], revision),
    ).fetchone()
    if conclusion is None:
        return None

    active_version = _table_version_at(connection, session, revision)
    all_laps = _readings_by_lap(connection, session["id"])

    pins: dict[int, int | None] = {}
    for row in connection.execute(
        """
        SELECT lap_number, reading_id, source, cumulative_ms
        FROM approvals
        WHERE session_id = ? AND revision <= ?
        ORDER BY revision, id
        """,
        (session["id"], revision),
    ):
        pins[row["lap_number"]] = row["reading_id"] if row["reading_id"] else None

    lap_entries = []
    for lap in _build_visible_laps(all_laps, active_version, pins):
        pin_entry = None
        if lap["pinned_reading_id"] is not None:
            reading = next(
                (r for r in lap["readings"] if r["id"] == lap["pinned_reading_id"]),
                None,
            )
            if reading is not None:
                pin_entry = {
                    "reading_id": reading["id"],
                    "table_version": reading["table_version"],
                    "source": reading["source"],
                    "cumulative_ms": reading["cumulative_ms"],
                }
        lap_entries.append(
            {
                "lap_number": lap["lap_number"],
                "readings": [
                    {
                        "source": reading["source"],
                        "cumulative_ms": reading["cumulative_ms"],
                    }
                    for reading in lap["readings"]
                ],
                "pin": pin_entry,
            }
        )

    return {
        "revision": revision,
        "laps_version": active_version,
        "min_lap_ms": conclusion["min_lap_ms"],
        "record_ms": conclusion["record_ms"],
        "created_at": conclusion["created_at"],
        "conclusion": {
            "lower_ms": conclusion["lower_ms"],
            "upper_ms": conclusion["upper_ms"],
            "status": conclusion["status"],
            "reason": conclusion["reason"],
        },
        "laps": lap_entries,
    }


def _snapshot_state(
    session: sqlite3.Row,
    all_laps: list[dict],
    revision: int,
    conclusion: sqlite3.Row,
) -> dict:
    lap_entries = []
    visible = _build_visible_laps(
        all_laps,
        session["laps_version"],
        {item["lap_number"]: item["pinned_reading_id"] for item in all_laps},
    )
    for lap in visible:
        lap_entries.append(
            {
                "lap_number": lap["lap_number"],
                "readings": [
                    {
                        "source": reading["source"],
                        "cumulative_ms": reading["cumulative_ms"],
                    }
                    for reading in lap["readings"]
                ],
                "pin": _pin_entry(
                    all_laps, lap["lap_number"], lap["pinned_reading_id"]
                ),
            }
        )
    return {
        "revision": revision,
        "laps_version": session["laps_version"],
        "min_lap_ms": session["min_lap_ms"],
        "record_ms": session["record_ms"],
        "created_at": conclusion["created_at"],
        "conclusion": {
            "lower_ms": conclusion["lower_ms"],
            "upper_ms": conclusion["upper_ms"],
            "status": conclusion["status"],
            "reason": conclusion["reason"],
        },
        "laps": lap_entries,
    }


def _insert_snapshot(
    connection: sqlite3.Connection,
    session_id: int,
    snapshot: dict,
) -> None:
    connection.execute(
        """
        INSERT INTO revisions (session_id, revision, laps_version, min_lap_ms,
                               record_ms, snapshot_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            snapshot["revision"],
            snapshot["laps_version"],
            snapshot["min_lap_ms"],
            snapshot["record_ms"],
            _snapshot_json(snapshot),
            snapshot["created_at"],
        ),
    )


def _candidates(
    laps: list[dict],
    target_lap: int | None = None,
    target_values: list[int] | None = None,
):
    """Build per-lap candidate values.

    ``target_lap`` is the lap currently being decided.  Its candidates are
    replaced by ``target_values`` (a single selected value after approval, or
    every reading after clearing the pin).  Every other lap contributes its
    latest stored pin, or all of its readings when it is still unpinned.
    """
    result = []
    for lap in laps:
        if target_lap is not None and lap["lap_number"] == target_lap:
            if target_values is not None:
                result.append(list(target_values))
            else:
                # Clearing a pin: all readings of this lap become selectable.
                result.append([reading["cumulative_ms"] for reading in lap["readings"]])
        elif lap["pinned_reading_id"] is not None:
            pinned = [
                reading["cumulative_ms"]
                for reading in lap["readings"]
                if reading["id"] == lap["pinned_reading_id"]
            ]
            if not pinned:  # Defensive; foreign keys and app logic prevent this.
                raise ReviewConflict(f"第 {lap['lap_number']} 圈的核定读数不存在")
            result.append(pinned)
        else:
            result.append([reading["cumulative_ms"] for reading in lap["readings"]])
    return result


def _latest_conclusion(connection: sqlite3.Connection, session_id: int):
    return connection.execute(
        "SELECT * FROM conclusions WHERE session_id = ? ORDER BY revision DESC, id DESC LIMIT 1",
        (session_id,),
    ).fetchone()


STATUS_TONE = {
    BREAKS: "good",
    NOT_BREAK: "neutral",
    REVIEW: "warn",
    INFEASIBLE: "bad",
}


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        DATABASE=os.environ.get(
            "DATABASE_PATH",
            os.path.join(app.instance_path, "race_review.sqlite"),
        ),
        MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES,
    )
    if test_config:
        app.config.update(test_config)

    database = app.config["DATABASE"]
    os.makedirs(os.path.dirname(os.path.abspath(database)), exist_ok=True)

    def get_db() -> sqlite3.Connection:
        if "db" not in g:
            g.db = _connect(database)
        return g.db

    app.get_db = get_db  # type: ignore[attr-defined]

    @app.teardown_appcontext
    def close_db(_exception=None):
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    connection = _connect(database, initialise=True)
    try:
        connection.executescript(SCHEMA)
        _migrate(connection)
        connection.executescript(INDEXES)
    finally:
        connection.close()

    @app.template_filter("lap_time")
    def lap_time(value):
        if value is None:
            return "—"
        seconds, millis = divmod(int(value), 1000)
        return f"{seconds}.{millis:03d} s"

    @app.context_processor
    def inject_helpers():
        return {"status_labels": STATUS_LABELS, "status_tone": STATUS_TONE}

    @app.errorhandler(413)
    def payload_too_large(_error):
        return render_index("上传内容过大（CSV 上限 2 MB）", 413)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    def list_sessions():
        connection = get_db()
        return connection.execute(
            """
            SELECT sessions.*,
                   (SELECT status FROM conclusions
                     WHERE conclusions.session_id = sessions.id
                     ORDER BY revision DESC, id DESC LIMIT 1) AS status,
                   (SELECT lower_ms FROM conclusions
                     WHERE conclusions.session_id = sessions.id
                     ORDER BY revision DESC, id DESC LIMIT 1) AS lower_ms,
                   (SELECT upper_ms FROM conclusions
                     WHERE conclusions.session_id = sessions.id
                     ORDER BY revision DESC, id DESC LIMIT 1) AS upper_ms,
                   (SELECT COUNT(*) FROM laps
                     WHERE laps.session_id = sessions.id) AS lap_count
            FROM sessions
            ORDER BY sessions.created_at DESC, sessions.id DESC
            """
        ).fetchall()

    def render_index(error: str | None = None, status_code: int = 200):
        return (
            render_template(
                "index.html",
                sessions=list_sessions(),
                error=error,
            ),
            status_code,
        )

    @app.get("/")
    def index():
        return render_index(request.args.get("error"))

    @app.post("/import")
    def import_csv():
        name = (request.form.get("name") or "").strip() or "未命名批次"
        if len(name) > 200:
            return render_index("批次名称过长", 400)

        try:
            min_lap_ms = int(request.form.get("min_lap_ms", ""))
        except ValueError:
            return render_index("最短可信圈时必须为整数毫秒", 400)
        if min_lap_ms <= 0:
            return render_index("最短可信圈时必须大于 0", 400)

        try:
            record_ms = int(request.form.get("record_ms", ""))
        except ValueError:
            return render_index("现有纪录必须为整数毫秒", 400)
        if record_ms < 0:
            return render_index("现有纪录不能为负数", 400)

        upload = request.files.get("csv_file")
        if upload is None or not upload.filename:
            return render_index("请选择 CSV 文件", 400)
        raw = upload.read(MAX_CSV_BYTES + 1)
        if len(raw) > MAX_CSV_BYTES:
            return render_index("CSV 文件过大（上限 2 MB）", 400)

        try:
            parsed = parse_csv(raw)
        except CsvError as exc:
            return render_index(str(exc), 400)

        candidates = [[value for _source, value in readings] for _lap, readings in parsed]
        result = solve(candidates, min_lap_ms, record_ms)
        if not result.feasible:
            return render_index(result.reason or "没有完整可行序列", 400)

        connection = get_db()
        timestamp = _now()
        with _transaction(connection):
            cursor = connection.execute(
                """
                INSERT INTO sessions (name, min_lap_ms, record_ms, revision,
                                      created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)
                """,
                (name, min_lap_ms, record_ms, timestamp, timestamp),
            )
            session_id = cursor.lastrowid
            for lap_number, readings in parsed:
                lap_cursor = connection.execute(
                    "INSERT INTO laps (session_id, lap_number) VALUES (?, ?)",
                    (session_id, lap_number),
                )
                lap_id = lap_cursor.lastrowid
                for source, cumulative in readings:
                    connection.execute(
                        """
                        INSERT INTO readings (lap_id, table_version, source, cumulative_ms)
                        VALUES (?, 1, ?, ?)
                        """,
                        (lap_id, source, cumulative),
                    )
            connection.execute(
                """
                INSERT INTO conclusions (session_id, revision, lower_ms, upper_ms,
                                         status, min_lap_ms, record_ms, reason, created_at)
                VALUES (?, 1, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    session_id,
                    result.lower_ms,
                    result.upper_ms,
                    result.status,
                    min_lap_ms,
                    record_ms,
                    timestamp,
                ),
            )
            session_row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            conclusion_row = connection.execute(
                "SELECT * FROM conclusions WHERE session_id = ? AND revision = 1",
                (session_id,),
            ).fetchone()
            all_laps = _readings_by_lap(connection, session_id)
            _insert_snapshot(
                connection,
                session_id,
                _snapshot_state(session_row, all_laps, 1, conclusion_row),
            )

        return redirect(url_for("review_session", session_id=session_id))

    def _session_revision_numbers(connection: sqlite3.Connection, session_id: int):
        return [
            row["revision"]
            for row in connection.execute(
                """
                SELECT revision FROM revisions WHERE session_id = ?
                UNION
                SELECT revision FROM conclusions WHERE session_id = ?
                ORDER BY revision DESC
                """,
                (session_id, session_id),
            )
        ]

    def _current_view(connection: sqlite3.Connection, session: sqlite3.Row) -> dict:
        all_laps = _readings_by_lap(connection, session["id"])
        pins = {lap["lap_number"]: lap["pinned_reading_id"] for lap in all_laps}
        return {
            "laps": _build_visible_laps(all_laps, session["laps_version"], pins),
            "conclusion": _latest_conclusion(connection, session["id"]),
            "readings_version": session["laps_version"],
            "readings_version_revision": session["revision"],
        }

    def render_review(
        session_id: int,
        error: str | None = None,
        status_code: int = 200,
        revision: int | None = None,
    ):
        connection = get_db()
        session = connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            return render_template("not_found.html"), 404

        revisions = _session_revision_numbers(connection, session_id)
        historical = False
        state = None
        if revision is not None:
            state = _state_from_revision(connection, session_id, revision)
            if state is None:
                # Data written before snapshots existed: replay the recorded
                # conclusion against the then-active reading table.
                state = _synthesize_state(connection, session, revision)
            if state is None and revision == session["revision"]:
                # No history at all for the current revision; the live state
                # is by definition that revision.
                revision = None
            elif state is None:
                return render_review(
                    session_id,
                    f"版本 {revision} 不存在。",
                    404,
                )

        if revision is not None:
            if revision == session["revision"]:
                # The current version: show live rows so the page can still be
                # used for approval.  Older versions come from their snapshot.
                view = _current_view(connection, session)
                min_lap_ms = session["min_lap_ms"]
                record_ms = session["record_ms"]
                approvals = connection.execute(
                    """
                    SELECT * FROM approvals
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT 20
                    """,
                    (session_id,),
                ).fetchall()
                corrections = connection.execute(
                    "SELECT * FROM corrections WHERE session_id = ? ORDER BY id DESC",
                    (session_id,),
                ).fetchall()
            else:
                historical = True
                laps = []
                for entry in state["laps"]:
                    readings = []
                    for index, reading in enumerate(entry["readings"]):
                        readings.append(
                            {
                                # Snapshots are immutable and read-only, so a
                                # local synthetic id is enough for the template
                                # to mark the approved reading.
                                "id": f"v{revision}-{entry['lap_number']}-{index}",
                                "source": reading["source"],
                                "cumulative_ms": reading["cumulative_ms"],
                                "table_version": None,
                            }
                        )
                    pin = entry.get("pin")
                    pinned_id = None
                    if pin is not None and pin.get("source") is not None:
                        match = next(
                            (
                                r
                                for r in readings
                                if r["source"] == pin["source"]
                                and r["cumulative_ms"] == pin["cumulative_ms"]
                            ),
                            None,
                        )
                        if match is not None:
                            pinned_id = match["id"]
                        else:
                            # The approved reading is not part of this
                            # snapshot's table (possible for legacy revisions
                            # after a correction); still show it explicitly.
                            pinned_id = f"v{revision}-{entry['lap_number']}-pin"
                            readings.append(
                                {
                                    "id": pinned_id,
                                    "source": pin["source"],
                                    "cumulative_ms": pin["cumulative_ms"],
                                    "table_version": pin["table_version"],
                                }
                            )
                    laps.append(
                        {
                            "lap_number": entry["lap_number"],
                            "pinned_reading_id": pinned_id,
                            "readings": readings,
                        }
                    )
                view = {
                    "laps": laps,
                    "conclusion": state["conclusion"],
                    "readings_version": state["laps_version"],
                    "readings_version_revision": revision,
                }
                min_lap_ms = state["min_lap_ms"]
                record_ms = state["record_ms"]
                approvals = connection.execute(
                    "SELECT * FROM approvals WHERE session_id = ? AND revision = ? ORDER BY id",
                    (session_id, revision),
                ).fetchall()
                corrections = connection.execute(
                    "SELECT * FROM corrections WHERE session_id = ? AND revision = ?",
                    (session_id, revision),
                ).fetchall()
        else:
            view = _current_view(connection, session)
            approvals = connection.execute(
                """
                SELECT * FROM approvals
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 20
                """,
                (session_id,),
            ).fetchall()
            corrections = connection.execute(
                "SELECT * FROM corrections WHERE session_id = ? ORDER BY id DESC",
                (session_id,),
            ).fetchall()
            min_lap_ms = session["min_lap_ms"]
            record_ms = session["record_ms"]

        conclusion = view["conclusion"]
        if historical and conclusion is not None:
            conclusion_for_template = dict(conclusion)
            conclusion_for_template["revision"] = revision
            conclusion_for_template["created_at"] = (
                state.get("created_at") or conclusion_for_template.get("created_at")
            )
        else:
            conclusion_for_template = conclusion

        return (
            render_template(
                "review.html",
                session=session,
                laps=view["laps"],
                conclusion=conclusion_for_template,
                approvals=approvals,
                corrections=corrections,
                error=error,
                revisions=revisions,
                viewing_revision=revision,
                readings_version=view["readings_version"],
                readings_version_revision=view["readings_version_revision"],
                historical=historical,
                view_min_lap_ms=min_lap_ms,
                view_record_ms=record_ms,
            ),
            status_code,
        )

    @app.get("/sessions/<int:session_id>")
    def review_session(session_id: int):
        revision_raw = request.args.get("revision", "")
        if revision_raw:
            try:
                return render_review(session_id, revision=int(revision_raw))
            except ValueError:
                return render_review(session_id, "版本号无效", 404)
        return render_review(session_id)

    @app.get("/sessions/<int:session_id>/versions/<int:revision>")
    def review_version(session_id: int, revision: int):
        return render_review(session_id, revision=revision)

    @app.post("/sessions/<int:session_id>/approve")
    def approve_reading(session_id: int):
        expected_pin_raw = request.form.get("expected_pin", "")
        expected_pin: int | None
        if expected_pin_raw in ("", "none"):
            expected_pin = None
        else:
            try:
                expected_pin = int(expected_pin_raw)
            except ValueError:
                return render_review(session_id, "缺少有效的核定状态，请刷新后重试", 409)

        lap_number_text = request.form.get("lap_number", "")
        try:
            lap_number = int(lap_number_text)
        except ValueError:
            return render_review(session_id, "圈号无效", 409)

        clear = request.form.get("action") == "clear"
        reading_id_text = request.form.get("reading_id", "")
        reading_id = None
        if not clear:
            try:
                reading_id = int(reading_id_text)
            except ValueError:
                return render_review(session_id, "请选择一条要核定的读数", 409)

        # The reading-table version the submitting page was rendered from.
        # A correction that lands in between replaces readings, so this
        # submission must be rejected instead of pinning a stale id.
        expected_version_raw = request.form.get("readings_version", "")
        expected_version: int | None
        if expected_version_raw in ("", "none"):
            expected_version = None
        else:
            try:
                expected_version = int(expected_version_raw)
            except ValueError:
                return render_review(session_id, "缺少有效的读数表版本，请刷新后重试", 409)

        connection = get_db()
        session = connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            return render_template("not_found.html"), 404

        try:
            with _transaction(connection):
                # Re-read inside the write transaction so the decision uses
                # the freshest revision and every lap's latest pin.
                session = connection.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if session is None:
                    raise ReviewConflict("该复核批次已不存在")

                if (
                    expected_version is not None
                    and session["laps_version"] != expected_version
                ):
                    raise ReviewConflict(
                        "计时组已上传更正版读数表，本页面上的读数已被替换。"
                        "本次提交未保存，原状态保持不变，请刷新后按最新读数表重新核定。"
                    )

                lap = connection.execute(
                    "SELECT * FROM laps WHERE session_id = ? AND lap_number = ?",
                    (session_id, lap_number),
                ).fetchone()
                if lap is None:
                    raise ReviewConflict(f"第 {lap_number} 圈不存在")

                # Per-lap optimistic concurrency.  Changes to *other* laps are
                # expected and deliberately included in the recalculation
                # below; only a concurrent change to this very lap makes the
                # submission stale, because it would silently overwrite a
                # colleague's decision for the same lap.
                if lap["pinned_reading_id"] != expected_pin:
                    raise ReviewConflict(
                        f"第 {lap_number} 圈的核定已被另一位裁判更新。"
                        "本次提交未保存，原状态保持不变，请刷新后重新提交。"
                    )

                reading = None
                if not clear:
                    reading = connection.execute(
                        """
                        SELECT * FROM readings
                        WHERE id = ? AND lap_id = ? AND table_version = ?
                        """,
                        (reading_id, lap["id"], session["laps_version"]),
                    ).fetchone()
                    if reading is None:
                        raise ReviewConflict(
                            "所选读数不属于该圈，或已被更正版读数表替换。"
                            "本次提交未保存，原状态保持不变，请刷新后按最新读数重新核定。"
                        )

                # Read *latest* pins for all laps inside the write transaction,
                # replacing the target lap with the newly selected reading.
                all_laps = _readings_by_lap(connection, session_id)
                pins = {item["lap_number"]: item["pinned_reading_id"] for item in all_laps}
                laps = _build_visible_laps(all_laps, session["laps_version"], pins)
                candidates = _candidates(
                    laps,
                    lap_number,
                    None if clear else [reading["cumulative_ms"]],
                )

                result = solve(candidates, session["min_lap_ms"], session["record_ms"])
                if not result.feasible:
                    reason = (result.reason or "没有完整可行序列").rstrip("。")
                    raise ReviewConflict(
                        reason + "。本次核定未保存，原状态保持不变。"
                    )

                new_revision = session["revision"] + 1
                timestamp = _now()
                connection.execute(
                    "UPDATE laps SET pinned_reading_id = ? WHERE id = ?",
                    (None if clear else reading_id, lap["id"]),
                )
                connection.execute(
                    "UPDATE sessions SET revision = ?, updated_at = ? WHERE id = ?",
                    (new_revision, timestamp, session_id),
                )
                connection.execute(
                    """
                    INSERT INTO conclusions (session_id, revision, lower_ms, upper_ms,
                                             status, min_lap_ms, record_ms, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        session_id,
                        new_revision,
                        result.lower_ms,
                        result.upper_ms,
                        result.status,
                        session["min_lap_ms"],
                        session["record_ms"],
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO approvals (session_id, lap_number, reading_id, source,
                                           cumulative_ms, revision, lower_ms, upper_ms,
                                           status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        lap_number,
                        None if clear else reading_id,
                        None if clear else reading["source"],
                        None if clear else reading["cumulative_ms"],
                        new_revision,
                        result.lower_ms,
                        result.upper_ms,
                        result.status,
                        timestamp,
                    ),
                )
                pins_after_conclusion = connection.execute(
                    "SELECT * FROM conclusions WHERE session_id = ? AND revision = ?",
                    (session_id, new_revision),
                ).fetchone()
                refreshed_session = connection.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                _insert_snapshot(
                    connection,
                    session_id,
                    _snapshot_state(
                        refreshed_session,
                        _readings_by_lap(connection, session_id),
                        new_revision,
                        pins_after_conclusion,
                    ),
                )
        except ReviewConflict as exc:
            return render_review(session_id, exc.message, 409)

        flash("核定已保存，结论已按最新核定重算。", "good")
        return redirect(url_for("review_session", session_id=session_id))

    @app.post("/sessions/<int:session_id>/correct")
    def correct_session(session_id: int):
        expected_version_raw = request.form.get("expected_version", "")
        try:
            expected_version = int(expected_version_raw)
        except ValueError:
            return render_review(
                session_id, "缺少有效的读数表版本，请刷新后重试", 409
            )

        upload = request.files.get("csv_file")
        if upload is None or not upload.filename:
            return render_review(session_id, "请选择更正版 CSV 文件", 400)
        raw = upload.read(MAX_CSV_BYTES + 1)
        if len(raw) > MAX_CSV_BYTES:
            return render_review(session_id, "CSV 文件过大（上限 2 MB）", 400)

        try:
            parsed = parse_csv(raw)
        except CsvError as exc:
            return render_review(session_id, str(exc), 400)

        connection = get_db()
        if connection.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone() is None:
            return render_template("not_found.html"), 404

        try:
            with _transaction(connection):
                session = connection.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if session is None:
                    raise ReviewConflict("该复核批次已不存在")

                if session["laps_version"] != expected_version:
                    raise ReviewConflict(
                        "读数表已被另一份更正版替换。本次上传未保存，"
                        "原状态保持不变，请刷新后核对最新读数表再上传。"
                    )

                all_laps = _readings_by_lap(connection, session_id)
                old_visible = _build_visible_laps(
                    all_laps,
                    session["laps_version"],
                    {item["lap_number"]: item["pinned_reading_id"] for item in all_laps},
                )

                new_lap_numbers = [lap_number for lap_number, _readings in parsed]
                old_lap_numbers = [item["lap_number"] for item in old_visible]
                if new_lap_numbers != old_lap_numbers:
                    raise ReviewConflict(
                        "更正版圈号范围与原批次不一致：原批次为 1–"
                        f"{old_lap_numbers[-1] if old_lap_numbers else 0} 圈，"
                        f"更正版为 1–{new_lap_numbers[-1] if new_lap_numbers else 0} 圈。"
                        "本次上传未保存，原状态保持不变。"
                    )

                lap_ids = {
                    row["lap_number"]: row["id"]
                    for row in connection.execute(
                        "SELECT id, lap_number FROM laps WHERE session_id = ?",
                        (session_id,),
                    )
                }

                # Carry every approved reading into the new table by
                # (lap, source), preserving lap number, source and milliseconds.
                new_readings: dict[tuple[int, str], int] = {}
                for lap_number, readings in parsed:
                    for source, cumulative in readings:
                        new_readings[(lap_number, source)] = cumulative

                carried: dict[int, tuple[int, str]] = {}
                lost: list[tuple[int, int | None, str | None]] = []
                for lap in old_visible:
                    pin_id = lap["pinned_reading_id"]
                    if pin_id is None:
                        continue
                    pinned = next(
                        (r for r in lap["readings"] if r["id"] == pin_id), None
                    )
                    if pinned is None:
                        lost.append((lap["lap_number"], None, None))
                        continue
                    match = new_readings.get((lap["lap_number"], pinned["source"]))
                    if match is None or match != pinned["cumulative_ms"]:
                        lost.append(
                            (lap["lap_number"], pinned["cumulative_ms"], pinned["source"])
                        )
                        continue
                    carried[lap["lap_number"]] = (
                        match,
                        pinned["source"],
                    )

                if lost:
                    parts = []
                    for lap_number, cumulative, source in lost:
                        if source is None:
                            parts.append(f"第 {lap_number} 圈核定读数缺失")
                        else:
                            parts.append(
                                f"第 {lap_number} 圈 {source} {cumulative} ms 缺失或被改动"
                            )
                    raise ReviewConflict(
                        "更正版未保留全部已核定读数："
                        + "；".join(parts)
                        + "。本次上传未保存，原状态保持不变。"
                    )

                new_version = session["laps_version"] + 1
                timestamp = _now()
                for lap_number, readings in parsed:
                    lap_id = lap_ids[lap_number]
                    for source, cumulative in readings:
                        connection.execute(
                            """
                            INSERT INTO readings (lap_id, table_version, source, cumulative_ms)
                            VALUES (?, ?, ?, ?)
                            """,
                            (lap_id, new_version, source, cumulative),
                        )

                new_pin_ids: dict[int, int | None] = {}
                for lap_number, lap_id in lap_ids.items():
                    if lap_number in carried:
                        _cumulative, source = carried[lap_number]
                        new_pin_ids[lap_number] = connection.execute(
                            """
                            SELECT id FROM readings
                            WHERE lap_id = ? AND table_version = ? AND source = ?
                            """,
                            (lap_id, new_version, source),
                        ).fetchone()["id"]
                    else:
                        new_pin_ids[lap_number] = None
                    connection.execute(
                        "UPDATE laps SET pinned_reading_id = ? WHERE id = ?",
                        (new_pin_ids[lap_number], lap_id),
                    )
                new_all_laps = _readings_by_lap(connection, session_id)
                candidates = _candidates(
                    _build_visible_laps(
                        new_all_laps,
                        new_version,
                        {
                            item["lap_number"]: item["pinned_reading_id"]
                            for item in new_all_laps
                        },
                    )
                )

                result = solve(candidates, session["min_lap_ms"], session["record_ms"])
                if not result.feasible:
                    reason = (result.reason or "没有完整可行序列").rstrip("。")
                    raise ReviewConflict(
                        "按最新核定在更正版读数表中找不到完整可行序列："
                        + reason
                        + "。本次上传未保存，原结果保持不变。"
                    )

                new_revision = session["revision"] + 1
                connection.execute(
                    "UPDATE sessions SET revision = ?, laps_version = ?, updated_at = ? WHERE id = ?",
                    (new_revision, new_version, timestamp, session_id),
                )
                connection.execute(
                    """
                    INSERT INTO conclusions (session_id, revision, lower_ms, upper_ms,
                                             status, min_lap_ms, record_ms, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        session_id,
                        new_revision,
                        result.lower_ms,
                        result.upper_ms,
                        result.status,
                        session["min_lap_ms"],
                        session["record_ms"],
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO corrections (session_id, previous_laps_version,
                                             laps_version, previous_revision, revision,
                                             kept_pins, unpinned_laps, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        session["laps_version"],
                        new_version,
                        session["revision"],
                        new_revision,
                        len(carried),
                        len(old_visible) - len(carried),
                        timestamp,
                    ),
                )
                refreshed_session_row = connection.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                conclusion_row = connection.execute(
                    "SELECT * FROM conclusions WHERE session_id = ? AND revision = ?",
                    (session_id, new_revision),
                ).fetchone()
                _insert_snapshot(
                    connection,
                    session_id,
                    _snapshot_state(
                        refreshed_session_row,
                        new_all_laps,
                        new_revision,
                        conclusion_row,
                    ),
                )
        except ReviewConflict as exc:
            return render_review(session_id, exc.message, 409)

        flash(
            f"已换用更正版读数表（表版本 {new_version}），"
            "已核定读数按圈号、来源和毫秒数保留并重新计算结论。",
            "good",
        )
        return redirect(url_for("review_session", session_id=session_id))

    return app
