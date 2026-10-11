"""Persistent, bounded journal of HTTP inference requests from every client."""
from __future__ import annotations

import json
from contextlib import contextmanager
import sqlite3
import threading
import time
from pathlib import Path

from core_models import data_root


class RequestJournal:
    def __init__(self, path: Path, limit: int = 200):
        self.path = path
        self.limit = limit
        self.lock = threading.RLock()
        self.initialized = False

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, date INTEGER, body TEXT)")
        if not self.initialized:
            # A restart cannot resume old in-flight inference.
            for identifier, body in db.execute("SELECT id, body FROM requests").fetchall():
                record = json.loads(body)
                # Recorded before the renames, under the names OmniBrain or MiniBrain.
                legacy = next((key for key in ('omnibrain', 'minibrain') if key in record["response"]), None)
                changed = legacy is not None
                if changed:
                    record["response"]["ghostswitch"] = record["response"].pop(legacy)
                if record["response"]["status"] in {'queued', 'running'}:
                    record["response"]["status"] = 'failed'
                    record["response"]["error"] = {'type': 'server_restarted', 'message': 'Server restarted before completion'}
                    changed = True
                if changed:
                    db.execute("UPDATE requests SET body=? WHERE id=?", (json.dumps(record, ensure_ascii=False), identifier))
            db.commit()
            self.initialized = True
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def start(self, identifier: str, payload, source: str):
        record = {'id': identifier, 'date': int(time.time() * 1000), 'source': source,
                  'request': payload, 'response': {'id': identifier, 'status': 'queued', 'choices': [], 'ghostswitch': {}}}
        with self.lock, self._connect() as db:
            db.execute('INSERT INTO requests VALUES (?, ?, ?)', (identifier, record['date'], json.dumps(record, ensure_ascii=False)))
            db.execute('DELETE FROM requests WHERE id IN (SELECT id FROM requests ORDER BY date DESC, rowid DESC LIMIT -1 OFFSET ?)', (self.limit,))

    def update(self, identifier: str, *, event=None, result=None):
        with self.lock, self._connect() as db:
            row = db.execute('SELECT body FROM requests WHERE id=?', (identifier,)).fetchone()
            if not row:
                return
            record = json.loads(row[0])
            if result is not None:
                record['response'] = {**result, 'id': identifier}
                record['finishedAt'] = int(time.time() * 1000)
            elif event and event.get('type') == 'status':
                record["response"]["status"] = 'running'
                record["response"]["ghostswitch"]['current_stage'] = event["status"]
                info = record["response"]["ghostswitch"]
                info['verification'] = {**info.get('verification', {}), 'attempts': event.get("attempt", 0)}
                info.pop('progress', None)
            elif event and event.get('type') == 'stage':
                info = record['response']['ghostswitch']
                stage = event['stage']
                info.setdefault('stages', []).append(stage)
                if stage['phase'] == 'verification':
                    info['verification'] = {'status': stage['status'], 'attempts': stage.get('attempt', 0)}
            elif event and event.get('type') == 'progress':
                record['response']['ghostswitch']['progress'] = {
                    key: value for key, value in event.items() if key != 'type'
                }
            db.execute('UPDATE requests SET body=? WHERE id=?', (json.dumps(record, ensure_ascii=False), identifier))

    def list(self):
        with self.lock, self._connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT body FROM requests ORDER BY date DESC, rowid DESC')]


JOURNAL = RequestJournal(data_root() / 'runtime' / 'request-journal.sqlite3')
