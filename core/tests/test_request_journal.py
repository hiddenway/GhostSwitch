import tempfile
import unittest
from pathlib import Path
from core.request_journal import RequestJournal


class LiveJournalTests(unittest.TestCase):
    def test_live_rejection_and_progress_survive_status_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = RequestJournal(Path(directory) / 'journal.sqlite3')
            journal.start('test', {}, 'OpenCode')
            stage = {'phase': 'verification', 'status': 'RETRY_WITH_FEEDBACK',
                     'attempt': 1, 'reason': 'tool_call_truncated', 'feedback': 'Reduce content'}
            journal.update('test', event={'type': 'stage', 'stage': stage})
            journal.update('test', event={'type': 'status', 'status': 'Генерация ответа', 'attempt': 2})
            journal.update('test', event={'type': 'progress', 'attempt': 2, 'generated_tokens': 32, 'max_tokens': 1024})
            info = journal.list()[0]['response']['ghostswitch']
            self.assertEqual(info['stages'], [stage])
            self.assertEqual(info['verification']['attempts'], 2)
            self.assertEqual(info['progress']['generated_tokens'], 32)
            journal.update('test', event={'type': 'status', 'status': 'Проверка ответа', 'attempt': 2})
            self.assertNotIn('progress', journal.list()[0]['response']['ghostswitch'])
