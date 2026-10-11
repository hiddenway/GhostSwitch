"""The verifier's exact checks: arithmetic parsed from the request, and JSON fields against the input."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core import verifier_v6
from scripts import verifier_eval

CASES = Path(__file__).parent / "data" / "verifier_cases.jsonl"
ACCEPT = '{"status":"ACCEPT","issue":null,"feedback":null,"check":null}'


def verify(request: str | list[dict], candidate: str, reviewer: str = ACCEPT):
    messages = [{"role": "user", "content": request}] if isinstance(request, str) else request
    with patch.object(verifier_v6.llm, "chat", return_value=reviewer) as chat:
        return verifier_v6.verify(messages, candidate, finish_reason="stop", review=True), chat


class RequestArithmeticTests(unittest.TestCase):
    def test_reads_one_expression_asked_about(self):
        cases = {
            "Сколько будет 17 * 23 + 144 / 12? Дай только число.": ("17*23+144/12", 403),
            "\"Сколько будет 17 * 23 + 144 / 12? Дай только число.\"": ("17*23+144/12", 403),
            "What is 48 × 37 − 296?": ("48×37−296", 1480),
            "Посчитай (125 - 37) * 4": ("(125-37)*4", 352),
            "Вычисли 7,5 * 4 + 0,25": ("7.5*4+0.25", 30.25),
            "Сколько будет 1 000 000 / 8?": ("1000000/8", 125000),
            "Сколько будет 15% от 2340?": ("15/100*2340", 351),
            "2^10 - 24 = ?": ("2^10-24", 1000),
        }
        for request, (expression, value) in cases.items():
            with self.subTest(request=request):
                self.assertEqual(verifier_v6._request_arithmetic(request), (expression, value))

    def test_ignores_requests_that_are_not_one_plain_expression(self):
        for request in (
            "Сколько будет 2 + 2 в двоичной системе?",          # another base
            "Сколько будет 3 + 5 км?",                          # units
            "Сколько будет 30 / 120 в процентах?",              # another form of the result
            "What is 12 * 4 if each box holds 6?",              # other numbers
            "Сколько будет 2+2 и 3+3?",                         # two expressions
            "Calculate 1,234 * 5",                              # thousands or decimals
            "Извлеки дату: 2026-04-17",                         # a date, not a question
            "Поезд проехал 240 км за 3 часа. Средняя скорость?",
            "Сколько будет 100 / 0?",
        ):
            with self.subTest(request=request):
                self.assertIsNone(verifier_v6._request_arithmetic(request))

    def test_contradiction_needs_numbers_none_of_which_is_the_value(self):
        self.assertTrue(verifier_v6._contradicts_arithmetic("553", 403))
        self.assertTrue(verifier_v6._contradicts_arithmetic("17 * 23 + 144 / 12 = 391 + 1,2 = 392,2", 403))
        for answer, exact in (("403", 403), ("391 + 12 = 403", 403), ("Ответ: 403 (391 + 12)", 403),
                              ("12,5", 12.5), ("≈ 33,33", 100 / 3), ("Ответ: 1/2", 0.5), ("минус 7", -7),
                              ("75%", 0.75),
                              ("≈ 1,1 × 10^12", 2 ** 40), ("четыреста три", 403)):
            with self.subTest(answer=answer):
                self.assertFalse(verifier_v6._contradicts_arithmetic(answer, exact))

    def test_wrong_result_is_retried_without_the_reviewer(self):
        result, chat = verify("Сколько будет 17 * 23 + 144 / 12? Дай только число.", "553")
        chat.assert_not_called()
        self.assertEqual(result["status"], "RETRY_WITH_FEEDBACK")
        self.assertEqual(result["issue"], "result")
        self.assertNotIn("403", result["feedback"])
        self.assertEqual(result["check"], {"expression": "17*23+144/12", "value": 403.0, "agrees": False,
                                           "override": "contradicted", "source": "request"})

    def test_request_expression_replaces_the_reviewers_check(self):
        typo = '{"status":"ACCEPT","issue":null,"feedback":null,"check":"17*23+144*12"}'
        result, chat = verify("Сколько будет 17 * 23 + 144 / 12?", "403", typo)
        chat.assert_called_once()
        self.assertEqual(result["status"], "ACCEPT")
        self.assertEqual(result["check"], {"expression": "17*23+144/12", "value": 403.0, "agrees": True,
                                           "source": "request"})
        wrong_call = '{"status":"RETRY_WITH_FEEDBACK","issue":"result","feedback":"Wrong.","check":null}'
        result, _ = verify("Сколько будет 17 * 23 + 144 / 12?", "391 + 12 = 403", wrong_call)
        self.assertEqual((result["status"], result["check"]["override"]), ("ACCEPT", "confirmed"))


class ApproximateValueTests(unittest.TestCase):
    LOAN = [{"role": "user", "content": "Кредит 1 000 000 рублей на 5 лет под 12% годовых. Ежемесячный платёж?"}]
    CHECK = "1000000*0.01*1.01**60/(1.01**60-1)"  # 22 244,45

    def review(self, candidate: str, status: str):
        verdict = json.dumps({"status": status, "issue": "result" if status != "ACCEPT" else None,
                              "feedback": "Пересчитайте платёж." if status != "ACCEPT" else None,
                              "check": self.CHECK})
        return verify(self.LOAN, candidate, verdict)[0]

    def test_marked_approximation_within_a_tenth_of_a_percent_is_confirmed(self):
        for candidate in ("Платёж ≈ 22 243,78 ₽.",
                          "Платёж = 1 000 000 × 0.01 × 1.01^60 / (1.01^60 − 1) ≈ 22 247.36 рублей. "
                          "Проверка: 22 247.36 × 60 ≈ 1 334 841.60, из чего 334 841.60 руб. — проценты.",
                          "Около **22 244** рублей в месяц."):
            with self.subTest(candidate=candidate):
                result = self.review(candidate, "RETRY_WITH_FEEDBACK")
                self.assertEqual((result["status"], result["check"]["override"]), ("ACCEPT", "confirmed"))

    def test_unmarked_or_distant_values_keep_the_rejection(self):
        for candidate in ("Платёж 22 243,78 ₽.", "Правильный платёж ≈ 23 115,54 рубля.", "≈ 20 000 рублей"):
            with self.subTest(candidate=candidate):
                self.assertEqual(self.review(candidate, "RETRY_WITH_FEEDBACK")["status"], "RETRY_WITH_FEEDBACK")


class JsonFieldTests(unittest.TestCase):
    def test_reads_requested_field_names_and_kinds(self):
        names, kinds = verifier_v6._requested_fields(
            "Turn this receipt text into JSON fields merchant/date/total: Green Bowl; 08 May 2026; paid $17.65.")
        self.assertEqual(names, {"merchant", "date", "total"})
        self.assertEqual(verifier_v6._requested_fields(
            "Извлеки продавца, дату и оплаченную сумму из OCR: NORTH PIER / 2026-04-17 / ИТОГО 28,40 USD."),
            (set(), {"merchant", "date", "total"}))
        # TAX in the quoted data is not a requested field.
        self.assertEqual(verifier_v6._requested_fields(
            "The OCR is noisy: 'M0ON CAFE 11/06/26 SUBT 12.00 TAX 0.96 PAID 12.96'. Extract the paid total and date."),
            (set(), {"total", "date"}))
        self.assertEqual(verifier_v6._requested_fields("Extract the fields from this receipt: ..."), (set(), set()))
        # Keys of JSON the user pasted as data are not requested fields.
        self.assertEqual(verifier_v6._requested_fields(
            'From this order {"order_id": 5, "total": 12.5} extract the total as JSON.')[0], set())

    def test_values_must_come_from_the_text(self):
        text = "Green Bowl; 08 May 2026; paid $17.65. M0ON CAFE. INVOICE #A-1043. ИТОГО 1 250,00 ₽"
        for value in ("GREEN BOWL", "2026-05-08", "08/05/2026", "MOON CAFE", 17.65, "17.65", "A-1043",
                      "1 250,00 ₽", 1250, None, ""):
            with self.subTest(value=value):
                self.assertTrue(verifier_v6._in_text(value, text))
        for value in ("08 MAY 2016", "1 520,00 ₽", "RM17.65", 17.56, "226, JALAN DUTA, 57000 KUALA LUMPUR"):
            with self.subTest(value=value):
                self.assertFalse(verifier_v6._in_text(value, text))

    def test_invented_and_unrequested_fields_are_retried_without_the_reviewer(self):
        request = "Извлеки продавца, дату и оплаченную сумму из OCR: NORTH PIER / 2026-04-17 / ИТОГО 28,40 USD."
        answer = '{"address": "226, JALAN DUTA, KUALA LUMPUR", "date": "2026-04-17", "total": "28,40"}'
        result, chat = verify(request, answer)
        chat.assert_not_called()
        self.assertEqual(result, {"status": "RETRY_WITH_FEEDBACK", "issue": "format", "override": "json_fields",
                                  "feedback": "Add the requested fields: merchant. "
                                              "Remove fields the user did not ask for: address."})
        result, _ = verify(request, '{"merchant": "NORTH PIER", "date": "2026-04-18", "total": "28,40"}')
        self.assertEqual(result["issue"], "result")
        self.assertIn("not in the input text: date", result["feedback"])
        self.assertNotIn("2026-04-17", result["feedback"])

    def test_faithful_extraction_goes_to_the_reviewer(self):
        request = "Turn this receipt text into JSON fields merchant/date/total: Green Bowl; 08 May 2026; paid $17.65."
        result, chat = verify(request, 'Here it is:\n```json\n{"merchant": "Green Bowl", "date": "2026-05-08", '
                                       '"total": 17.65}\n```')
        chat.assert_called_once()
        self.assertEqual(result, {"status": "ACCEPT", "feedback": None})

    def test_only_an_answer_that_is_json_counts(self):
        self.assertEqual(verifier_v6._candidate_json('{"a": 1}'), {"a": 1})
        self.assertEqual(verifier_v6._candidate_json('Вот JSON:\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(verifier_v6._candidate_json('Вот JSON: {"a": 1}'), {"a": 1})
        for answer in ('```python\ndef parse(text):\n    return {"total": 1}\n```',
                       'def parse(text):\n    return {"total": 1}',
                       '```json\n{"a": 1}\n```\nand\n```json\n{"b": 2}\n```',
                       "The total is 12.96 and the date is 11/06/26."):
            with self.subTest(answer=answer):
                self.assertIsNone(verifier_v6._candidate_json(answer))
        request = "Write a function to parse a receipt and return the total and date."
        result, chat = verify(request, '```python\ndef parse(text):\n    return {"total": 1.5, "date": None}\n```')
        chat.assert_called_once()

    def test_unknown_keys_may_hold_requested_fields(self):
        request = "Извлеки продавца и сумму: ООО «Ромашка», ИТОГО 1 250,00 ₽"
        result, chat = verify(request, '{"name": "ООО «Ромашка»", "amount": "1 250,00 ₽"}')
        chat.assert_called_once()
        result, chat = verify(request, '{"amount": "1 250,00 ₽"}')
        self.assertEqual(result["feedback"], "Add the requested fields: merchant.")

    def test_generated_examples_and_images_skip_the_value_check(self):
        result, chat = verify("Придумай пример JSON профиля пользователя с полями name, age, city.",
                              '{"name": "Анна", "age": 28, "city": "Казань"}')
        self.assertEqual(result["status"], "ACCEPT")
        result, _ = verify("Придумай пример JSON профиля пользователя с полями name, age, city.",
                           '{"name": "Анна", "age": 28}')
        self.assertEqual(result["feedback"], "Add the requested fields: city.")
        photo = [{"role": "user", "content": "[Image 1: receipt.png]\nExtract the merchant and total.",
                  "images": [SimpleNamespace(label="[Image 1: receipt.png]")]}]
        result, chat = verify(photo, '{"merchant": "Blue Cafe", "total": 9.5}')
        chat.assert_called_once()
        self.assertEqual(result["status"], "ACCEPT")


class LabeledCaseTests(unittest.TestCase):
    """The exact checks alone, with a reviewer that accepts everything, on the labeled set."""

    def test_exact_checks_reject_no_correct_answer_and_catch_the_known_failures(self):
        cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(cases), 40)
        rejected = set()
        for case in cases:
            with patch.object(verifier_v6.llm, "chat", return_value=ACCEPT):
                result = verifier_v6.verify(case["messages"], case["candidate"], finish_reason="stop",
                                            sources=case.get("sources"))
            if result["status"] != "ACCEPT":
                rejected.add(case["id"])
        correct = {case["id"] for case in cases if case["label"] == "correct"}
        self.assertEqual(rejected & correct, set())
        self.assertLessEqual({case["id"] for case in cases if case["category"] == "arithmetic"
                              and case["label"] == "wrong"}, rejected)
        self.assertLessEqual({"receipt-pier-address", "receipt-bowl-address", "receipt-bowl-2016",
                              "receipt-bowl-extra", "receipt-ru-total", "json-profile-missing"}, rejected)

    def test_eval_script_reports_a_confusion_matrix(self):
        out = io.StringIO()
        with redirect_stdout(out):
            verifier_eval.main(["--llm", "stub", "--only", "arithmetic"])
        report = out.getvalue()
        self.assertIn("wrong answers accepted (false ACCEPT)", report)
        self.assertIn("0/8", report)


if __name__ == "__main__":
    unittest.main()
