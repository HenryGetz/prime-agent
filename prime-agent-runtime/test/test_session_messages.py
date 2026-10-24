from __future__ import annotations

import asyncio
import importlib
import unittest
from unittest.mock import Mock, patch


rlm_module = importlib.import_module("rlm")


def user(text: str) -> dict:
    return {"role": "user", "content": text, "timestamp": 1000}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}], "timestamp": 1000}


class FakeHost:
    """The daemon's page contract over a list of (entry id, message) rows."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = [(f"e{index}", row) for index, row in enumerate(rows)]
        self.pages: list[dict] = []

    def position(self, entry_id: str | None, default: int) -> int:
        return default if entry_id is None else [id for id, _ in self.rows].index(entry_id)

    async def __call__(self, request_type: str, payload: dict) -> dict:
        assert request_type == "rlm.session.messages"
        page = payload["page"]
        self.pages.append(page)
        limit = page["limit"]
        if page["from"] == "end":
            end = self.position(page["before"], len(self.rows))
            start = max(0, end - limit)
            more = start > 0
        else:
            start = self.position(page["after"], -1) + 1
            stop = self.position(page["until"], len(self.rows))
            end = min(stop, start + limit)
            more = end < stop
        rows = self.rows[start:end]
        return {"messages": [row for _, row in rows], "entryIds": [id for id, _ in rows], "more": more}


def run(host: FakeHost, call):
    emit = Mock()
    with (
        patch.object(rlm_module, "host_request", host),
        patch.object(rlm_module, "emit", emit),
        patch.object(rlm_module, "_PAGE_SIZE", 3),
    ):
        result = asyncio.run(call(rlm_module.rlm.session))
    page_sizes = [args.args[0][rlm_module._TRANSCRIPT_READ_MIME]["messages"] for args in emit.call_args_list]
    return result, page_sizes


class SessionMessagesTest(unittest.TestCase):
    def test_cursors_continue_in_both_directions(self) -> None:
        host = FakeHost([user(f"m{index}") for index in range(8)])

        tail, _ = run(host, lambda session: session.messages(limit=4))
        self.assertEqual([m.id for m in tail], ["e7", "e6", "e5", "e4"])
        older, _ = run(host, lambda session: session.messages(limit=10, before=tail[-1].id))
        self.assertEqual([m.id for m in older], ["e3", "e2", "e1", "e0"])
        newer, _ = run(host, lambda session: session.messages(limit=3, after="e1"))
        self.assertEqual([m.text for m in newer], ["m2", "m3", "m4"])
        between, _ = run(host, lambda session: session.messages(limit=10, after="e1", before="e6"))
        self.assertEqual([m.id for m in between], ["e2", "e3", "e4", "e5"])

    def test_role_filter_pages_past_a_page_without_a_match(self) -> None:
        # Pages of three from the tail: e8-e6, e5-e3 (no user row), e2-e0.
        rows = [user("u0"), assistant("a1"), user("u2")]
        rows += [assistant(f"a{index}") for index in range(3, 6)]
        rows += [user("u6"), assistant("a7"), assistant("a8")]
        host = FakeHost(rows)

        users, page_sizes = run(host, lambda session: session.messages(limit=3, role="user"))

        self.assertEqual([m.text for m in users], ["u6", "u2", "u0"])
        self.assertEqual(page_sizes, [3, 3, 3])
        self.assertEqual([page.get("before") for page in host.pages], [None, "e6", "e3"])

    def test_messages_read_summaries_custom_rows_and_assistant_text(self) -> None:
        host = FakeHost(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "plan"},
                        {"type": "text", "text": "answer"},
                        {"type": "toolCall", "id": "c1", "name": "python_repl", "arguments": {"code": "1"}},
                    ],
                    "timestamp": 1001,
                },
                {"role": "compactionSummary", "summary": "the story so far", "tokensBefore": 4000, "timestamp": 1003},
                {
                    "role": "custom",
                    "customType": "agent_message",
                    "content": "[task from parent]",
                    "display": True,
                    "timestamp": "1970-01-01T00:00:01.004Z",
                },
            ]
        )

        messages, _ = run(host, lambda session: session.messages())

        self.assertEqual(
            [(m.role, m.text, m.timestamp) for m in messages],
            [
                ("custom", "[task from parent]", 1004),
                ("compactionSummary", "the story so far", 1003),
                ("assistant", "answer", 1001),
            ],
        )
        self.assertEqual(messages[-1].raw["content"][2]["name"], "python_repl")


if __name__ == "__main__":
    unittest.main()
