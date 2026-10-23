from __future__ import annotations

import asyncio
import importlib
import unittest
from unittest.mock import AsyncMock, Mock, patch


rlm_module = importlib.import_module("rlm")

TRANSCRIPT = [
    {"role": "user", "content": "first prompt", "timestamp": 1000},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan"},
            {"type": "text", "text": "answer"},
            {"type": "toolCall", "id": "c1", "name": "python_repl", "arguments": {"code": "1"}},
        ],
        "timestamp": 1001,
    },
    {
        "role": "toolResult",
        "toolCallId": "c1",
        "toolName": "python_repl",
        "content": [{"type": "text", "text": "1"}],
        "isError": False,
        "timestamp": 1002,
    },
    {"role": "compactionSummary", "summary": "the story so far", "tokensBefore": 4000, "timestamp": 1003},
    {
        "role": "custom",
        "customType": "agent_message",
        "content": "[task from parent]",
        "display": True,
        "timestamp": "1970-01-01T00:00:01.004Z",
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "second prompt"}, {"type": "image", "data": "", "mimeType": "image/png"}],
        "timestamp": 1005,
    },
]


def read(**kwargs):
    host_request = AsyncMock(return_value={"messages": TRANSCRIPT})
    emit = Mock()
    with patch.object(rlm_module, "host_request", host_request), patch.object(rlm_module, "emit", emit):
        messages = asyncio.run(rlm_module.rlm.session.messages(**kwargs))
    host_request.assert_awaited_once_with("rlm.session.messages")
    (record,) = emit.call_args.args[0].values()
    return messages, record


class SessionMessagesTest(unittest.TestCase):
    def test_role_filter_keeps_transcript_indices(self) -> None:
        messages, record = read(role="user")

        self.assertEqual(
            [(m.index, m.text, m.timestamp) for m in messages],
            [(0, "first prompt", 1000), (5, "second prompt", 1005)],
        )
        self.assertEqual(record, {"count": 2, "total": 6, "first": 0, "last": 5, "role": "user"})

    def test_slice_reads_summaries_custom_rows_and_assistant_text(self) -> None:
        messages, record = read(start=1, end=-1)

        self.assertEqual(
            [(m.index, m.role, m.text, m.timestamp) for m in messages],
            [
                (1, "assistant", "answer", 1001),
                (2, "toolResult", "1", 1002),
                (3, "compactionSummary", "the story so far", 1003),
                (4, "custom", "[task from parent]", 1004),
            ],
        )
        self.assertEqual(messages[0].raw["content"][2]["name"], "python_repl")
        self.assertEqual(record, {"count": 4, "total": 6, "first": 1, "last": 4, "role": None})


if __name__ == "__main__":
    unittest.main()
