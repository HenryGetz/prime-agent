//! A turn whose only output is reasoning (no text, no tool calls) replays with its
//! reasoning; only a turn with nothing at all is left out of the next request.

use super::*;
use crate::providers::openai_completions::convert::convert_messages;
use crate::providers::openai_completions::get_compat;

fn chunk(delta: &Value, finish_reason: Option<&str>) -> String {
    let chunk = json!({
        "id": "c1",
        "object": "chat.completion.chunk",
        "model": "gpt-oss-test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    });
    format!("data: {chunk}\n\n")
}

/// Streams one turn of `deltas`, then replays it after a user message.
async fn replayed(deltas: &[Value]) -> Vec<Value> {
    let mut model = completions_model("gpt-oss-test", "intercept", 0.0, 0.0);
    model["reasoning"] = json!(true);
    let mut body: String = deltas.iter().map(|delta| chunk(delta, None)).collect();
    body.push_str(&chunk(&json!({}), Some("stop")));
    body.push_str(DONE);
    let message = stream_final_message(model.clone(), body).await;
    model["baseUrl"] = json!("http://127.0.0.1:0");
    let model: Model = serde_json::from_value(model).unwrap();
    let context = Context {
        system_prompt: None,
        messages: vec![
            crate::types::Message::Assistant(message),
            crate::types::Message::User(crate::types::UserMessage {
                content: crate::types::UserMessageContent::Text("next".into()),
                timestamp: 0,
                rest: Map::default(),
            }),
        ],
        tools: None,
    };
    convert_messages(&model, &context, &get_compat(&model))
}

#[tokio::test]
async fn a_reasoning_only_turn_is_replayed_with_its_reasoning() {
    let params = replayed(&[
        json!({"role": "assistant", "reasoning_content": "send the result"}),
        json!({"reasoning_content": " to the parent"}),
    ])
    .await;
    assert_eq!(params.len(), 2, "{params:?}");
    assert_eq!(
        params[0]["reasoning_content"],
        json!("send the result to the parent")
    );
    assert_eq!(params[0]["content"], json!(""));
    assert!(params[0].get("tool_calls").is_none());
}

#[tokio::test]
async fn a_turn_with_nothing_in_it_is_left_out() {
    let params = replayed(&[json!({"role": "assistant"})]).await;
    assert_eq!(params.len(), 1, "{params:?}");
    assert_eq!(params[0]["role"], json!("user"));
}
