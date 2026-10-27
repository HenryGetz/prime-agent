//! Tool-call argument text replays exactly as the model streamed it. A model can
//! write arguments that aren't valid JSON (raw code, a missing closing brace) or
//! that serialize differently once parsed; replaying the parse instead would make
//! the history disagree with what the model sampled.

use super::*;
use crate::providers::openai_completions::convert::convert_messages;
use crate::providers::openai_completions::get_compat;

fn chunk(delta: &Value) -> String {
    let chunk = json!({
        "id": "c1",
        "object": "chat.completion.chunk",
        "model": "gpt-oss-test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
    });
    format!("data: {chunk}\n\n")
}

const TOOLCALL_FINISH: &str = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-oss-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n";

const ARGUMENTS: [&str; 4] = [
    "print(1)",
    "{\"code\":\"print(1)\"",
    "{\"code\": \"print(1)\"}",
    "{\"code\":\"print(1)\"}",
];

/// Streams one tool call per entry of [`ARGUMENTS`], each split across two deltas.
async fn stream_calls() -> (Model, AssistantMessage) {
    let mut chunks = String::new();
    for (index, arguments) in ARGUMENTS.iter().enumerate() {
        let (head, tail) = arguments.split_at(arguments.len() / 2);
        chunks.push_str(&chunk(&json!({"tool_calls": [{
            "index": index,
            "id": format!("call_{index}"),
            "function": {"name": "ipython", "arguments": head},
        }]})));
        chunks.push_str(&chunk(&json!({"tool_calls": [{
            "index": index,
            "function": {"arguments": tail},
        }]})));
    }
    let model = completions_model("gpt-oss-test", "intercept", 0.0, 0.0);
    let message =
        stream_final_message(model.clone(), format!("{chunks}{TOOLCALL_FINISH}{DONE}")).await;
    (replay_model(model), message)
}

fn replay_model(mut model: Value) -> Model {
    model["baseUrl"] = json!("http://127.0.0.1:0");
    serde_json::from_value(model).unwrap()
}

fn replayed_arguments(model: &Model, message: AssistantMessage) -> Vec<String> {
    let context = Context {
        system_prompt: None,
        messages: vec![crate::types::Message::Assistant(message)],
        tools: None,
    };
    let params = convert_messages(model, &context, &get_compat(model));
    params[0]["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|call| call["function"]["arguments"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn the_streamed_argument_text_is_kept_only_where_the_parse_differs() {
    let (_, message) = stream_calls().await;
    let calls: Vec<&ToolCall> = message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    let raw: Vec<Option<&str>> = calls
        .iter()
        .map(|call| call.raw_arguments.as_deref())
        .collect();
    assert_eq!(
        raw,
        [
            Some(ARGUMENTS[0]),
            Some(ARGUMENTS[1]),
            Some(ARGUMENTS[2]),
            None
        ]
    );
    // Execution still sees the parsed (or repaired) arguments.
    assert_eq!(
        calls[1].arguments,
        *json!({"code": "print(1)"}).as_object().unwrap()
    );
}

#[tokio::test]
async fn the_same_model_gets_its_argument_text_back_verbatim() {
    let (model, message) = stream_calls().await;
    assert_eq!(replayed_arguments(&model, message), ARGUMENTS);
}

#[tokio::test]
async fn another_model_gets_the_parsed_arguments() {
    let (_, message) = stream_calls().await;
    let other = replay_model(completions_model("other-model", "intercept", 0.0, 0.0));
    assert_eq!(
        replayed_arguments(&other, message),
        ["{}", ARGUMENTS[3], ARGUMENTS[3], ARGUMENTS[3]]
    );
}
