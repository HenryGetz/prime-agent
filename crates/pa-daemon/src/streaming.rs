//! Live token-stream coalescing: the emit path parks `message_update` frames in a
//! single-slot coalescer — a flusher emits at most one parked update per interval;
//! every other frame flushes the parked update first, so wire order stays identical.
//! The deltas are ADDITIVE, so the parked run merges delta text of one stream-event
//! kind; a kind switch or a block-end event flushes it.

use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Map};

use crate::engine::AssistantSnapshot;
use crate::protocol::{create_daemon_event_meta, DaemonOutbound};
use crate::worker::OutboundFrame;

/// One parked update flushes per interval; anything parked longer is a
/// stream stall, so this bounds both broadcast rate and update staleness.
pub(crate) const UPDATE_FLUSH_INTERVAL: Duration = Duration::from_millis(50);

/// The session identity a flushed frame is stamped with (the frame is
/// built at flush time, not at park time).
#[derive(Clone)]
struct SessionIdentity {
    active_session_id: String,
    generation: String,
}

/// The parked `message_update`: the newest snapshot plus the merged
/// additive delta run of the superseded frames.
struct PendingUpdate {
    message: AssistantSnapshot,
    /// The stream event kind of the parked run (`text_delta`, ...).
    kind: String,
    /// The concatenated delta text of every merged frame of the same kind.
    delta: String,
    /// The broadcast sequence the newest parked frame consumed.
    sequence: u64,
}

/// Single-slot coalescer for one turn's `message_update` frames. Shared
/// between the turn's emit path (park/direct-send) and the flusher task;
/// every broadcast happens under `inner`, so frame order is total.
pub(crate) struct TurnStreamCoalescer {
    inner: Mutex<CoalescerInner>,
    session: SessionIdentity,
}

struct CoalescerInner {
    /// The latest parked update. Replaced by newer snapshots of the same
    /// stream-event kind (their delta text merges into `delta`).
    pending: Option<PendingUpdate>,
    /// The turn ended: park no more, flush nothing, stop the flusher.
    closed: bool,
}

impl TurnStreamCoalescer {
    pub(crate) fn new(active_session_id: String, generation: String) -> Self {
        TurnStreamCoalescer {
            inner: Mutex::new(CoalescerInner {
                pending: None,
                closed: false,
            }),
            session: SessionIdentity {
                active_session_id,
                generation,
            },
        }
    }

    /// Park one streamed update: the newest snapshot wins; the delta merges into the
    /// parked run (same kind) or starts a fresh one after broadcasting the parked run.
    /// `false` when the turn already ended.
    pub(crate) fn park_update(
        &self,
        message: AssistantSnapshot,
        kind: &str,
        delta: &str,
        sequence: u64,
        events: &crate::worker::EventPump,
    ) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.closed {
            return false;
        }
        match &mut inner.pending {
            // Same kind: the newer snapshot wins and its delta text joins
            // the run (deltas are additive for delta-mapping consumers).
            Some(pending) if pending.kind == kind => {
                pending.delta.push_str(delta);
                pending.message = message;
                pending.sequence = sequence;
            }
            // Kind switch: the parked run's delta text belongs to another block
            // (a chunk can carry text and reasoning; a block can start while
            // another is open), so it goes out before the new run starts.
            _ => {
                self.flush_locked(&mut inner, events);
                inner.pending = Some(PendingUpdate {
                    message,
                    kind: kind.to_string(),
                    delta: delta.to_string(),
                    sequence,
                });
            }
        }
        true
    }

    /// Broadcast every `payloads` frame directly, after flushing any parked
    /// update first. All sends happen under the coalescer lock, so the flusher
    /// can never interleave between the parked update and its settling frame.
    pub(crate) fn send_direct(&self, payloads: &[Vec<u8>], events: &crate::worker::EventPump) {
        let mut inner = self.inner.lock().unwrap();
        self.flush_locked(&mut inner, events);
        for payload in payloads {
            events.send(OutboundFrame::session_event(payload.clone()));
        }
    }

    /// Flusher tick: broadcast the parked update when one is waiting.
    /// Returns `false` once the turn closed and the flusher should stop.
    pub(crate) fn flush_pending(&self, events: &crate::worker::EventPump) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.closed {
            return false;
        }
        self.flush_locked(&mut inner, events);
        true
    }

    /// End of turn: nothing parked after this point is broadcast (an aborted
    /// turn's stale partial must not appear after its settle events), and
    /// anything still parked is dropped.
    pub(crate) fn close(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        inner.pending = None;
    }

    /// Serialize and broadcast the parked update (the caller holds `inner`):
    /// the wire conversion runs here, once per flush.
    fn flush_locked(&self, inner: &mut CoalescerInner, events: &crate::worker::EventPump) {
        let Some(pending) = inner.pending.take() else {
            return;
        };
        let Some(message) = pending.message.into_wire() else {
            return;
        };
        let mut event = json!({
            "type": "message_update",
            "message": message,
        });
        if !pending.kind.is_empty() {
            let mut stream_event = json!({ "type": pending.kind });
            if !pending.delta.is_empty() {
                stream_event["delta"] = json!(pending.delta);
            }
            event["assistantMessageEvent"] = stream_event;
        }
        let meta = create_daemon_event_meta(
            &self.session.active_session_id,
            pending.sequence,
            None,
            Some(&self.session.generation),
        );
        let outbound = DaemonOutbound::SessionEvent {
            active_session_id: self.session.active_session_id.clone(),
            event,
            meta: Some(meta),
            rest: Map::default(),
        };
        let payload = serde_json::to_vec(&outbound).unwrap_or_default();
        events.send(OutboundFrame::session_event(payload));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event_payload(event: &Value) -> Vec<u8> {
        serde_json::to_vec(&json!({ "event": event })).unwrap_or_default()
    }

    fn subscribe() -> (
        crate::worker::EventPump,
        broadcast::Receiver<Arc<OutboundFrame>>,
    ) {
        let pump = crate::worker::EventPump::new();
        let rx = pump.subscribe();
        (pump, rx)
    }

    fn session_event_of(frame: &std::sync::Arc<OutboundFrame>) -> Value {
        serde_json::from_slice::<Value>(&frame.payload).expect("session event payload")
    }

    use serde_json::Value;
    use std::sync::Arc;
    use tokio::sync::broadcast;

    #[test]
    fn direct_sends_flush_the_parked_update_first() {
        let (pump, mut rx) = subscribe();
        let coalescer = TurnStreamCoalescer::new("s1".to_string(), "g1".to_string());
        assert!(coalescer.park_update(
            AssistantSnapshot::Wire(json!({ "text": "partial" })),
            "text_delta",
            "part",
            1,
            &pump
        ));
        // The parked update waits for the flusher until a direct frame
        // arrives: the direct send must broadcast the parked snapshot first.
        coalescer.send_direct(&[event_payload(&json!({ "type": "message_end" }))], &pump);
        let first = session_event_of(&rx.try_recv().unwrap());
        assert_eq!(first["event"]["type"], "message_update");
        assert_eq!(first["event"]["message"]["text"], "partial");
        let second = session_event_of(&rx.try_recv().unwrap());
        assert_eq!(second["event"]["type"], "message_end");
        assert!(rx.try_recv().is_err(), "no further frames");
    }

    #[test]
    fn a_newer_snapshot_merges_the_superseded_deltas() {
        let (pump, mut rx) = subscribe();
        let coalescer = TurnStreamCoalescer::new("s1".to_string(), "g1".to_string());
        assert!(coalescer.park_update(
            AssistantSnapshot::Wire(json!({ "text": "The" })),
            "text_delta",
            "The ",
            1,
            &pump
        ));
        assert!(coalescer.park_update(
            AssistantSnapshot::Wire(json!({ "text": "The Thames" })),
            "text_delta",
            "Thames ",
            2,
            &pump
        ));
        assert!(coalescer.park_update(
            AssistantSnapshot::Wire(json!({ "text": "The Thames flows" })),
            "text_delta",
            "flows",
            3,
            &pump
        ));
        assert!(coalescer.flush_pending(&pump));
        let flushed = session_event_of(&rx.try_recv().unwrap());
        assert_eq!(flushed["event"]["message"]["text"], "The Thames flows");
        assert_eq!(
            flushed["event"]["assistantMessageEvent"]["delta"],
            "The Thames flows"
        );
        // The flushed frame carries the newest frame's broadcast sequence.
        assert_eq!(flushed["meta"]["sequence"], 3);
        assert!(
            rx.try_recv().is_err(),
            "the superseded snapshots are merged"
        );
    }

    /// Every broadcast frame as `(event type, stream kind, delta)`.
    fn drain(rx: &mut broadcast::Receiver<Arc<OutboundFrame>>) -> Vec<(String, String, String)> {
        std::iter::from_fn(|| rx.try_recv().ok())
            .map(|frame| {
                let event = &session_event_of(&frame)["event"];
                let field = |value: &Value| value.as_str().unwrap_or_default().to_string();
                (
                    field(&event["type"]),
                    field(&event["assistantMessageEvent"]["type"]),
                    field(&event["assistantMessageEvent"]["delta"]),
                )
            })
            .collect()
    }

    fn frame(event_type: &str, kind: &str, delta: &str) -> (String, String, String) {
        (event_type.to_string(), kind.to_string(), delta.to_string())
    }

    #[test]
    fn a_kind_switch_flushes_the_parked_run() {
        // A chunk carrying both `content` and `reasoning_content` streams a text
        // delta, then a thinking delta, then more text, all inside one interval.
        let (pump, mut rx) = subscribe();
        let coalescer = TurnStreamCoalescer::new("s1".to_string(), "g1".to_string());
        for (sequence, (kind, delta)) in [
            ("text_delta", "I"),
            ("thinking_delta", "."),
            ("text_delta", " don't"),
            ("text_delta", " know"),
        ]
        .into_iter()
        .enumerate()
        {
            assert!(coalescer.park_update(
                AssistantSnapshot::Wire(json!({ "text": delta })),
                kind,
                delta,
                sequence as u64 + 1,
                &pump
            ));
        }
        coalescer.send_direct(&[event_payload(&json!({ "type": "turn_end" }))], &pump);
        assert_eq!(
            drain(&mut rx),
            vec![
                frame("message_update", "text_delta", "I"),
                frame("message_update", "thinking_delta", "."),
                frame("message_update", "text_delta", " don't know"),
                frame("turn_end", "", ""),
            ]
        );
    }

    #[test]
    fn close_stops_parking_and_flushing() {
        let (pump, mut rx) = subscribe();
        let coalescer = TurnStreamCoalescer::new("s1".to_string(), "g1".to_string());
        assert!(coalescer.park_update(
            AssistantSnapshot::Wire(json!({ "text": "partial" })),
            "text_delta",
            "p",
            1,
            &pump
        ));
        coalescer.close();
        assert!(!coalescer.park_update(
            AssistantSnapshot::Wire(json!({ "text": "stale" })),
            "text_delta",
            "s",
            2,
            &pump
        ));
        assert!(!coalescer.flush_pending(&pump), "the flusher stops");
        assert!(rx.try_recv().is_err(), "a closed turn parks nothing");
    }

    #[test]
    fn flush_without_parked_frames_is_a_kept_alive_noop() {
        let (pump, mut rx) = subscribe();
        let coalescer = TurnStreamCoalescer::new("s1".to_string(), "g1".to_string());
        assert!(coalescer.flush_pending(&pump));
        assert!(rx.try_recv().is_err());
    }
}
