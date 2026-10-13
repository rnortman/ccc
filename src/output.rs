//! Rendering a finished turn: the subagent's message plus a trailer line, or
//! one JSON object.

use std::path::{Path, PathBuf};

use serde_json::json;

use crate::config::{Effort, Harness};

/// Everything the caller is told about a completed turn.
#[derive(Debug, Clone)]
pub struct RunOutput {
    pub session: String,
    pub harness: Harness,
    pub model: String,
    pub effort: Option<Effort>,
    pub profile: Option<String>,
    pub agent: Option<String>,
    pub cwd: PathBuf,
    pub result: String,
    /// Harness-native usage and cost fields.
    pub native: serde_json::Value,
}

/// Placeholder for an absent optional field in the trailer.
const NONE: &str = "-";

impl RunOutput {
    /// The one-line summary that follows the message in text mode.
    pub fn trailer(&self) -> String {
        format!(
            "[ccc] session={} harness={} model={} effort={} profile={}",
            self.session,
            self.harness,
            self.model,
            self.effort.map(Effort::as_str).unwrap_or(NONE),
            self.profile.as_deref().unwrap_or(NONE),
        )
    }

    /// The message, a blank line, then the trailer.
    pub fn text(&self) -> String {
        format!("{}\n\n{}", self.result.trim_end(), self.trailer())
    }

    /// The structured form for `--json` output.
    pub fn json(&self) -> serde_json::Value {
        json!({
            "session": self.session,
            "harness": self.harness.as_str(),
            "model": self.model,
            "effort": self.effort.map(Effort::as_str),
            "profile": self.profile,
            "agent": self.agent,
            "cwd": path_str(&self.cwd),
            "result": self.result,
            "native": self.native,
        })
    }

    /// Text mode or `--json`, as a single string without a trailing newline.
    pub fn render(&self, as_json: bool) -> String {
        if as_json {
            serde_json::to_string_pretty(&self.json()).expect("rendering JSON output")
        } else {
            self.text()
        }
    }
}

fn path_str(p: &Path) -> String {
    p.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out() -> RunOutput {
        RunOutput {
            session: "9f22dd1b".into(),
            harness: Harness::Claude,
            model: "claude-haiku-4-5[1m]".into(),
            effort: Some(Effort::Low),
            profile: Some("team03".into()),
            agent: Some("explorer".into()),
            cwd: PathBuf::from("/srv/work"),
            result: "the report\n".into(),
            native: json!({"total_cost_usd": 0.0031}),
        }
    }

    #[test]
    fn trailer_format() {
        assert_eq!(
            out().trailer(),
            "[ccc] session=9f22dd1b harness=claude model=claude-haiku-4-5[1m] effort=low profile=team03"
        );
    }

    #[test]
    fn absent_effort_and_profile_render_as_dashes() {
        let mut o = out();
        o.effort = None;
        o.profile = None;
        o.harness = Harness::Codex;
        assert!(o.trailer().ends_with("effort=- profile=-"));
        assert!(o.trailer().contains("harness=codex"));
    }

    #[test]
    fn text_is_the_message_then_a_blank_line_then_the_trailer() {
        let o = out();
        let text = o.text();
        let (body, trailer) = text.rsplit_once("\n\n").unwrap();
        assert_eq!(body, "the report");
        assert_eq!(trailer, o.trailer());
    }

    #[test]
    fn json_shape() {
        let v = out().json();
        assert_eq!(v["session"], "9f22dd1b");
        assert_eq!(v["harness"], "claude");
        assert_eq!(v["effort"], "low");
        assert_eq!(v["profile"], "team03");
        assert_eq!(v["agent"], "explorer");
        assert_eq!(v["cwd"], "/srv/work");
        assert_eq!(v["result"], "the report\n");
        assert_eq!(v["native"]["total_cost_usd"], json!(0.0031));

        let mut o = out();
        o.effort = None;
        o.profile = None;
        o.agent = None;
        let v = o.json();
        assert!(v["effort"].is_null());
        assert!(v["profile"].is_null());
        assert!(v["agent"].is_null());
    }

    #[test]
    fn render_picks_the_mode() {
        let o = out();
        assert!(o.render(false).starts_with("the report"));
        assert!(o.render(true).starts_with('{'));
    }
}
