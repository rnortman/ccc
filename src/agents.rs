//! Agent definitions: `<type>.md` files with YAML frontmatter and a prose body
//! that becomes the system prompt / developer instructions.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::de::{Deserializer, Error as _};
use serde::Deserialize;

use crate::config::{blank_to_none, Harness};

/// Bodies larger than this cannot be passed on the command line safely
/// (Linux `MAX_ARG_STRLEN` is 128 KiB).
pub const MAX_BODY_BYTES: usize = 100 * 1024;

/// Codex sandbox mode for an agent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Sandbox {
    ReadOnly,
    #[default]
    WorkspaceWrite,
}

impl Sandbox {
    pub fn as_str(self) -> &'static str {
        match self {
            Sandbox::ReadOnly => "read-only",
            Sandbox::WorkspaceWrite => "workspace-write",
        }
    }
}

/// The YAML header of a definition file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Frontmatter {
    name: String,
    description: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    effort: Option<String>,
    #[serde(default, deserialize_with = "deserialize_tools")]
    tools: Option<StringList>,
    #[serde(default, deserialize_with = "deserialize_harness")]
    harness: Option<StringList>,
    #[serde(default)]
    codex_model: Option<String>,
    #[serde(default)]
    codex_effort: Option<String>,
    #[serde(default)]
    codex_sandbox: Option<Sandbox>,
}

/// A string field may be comma-separated; a YAML list is also accepted.
#[derive(Debug, Clone)]
enum StringList {
    Csv(String),
    List(Vec<String>),
}

fn deserialize_tools<'de, D>(deserializer: D) -> std::result::Result<Option<StringList>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_string_list(deserializer, "tools")
}

fn deserialize_harness<'de, D>(deserializer: D) -> std::result::Result<Option<StringList>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_string_list(deserializer, "harness")
}

fn deserialize_string_list<'de, D>(
    deserializer: D,
    key: &'static str,
) -> std::result::Result<Option<StringList>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_yaml::Value::deserialize(deserializer)?;
    let wrong_type = || D::Error::custom(format!("`{key}` must be a string or a list of strings"));
    match value {
        serde_yaml::Value::Null => Ok(None),
        serde_yaml::Value::String(value) => Ok(Some(StringList::Csv(value))),
        serde_yaml::Value::Sequence(values) => Ok(Some(StringList::List(
            values
                .into_iter()
                .map(|value| match value {
                    serde_yaml::Value::String(value) => Ok(value),
                    _ => Err(wrong_type()),
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
        ))),
        _ => Err(wrong_type()),
    }
}

impl StringList {
    /// Normalizes entries, returning `None` when nothing is named.
    fn into_vec(self) -> Option<Vec<String>> {
        let values: Vec<String> = match self {
            StringList::Csv(s) => s.split(',').map(str::to_string).collect(),
            StringList::List(v) => v,
        };
        let values: Vec<String> = values
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect();
        (!values.is_empty()).then_some(values)
    }
}

fn parse_harnesses(value: StringList) -> Result<Option<Vec<Harness>>> {
    let Some(values) = value.into_vec() else {
        return Ok(None);
    };
    let mut harnesses = Vec::new();
    for value in values {
        let harness = Harness::parse(&value)?;
        if harnesses.contains(&harness) {
            bail!("duplicate harness {harness} in definition policy");
        }
        harnesses.push(harness);
    }
    Ok(Some(harnesses))
}

/// A parsed `<type>.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentDef {
    pub name: String,
    pub description: String,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub tools: Option<Vec<String>>,
    pub codex_model: Option<String>,
    pub codex_effort: Option<String>,
    pub codex_sandbox: Sandbox,
    /// Allowed harnesses; `Some` must be non-empty. Parsed empty input is `None`.
    pub harness: Option<Vec<Harness>>,
    pub body: String,
    pub overlays: BTreeMap<Harness, String>,
}

impl AgentDef {
    /// Loads `<dir>/<name>.md`.
    pub fn load(dir: &Path, name: &str) -> Result<AgentDef> {
        if name.is_empty() || name.contains('/') || name.contains("..") {
            bail!("invalid agent type {name:?}");
        }
        if let Some(harness) = Harness::ALL
            .iter()
            .find(|h| name.ends_with(&format!(".{h}")))
        {
            let suffix = format!(".{harness}");
            let base = &name[..name.len() - suffix.len()];
            bail!(
                "agent type {name:?} ends with reserved suffix {suffix:?}; {base}.claude.md / {base}.codex.md is an overlay of {base}.md"
            );
        }
        let path = dir.join(format!("{name}.md"));
        let text = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "no definition for agent type {name:?} (looked for {})",
                path.display()
            )
        })?;
        let mut def =
            AgentDef::parse(&text, name).with_context(|| format!("parsing {}", path.display()))?;
        for harness in Harness::ALL {
            let overlay_path = dir.join(format!("{name}.{harness}.md"));
            let overlay = match std::fs::read_to_string(&overlay_path) {
                Ok(text) => text,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    return Err(e)
                        .with_context(|| format!("reading overlay {}", overlay_path.display()))
                }
            };
            let overlay = overlay
                .strip_prefix('\u{feff}')
                .unwrap_or(&overlay)
                .trim()
                .to_string();
            if overlay.is_empty() {
                bail!("overlay {} is empty", overlay_path.display());
            }
            if overlay.len() > MAX_BODY_BYTES {
                bail!(
                    "overlay {} is {} bytes, over the {MAX_BODY_BYTES}-byte command-line limit",
                    overlay_path.display(),
                    overlay.len()
                );
            }
            if overlay
                .lines()
                .next()
                .is_some_and(|line| line.trim() == "---")
            {
                bail!(
                    "overlay {} carries no frontmatter; put keys in the base {}.md",
                    overlay_path.display(),
                    name
                );
            }
            def.overlays.insert(harness, overlay);
        }
        Ok(def)
    }

    /// Parses a definition; `stem` is the file name without `.md`.
    pub fn parse(text: &str, stem: &str) -> Result<AgentDef> {
        let (header, body) = split_frontmatter(text)?;
        let fm: Frontmatter = serde_yaml::from_str(header).context("invalid YAML frontmatter")?;

        let model = blank_to_none(fm.model);
        let effort = blank_to_none(fm.effort);
        let codex_model = blank_to_none(fm.codex_model);
        let codex_effort = blank_to_none(fm.codex_effort);
        let harness = fm.harness.map(parse_harnesses).transpose()?.flatten();

        if fm.name != stem {
            bail!(
                "frontmatter name {:?} does not match file stem {stem:?}",
                fm.name
            );
        }
        if let Some(allowed) = &harness {
            if !allowed.iter().any(|h| match h {
                Harness::Claude => model.is_some(),
                Harness::Codex => codex_model.is_some(),
            }) {
                let allowed_text = allowed
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                let missing = allowed
                    .iter()
                    .filter_map(|h| match h {
                        Harness::Claude if model.is_none() => Some("model"),
                        Harness::Codex if codex_model.is_none() => Some("codex_model"),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!(
                    "harness policy allows {allowed_text}, but no corresponding model is configured; eligible missing key(s): {missing}"
                );
            }
        } else if model.is_none() && codex_model.is_none() {
            bail!("at least one of `model` or `codex_model` is required");
        }
        let body = body.trim().to_string();
        if body.is_empty() {
            bail!("definition body (the system prompt) is empty");
        }
        if body.len() > MAX_BODY_BYTES {
            bail!(
                "definition body is {} bytes, over the {MAX_BODY_BYTES}-byte command-line limit",
                body.len()
            );
        }

        Ok(AgentDef {
            name: fm.name,
            description: fm.description,
            model,
            effort,
            tools: fm.tools.and_then(StringList::into_vec),
            codex_model,
            codex_effort,
            codex_sandbox: fm.codex_sandbox.unwrap_or_default(),
            harness,
            body,
            overlays: BTreeMap::new(),
        })
    }

    /// Whether the definition policy permits this harness (true when no policy).
    pub fn allows(&self, h: Harness) -> bool {
        self.harness
            .as_ref()
            .is_none_or(|allowed| allowed.contains(&h))
    }

    /// The first entry in the harness policy, or `None` when no policy is set.
    pub fn preferred_harness(&self) -> Option<Harness> {
        self.harness.as_ref().and_then(|h| h.first().copied())
    }

    /// The harness-specific overlay body, falling back to the base body.
    pub fn body_for(&self, h: Harness) -> &str {
        self.overlays
            .get(&h)
            .map(String::as_str)
            .unwrap_or(&self.body)
    }
}

/// Splits `---\n<yaml>\n---\n<body>`.
fn split_frontmatter(text: &str) -> Result<(&str, &str)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let rest = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
        .ok_or_else(|| anyhow::anyhow!("file does not start with a `---` frontmatter fence"))?;
    for (idx, line) in line_offsets(rest) {
        if line.trim_end() == "---" {
            let header = &rest[..idx];
            let after = &rest[idx + line.len()..];
            return Ok((header, after));
        }
    }
    bail!("frontmatter is not closed by a `---` line")
}

/// Yields `(byte offset, line with its terminator)` for each line.
fn line_offsets(s: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        if pos >= s.len() {
            return None;
        }
        let start = pos;
        let end = match s[pos..].find('\n') {
            Some(i) => pos + i + 1,
            None => s.len(),
        };
        pos = end;
        Some((start, &s[start..end]))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = "---\nname: explorer\ndescription: Investigates code.\nmodel: claude-opus-5[1M]\neffort: low\ntools: Read, Grep, Glob, Bash\ncodex_model: gpt-5.6-terra\ncodex_effort: low\ncodex_sandbox: read-only\n---\nYou are an explorer.\n\nCite files.\n";

    #[test]
    fn parses_a_full_definition() {
        let d = AgentDef::parse(FULL, "explorer").unwrap();
        assert_eq!(d.name, "explorer");
        assert_eq!(d.description, "Investigates code.");
        assert_eq!(d.model.as_deref(), Some("claude-opus-5[1M]"));
        assert_eq!(d.effort.as_deref(), Some("low"));
        assert_eq!(
            d.tools.as_deref(),
            Some(
                ["Read", "Grep", "Glob", "Bash"]
                    .map(String::from)
                    .as_slice()
            )
        );
        assert_eq!(d.codex_model.as_deref(), Some("gpt-5.6-terra"));
        assert_eq!(d.codex_effort.as_deref(), Some("low"));
        assert_eq!(d.codex_sandbox, Sandbox::ReadOnly);
        assert_eq!(d.body, "You are an explorer.\n\nCite files.");
    }

    #[test]
    fn parses_a_review_chain_style_definition_without_optional_keys() {
        let text = "---\nname: scribe\ndescription: Writes files.\nmodel: claude-fable-5-1[1m]\n---\nWrite what you are told.\n";
        let d = AgentDef::parse(text, "scribe").unwrap();
        assert_eq!(d.effort, None);
        assert_eq!(d.tools, None);
        assert_eq!(d.codex_model, None);
        assert_eq!(d.codex_sandbox, Sandbox::WorkspaceWrite);
    }

    #[test]
    fn tools_may_be_a_yaml_list() {
        let text =
            "---\nname: scribe\ndescription: d\nmodel: m\ntools:\n  - Read\n  - Write\n---\nbody\n";
        let d = AgentDef::parse(text, "scribe").unwrap();
        assert_eq!(d.tools.unwrap(), vec!["Read", "Write"]);
    }

    #[test]
    fn string_list_type_errors_name_their_field() {
        for (key, value) in [
            ("tools", "42"),
            ("tools", "{}"),
            ("harness", "42"),
            ("harness", "{}"),
        ] {
            let text =
                format!("---\nname: c\ndescription: d\nmodel: m\n{key}: {value}\n---\nbody\n");
            let err = format!("{:#}", AgentDef::parse(&text, "c").unwrap_err());
            assert!(
                err.contains(&format!("`{key}` must be a string or a list of strings")),
                "{err}"
            );
        }
    }

    #[test]
    fn codex_only_definition_is_valid() {
        let text = "---\nname: c\ndescription: d\ncodex_model: gpt-5.6-luna\n---\nbody\n";
        let d = AgentDef::parse(text, "c").unwrap();
        assert!(d.model.is_none());
        assert_eq!(d.codex_model.as_deref(), Some("gpt-5.6-luna"));
    }

    #[test]
    fn rejects_missing_models() {
        let text = "---\nname: c\ndescription: d\neffort: low\n---\nbody\n";
        let err = AgentDef::parse(text, "c").unwrap_err().to_string();
        assert!(err.contains("model"), "{err}");
    }

    #[test]
    fn rejects_name_stem_mismatch() {
        let err = AgentDef::parse(FULL, "explorer2").unwrap_err().to_string();
        assert!(err.contains("does not match file stem"), "{err}");
    }

    #[test]
    fn rejects_unknown_keys() {
        let text = "---\nname: c\ndescription: d\nmodel: m\ncodex_sandox: read-only\n---\nbody\n";
        assert!(AgentDef::parse(text, "c").is_err());
    }

    #[test]
    fn rejects_missing_fence_and_empty_body() {
        assert!(AgentDef::parse("name: c\n", "c").is_err());
        assert!(AgentDef::parse("---\nname: c\ndescription: d\nmodel: m\n", "c").is_err());
        assert!(
            AgentDef::parse("---\nname: c\ndescription: d\nmodel: m\n---\n   \n", "c").is_err()
        );
    }

    #[test]
    fn rejects_oversized_body() {
        let body = "x".repeat(MAX_BODY_BYTES + 1);
        let text = format!("---\nname: c\ndescription: d\nmodel: m\n---\n{body}\n");
        let err = AgentDef::parse(&text, "c").unwrap_err().to_string();
        assert!(err.contains("command-line limit"), "{err}");
    }

    #[test]
    fn body_keeps_internal_fences() {
        let text = "---\nname: c\ndescription: d\nmodel: m\n---\nintro\n\n---\n\nmore\n";
        let d = AgentDef::parse(text, "c").unwrap();
        assert_eq!(d.body, "intro\n\n---\n\nmore");
    }

    #[test]
    fn rejects_path_traversal_in_type_names() {
        let dir = Path::new("/nonexistent");
        assert!(AgentDef::load(dir, "../etc/passwd").is_err());
        assert!(AgentDef::load(dir, "").is_err());
    }

    #[test]
    fn crlf_and_a_byte_order_mark_still_parse() {
        let crlf = FULL.replace('\n', "\r\n");
        let d = AgentDef::parse(&crlf, "explorer").unwrap();
        assert_eq!(d.name, "explorer");
        assert_eq!(d.body, "You are an explorer.\r\n\r\nCite files.");

        let bom = format!("\u{feff}{FULL}");
        let d = AgentDef::parse(&bom, "explorer").unwrap();
        assert_eq!(d.body, "You are an explorer.\n\nCite files.");
    }

    #[test]
    fn a_blank_value_is_an_absent_value() {
        let text = "---\nname: c\ndescription: d\nmodel: \"\"\ncodex_model: gpt-5.6-luna\ncodex_effort: \"  \"\n---\nbody\n";
        let d = AgentDef::parse(text, "c").unwrap();
        assert_eq!(d.model, None);
        assert_eq!(d.codex_effort, None);
        let only_blank = "---\nname: c\ndescription: d\nmodel: \"\"\n---\nbody\n";
        let err = AgentDef::parse(only_blank, "c").unwrap_err().to_string();
        assert!(err.contains("model"), "{err}");
    }

    #[test]
    fn an_empty_tools_value_means_the_default_tool_set() {
        for value in ["tools: \"\"", "tools: \", ,\"", "tools: []"] {
            let text = format!("---\nname: c\ndescription: d\nmodel: m\n{value}\n---\nbody\n");
            let d = AgentDef::parse(&text, "c").unwrap();
            assert_eq!(d.tools, None, "{value}");
        }
        let text = "---\nname: c\ndescription: d\nmodel: m\ntools: Read, , Grep\n---\nbody\n";
        assert_eq!(
            AgentDef::parse(text, "c").unwrap().tools.unwrap(),
            vec!["Read", "Grep"]
        );
    }

    #[test]
    fn harness_policy_accepts_scalar_csv_and_list_forms() {
        for (value, expected) in [
            ("harness: CLAUDE", vec![Harness::Claude]),
            (
                "harness: ' codex, , CLAUDE '",
                vec![Harness::Codex, Harness::Claude],
            ),
            (
                "harness:\n  - CoDeX\n  - claude",
                vec![Harness::Codex, Harness::Claude],
            ),
        ] {
            let text = format!(
                "---\nname: c\ndescription: d\nmodel: m\ncodex_model: g\n{value}\n---\nbody\n"
            );
            let d = AgentDef::parse(&text, "c").unwrap();
            assert_eq!(d.harness, Some(expected.clone()));
            assert!(d.allows(Harness::Claude) == expected.contains(&Harness::Claude));
            assert_eq!(d.preferred_harness(), expected.first().copied());
        }
    }

    #[test]
    fn blank_or_empty_harness_policy_is_absent_and_bad_values_fail() {
        for value in ["harness: ''", "harness: ', ,'", "harness: []"] {
            let text = format!("---\nname: c\ndescription: d\nmodel: m\n{value}\n---\nbody\n");
            assert_eq!(AgentDef::parse(&text, "c").unwrap().harness, None);
        }
        let err = AgentDef::parse(
            "---\nname: c\ndescription: d\nmodel: m\nharness: unknown\n---\nbody\n",
            "c",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("unknown"), "{err}");
        let err = AgentDef::parse(
            "---\nname: c\ndescription: d\nmodel: m\nharness: claude, CLAUDE\n---\nbody\n",
            "c",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("duplicate harness claude"), "{err}");
    }

    #[test]
    fn harness_policy_generalizes_model_validation() {
        let codex_only = "---\nname: c\ndescription: d\nharness: codex\n---\nbody\n";
        let err = AgentDef::parse(codex_only, "c").unwrap_err().to_string();
        assert!(
            err.contains("codex") && err.contains("codex_model"),
            "{err}"
        );
        let codex_only = codex_only.replace("harness: codex", "harness: codex\ncodex_model: g");
        assert!(AgentDef::parse(&codex_only, "c").is_ok());
        let claude_only = "---\nname: c\ndescription: d\nharness: claude\n---\nbody\n";
        let err = AgentDef::parse(claude_only, "c").unwrap_err().to_string();
        assert!(err.contains("claude") && err.contains("model"), "{err}");
        let both = "---\nname: c\ndescription: d\nharness: claude, codex\n---\nbody\n";
        let err = AgentDef::parse(both, "c").unwrap_err().to_string();
        assert!(
            err.contains("claude, codex") && err.contains("model") && err.contains("codex_model"),
            "{err}"
        );
        assert!(AgentDef::parse(
            "---\nname: c\ndescription: d\nharness: claude, codex\nmodel: m\n---\nbody\n",
            "c"
        )
        .is_ok());
    }

    #[test]
    fn loads_and_normalizes_both_harness_overlays() {
        let dir = crate::testutil::TempDir::new("agents-overlays");
        dir.write_file(
            "c.md",
            "---\nname: c\ndescription: d\nmodel: m\n---\nbase\n",
            0o644,
        );
        dir.write_file("c.claude.md", "\u{feff}  claude overlay  \n", 0o644);
        dir.write_file("c.codex.md", "\n codex overlay\n\n", 0o644);
        let d = AgentDef::load(dir.path(), "c").unwrap();
        assert_eq!(d.body_for(Harness::Claude), "claude overlay");
        assert_eq!(d.body_for(Harness::Codex), "codex overlay");
        let no_codex =
            AgentDef::parse("---\nname: c\ndescription: d\nmodel: m\n---\nbase\n", "c").unwrap();
        assert_eq!(no_codex.body_for(Harness::Codex), "base");
    }

    #[test]
    fn loads_a_disallowed_overlay_for_resume_reconstruction() {
        let dir = crate::testutil::TempDir::new("policy-overlay");
        dir.write_file(
            "c.md",
            "---\nname: c\ndescription: d\nharness: codex\ncodex_model: g\n---\nbase\n",
            0o644,
        );
        dir.write_file("c.claude.md", "claude overlay\n", 0o644);
        let d = AgentDef::load(dir.path(), "c").unwrap();
        assert!(!d.allows(Harness::Claude));
        assert_eq!(d.body_for(Harness::Claude), "claude overlay");
    }

    #[test]
    fn rejects_invalid_overlays_with_path_context() {
        let cases = [("---\nkeys: no\n", "frontmatter"), ("  \n", "empty")];
        for (overlay, expected) in cases {
            let dir = crate::testutil::TempDir::new("bad-overlay");
            dir.write_file(
                "c.md",
                "---\nname: c\ndescription: d\nmodel: m\n---\nbase\n",
                0o644,
            );
            dir.write_file("c.claude.md", overlay, 0o644);
            let err = format!("{:#}", AgentDef::load(dir.path(), "c").unwrap_err());
            assert!(err.contains("c.claude.md"), "{err}");
            assert!(err.contains(expected), "{err}");
        }
        let dir = crate::testutil::TempDir::new("large-overlay");
        dir.write_file(
            "c.md",
            "---\nname: c\ndescription: d\nmodel: m\n---\nbase\n",
            0o644,
        );
        dir.write_file("c.claude.md", &"x".repeat(MAX_BODY_BYTES + 1), 0o644);
        let err = format!("{:#}", AgentDef::load(dir.path(), "c").unwrap_err());
        assert!(
            err.contains("c.claude.md") && err.contains("command-line limit"),
            "{err}"
        );
    }

    #[test]
    fn reports_unreadable_overlay_paths() {
        let dir = crate::testutil::TempDir::new("unreadable-overlay");
        dir.write_file(
            "c.md",
            "---\nname: c\ndescription: d\nmodel: m\n---\nbase\n",
            0o644,
        );
        std::fs::create_dir(dir.path().join("c.claude.md")).unwrap();
        let err = format!("{:#}", AgentDef::load(dir.path(), "c").unwrap_err());
        assert!(
            err.contains("reading overlay") && err.contains("c.claude.md"),
            "{err}"
        );
    }

    #[test]
    fn reserves_harness_suffixes_but_allows_other_dotted_names() {
        let temp = crate::testutil::TempDir::new("dotted-agent");
        for stem in ["review.claude", "review.codex"] {
            temp.write_file(
                &format!("{stem}.md"),
                &format!("---\nname: {stem}\ndescription: d\nmodel: m\n---\nbody\n"),
                0o644,
            );
            let err = AgentDef::load(temp.path(), stem).unwrap_err().to_string();
            assert!(
                err.contains("reserved suffix") && err.contains("overlay"),
                "{err}"
            );
            assert!(err.contains(&format!("{stem}.md")), "{err}");
        }
        temp.write_file(
            "review.v2.md",
            "---\nname: review.v2\ndescription: d\nmodel: m\n---\nbody\n",
            0o644,
        );
        temp.write_file("review.v2.claude.md", "overlay\n", 0o644);
        assert_eq!(
            AgentDef::load(temp.path(), "review.v2")
                .unwrap()
                .body_for(Harness::Claude),
            "overlay"
        );
    }

    #[test]
    fn loads_a_definition_from_a_directory() {
        let dir = crate::testutil::TempDir::new("agents-load");
        dir.write_file("explorer.md", FULL, 0o644);
        let d = AgentDef::load(dir.path(), "explorer").unwrap();
        assert_eq!(d.model.as_deref(), Some("claude-opus-5[1M]"));

        let err = format!("{:#}", AgentDef::load(dir.path(), "missing").unwrap_err());
        assert!(err.contains("no definition for agent type"), "{err}");
        assert!(err.contains("missing.md"), "{err}");

        dir.write_file("broken.md", "---\nname: broken\n---\nbody\n", 0o644);
        let err = format!("{:#}", AgentDef::load(dir.path(), "broken").unwrap_err());
        assert!(err.contains("broken.md"), "{err}");
    }

    #[test]
    fn the_shipped_definitions_parse() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("agents");
        let mut seen = 0;
        for entry in std::fs::read_dir(&dir).expect("reading agents/") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            if Harness::ALL
                .iter()
                .any(|h| stem.ends_with(&format!(".{h}")))
            {
                let base = stem.rsplit_once('.').unwrap().0;
                assert!(
                    dir.join(format!("{base}.md")).is_file(),
                    "{stem} has no unsuffixed base"
                );
                continue;
            }
            let d =
                AgentDef::load(&dir, &stem).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            assert!(!d.body.is_empty(), "{stem} has an empty body");
            assert!(
                d.model.is_some() || d.codex_model.is_some(),
                "{stem} names no model"
            );
            for (e, harness) in [
                (&d.effort, crate::config::Harness::Claude),
                (&d.codex_effort, crate::config::Harness::Codex),
            ] {
                let Some(e) = e else { continue };
                crate::config::Effort::parse_for(e, harness)
                    .unwrap_or_else(|err| panic!("{stem}: {err}"));
            }
            seen += 1;
        }
        assert!(seen >= 3, "expected the shipped stubs, found {seen}");
    }
}
