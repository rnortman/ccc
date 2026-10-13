//! Agent type -> (harness, model, effort, tools/sandbox).
//!
//! Per field the first hit wins: CLI flag, per-agent env, `_ALL` env,
//! `[agents.<type>]`, `[agents."*"]`, the agent definition, the global default.

use anyhow::{bail, Context, Result};

use crate::agents::{AgentDef, Sandbox};
use crate::config::{
    blank_to_none, env_agent_key, env_override, env_value, AgentOverrides, Config, Effort,
    EnvSource, Harness,
};

/// Everything a harness driver needs about one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// `None` for a roleless launch.
    pub agent: Option<String>,
    pub description: String,
    pub harness: Harness,
    pub model: String,
    pub effort: Option<Effort>,
    /// Claude only; `None` means the harness default tool set.
    pub tools: Option<Vec<String>>,
    /// Codex only.
    pub sandbox: Sandbox,
    pub body: String,
}

impl Resolved {
    /// The agent type, or `-` for a roleless run.
    pub fn label(&self) -> &str {
        self.agent.as_deref().unwrap_or("-")
    }
}

/// Which rung of the routing ladder supplied the harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessSource {
    Cli,
    EnvAgent(String),
    EnvAll,
    ConfigAgent,
    ConfigWildcard,
    Definition,
    Default,
}

fn source_name(source: &HarnessSource, agent: &str) -> String {
    match source {
        HarnessSource::Cli => "--harness".to_string(),
        HarnessSource::EnvAgent(key) => key.clone(),
        HarnessSource::EnvAll => "CCC_HARNESS_ALL".to_string(),
        HarnessSource::ConfigAgent => format!("[agents.{agent}]"),
        HarnessSource::ConfigWildcard => "[agents.\"*\"]".to_string(),
        HarnessSource::Definition => "definition".to_string(),
        HarnessSource::Default => "default_harness".to_string(),
    }
}

fn allowed_text(def: &AgentDef) -> String {
    def.harness
        .as_ref()
        .map(|v| {
            v.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|| "claude, codex".to_string())
}

fn choose_harness(
    def: &AgentDef,
    h: Harness,
    source: HarnessSource,
    notes: &mut Vec<String>,
) -> Result<Option<(Harness, HarnessSource)>> {
    if def.allows(h) {
        return Ok(Some((h, source)));
    }
    let source_name = source_name(&source, &def.name);
    match source {
        HarnessSource::Cli => bail!(
            "agent {:?} rejects harness {h} from {source_name}; allowed harnesses: {}",
            def.name,
            allowed_text(def)
        ),
        HarnessSource::EnvAgent(_) => bail!(
            "agent {:?} rejects harness {h} from {source_name}; allowed harnesses: {}",
            def.name,
            allowed_text(def)
        ),
        HarnessSource::ConfigAgent => bail!(
            "agent {:?} rejects harness {h} from {source_name}; allowed harnesses: {}",
            def.name,
            allowed_text(def)
        ),
        HarnessSource::EnvAll => {
            notes.push(format!(
                "{source_name} selected {h} for agent {:?}, but allowed harnesses are: {}; skipped",
                def.name,
                allowed_text(def)
            ));
            Ok(None)
        }
        HarnessSource::ConfigWildcard => {
            notes.push(format!(
                "{source_name} selected {h} for agent {:?}, but allowed harnesses are: {}; skipped",
                def.name,
                allowed_text(def)
            ));
            Ok(None)
        }
        HarnessSource::Default => {
            notes.push(format!(
                "{source_name} selected {h} for agent {:?}, but allowed harnesses are: {}; skipped",
                def.name,
                allowed_text(def)
            ));
            Ok(None)
        }
        HarnessSource::Definition => {
            bail!("agent {:?} has disallowed definition harness {h}", def.name)
        }
    }
}

/// Resolves the harness using the explicit routing ladder.
pub fn resolve_harness(
    cfg: &Config,
    env: &dyn EnvSource,
    def: &AgentDef,
    cli_harness: Option<Harness>,
) -> Result<(Harness, HarnessSource, Vec<String>)> {
    let agent = def.name.as_str();
    let mut notes = Vec::new();
    if let Some(h) = cli_harness {
        let Some((h, s)) = choose_harness(def, h, HarnessSource::Cli, &mut notes)? else {
            bail!(
                "harness selection for agent {:?} unexpectedly had no allowed result",
                def.name
            );
        };
        return Ok((h, s, notes));
    }
    let agent_key = format!("CCC_HARNESS_{}", env_agent_key(agent));
    if let Some(v) = env_value(env, &agent_key) {
        let h = Harness::parse(&v).with_context(|| format!("harness from {agent_key}"))?;
        let Some((h, s)) = choose_harness(
            def,
            h,
            HarnessSource::EnvAgent(agent_key.clone()),
            &mut notes,
        )?
        else {
            bail!(
                "harness selection for agent {:?} unexpectedly had no allowed result",
                def.name
            );
        };
        return Ok((h, s, notes));
    }
    if let Some(v) = env_value(env, "CCC_HARNESS_ALL") {
        let h = Harness::parse(&v).with_context(|| "harness from CCC_HARNESS_ALL")?;
        if let Some(chosen) = choose_harness(def, h, HarnessSource::EnvAll, &mut notes)? {
            return Ok((chosen.0, chosen.1, notes));
        }
    }
    if let Some(h) = cfg.agents.get(agent).and_then(|o| o.harness) {
        let Some((h, s)) = choose_harness(def, h, HarnessSource::ConfigAgent, &mut notes)? else {
            bail!(
                "harness selection for agent {:?} unexpectedly had no allowed result",
                def.name
            );
        };
        return Ok((h, s, notes));
    }
    if let Some(h) = cfg.agents.get("*").and_then(|o| o.harness) {
        if let Some(chosen) = choose_harness(def, h, HarnessSource::ConfigWildcard, &mut notes)? {
            return Ok((chosen.0, chosen.1, notes));
        }
    }
    if let Some(h) = def.preferred_harness() {
        return Ok((h, HarnessSource::Definition, notes));
    }
    let h = cfg.default_harness;
    if let Some(chosen) = choose_harness(def, h, HarnessSource::Default, &mut notes)? {
        return Ok((chosen.0, chosen.1, notes));
    }
    bail!("agent {:?} has no allowed fallback harness", def.name)
}

/// One rung-by-rung lookup: per-agent env, `_ALL` env, `[agents.<type>]`,
/// `[agents."*"]`, the definition.
fn pick(
    cfg: &Config,
    env: &dyn EnvSource,
    agent: &str,
    env_suffix: &str,
    from_cfg: impl Fn(&AgentOverrides) -> Option<String>,
    from_def: impl FnOnce() -> Option<String>,
) -> Option<String> {
    env_override(env, env_suffix, agent)
        .or_else(|| {
            cfg.agent_overrides(agent)
                .into_iter()
                .find_map(|o| blank_to_none(from_cfg(o)))
        })
        .or_else(|| blank_to_none(from_def()))
}

/// Full resolution for `def`, whose `name` is the agent type.
pub fn resolve(
    cfg: &Config,
    env: &dyn EnvSource,
    def: &AgentDef,
    cli_harness: Option<Harness>,
) -> Result<(Resolved, Vec<String>)> {
    let agent = def.name.as_str();
    let (harness, _source, notes) = resolve_harness(cfg, env, def, cli_harness)?;

    let (model_env, effort_env) = match harness {
        Harness::Claude => ("MODEL", "EFFORT"),
        Harness::Codex => ("CODEX_MODEL", "CODEX_EFFORT"),
    };

    let model = pick(
        cfg,
        env,
        agent,
        model_env,
        |o| match harness {
            Harness::Claude => o.model.clone(),
            Harness::Codex => o.codex_model.clone(),
        },
        || match harness {
            Harness::Claude => def.model.clone(),
            Harness::Codex => def.codex_model.clone(),
        },
    );

    let Some(model) = model else {
        bail!(
            "agent type {agent:?} has no {harness} model: its definition names no `{}` and nothing overrides one",
            match harness {
                Harness::Claude => "model",
                Harness::Codex => "codex_model",
            }
        );
    };

    let effort = pick(
        cfg,
        env,
        agent,
        effort_env,
        |o| match harness {
            Harness::Claude => o.effort.clone(),
            Harness::Codex => o.codex_effort.clone(),
        },
        || match harness {
            Harness::Claude => def.effort.clone(),
            Harness::Codex => def.codex_effort.clone(),
        },
    );
    let effort = match effort {
        Some(e) => Some(
            Effort::parse_for(&e, harness)
                .with_context(|| format!("effort for agent type {agent:?}"))?,
        ),
        None => None,
    };

    Ok((
        Resolved {
            agent: Some(agent.to_string()),
            description: def.description.clone(),
            harness,
            model,
            effort,
            tools: def.tools.clone(),
            sandbox: def.codex_sandbox,
            body: def.body_for(harness).to_string(),
        },
        notes,
    ))
}

/// Resolution for a `ccc launch` with no agent type: Claude, with the model and
/// effort from the CLI, else `[launch]`, else the built-in default. The
/// per-agent routing ladder does not apply.
pub fn resolve_roleless(
    cfg: &Config,
    cli_model: Option<&str>,
    cli_effort: Option<Effort>,
) -> Result<Resolved> {
    let harness = Harness::Claude;
    let name = cli_model.unwrap_or(&cfg.launch.model);
    let model = cfg.launch.expand(name);
    if model.trim().is_empty() {
        if name.trim().is_empty() {
            bail!("[launch] model is blank");
        }
        bail!("[launch.aliases] {name} is blank");
    }
    let effort = match cli_effort {
        Some(e) => Some(e),
        None => match &cfg.launch.effort {
            Some(s) => Some(Effort::parse_for(s, harness).context("[launch] effort")?),
            None => None,
        },
    };
    Ok(Resolved {
        agent: None,
        description: String::new(),
        harness,
        model,
        effort,
        tools: None,
        sandbox: Sandbox::default(),
        body: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConfigPaths;
    use std::collections::BTreeMap;

    fn def() -> AgentDef {
        AgentDef::parse(
            "---\nname: explorer\ndescription: d\nmodel: def-claude\neffort: low\ntools: Read, Grep\ncodex_model: def-codex\ncodex_effort: minimal\ncodex_sandbox: read-only\n---\nbody\n",
            "explorer",
        )
        .unwrap()
    }

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn paths() -> ConfigPaths {
        ConfigPaths::in_env(&env(&[("HOME", "/home/u")]))
    }

    fn site_cfg(text: &str) -> Config {
        Config::from_texts(Some(text), None, &paths()).unwrap()
    }

    fn agents_cfg(text: &str) -> Config {
        Config::from_texts(None, Some(text), &paths()).unwrap()
    }

    #[test]
    fn roleless_precedence_is_cli_then_launch_config_then_built_in() {
        let r = resolve_roleless(&agents_cfg(""), None, None).unwrap();
        assert_eq!(r.model, "claude-opus-5-5[1M]");
        assert_eq!(r.effort, None);
        assert_eq!(r.agent, None);
        assert_eq!(r.harness, Harness::Claude);
        assert_eq!(r.body, "");
        assert_eq!(r.tools, None);

        let c = site_cfg("[launch]\nmodel = \"fable\"\neffort = \"medium\"\n");
        let r = resolve_roleless(&c, None, None).unwrap();
        assert_eq!(r.model, "claude-fable-5-1[1M]");
        assert_eq!(r.effort, Some(Effort::Medium));

        let r = resolve_roleless(&c, Some("opus"), Some(Effort::Max)).unwrap();
        assert_eq!(r.model, "claude-opus-5-5[1M]");
        assert_eq!(r.effort, Some(Effort::Max));

        let r = resolve_roleless(&c, Some("claude-sonnet-4-5"), None).unwrap();
        assert_eq!(r.model, "claude-sonnet-4-5");
    }

    #[test]
    fn roleless_effort_follows_the_agent_spelling_rules_and_names_its_key() {
        let r = resolve_roleless(&site_cfg("[launch]\neffort = \"High\"\n"), None, None).unwrap();
        assert_eq!(r.effort, Some(Effort::High));
        for bad in ["minimal", "lo"] {
            let c = site_cfg(&format!("[launch]\neffort = \"{bad}\"\n"));
            let err = format!("{:#}", resolve_roleless(&c, None, None).unwrap_err());
            assert!(err.contains("[launch] effort"), "{err}");
            let r = resolve_roleless(&c, None, Some(Effort::High)).unwrap();
            assert_eq!(r.effort, Some(Effort::High));
        }
    }

    #[test]
    fn a_blank_roleless_model_names_its_source() {
        let err = format!(
            "{:#}",
            resolve_roleless(&site_cfg("[launch]\nmodel = \"  \"\n"), None, None).unwrap_err()
        );
        assert!(err.contains("[launch] model"), "{err}");

        let via_config = site_cfg("[launch]\nmodel = \"fable\"\n[launch.aliases]\nfable = \"\"\n");
        let err = format!(
            "{:#}",
            resolve_roleless(&via_config, None, None).unwrap_err()
        );
        assert!(err.contains("[launch.aliases] fable"), "{err}");

        let blank_alias = site_cfg("[launch.aliases]\nfable = \"\"\n");
        let err = format!(
            "{:#}",
            resolve_roleless(&blank_alias, Some("fable"), None).unwrap_err()
        );
        assert!(err.contains("[launch.aliases] fable"), "{err}");

        let unused = blank_alias;
        let r = resolve_roleless(&unused, None, None).unwrap();
        assert_eq!(r.model, "claude-opus-5-5[1M]");
    }

    #[test]
    fn roleless_resolution_ignores_the_agent_population_overrides() {
        let c = agents_cfg(
            "default_harness = \"codex\"\n[agents.\"*\"]\nmodel = \"star\"\neffort = \"low\"\nharness = \"codex\"\n",
        );
        assert_eq!(
            resolve_roleless(&c, None, None).unwrap(),
            resolve_roleless(&agents_cfg(""), None, None).unwrap()
        );
    }

    #[test]
    fn falls_back_to_definition_and_default_harness() {
        let (r, _) = resolve(&agents_cfg(""), &env(&[]), &def(), None).unwrap();
        assert_eq!(r.harness, Harness::Claude);
        assert_eq!(r.model, "def-claude");
        assert_eq!(r.effort, Some(Effort::Low));
        assert_eq!(r.tools.unwrap(), vec!["Read", "Grep"]);
        assert_eq!(r.sandbox, Sandbox::ReadOnly);
        assert_eq!(r.body, "body");
    }

    #[test]
    fn codex_harness_uses_the_codex_fields() {
        let (r, _) = resolve(
            &agents_cfg("default_harness = \"codex\""),
            &env(&[]),
            &def(),
            None,
        )
        .unwrap();
        assert_eq!(r.model, "def-codex");
        assert_eq!(r.effort, Some(Effort::Minimal));
    }

    #[test]
    fn harness_precedence_cli_env_agent_wildcard_default() {
        let c =
            agents_cfg("default_harness = \"claude\"\n[agents.explorer]\nharness = \"codex\"\n");
        // CLI flag wins over everything.
        assert_eq!(
            resolve_harness(
                &c,
                &env(&[("CCC_HARNESS_EXPLORER", "codex")]),
                &def(),
                Some(Harness::Claude)
            )
            .unwrap(),
            (Harness::Claude, HarnessSource::Cli, Vec::new())
        );
        // Per-agent env beats _ALL env beats config.
        assert_eq!(
            resolve_harness(
                &c,
                &env(&[
                    ("CCC_HARNESS_EXPLORER", "claude"),
                    ("CCC_HARNESS_ALL", "codex")
                ]),
                &def(),
                None
            )
            .unwrap(),
            (
                Harness::Claude,
                HarnessSource::EnvAgent("CCC_HARNESS_EXPLORER".into()),
                Vec::new()
            )
        );
        assert_eq!(
            resolve_harness(&c, &env(&[("CCC_HARNESS_ALL", "claude")]), &def(), None).unwrap(),
            (Harness::Claude, HarnessSource::EnvAll, Vec::new())
        );
        // Config per-agent beats wildcard beats default.
        assert_eq!(
            resolve_harness(&c, &env(&[]), &def(), None).unwrap(),
            (Harness::Codex, HarnessSource::ConfigAgent, Vec::new())
        );
        let c2 = agents_cfg("default_harness = \"claude\"\n[agents.\"*\"]\nharness = \"codex\"\n");
        assert_eq!(
            resolve_harness(
                &c2,
                &env(&[]),
                &AgentDef::parse(
                    "---\nname: scribe\ndescription: d\nmodel: m\n---\nbody\n",
                    "scribe"
                )
                .unwrap(),
                None
            )
            .unwrap(),
            (Harness::Codex, HarnessSource::ConfigWildcard, Vec::new())
        );
        assert_eq!(
            resolve_harness(
                &agents_cfg(""),
                &env(&[]),
                &AgentDef::parse(
                    "---\nname: scribe\ndescription: d\nmodel: m\n---\nbody\n",
                    "scribe"
                )
                .unwrap(),
                None
            )
            .unwrap(),
            (Harness::Claude, HarnessSource::Default, Vec::new())
        );
    }

    #[test]
    fn every_higher_priority_rung_precedes_definition_preference() {
        let d = AgentDef::parse(
            "---\nname: explorer\ndescription: d\nharness: codex, claude\nmodel: m\ncodex_model: g\n---\nbody\n",
            "explorer",
        )
        .unwrap();
        let cases = [
            (
                vec![("CCC_HARNESS_EXPLORER", "claude")],
                "",
                HarnessSource::EnvAgent("CCC_HARNESS_EXPLORER".into()),
            ),
            (
                vec![("CCC_HARNESS_ALL", "claude")],
                "",
                HarnessSource::EnvAll,
            ),
            (
                vec![],
                "[agents.explorer]\nharness = \"claude\"",
                HarnessSource::ConfigAgent,
            ),
            (
                vec![],
                "[agents.\"*\"]\nharness = \"claude\"",
                HarnessSource::ConfigWildcard,
            ),
        ];
        for (environment, config, expected_source) in cases {
            let environment = env(&environment);
            let (harness, source, _) =
                resolve_harness(&agents_cfg(config), &environment, &d, None).unwrap();
            assert_eq!(harness, Harness::Claude);
            assert_eq!(source, expected_source);
        }
    }

    #[test]
    fn model_precedence_for_each_field() {
        let c = agents_cfg(
            "[agents.explorer]\nmodel = \"agent-claude\"\neffort = \"medium\"\ncodex_model = \"agent-codex\"\ncodex_effort = \"medium\"\n[agents.\"*\"]\nmodel = \"star-claude\"\ncodex_model = \"star-codex\"\n",
        );
        // Per-agent env beats config.
        let (r, _) = resolve(
            &c,
            &env(&[
                ("CCC_MODEL_EXPLORER", "env-claude"),
                ("CCC_EFFORT_EXPLORER", "high"),
            ]),
            &def(),
            None,
        )
        .unwrap();
        assert_eq!(r.model, "env-claude");
        assert_eq!(r.effort, Some(Effort::High));
        // _ALL env beats config.
        let (r, _) = resolve(&c, &env(&[("CCC_MODEL_ALL", "all-claude")]), &def(), None).unwrap();
        assert_eq!(r.model, "all-claude");
        // Config per-agent beats wildcard.
        let (r, _) = resolve(&c, &env(&[]), &def(), None).unwrap();
        assert_eq!(r.model, "agent-claude");
        assert_eq!(r.effort, Some(Effort::Medium));
        // Codex side, same ladder.
        let (r, _) = resolve(
            &c,
            &env(&[("CCC_CODEX_MODEL_EXPLORER", "env-codex")]),
            &def(),
            Some(Harness::Codex),
        )
        .unwrap();
        assert_eq!(r.model, "env-codex");
        let (r, _) = resolve(
            &c,
            &env(&[("CCC_CODEX_EFFORT_ALL", "xhigh")]),
            &def(),
            Some(Harness::Codex),
        )
        .unwrap();
        assert_eq!(r.model, "agent-codex");
        assert_eq!(r.effort, Some(Effort::XHigh));
    }

    #[test]
    fn an_unknown_effort_fails_before_any_spawn() {
        let err = format!(
            "{:#}",
            resolve(
                &agents_cfg(""),
                &env(&[("CCC_EFFORT_ALL", "lo")]),
                &def(),
                None
            )
            .unwrap_err()
        );
        assert!(err.contains("effort"), "{err}");
        assert!(err.contains("lo"), "{err}");

        let d = AgentDef::parse(
            "---\nname: a\ndescription: d\nmodel: m\neffort: lo\n---\nbody\n",
            "a",
        )
        .unwrap();
        let err = format!(
            "{:#}",
            resolve(&agents_cfg(""), &env(&[]), &d, None).unwrap_err()
        );
        assert!(err.contains("effort"), "{err}");
        assert!(err.contains("lo"), "{err}");
    }

    #[test]
    fn an_effort_the_resolved_harness_rejects_fails_before_any_spawn() {
        let err = format!(
            "{:#}",
            resolve(
                &agents_cfg(""),
                &env(&[("CCC_EFFORT_ALL", "minimal")]),
                &def(),
                None
            )
            .unwrap_err()
        );
        assert!(err.contains("claude"), "{err}");
        assert!(err.contains("minimal"), "{err}");

        let err = format!(
            "{:#}",
            resolve(
                &agents_cfg(""),
                &env(&[("CCC_CODEX_EFFORT_ALL", "max")]),
                &def(),
                Some(Harness::Codex)
            )
            .unwrap_err()
        );
        assert!(err.contains("codex"), "{err}");
        assert!(err.contains("max"), "{err}");
    }

    #[test]
    fn a_blank_override_does_not_shadow_the_definition() {
        let c = agents_cfg("[agents.\"*\"]\nmodel = \"\"\neffort = \"  \"\n");
        let (r, _) = resolve(&c, &env(&[]), &def(), None).unwrap();
        assert_eq!(r.model, "def-claude");
        assert_eq!(r.effort, Some(Effort::Low));

        let d = AgentDef::parse(
            "---\nname: a\ndescription: d\nmodel: m\ncodex_model: \"\"\n---\nbody\n",
            "a",
        )
        .unwrap();
        let err = resolve(&agents_cfg(""), &env(&[]), &d, Some(Harness::Codex))
            .unwrap_err()
            .to_string();
        assert!(err.contains("codex_model"), "{err}");
    }

    #[test]
    fn wildcard_applies_to_unlisted_agents() {
        let c = agents_cfg("[agents.\"*\"]\nmodel = \"star-claude\"\n");
        let (r, _) = resolve(&c, &env(&[]), &def(), None).unwrap();
        assert_eq!(r.model, "star-claude");
    }

    #[test]
    fn sentinel_env_values_do_not_override() {
        let c = agents_cfg("");
        let (r, _) = resolve(
            &c,
            &env(&[("CCC_MODEL_EXPLORER", "inherit"), ("CCC_EFFORT_ALL", "-")]),
            &def(),
            None,
        )
        .unwrap();
        assert_eq!(r.model, "def-claude");
        assert_eq!(r.effort, Some(Effort::Low));
    }

    #[test]
    fn definition_preference_beats_global_default() {
        let d = AgentDef::parse(
            "---\nname: explorer\ndescription: d\nharness: codex, claude\nmodel: m\ncodex_model: g\n---\nbody\n",
            "explorer",
        )
        .unwrap();
        let (r, _) = resolve(
            &agents_cfg("default_harness = \"claude\""),
            &env(&[]),
            &d,
            None,
        )
        .unwrap();
        assert_eq!(r.harness, Harness::Codex);
    }

    #[test]
    fn specific_harness_rungs_reject_disallowed_choices() {
        let d = AgentDef::parse(
            "---\nname: explorer\ndescription: d\nharness: claude\nmodel: m\n---\nbody\n",
            "explorer",
        )
        .unwrap();
        for (cfg_text, env_pairs, cli, source) in [
            ("", vec![], Some(Harness::Codex), "--harness"),
            (
                "",
                vec![("CCC_HARNESS_EXPLORER", "codex")],
                None,
                "CCC_HARNESS_EXPLORER",
            ),
            (
                "[agents.explorer]\nharness = \"codex\"",
                vec![],
                None,
                "[agents.explorer]",
            ),
        ] {
            let err = resolve_harness(&agents_cfg(cfg_text), &env(&env_pairs), &d, cli)
                .unwrap_err()
                .to_string();
            assert!(err.contains("explorer") && err.contains("codex"), "{err}");
            assert!(
                err.contains(source)
                    || (source == "[agents.explorer]" && err.contains("agents.explorer")),
                "{err}"
            );
            assert!(err.contains("claude"), "{err}");
        }
    }

    #[test]
    fn invalid_harness_environment_values_name_their_rung() {
        let d = def();
        for (key, value) in [
            ("CCC_HARNESS_EXPLORER", "not-a-harness"),
            ("CCC_HARNESS_ALL", "also-not-a-harness"),
        ] {
            let err = format!(
                "{:#}",
                resolve_harness(&agents_cfg(""), &env(&[(key, value)]), &d, None).unwrap_err()
            );
            assert!(err.contains(key) && err.contains(value), "{err}");
        }
    }

    #[test]
    fn population_harness_defaults_skip_with_notes() {
        let d = AgentDef::parse(
            "---\nname: explorer\ndescription: d\nharness: claude\nmodel: m\n---\nbody\n",
            "explorer",
        )
        .unwrap();
        let (h, source, notes) = resolve_harness(
            &agents_cfg("[agents.\"*\"]\nharness = \"codex\""),
            &env(&[("CCC_HARNESS_ALL", "codex")]),
            &d,
            None,
        )
        .unwrap();
        assert_eq!(h, Harness::Claude);
        assert_eq!(source, HarnessSource::Definition);
        assert_eq!(notes.len(), 2);
        assert!(notes[0].contains("CCC_HARNESS_ALL") && notes[0].contains("skipped"));
        assert!(notes[1].contains("[agents.\"*\"]") && notes[1].contains("skipped"));
    }

    #[test]
    fn ordered_two_harness_policy_accepts_either_allowed_choice() {
        let d = AgentDef::parse(
            "---\nname: explorer\ndescription: d\nharness: codex, claude\nmodel: m\ncodex_model: g\n---\nbody\n",
            "explorer",
        )
        .unwrap();
        assert_eq!(
            resolve_harness(&agents_cfg(""), &env(&[]), &d, Some(Harness::Claude))
                .unwrap()
                .0,
            Harness::Claude
        );
        assert_eq!(
            resolve_harness(&agents_cfg(""), &env(&[]), &d, Some(Harness::Codex))
                .unwrap()
                .0,
            Harness::Codex
        );
    }

    #[test]
    fn an_empty_mutable_harness_policy_returns_an_error() {
        let mut d = def();
        d.harness = Some(vec![]);
        let err = resolve_harness(&agents_cfg(""), &env(&[]), &d, None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("explorer") && err.contains("allowed fallback"),
            "{err}"
        );
    }

    #[test]
    fn selected_harness_overlay_body_is_resolved_with_base_fallback() {
        let mut d = def();
        d.overlays.insert(Harness::Codex, "codex body".into());
        let (r, _) = resolve(
            &agents_cfg("default_harness = \"codex\""),
            &env(&[]),
            &d,
            None,
        )
        .unwrap();
        assert_eq!(r.body, "codex body");
        let (r, _) = resolve(&agents_cfg(""), &env(&[]), &d, None).unwrap();
        assert_eq!(r.body, "body");
    }

    #[test]
    fn missing_model_for_the_resolved_harness_is_an_error() {
        let d = AgentDef::parse(
            "---\nname: claude-only\ndescription: d\nmodel: m\n---\nbody\n",
            "claude-only",
        )
        .unwrap();
        let err = resolve(&agents_cfg(""), &env(&[]), &d, Some(Harness::Codex))
            .unwrap_err()
            .to_string();
        assert!(err.contains("codex_model"), "{err}");
        // An override supplies it.
        let (r, _) = resolve(
            &agents_cfg(""),
            &env(&[("CCC_CODEX_MODEL_ALL", "gpt-5.6-luna")]),
            &d,
            Some(Harness::Codex),
        )
        .unwrap();
        assert_eq!(r.model, "gpt-5.6-luna");
    }

    #[test]
    fn effort_may_be_absent_everywhere() {
        let d =
            AgentDef::parse("---\nname: a\ndescription: d\nmodel: m\n---\nbody\n", "a").unwrap();
        let (r, _) = resolve(&agents_cfg(""), &env(&[]), &d, None).unwrap();
        assert_eq!(r.effort, None);
        assert_eq!(r.tools, None);
    }
}
