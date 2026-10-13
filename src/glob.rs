//! Minimal `*`-only glob matching.

/// Matches `pattern` against `text`, where `*` matches any run of characters
/// (including none) and every other character is literal. Case-insensitive.
#[cfg(test)]
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let t: Vec<char> = text.to_lowercase().chars().collect();
    match_lowered(pattern, &t)
}

fn match_lowered(pattern: &str, t: &[char]) -> bool {
    let p: Vec<char> = pattern.to_lowercase().chars().collect();

    // Greedy scan with backtracking to the last `*`.
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut resume) = (None, 0usize);
    while ti < t.len() {
        // A `*` in the pattern is always a wildcard, even against a literal
        // `*` in the text, so the wildcard branch is tried first.
        if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            resume = ti;
            pi += 1;
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            resume += 1;
            ti = resume;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// True if any pattern matches.
pub fn any_match<S: AsRef<str>>(patterns: &[S], text: &str) -> bool {
    let t: Vec<char> = text.to_lowercase().chars().collect();
    patterns.iter().any(|p| match_lowered(p.as_ref(), &t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_and_wildcard() {
        assert!(glob_match("seven_day", "seven_day"));
        assert!(!glob_match("seven_day", "seven_days"));
        assert!(glob_match("7-day_opus*", "7-day_opus_4_8"));
        assert!(glob_match("7-day_opus*", "7-day_opus"));
        assert!(!glob_match("7-day_opus*", "7-day_fable"));
        assert!(glob_match("*fable*", "claude-fable-5-1[1m]"));
        assert!(!glob_match("*fable*", "claude-opus-5[1M]"));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn case_insensitive_and_multi_star() {
        assert!(glob_match("*FABLE*", "claude-fable-5-1"));
        assert!(glob_match("a*b*c", "axxbyyc"));
        assert!(!glob_match("a*b*c", "axxbyy"));
        assert!(glob_match("a*a", "aaa"));
    }

    #[test]
    fn a_star_in_the_text_is_matched_by_a_wildcard() {
        assert!(glob_match("*b", "*ab"));
        assert!(glob_match("*", "*"));
        assert!(glob_match("7-day_*", "7-day_*scoped"));
        assert!(glob_match("a*c", "a*c"));
    }

    #[test]
    fn empty_pattern_and_empty_text() {
        assert!(glob_match("", ""));
        assert!(!glob_match("", "x"));
        assert!(glob_match("*", ""));
        assert!(!glob_match("x", ""));
    }

    #[test]
    fn any_of_several() {
        let pats = vec!["seven_day".to_string(), "7-day_opus*".to_string()];
        assert!(any_match(&pats, "7-day_opus_4_8"));
        assert!(any_match(&pats, "seven_day"));
        assert!(!any_match(&pats, "five_hour"));
        assert!(!any_match(&[] as &[String], "seven_day"));
    }
}
