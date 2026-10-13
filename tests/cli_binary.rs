//! An integration-test target, so `cargo test` builds the `ccc` binary that
//! the CLI unit tests in `src/exec.rs` spawn from `target/<profile>/ccc`.

#[test]
fn cargo_builds_the_ccc_binary_for_the_tests() {
    let binary = std::path::Path::new(env!("CARGO_BIN_EXE_ccc"));
    assert!(binary.is_file(), "{} is missing", binary.display());
}
