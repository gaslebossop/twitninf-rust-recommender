# Contributing to TwitNinf Rust Recommender

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person

## Getting Started

### Prerequisites
- Rust 1.70+
- PostgreSQL 14+
- Redis 6.0+
- Git

### Setup

```bash
git clone https://github.com/yourusername/twitninf-rust-recommender.git
cd twitninf-rust-recommender

# Install dependencies
cargo build

# Copy environment template
cp .env.example .env
# Edit .env with your local credentials
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_sigmoid -- --nocapture
```

### Running Locally

```bash
RUST_LOG=twitninf_recommender=debug cargo run --release
```

## Development Guidelines

### Code Organization

- **Each dimension gets its own file** in `src/algorithm/dimensions/`
- **Utility functions** go in `src/utils/`
- **Constants** go in `src/constants.rs`
- **Models** go in `src/models/`

### Code Style

- Use `rustfmt` for formatting:
  ```bash
  cargo fmt
  ```

- Use `clippy` for linting:
  ```bash
  cargo clippy -- -D warnings
  ```

- Follow Rust naming conventions:
  - `CamelCase` for types
  - `snake_case` for functions and variables
  - `SCREAMING_SNAKE_CASE` for constants

### Adding a New Dimension

If adding a new scoring dimension:

1. Create `src/algorithm/dimensions/d{n}_{name}.rs`
2. Implement `pub fn calculate(...)` function
3. Add to `src/algorithm/dimensions/mod.rs`
4. Add constant weights to `src/constants.rs`
5. Add tests

Example:
```rust
pub fn calculate(t: &RawTweet, profile: &UserProfile) -> f64 {
    // Your dimension logic
    let score = /* calculation */;
    debug!(score, "D9 Final");
    score
}
```

### Performance Considerations

- Use `#[inline]` for small utility functions
- Avoid unnecessary allocations (clones, vecs)
- Prefer references over owned values
- Profile with flamegraph:
  ```bash
  cargo flamegraph --release --bin twitninf-recommender
  ```

### Logging

Use `tracing` macros:
```rust
trace!("Fine-grained debug info");  // Most verbose
debug!("Key decision points");       // Development
info!("Major operations");           // Production
warn!("Potential issues");
error!("Failures");
```

### Testing

Write tests for all public functions:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        let result = calculate(input);
        assert_eq!(result, expected);
    }
}
```

## Commit Messages

Use clear, descriptive commit messages:

```
feat: Add new dimension D9 for X

- Implemented calculate_d9 function
- Added weights to constants
- 95% test coverage

Closes #123
```

### Commit Message Format

- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code restructuring
- `perf:` Performance improvement
- `test:` Test additions/changes
- `docs:` Documentation
- `ci:` CI/CD changes

## Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and test locally
4. Run `cargo fmt` and `cargo clippy`
5. Commit with clear messages
6. Push to your fork
7. Create Pull Request with:
   - Clear title
   - Description of changes
   - Tests added/updated
   - Performance impact (if any)

### PR Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass (`cargo test`)
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] No performance regressions
- [ ] Commits are well-organized

## Reporting Bugs

Create an issue with:

- **Clear title** describing the bug
- **Steps to reproduce**
- **Expected behavior** vs **actual behavior**
- **Environment**: OS, Rust version, config
- **Logs** (with `RUST_LOG=debug`)

## Suggesting Enhancements

Create an issue with:

- **Motivation**: Why is this needed?
- **Proposed solution**: How would it work?
- **Alternative approaches**: Other options?
- **Impact**: Performance, complexity, benefits?

## Performance Standards

- **Recommendation latency**: < 500ms p99
- **Cache hit rate**: > 70%
- **Memory usage**: < 500MB
- **Test coverage**: > 80%

## Documentation

- Update README.md for user-facing changes
- Update LOGS_GUIDE.md for logging changes
- Add module-level comments (`//!`) to new files
- Keep comments focused on WHY, not WHAT

## Questions?

- Check existing issues/discussions first
- Create a discussion for questions
- Check documentation: README.md, LOGS_GUIDE.md

---

**Thank you for contributing!** 🚀
