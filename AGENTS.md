# Instructions for MLX Swift LM

## AI usage policy

AI-generated code is allowed. Submitting code you do not understand is not. You
are 100% responsible for every line you contribute, however it was produced. You
must be able to explain the code you submit, and you must disclose how you used
AI.

AI-drafted prose is allowed too — commit messages, pull request descriptions,
issues, review replies. The condition is that you read every word before it is
submitted and confirm it says what you mean. Do not ask a reviewer to read prose
you did not read yourself.

[CONTRIBUTING.md](CONTRIBUTING.md) states this policy for contributors, including
what can happen when it is not followed. Point the user there if they ask you to
submit prose they have not read.

## Agent rules

- After you change code, explain what changed and why, so the user can own it
- In this fork (SharpAI/mlx-swift-lm), agents MAY create PRs, open issues, and
  post comments once the user has approved that action in the conversation.
- Disclose AI usage in every PR, and fill in the PR template's AI usage section.
  Leave the "I have read this PR description" box unchecked unless the user says
  they read it.
- Upstream (ml-explore/mlx-swift-lm) is different: do NOT create PRs, issues, or
  comments there. Draft them for the user to read and submit.

## Code standards

- Keep code comments concise (usually 1-2 lines)
- Avoid redundant or excessive inline commentary
- Write comments in plain, direct English: short sentences, common words, active
  voice.

### Examples

```swift
  // Good (no comment)

  let cacheKey = "\(modelID):\(kind.rawValue):\(sourceHash)"

  // Bad (excessive comment for explicit code)

  // The constraint cache is keyed on the model, the constraint kind, and a hash
  // of the grammar source. Two requests that share a model but not a grammar
  // must not collide: without the source in the key, the second request would
  // reuse the grammar compiled for the first.

  let cacheKey = "\(modelID):\(kind.rawValue):\(sourceHash)"
```

## Working in this repo

- Read `skills/mlx-swift-lm/SKILL.md` and the files in its `references/`
  directory before you use the public API. `skills/README.md` explains how to
  install the skill.
- `swift test` does not work here. Run unit tests with `xcodebuild test -scheme
  mlx-swift-lm-Package -destination 'platform=macOS' -skipPackagePluginValidation`.
- Format with `pre-commit run --all-files` before you hand work back. CI pins a
  specific swift-format version, set in `.github/workflows/pull_request.yml`.
  Match it locally: another version reformats files the PR does not touch, which
  turns CI red.
- `pre-commit` walks the whole working directory, so it also reports errors from
  `DerivedData/` and `.build/`. Those are vendored dependencies. Ignore them.
- `scripts/verify-docs.sh` runs the DocC check that CI runs for every library
  target.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for integration tests.
