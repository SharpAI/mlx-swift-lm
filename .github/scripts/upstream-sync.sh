#!/usr/bin/env bash
#
# Upstream sync bot (shared by SharpAI/mlx-swift and SharpAI/mlx-swift-lm).
#
# Merges the upstream default branch into this fork as a REAL merge commit,
# pushes it to a NEW branch and opens a pull request for a human to review.
#
# THE BOT NEVER APPROVES. The only pull-request call in this file is
# `gh pr create`: no review, no approve, no merge, no auto-merge, no label or
# comment on anyone's PR. (GitHub's repo checkbox is worded "create and approve
# pull requests" and cannot split the two, so the guarantee lives in this file:
# before adding any `gh` call here, check that it cannot approve, review or
# merge.) It also never force-pushes and never writes to a branch it did not
# just create.
#
# Outcome of a run:
#   green  nothing for a person to do right now
#            - the fork already contains upstream
#            - a sync PR is already open (by branch name, or because an open PR
#              already contains the upstream commit): someone is on it
#            - the bot opened the PR (review and merge it)
#   red    a person has to act (the failed run is the alert; Issues are off)
#            - the merge conflicts            -> nothing is pushed; the
#              conflicted files are in the log and the step summary
#            - the merge is clean but no PR could be opened (no token may do
#              it, or a PR for this branch was closed earlier) -> the branch
#              bot/upstream-<upstream sha7>-<base sha7> is on origin and the
#              summary has a one-click "open the PR" link
#            - anything else went wrong       -> ::error:: with the reason
#
# The branch name encodes both inputs, so one (upstream, base) pair is pushed at
# most once. A day later, with the same inputs, the bot finds the branch and
# does not push again; it opens the PR only if none was ever opened from that
# branch (a PR a person closed is a decision, not a failure, and is not reopened
# or duplicated). If main or upstream moved, the next push is a new branch.
#
# PR creation order: SWIFTLM_PR_TOKEN (PR_TOKEN, a personal token; the PR then
# starts CI normally), else GITHUB_TOKEN (needs Settings > Actions > General >
# "Allow GitHub Actions to create ... pull requests"; checks do NOT start on
# such a PR until it is closed and reopened by a person), else the link.
#
# .github/workflows is never taken from upstream (GITHUB_TOKEN cannot push
# workflow changes); the fork's version is restored before the merge commit.
#
# Safety: outside GitHub Actions the script is a dry run (reads from GitHub,
# writes nothing). Set DRY_RUN=0 to override, DRY_RUN=1 to force a dry run.
# A dry run exits 0 unless the merge conflicts; it prints what a real run would do.
#
# Required env:  UPSTREAM_REPO (e.g. ml-explore/mlx-swift-lm)
#                GITHUB_REPOSITORY (set by Actions; e.g. SharpAI/mlx-swift-lm)
# Tokens:        GH_TOKEN (Actions: secrets.GITHUB_TOKEN with contents: write for
#                the push and pull-requests: write for `gh pr create`)
#                PR_TOKEN (optional: a personal token with Pull requests: write,
#                tried first for `gh pr create`)
# Optional env:  BASE_BRANCH (main), BASE_REF (origin/$BASE_BRANCH),
#                UPSTREAM_REF (upstream/main), BRANCH_PREFIX (bot/upstream-)
#                WATCH_PATHS    space-separated paths worth a human look when a
#                               merge touches them (listed in the PR description)
#                KEEP_LINES     one "path::fixed string" per line; flagged if the
#                               base has the string in the file and the merge
#                               result does not (fork-only lines upstream may drop)
#                CONFLICT_HINTS free text appended to the conflict report
set -euo pipefail

UPSTREAM_REPO="${UPSTREAM_REPO:?set UPSTREAM_REPO, e.g. ml-explore/mlx-swift-lm}"
REPO="${GITHUB_REPOSITORY:?set GITHUB_REPOSITORY, e.g. SharpAI/mlx-swift-lm}"
BASE_BRANCH="${BASE_BRANCH:-main}"
BASE_REF="${BASE_REF:-origin/$BASE_BRANCH}"
UPSTREAM_REF="${UPSTREAM_REF:-upstream/main}"
BRANCH_PREFIX="${BRANCH_PREFIX:-bot/upstream-}"
WATCH_PATHS="${WATCH_PATHS:-}"
KEEP_LINES="${KEEP_LINES:-}"
CONFLICT_HINTS="${CONFLICT_HINTS:-}"
PR_TOKEN="${PR_TOKEN:-}"
# Keep the personal token out of every child's environment (git, perl, iconv,
# docker, `gh pr list`): it is handed to exactly one process, as
# `GH_TOKEN="$1" gh pr create`. Without this it is inherited by all of them, so
# anything that dumps or traces its environment would print a token that outlives
# the run. (Hygiene only: it does not hide it from a process that reads its
# parent's original environment block.)
export -n PR_TOKEN
MAX_LISTED=40
STALE_PR_DAYS=7
# Longest "open PR" link we pre-fill. github.com answers HTTP 500 (before it even
# routes the request) once the request line plus ALL request headers exceed about
# 7000 bytes, and a signed-in browser sends 0.7 KB of headers plus 2-4 KB of
# cookies. Measured with browser-like headers: largest working link = ~6480 minus
# the cookie size (3500 B cookie -> ~2975). Past this the link falls back to a
# shorter description, then to the title only (the full description stays in the
# step summary).
MAX_URL_LEN=2500

DRY_RUN="${DRY_RUN:-}"
if [ -z "$DRY_RUN" ] && [ "${GITHUB_ACTIONS:-}" != "true" ]; then
  DRY_RUN=1
fi
if [ "$DRY_RUN" = "0" ]; then
  DRY_RUN=""
fi

if [ -z "${GH_TOKEN:-}" ] && [ -n "${GITHUB_TOKEN:-}" ]; then
  export GH_TOKEN="$GITHUB_TOKEN"
fi

# The merge commit needs an identity, and a runner does not always have one.
BOT_NAME="github-actions[bot]"
BOT_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
export GIT_AUTHOR_NAME="$BOT_NAME" GIT_AUTHOR_EMAIL="$BOT_EMAIL"
export GIT_COMMITTER_NAME="$BOT_NAME" GIT_COMMITTER_EMAIL="$BOT_EMAIL"

# The bot must not be subject to the invoking clone's hooks (post-checkout,
# pre-commit, commit-msg, pre-push, ...) or signing setup (commit.gpgsign with
# gpg/ssh keys, merge.verifySignatures): none of it exists on a runner, and a
# local dry run must not die (or wait for a hardware key) because of it. These
# are appended to any GIT_CONFIG_COUNT entries the caller already has, and they
# apply to every git call below (worktree add, merge, commit, push).
git_env_config() { # <key> <value>
  local n="${GIT_CONFIG_COUNT:-0}"
  export "GIT_CONFIG_KEY_$n=$1" "GIT_CONFIG_VALUE_$n=$2" "GIT_CONFIG_COUNT=$((n + 1))"
}
git_env_config core.hooksPath /dev/null
git_env_config commit.gpgsign false
git_env_config merge.verifySignatures false
# The one push below must create exactly one ref. A user or repo level
# push.followTags=true would also push every annotated tag that the remote lacks
# and that the merge makes reachable (upstream's release tags), and
# push.recurseSubmodules=only would push no ref at all (yet exit 0). Neither is
# set on a hosted runner; a local DRY_RUN=0 run inherits whatever ~/.gitconfig says.
git_env_config push.followTags false
git_env_config push.recurseSubmodules no

log() { echo "[upstream-sync] $*"; }
summary() {
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    echo "$*" >> "$GITHUB_STEP_SUMMARY"
  fi
}
# Show a report file in the log (collapsed group) and in the step summary.
report() { # <log group title> <markdown file>
  echo "::group::$1"
  cat "$2"
  echo "::endgroup::"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    cat "$2" >> "$GITHUB_STEP_SUMMARY"
  fi
}
# Run a state-changing command, or only print it in a dry run.
write() {
  if [ -n "$DRY_RUN" ]; then
    echo "[dry-run] would run: $*"
    return 0
  fi
  "$@"
}

# Upstream commit text and file names are untrusted input for Markdown. Render
# them as an inline code span so they cannot become @mentions, #N / owner/repo#N
# backlinks, or (via a bare CR, which CommonMark treats as a line ending) forged
# headings and checkboxes. Control characters become spaces, backticks become
# apostrophes, long text is truncated.
md_code() { # <text> [max-chars]
  local s="$1" max="${2:-160}"
  # iconv -c first: invalid UTF-8 (a Latin-1 commit) must not reach the summary, and BSD tr/sed
  # fail on it. glibc iconv can exit 1 when it drops bytes, hence `|| true`. The tr/sed run
  # under LC_ALL=C so a UTF-8 locale cannot make them reject what is left.
  s="$(printf '%s' "$s" | { iconv -c -f UTF-8 -t UTF-8 2>/dev/null || true; } | LC_ALL=C tr '\000-\037\177' ' ' | LC_ALL=C tr '`' "'" | LC_ALL=C sed 's/  */ /g; s/^ //; s/ $//')"
  if [ "${#s}" -gt "$max" ]; then
    # In a non-UTF-8 locale bash counts and cuts bytes; iconv -c drops a torn
    # trailing multibyte sequence (glibc exits 1 on it, hence `|| true`).
    s="$(printf '%s' "${s:0:$((max - 1))}" | iconv -c -f UTF-8 -t UTF-8 2>/dev/null || true)…"
  fi
  [ -n "$s" ] || s='(empty)'
  printf '`%s`' "$s"
}
md_list() { # <items, one per line> [max-items]
  local items="$1" max="${2:-30}" n=0 line total
  total="$(printf '%s\n' "$items" | wc -l | tr -d ' ')"
  while IFS= read -r line; do
    [ "$n" -lt "$max" ] || break
    n=$((n + 1))
    printf -- '- %s\n' "$(md_code "$line" 200)"
  done <<< "$items"
  if [ "$total" -gt "$max" ]; then
    printf -- '- … and %s more\n' "$((total - max))"
  fi
}
commit_lines() { # <range> <max-commits>
  local h s an
  # The trailing "." field keeps a raw Latin-1 author name from being the last thing before the
  # newline: bash 5 read swallows the newline after an invalid UTF-8 lead byte and glues the next commit on.
  git log --no-merges -n "$2" --format='%h%x1f%s%x1f%an%x1f.' "$1" |
    while IFS=$'\x1f' read -r h s an _; do
      printf -- '- `%s` %s (%s)\n' "$h" "$(md_code "$s")" "$(md_code "$an" 60)"
    done
}
urlenc() { # stdin -> percent-encoded stdout (bytes, so UTF-8 stays valid)
  perl -pe 's/([^A-Za-z0-9_.~-])/sprintf("%%%02X", ord($1))/ge' 2>/dev/null
}

ORIG_DIR="$(pwd)"
TMP="$(mktemp -d)"
WORK="$TMP/worktree"
WORK_ADMIN=""
# No `git worktree prune` here: it deletes the admin entry of EVERY registered
# worktree whose directory is currently missing (an unmounted volume, say), not
# just ours, which a local dry run must never do. `worktree remove` already
# drops our own entry; if it fails, remove only that entry by hand.
cleanup() {
  cd "$ORIG_DIR" || true
  git worktree remove --force "$WORK" >/dev/null 2>&1 || {
    case "$WORK_ADMIN" in
      */worktrees/?*) rm -rf "$WORK_ADMIN" ;;
    esac
  }
  rm -rf "$TMP"
}
trap cleanup EXIT

if ! git remote get-url upstream >/dev/null 2>&1; then
  git remote add upstream "https://github.com/$UPSTREAM_REPO.git"
fi
git fetch --quiet origin "$BASE_BRANCH" || {
  echo "::error::cannot fetch branch '$BASE_BRANCH' from origin ($REPO); set BASE_BRANCH if the fork's default branch is not 'main'"
  exit 1
}
git fetch --quiet upstream

base_sha="$(git rev-parse --verify --quiet "$BASE_REF^{commit}")" || {
  echo "::error::$BASE_REF does not exist; check BASE_BRANCH/BASE_REF"
  exit 1
}
base_short="$(git rev-parse --short=7 "$base_sha")"
up_sha="$(git rev-parse --verify --quiet "$UPSTREAM_REF^{commit}")" || {
  echo "::error::$UPSTREAM_REF does not exist in $UPSTREAM_REPO; did it rename its default branch? set UPSTREAM_REF"
  exit 1
}
up_short="$(git rev-parse --short=7 "$up_sha")"
behind="$(git rev-list --count "$base_sha..$up_sha")"
log "$REPO $BASE_REF=$base_short, $UPSTREAM_REPO $UPSTREAM_REF=$up_short, upstream commits not in the fork: $behind${DRY_RUN:+ (dry run)}"

# -------------------------------------------------------- already synced ----

if [ "$behind" = "0" ]; then
  log "fork already contains $UPSTREAM_REF; nothing to do"
  summary "Fork is up to date with \`$UPSTREAM_REPO\` (\`$up_short\`)."
  exit 0
fi

# Someone is already carrying a sync PR: stay quiet. Only PRs into $BASE_BRANCH
# count (a PR into another branch cannot sync this one), and only when the head
# branch lives in THIS repo (push access = trusted). A PR from any other
# repository is ignored, the upstream repo's own main included: anyone can open
# a PR from upstream's main (or from their own fork, under any branch name) into
# this fork, such a PR always "contains the upstream commit" (and cannot take
# conflict fixes), and it would keep the bot quiet for as long as it stays open.
# A trusted PR counts when its branch is named like a sync branch, or when it
# already contains the upstream commit (whatever its branch is called).
if ! open_prs="$(gh pr list --repo "$REPO" --state open --base "$BASE_BRANCH" --limit 100 \
  --json number,headRefName,createdAt,isCrossRepository,headRepositoryOwner \
  --jq '.[] | [.number, .headRefName, (((now - (.createdAt | fromdateiso8601)) / 86400) | floor), (if .isCrossRepository then (.headRepositoryOwner.login // "(deleted)") else "(this repo)" end)] | @tsv' 2>"$TMP/gh.err")"; then
  cat "$TMP/gh.err"
  echo "::error::could not list open PRs (needs pull-requests: read and a GH_TOKEN); not guessing"
  exit 1
fi
sync_prs=""
stale_pr=""
while IFS=$'\t' read -r num head age owner; do
  [ -n "$num" ] || continue
  [ "$owner" = "(this repo)" ] || continue
  how=""
  case "$head" in
    sync/upstream-*|"$BRANCH_PREFIX"*) how="branch name" ;;
    *)
      if git fetch --quiet origin "refs/pull/$num/head" 2>"$TMP/pr-fetch.err"; then
        if git merge-base --is-ancestor "$up_sha" FETCH_HEAD 2>/dev/null; then
          how="contains $up_short"
        fi
      else
        echo "::warning::could not fetch refs/pull/$num/head, so PR #$num was not checked for the upstream commit: $(head -c 300 "$TMP/pr-fetch.err" | tr '\r\n' '  ')"
      fi
      ;;
  esac
  if [ -n "$how" ]; then
    sync_prs="$sync_prs #$num ($(md_code "$head" 100), open $age days, $how)"
    if [ "$age" -ge "$STALE_PR_DAYS" ]; then
      stale_pr="$stale_pr #$num"
    fi
  fi
done <<< "$open_prs"
if [ -n "$sync_prs" ]; then
  log "an upstream sync PR is already open, nothing to do:$sync_prs"
  summary "Upstream sync PR already open:$sync_prs. The fork is $behind commit(s) behind \`$UPSTREAM_REPO\` (\`$up_short\`)."
  if [ -n "$stale_pr" ]; then
    echo "::warning::upstream sync PR$stale_pr has been open for $STALE_PR_DAYS+ days; the fork is $behind commit(s) behind $UPSTREAM_REPO"
  fi
  exit 0
fi

# ----------------------------------------------------------------- merge ----

# Checked after the PR guard on purpose: with no shared history (upstream
# rewrote its history) a person has to merge by hand, and the sync PR they open
# must quiet the run like any other.
if ! merge_base="$(git merge-base "$base_sha" "$up_sha")"; then
  echo "::error::$BASE_REF and $UPSTREAM_REF share no history; refusing to merge unrelated histories (merge by hand with --allow-unrelated-histories and open a sync/upstream-* PR; the bot stays quiet while one is open)"
  exit 1
fi

git worktree add -q --detach "$WORK" "$base_sha"
WORK_ADMIN="$(git -C "$WORK" rev-parse --absolute-git-dir)"
cd "$WORK"
branch="${BRANCH_PREFIX}${up_short}-${base_short}"
compare_url="https://github.com/$UPSTREAM_REPO/compare/$(git rev-parse --short "$merge_base")...$up_short"

merge_rc=0
git merge --no-ff --no-commit "$up_sha" > "$TMP/merge.log" 2>&1 || merge_rc=$?
if [ "$merge_rc" -ne 0 ] && [ -z "$(git diff --name-only --diff-filter=U)" ]; then
  cat "$TMP/merge.log"
  echo "::error::git merge failed without conflicts (see output above)"
  exit 1
fi

# The fork's .github/workflows is restored wholesale BEFORE conflicts are
# judged: the bot never applies upstream workflow changes (GITHUB_TOKEN cannot
# push a commit that touches .github/workflows/*), so a conflict there (say a
# modify/delete on an upstream workflow the fork does not carry) is moot and
# must not block the sync. The `rm -rf` matters: `git checkout <ref> -- dir`
# leaves files that exist only on the upstream side in place, and `git add -A`
# is what clears the unmerged index entries of those removed paths.
workflow_changes="$(git -c core.quotepath=false diff --name-only "$merge_base" "$up_sha" -- .github/workflows)"
workflow_conflicts="$(git -c core.quotepath=false diff --name-only --diff-filter=U -- .github/workflows)"
rm -rf .github/workflows
if git cat-file -e "$base_sha:.github/workflows" 2>/dev/null; then
  git checkout "$base_sha" -- .github/workflows
fi
git add -A -- .github/workflows 2>/dev/null || true

conflicted="$(git -c core.quotepath=false diff --name-only --diff-filter=U)"
if [ -n "$conflicted" ]; then
  git merge --abort 2>/dev/null || true
  n_conf="$(printf '%s\n' "$conflicted" | wc -l | tr -d ' ')"
  log "merge has $n_conf conflicted file(s); nothing will be pushed"
  {
    echo "## Upstream sync blocked: $n_conf file(s) conflict"
    echo
    echo "The daily upstream sync could not merge \`$UPSTREAM_REPO\` main (\`$up_short\`, $behind commit(s) not yet in this fork) into \`$BASE_BRANCH\`. The bot never resolves conflicts and pushed nothing."
    echo
    echo "Upstream changes: $compare_url"
    echo
    echo "Conflicted files:"
    md_list "$conflicted" 60
    if printf '%s\n' "$conflicted" | grep -qE '~[0-9a-f]{40}$'; then
      echo
      echo "Entries like \`path~<sha>\` are directory/file conflicts: one side has a directory (or vendored tree) where the other has a file or submodule link."
    fi
    echo
    echo "How to resolve (real merge only, never cherry-pick or \`-X ours/theirs\`):"
    echo '```'
    echo "git remote add upstream https://github.com/$UPSTREAM_REPO.git   # once"
    echo "git fetch origin && git fetch upstream"
    echo "git switch -c sync/upstream-$up_short-merge origin/$BASE_BRANCH"
    echo "git merge upstream/main        # resolve the conflicts by hand"
    echo "# silent-revert check: for each file upstream changed, 'git diff upstream/main HEAD --numstat -- <file>'"
    echo "# must show only differences that are intentional fork behavior"
    echo '```'
    echo "Open the PR into \`$BASE_BRANCH\` from a branch of this repository named \`sync/upstream-...\` (the bot stays quiet while one is open) and merge it with a **merge commit** (not squash/rebase) so downstream submodule pins stay valid."
    echo
    echo "If an earlier sync PR was squash- or rebase-merged, upstream is not an ancestor of \`$BASE_BRANCH\` and the same conflicts keep coming back: resolve them once more with the real merge above and merge that PR with a merge commit."
    if [ -n "$workflow_conflicts" ]; then
      echo
      echo "Upstream also changed workflow files that conflict with the fork's; the bot ignores workflow files, so these are not in the list above:"
      md_list "$workflow_conflicts" 20
    fi
    if [ -n "$CONFLICT_HINTS" ]; then
      echo
      echo "Known fork-specific spots:"
      printf '%s\n' "$CONFLICT_HINTS"
    fi
    echo
    echo "_Maintained by the upstream-sync workflow. This run stays red while the fork is behind and the merge conflicts; it turns green once a sync PR is open or the fork contains upstream._"
  } > "$TMP/blocked.md"
  report "Upstream sync blocked: $UPSTREAM_REPO main conflicts with the fork ($n_conf files)" "$TMP/blocked.md"
  echo "::error::upstream sync blocked: merging $UPSTREAM_REPO $up_short into $BASE_BRANCH conflicts in $n_conf file(s); nothing was pushed"
  exit 1
fi

git commit -q \
  -m "Merge $UPSTREAM_REPO main ($up_short) into $BASE_BRANCH" \
  -m "Automated merge by the upstream-sync workflow. $behind upstream commit(s). Merge with a merge commit, not squash/rebase."
if [ "$(git rev-list --parents -n 1 HEAD | wc -w | tr -d ' ')" != "3" ]; then
  echo "::error::expected a two-parent merge commit"
  exit 1
fi
merged_sha="$(git rev-parse HEAD)"
# Check the RESULT of the workflow restore, not the steps meant to produce it:
# `git add -A` re-applies the clean filter, so an upstream .gitattributes
# (`* text eol=lf`, `* ident`, a filter) can rewrite the bytes of the restored
# workflow files. The merge commit must be byte-identical to the base there.
if ! git diff --quiet "$base_sha" "$merged_sha" -- .github/workflows; then
  echo "::error::refusing to push: the merge commit changes .github/workflows relative to $BASE_REF (an upstream .gitattributes rewriting the restored files?); the bot never touches workflow files. Merge by hand."
  exit 1
fi

# ----------------------------------------------------------------- checks ----

# Git can merge cleanly into invalid Swift (two nearby edits combined into a
# dropped signature or an orphaned body). Parse the Swift files the merge
# changed. Not a build (no Metal toolchain here), only a structure check. The
# check is advisory: if docker itself fails, the run says so and carries on.
syntax_failed=""
syntax_skipped=""
# NUL-delimited (-z): without it git C-quotes names that contain a double quote, a backslash, TAB or LF
# (as "a\"b.swift", with the quotes), swiftc cannot open the quoted text and a valid
# file is reported as a syntax error. Counting NULs also keeps a name with a newline at one.
swift_z() { git -c core.quotepath=false diff -z --name-only --diff-filter=ACMR "$base_sha" HEAD -- '*.swift'; }
n_swift="$(swift_z | tr -cd '\000' | wc -c | tr -d ' ')"
if [ "$n_swift" = "0" ]; then
  syntax_note="no Swift files changed"
elif [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  # The container's exit status is 0 whenever the loop ran (a parse failure only
  # echoes the file name), so non-zero means docker itself failed (daemon, image
  # pull, rate limit, OOM kill). "./" keeps a name that starts with "-" from
  # being read as a swiftc option.
  if syntax_failed="$(swift_z | docker run --rm -i -v "$PWD:/src" -w /src swift:6.2 bash -c '
      while IFS= read -r -d "" f; do
        swiftc -parse -suppress-warnings "./$f" >&2 || echo "$f"
      done')"; then
    syntax_note="$n_swift changed Swift file(s) parsed with swiftc 6.2"
  else
    docker_rc=$?
    syntax_failed=""
    syntax_skipped=1
    syntax_note="**SKIPPED**: \`docker run swift:6.2\` failed (exit $docker_rc), so the changed Swift files were NOT parsed"
    echo "::warning::Swift syntax check skipped: 'docker run swift:6.2' failed (exit $docker_rc); the check is advisory"
  fi
elif command -v swiftc >/dev/null 2>&1; then
  syntax_failed="$(swift_z | while IFS= read -r -d "" f; do
    swiftc -parse -suppress-warnings "./$f" >&2 || echo "$f"
  done)"
  syntax_note="$n_swift changed Swift file(s) parsed with local swiftc"
else
  syntax_note="skipped (no swiftc and not running in Actions)"
fi

watched=""
if [ -n "$WATCH_PATHS" ]; then
  # shellcheck disable=SC2086
  watched="$(git -c core.quotepath=false diff --name-only "$base_sha" HEAD -- $WATCH_PATHS)"
fi

# Fork-only lines the merge must not lose (e.g. the local path dependency).
lost_lines=""
while IFS= read -r spec; do
  [ -n "$spec" ] || continue
  keep_file="${spec%%::*}"
  keep_text="${spec#*::}"
  base_text="$(git show "$base_sha:$keep_file" 2>/dev/null || true)"
  if grep -qF -- "$keep_text" <<< "$base_text" && ! grep -qF -- "$keep_text" "$keep_file" 2>/dev/null; then
    lost_lines="$lost_lines$keep_file :: $keep_text"$'\n'
  fi
done <<< "$KEEP_LINES"
lost_lines="${lost_lines%$'\n'}"

removed="$(git -c core.quotepath=false diff --name-only --diff-filter=D "$base_sha" HEAD)"
shortstat="$(git diff --shortstat "$base_sha" HEAD | sed 's/^ *//')"
empty_merge=""
if [ -z "$shortstat" ]; then
  empty_merge=1
  shortstat="no file changes outside .github/workflows"
fi

# ------------------------------------------------------------- PR text ----

n_listed="$(git rev-list --no-merges --count "$base_sha..$up_sha")"
commit_list="$(commit_lines "$base_sha..$up_sha" "$MAX_LISTED")"
if [ "$n_listed" -gt "$MAX_LISTED" ]; then
  commit_list="$commit_list"$'\n'"- … and $((n_listed - MAX_LISTED)) more (see the compare link)"
fi

{
  if [ -n "$empty_merge" ]; then
    echo "- **This merge changes no file outside \`.github/workflows\`**: either upstream only touched workflows, or an earlier sync PR was squash- or rebase-merged and \`$BASE_BRANCH\` already has upstream's content without upstream's commits. Merge THIS PR with a merge commit; that records upstream as an ancestor and ends the daily red runs."
  fi
  if [ -n "$syntax_failed" ]; then
    echo "- **Swift syntax check failed** for:"
    md_list "$syntax_failed" 20 | sed 's/^/  /'
  fi
  if [ -n "$syntax_skipped" ]; then
    echo "- The Swift syntax check could not run (docker failure, see the workflow log); rely on CI or parse the changed Swift files yourself."
  fi
  if [ -n "$lost_lines" ]; then
    echo "- A fork-only line is **gone** after the merge:"
    md_list "$lost_lines" 10 | sed 's/^/  /'
  fi
  if [ -n "$watched" ]; then
    echo "- Fork-sensitive paths changed by this merge:"
    md_list "$watched" 30 | sed 's/^/  /'
  fi
  if [ -n "$removed" ]; then
    echo "- The merge deletes files that upstream deleted:"
    md_list "$removed" 15 | sed 's/^/  /'
  fi
  if [ -n "$workflow_changes" ]; then
    echo "- Upstream also changed workflow files; the bot **does not apply** them (the default token cannot push workflow changes), apply by hand if wanted:"
    md_list "$workflow_changes" 15 | sed 's/^/  /'
  fi
  if [ -n "$workflow_conflicts" ]; then
    echo "- Of these, the following conflicted with the fork's version and were settled by keeping the fork's:"
    md_list "$workflow_conflicts" 15 | sed 's/^/  /'
  fi
} > "$TMP/flags.md"

pr_body() { # <1: include the commit list | 0: leave it out (it is in the PR anyway)>
  echo "Automated **real merge** of \`$UPSTREAM_REPO\` main (\`$up_short\`, $behind commit(s) not yet in this fork) into \`$BASE_BRANCH\`. Upstream changes: $compare_url"
  echo
  echo "**Merge this PR with a merge commit (\"Create a merge commit\"), not squash or rebase.** Squashing orphans the upstream commits and the fork commits that downstream repos pin (submodule SHAs)."
  echo
  if [ "$1" = "2" ]; then
    # Fallback when even the short text does not fit in a link: keep the merge
    # warning, the disclosure and the checkbox; the flags and checks stay in the
    # run summary.
    echo "The flags to look at and the checks that were run are in the step summary of the \`upstream-sync\` workflow run${run_url:+ ($run_url)}; they are too long for a link."
    echo
    echo "### Bot disclosure"
    echo "Prepared by the \`upstream-sync\` GitHub Actions workflow. The workflow never approves, reviews or merges; no human has reviewed this merge yet."
    echo
    echo "- [ ] I have read this PR description in full and approve it as my own"
    return 0
  fi
  if [ "$1" = "1" ]; then
    echo "### Upstream commits"
    printf '%s\n' "$commit_list"
    echo
  fi
  echo "### Needs a human look"
  if [ -s "$TMP/flags.md" ]; then
    cat "$TMP/flags.md"
  else
    echo "- Nothing flagged."
  fi
  echo
  echo "### What was checked"
  if [ -n "$workflow_conflicts" ]; then
    echo "- Git merged with no conflicts outside \`.github/workflows\` (no \`-X ours/theirs\`); the fork's workflows were restored unchanged."
  else
    echo "- Git merged cleanly (no conflicts, no \`-X ours/theirs\`); the fork's workflows were restored unchanged."
  fi
  echo "- Size: $shortstat."
  echo "- Syntax: $syntax_note."
  echo "- Not run by the bot: a build, the tests, a Metal toolchain. CI on this PR must be green before merging."
  echo
  echo "### Bot disclosure"
  echo "Prepared by the \`upstream-sync\` GitHub Actions workflow. The workflow never approves, reviews or merges; no human has reviewed this merge yet."
  echo
  echo "- [ ] I have read this PR description in full and approve it as my own"
}

pr_title="🔄 Auto-Sync: $UPSTREAM_REPO main ($up_short, $behind commit(s))"
if [ -n "$syntax_failed" ]; then
  pr_title="🔄⚠️ Auto-Sync: $UPSTREAM_REPO main ($up_short): SYNTAX ERRORS, do not merge as-is"
fi
run_url=""
if [ -n "${GITHUB_RUN_ID:-}" ]; then
  run_url="${GITHUB_SERVER_URL:-https://github.com}/$REPO/actions/runs/$GITHUB_RUN_ID"
fi
pr_body 1 > "$TMP/pr-full.md"
pr_body 0 > "$TMP/pr-short.md"
pr_body 2 > "$TMP/pr-min.md"

# One click opens the PR form with title and description filled in. The short
# description keeps the URL under GitHub's length limit; the full text is in the
# step summary. If even that is too long (many flagged files), the minimal
# description (merge warning, disclosure, checkbox) is used, and only then the
# title alone. url_note says which one the link carries.
plain_url="https://github.com/$REPO/compare/$BASE_BRANCH...$branch?quick_pull=1"
open_url="$plain_url"
url_note="nothing is pre-filled, perl is missing; copy the title and description from the block below"
if command -v perl >/dev/null 2>&1; then
  title_enc="$(printf '%s' "$pr_title" | urlenc)"
  open_url="$plain_url&title=$title_enc"
  url_note="only the title is pre-filled, the description is too long for a link; copy it from the block below"
  for kind in short min; do
    candidate="$plain_url&title=$title_enc&body=$(urlenc < "$TMP/pr-$kind.md")"
    if [ "${#candidate}" -le "$MAX_URL_LEN" ]; then
      open_url="$candidate"
      if [ "$kind" = short ]; then
        url_note="title and description are pre-filled"
      else
        url_note="title and a short description are pre-filled, the flags are too long for a link; copy the full description from the block below"
      fi
      break
    fi
  done
fi

# ------------------------------------------------------------------- push ----

# Only ever CREATE the branch: no force, no update. The name encodes the two
# commits that were merged, so an existing branch should be the same sync; that
# is verified below (same tree as this run's merge), not assumed.
ls_rc=0
git ls-remote --exit-code --heads origin "refs/heads/$branch" > "$TMP/ls-remote.txt" 2>&1 || ls_rc=$?
case "$ls_rc" in
  0) branch_exists=1 ;;
  2) branch_exists="" ;;
  *)
    cat "$TMP/ls-remote.txt"
    echo "::error::could not list branches on origin (git exit $ls_rc)"
    exit 1
    ;;
esac

push_note=""
if [ -n "$branch_exists" ]; then
  # The name only says which two commits were meant. A branch of this name that does not hold
  # the merge computed above (same tree, both parents in its history) was not made by this
  # sync: the PR text (checks, size, syntax, "workflows restored unchanged") would describe a
  # commit that is not in the PR. A rerun's merge commit has a new SHA (timestamp) but the
  # same tree, so a branch from an earlier run of this bot passes.
  if ! git fetch --quiet origin "refs/heads/$branch" 2>"$TMP/br-fetch.err"; then
    cat "$TMP/br-fetch.err"
    echo "::error::could not fetch $branch from origin to check it; not opening a PR from an unchecked branch"
    exit 1
  fi
  if [ "$(git rev-parse 'FETCH_HEAD^{tree}')" != "$(git rev-parse "$merged_sha^{tree}")" ] ||
     ! git merge-base --is-ancestor "$up_sha" FETCH_HEAD ||
     ! git merge-base --is-ancestor "$base_sha" FETCH_HEAD; then
    branch_err="$branch already exists on origin (at $(git rev-parse --short=7 FETCH_HEAD)) but is not the merge of $UPSTREAM_REF ($up_short) into $BASE_REF ($base_short) computed by this run; the bot never writes to a branch it did not just create and opens no PR from it. Delete or rename that branch."
    summary "## Upstream sync: stopped"$'\n\n'"$branch_err"
    echo "::error::$branch_err"
    exit 1
  fi
  push_note="\`$branch\` is already on origin from an earlier run (same upstream commit, same base); nothing new was pushed."
  log "$branch already exists on origin; not pushing again"
elif [ -n "$DRY_RUN" ]; then
  push_note="[dry run] \`$branch\` would be pushed ($merged_sha)."
  log "[dry-run] would push $merged_sha to refs/heads/$branch"
else
  if ! git push origin "$merged_sha:refs/heads/$branch" > "$TMP/push.log" 2>&1; then
    cat "$TMP/push.log"
    {
      echo "## Upstream sync: the push failed"
      echo
      echo "The merge of \`$UPSTREAM_REPO\` main (\`$up_short\`) is clean, but pushing \`$branch\` failed:"
      echo '```'
      sed -n '1,20p' "$TMP/push.log"
      echo '```'
      echo "Common causes: a branch ruleset that blocks \`$BRANCH_PREFIX*\`, or the token lacking \`contents: write\`."
    } > "$TMP/push-failed.md"
    summary "$(cat "$TMP/push-failed.md")"
    echo "::error::upstream sync could not push $branch (see the log)"
    exit 1
  fi
  push_note="Pushed \`$branch\` ($merged_sha)."
  log "pushed $branch"
fi

# Older bot branches that are still on origin (no PR can be open on them, or
# the guard above would have returned): just a reminder to clean up.
older="$(git ls-remote --heads origin "refs/heads/${BRANCH_PREFIX}*" 2>/dev/null | sed 's|.*refs/heads/||' | grep -vxF "$branch" || true)"

# ----------------------------------------------------------------- open PR ----

# The one and only PR call in this script is `gh pr create`.
pr_url=""
pr_via=""
pr_existed=""
pr_errors=""
pr_note=""

create_pr() { # <token> <label>
  if GH_TOKEN="$1" gh pr create --repo "$REPO" --base "$BASE_BRANCH" --head "$branch" \
       --title "$pr_title" --body-file "$TMP/pr-full.md" > "$TMP/pr.out" 2> "$TMP/pr.err"; then
    pr_url="$(grep -Eo 'https://github\.com/[^ ]+/pull/[0-9]+' "$TMP/pr.out" | tail -n 1 || true)"
    [ -n "$pr_url" ] || pr_url="(created, URL not reported)"
    pr_via="$2"
    return 0
  fi
  # gh answers "a pull request for branch X into branch Y already exists: <url>" when
  # the PR is there (an earlier attempt that timed out client-side after GitHub had
  # created it, say): that PR is the goal, so it is not a failure.
  if grep -q 'already exists' "$TMP/pr.err"; then
    pr_url="$(grep -Eo 'https://github\.com/[^ ]+/pull/[0-9]+' "$TMP/pr.err" | tail -n 1 || true)"
    if [ -n "$pr_url" ]; then
      pr_via="$2"
      pr_existed=1
      return 0
    fi
  fi
  pr_errors="$pr_errors- $2: $(md_code "$(sed -n '1,3p' "$TMP/pr.err")" 300)"$'\n'
  return 1
}

# A PR that was ever opened from this exact branch (even one a person closed)
# is not opened again: a closed PR is a decision, not a failure. `gh pr list
# --head` matches the branch NAME only ("<owner>:<branch>" is not supported), so
# PRs from other people's forks that happen to use this (public, predictable)
# name are dropped: only a head in THIS repo counts, like in the guard above.
# That cut happens after --limit (newest first), so the limit is generous: a pile
# of foreign PRs with this name must not push the real one out of view.
# If the list itself fails the branch is already on origin, so do not stop here:
# nothing is opened (nothing is guessed), but the run still ends red with the
# report and the link.
prior_unknown=""
if ! prior_prs="$(gh pr list --repo "$REPO" --state all --head "$branch" --limit 1000 \
    --json number,state,isCrossRepository \
    --jq '.[] | select(.isCrossRepository | not) | "#\(.number) (\(.state | ascii_downcase))"' 2>"$TMP/gh.err")"; then
  cat "$TMP/gh.err"
  echo "::error::could not list PRs of $branch (needs pull-requests: read); not guessing, no PR is opened"
  prior_prs=""
  prior_unknown=1
fi
prior_prs="$(printf '%s\n' "$prior_prs" | tr '\n' ' ' | sed 's/ *$//')"

if [ -n "$prior_unknown" ]; then
  pr_note="Could not check whether a pull request from \`$branch\` already exists (the PR list call failed, see the log), so the bot did not open one."
  log "could not list the PRs of $branch; not opening one"
elif [ -n "$prior_prs" ]; then
  pr_note="A pull request from \`$branch\` was already opened before ($prior_prs) and none is open into \`$BASE_BRANCH\` now, so the bot does not open another."
  log "PR(s) of $branch already exist ($prior_prs); not opening another"
elif [ -n "$DRY_RUN" ]; then
  if [ -n "$PR_TOKEN" ]; then
    pr_note="[dry run] a PR would be opened, with SWIFTLM_PR_TOKEN first, then GITHUB_TOKEN."
  else
    pr_note="[dry run] a PR would be opened with GITHUB_TOKEN (no SWIFTLM_PR_TOKEN set)."
  fi
  log "[dry-run] would run: gh pr create --base $BASE_BRANCH --head $branch"
else
  if [ -n "$PR_TOKEN" ]; then
    create_pr "$PR_TOKEN" "SWIFTLM_PR_TOKEN" || true
  fi
  if [ -z "$pr_url" ] && [ -n "${GH_TOKEN:-}" ] && [ "${GH_TOKEN:-}" != "$PR_TOKEN" ]; then
    create_pr "$GH_TOKEN" "GITHUB_TOKEN" || true
  fi
  if [ -z "$pr_url" ]; then
    printf '%s' "$pr_errors"
    pr_note="No token could open the PR:"$'\n'"$pr_errors""To get automatic PRs: give \`SWIFTLM_PR_TOKEN\` Pull requests: write on this repo, or allow GitHub Actions to open PRs (Settings > Actions > General; GitHub words that checkbox \"create and approve pull requests\", the bot only ever creates them)."
    echo "::warning::could not open a PR for $branch: $(printf '%s' "$pr_errors" | tr '\n' ' ')"
  fi
fi

# ----------------------------------------------------------------- report ----

if [ -n "$pr_url" ]; then
  {
    echo "## Upstream sync: pull request opened"
    echo
    if [ -n "$pr_existed" ]; then
      opened_how="GitHub said this PR already existed when the bot tried to open it (an earlier request that looked failed had created it)."
    else
      opened_how="Opened with $pr_via."
    fi
    echo "$pr_url merges \`$UPSTREAM_REPO\` main (\`$up_short\`, $behind commit(s)) into \`$BASE_BRANCH\`. $push_note $opened_how The bot does not approve or merge it: review it, wait for CI, and merge it with a **merge commit**."
    if [ "$pr_via" = "GITHUB_TOKEN" ] && [ -z "$pr_existed" ]; then
      echo
      echo "GitHub does not start workflows for a PR opened with GITHUB_TOKEN: if no checks appear, close and reopen the PR once. (A personal \`SWIFTLM_PR_TOKEN\` with Pull requests: write avoids this.)"
      if [ -n "$pr_errors" ]; then
        echo
        echo "**\`SWIFTLM_PR_TOKEN\` was tried first and failed** (expired, revoked or missing Pull requests: write?), which is why this PR has no CI yet:"
        printf '%s' "$pr_errors"
      fi
    fi
    echo
    echo "### Needs a human look"
    if [ -s "$TMP/flags.md" ]; then
      cat "$TMP/flags.md"
    else
      echo "- Nothing flagged."
    fi
    if [ -n "$older" ]; then
      echo
      echo "Older bot branches still on origin (delete them when no longer needed):"
      md_list "$older" 10
    fi
  } > "$TMP/opened.md"
  report "Upstream sync: opened $pr_url" "$TMP/opened.md"
  if [ "$pr_via" = "GITHUB_TOKEN" ] && [ -z "$pr_existed" ] && [ -n "$pr_errors" ]; then
    echo "::warning::SWIFTLM_PR_TOKEN failed, so $pr_url was opened with GITHUB_TOKEN (no CI until it is closed and reopened): $(printf '%s' "$pr_errors" | tr '\n' ' ')"
  fi
  echo "::notice::upstream sync: opened $pr_url (not approved, not merged; review it)"
  exit 0
fi

{
  echo "## Upstream sync: a clean merge is ready, open the PR"
  echo
  echo "\`$branch\` merges \`$UPSTREAM_REPO\` main (\`$up_short\`, $behind commit(s)) into \`$BASE_BRANCH\` with no conflicts. $push_note"
  echo
  [ -z "$pr_note" ] || { printf '%s\n' "$pr_note"; echo; }
  echo "**[Open the pull request]($open_url)** ($url_note; merge it with a merge commit)"
  echo
  if [ -n "$DRY_RUN" ]; then
    echo "(Dry run: nothing was pushed and no PR was opened.)"
  else
    echo "This run is red on purpose: no PR is open for this sync, so a person has to act. It turns green once a sync PR is open or the fork contains upstream."
  fi
  echo
  echo "### Needs a human look"
  if [ -s "$TMP/flags.md" ]; then
    cat "$TMP/flags.md"
  else
    echo "- Nothing flagged."
  fi
  if [ -n "$older" ]; then
    echo
    echo "Older bot branches still on origin (delete them when no longer needed):"
    md_list "$older" 10
  fi
  echo
  echo "<details><summary>Full pull request description</summary>"
  echo
  echo '```markdown'
  echo "$pr_title"
  echo
  cat "$TMP/pr-full.md"
  echo '```'
  echo
  echo "</details>"
} > "$TMP/ready.md"

if [ -n "$DRY_RUN" ]; then
  # A prior PR from this branch (or a failed lookup) is the one case where a real
  # run skips PR creation and always ends red.
  if [ -n "$prior_prs" ] || [ -n "$prior_unknown" ]; then
    echo "[dry-run] a real run would NOT try to open a PR and would end RED (exit 1) with this report:"
  else
    echo "[dry-run] a real run would open the PR (exit 0) or, if that fails, end RED with this report:"
  fi
fi
report "Upstream sync: clean merge ready on $branch, open the PR" "$TMP/ready.md"
if [ -n "$DRY_RUN" ]; then
  echo "[dry-run] open-PR link: $open_url"
  [ -z "$prior_unknown" ] || exit 1
  exit 0
fi
echo "::error::upstream sync: $branch is ready but no PR is open, a person has to open it: $plain_url"
exit 1
