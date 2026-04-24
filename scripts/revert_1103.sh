#!/usr/bin/env bash
# scripts/revert_1103.sh
#
# Single-command revert for the Phase 1103 migration commit:
#   feat(1103-01): migrate v13.1 audit-chain integration to production
#
# Locates the migration commit by its subject line, runs `git revert --no-edit`,
# and verifies the working tree returns to the pre-migration byte-identity
# invariant (`git diff v13.0..HEAD -- antigence_subnet/` empty) and that the
# experiment directories are restored.
#
# Flags:
#   --dry-run    Print the operations that would run without executing them.
#                Exits 0 if all preconditions (commit found OR graceful no-op)
#                are satisfied.
#
# Exit codes:
#   0  success (or dry-run would succeed)
#   1  commit not found in history and not in dry-run no-op mode
#   2  `git revert` failed
#   3  post-revert invariants not restored

set -eu

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 64 ;;
    esac
done

MIGRATION_SUBJECT='feat(1103-01): migrate v13.1 audit-chain integration to production'

log() { printf '[revert_1103] %s\n' "$*"; }

SHA="$(git log --grep="^${MIGRATION_SUBJECT}$" --format=%H -n 1 || true)"

if [ -z "${SHA}" ]; then
    if [ "${DRY_RUN}" -eq 1 ]; then
        log "no migration commit found; revert is a no-op (dry-run)"
        exit 0
    fi
    log "ERROR: migration commit not found. Fall back to manual revert:" >&2
    log "  git log --oneline | grep '1103-01'" >&2
    log "  git revert <sha-of-migration>" >&2
    exit 1
fi

log "located migration commit: ${SHA}"

if [ "${DRY_RUN}" -eq 1 ]; then
    log "DRY-RUN: would run 'git revert --no-edit ${SHA}'"
    log "DRY-RUN: would verify 'git diff v13.0..HEAD -- antigence_subnet/' is empty"
    log "DRY-RUN: would verify experiments/v13.1-migration/ is restored"
    log "DRY-RUN: would verify tests/experiments/v13_1_migration/ is restored"
    exit 0
fi

log "executing: git revert --no-edit ${SHA}"
if ! git revert --no-edit "${SHA}"; then
    log "ERROR: git revert failed. Resolve conflicts manually or run 'git revert --abort'." >&2
    exit 2
fi

# Post-revert invariant verification.
DIFF_LINES="$(git diff v13.0..HEAD -- antigence_subnet/ | wc -l | tr -d ' ')"
if [ "${DIFF_LINES}" != "0" ]; then
    log "ERROR: post-revert 'git diff v13.0..HEAD -- antigence_subnet/' is ${DIFF_LINES} lines, expected 0" >&2
    exit 3
fi

if [ ! -d experiments/v13.1-migration ]; then
    log "ERROR: experiments/v13.1-migration/ not restored by revert" >&2
    exit 3
fi

if [ ! -d tests/experiments/v13_1_migration ]; then
    log "ERROR: tests/experiments/v13_1_migration/ not restored by revert" >&2
    exit 3
fi

log "revert complete; working tree restored to pre-1103 state"
exit 0
