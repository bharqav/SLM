#!/usr/bin/env bash
#
# Phase 5.1: Publish the repository and model weights to GitHub.
#
# This script PUBLISHES PUBLIC CONTENT and uploads large binaries. It refuses
# to do anything without --confirm, and prints exactly what it would do first.
#
#   ./publish.sh                      # dry run: show the plan, change nothing
#   ./publish.sh --confirm            # create repo, push, upload release assets
#   ./publish.sh --confirm --private  # same, but create the repo private
#
set -euo pipefail

REPO_NAME="${REPO_NAME:-slm-training}"
TAG="${TAG:-v1.0-trained}"
VISIBILITY="--public"
CONFIRM=0

for arg in "$@"; do
  case "$arg" in
    --confirm) CONFIRM=1 ;;
    --private) VISIBILITY="--private" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")"

# ----------------------------------------------------------------- preflight
command -v gh >/dev/null || { echo "gh CLI not found" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh not authenticated; run: gh auth login" >&2; exit 1; }
OWNER="$(gh api user --jq .login)"

MISSING=""
for f in README.md TRAINING_REPORT.md API.md QUICKSTART.md LICENSE requirements.txt; do
  [ -f "$f" ] || MISSING="$MISSING $f"
done
[ -n "$MISSING" ] && { echo "missing required files:$MISSING" >&2; exit 1; }

# Refuse to publish a write-up that still has unfilled placeholders.
if grep -lq 'TK' TRAINING_REPORT.md 2>/dev/null; then
  echo "TRAINING_REPORT.md still contains TK placeholders; regenerate with make_report.py" >&2
  exit 1
fi

ASSETS=""
for f in best_model.pt final_model.pt losses.json inference_samples.txt loss_curves.png; do
  [ -f "$f" ] && ASSETS="$ASSETS $f"
done

echo "=============================================================="
echo "  Target repo : $OWNER/$REPO_NAME  ($( [ "$VISIBILITY" = "--public" ] && echo PUBLIC || echo private ))"
echo "  Release tag : $TAG"
echo "  Commits     : $(git rev-list --count HEAD 2>/dev/null || echo 0)"
echo "  Tracked     : $(git ls-files | wc -l) files"
echo "  Assets      :$( [ -n "$ASSETS" ] && echo "$ASSETS" || echo " (none found)" )"
for f in $ASSETS; do
  printf "                %-24s %s\n" "$f" "$(du -h "$f" | cut -f1)"
done
echo "=============================================================="

if [ "$CONFIRM" -ne 1 ]; then
  echo
  echo "DRY RUN - nothing was published. Re-run with --confirm to proceed."
  exit 0
fi

# --------------------------------------------------------------------- push
if gh repo view "$OWNER/$REPO_NAME" >/dev/null 2>&1; then
  echo "Repository already exists; pushing to it."
  git remote get-url origin >/dev/null 2>&1 || \
    git remote add origin "https://github.com/$OWNER/$REPO_NAME.git"
else
  echo "Creating repository $OWNER/$REPO_NAME ..."
  gh repo create "$REPO_NAME" $VISIBILITY \
    --source=. --remote=origin \
    --description="An 81.6M parameter language model trained from scratch on GitHub docstrings and code comments"
fi

git branch -M main
git push -u origin main

# ------------------------------------------------------------------ release
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "Release $TAG exists; uploading assets with --clobber."
  # shellcheck disable=SC2086
  [ -n "$ASSETS" ] && gh release upload "$TAG" $ASSETS --clobber
else
  echo "Creating release $TAG ..."
  # shellcheck disable=SC2086
  gh release create "$TAG" $ASSETS \
    --title "Code-Aware SLM v1.0 (81.6M params)" \
    --notes-file TRAINING_REPORT.md
fi

echo
echo "Published: https://github.com/$OWNER/$REPO_NAME"
echo "Release  : https://github.com/$OWNER/$REPO_NAME/releases/tag/$TAG"
