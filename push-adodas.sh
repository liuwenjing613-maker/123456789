#!/usr/bin/env bash
# Push AdoDAS code from home monorepo (~), same workflow as before.
set -euo pipefail
cd "$(dirname "$0")"

MSG="${1:-Update code}"

git add -A AdoDAS2026 AdoDAS2026_folder_pth .gitignore push-adodas.sh
if git diff --cached --quiet; then
  echo "Nothing to commit under AdoDAS2026*."
  exit 0
fi

git commit -m "$MSG"
git push origin main
echo "Pushed to origin/main ($(git remote get-url origin))"
