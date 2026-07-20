#!/usr/bin/env bash
# Create an annotated release tag and push it to origin.
# Usage: ./.github/scripts/create-release-tag.sh v1.2.0 "Synapse OMNIA 1.2.0"
set -euo pipefail

TAG="${1:?Usage: create-release-tag.sh vX.Y.Z [message]}"
MSG="${2:-Synapse OMNIA ${TAG#v}}"

git checkout main
git pull origin main
git tag -a "$TAG" -m "$MSG"
git push origin "$TAG"

echo "Tag $TAG pushed. The release.yml workflow will publish multi-platform artifacts on GitHub Releases."
