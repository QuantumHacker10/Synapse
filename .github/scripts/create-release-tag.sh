#!/usr/bin/env bash
# Crée un tag annoté de release et le pousse sur origin.
# Usage : ./.github/scripts/create-release-tag.sh v1.2.0 "Synapse OMNIA 1.2.0"
set -euo pipefail

TAG="${1:?Usage: create-release-tag.sh vX.Y.Z [message]}"
MSG="${2:-Synapse OMNIA ${TAG#v}}"

git checkout main
git pull origin main
git tag -a "$TAG" -m "$MSG"
git push origin "$TAG"

echo "Tag $TAG pousse. Le workflow release.yml publiera l'artefact Windows."
