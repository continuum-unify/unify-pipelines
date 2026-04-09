#!/bin/bash

# =============================================================================
# Auto-update Open WebUI Pipelines from GitHub repository
# Usage: ./update-all-pipelines.sh
#
# CHANGES FROM PREVIOUS VERSION:
#   [CRITICAL] Fixed GITHUB_ORG: was "Continuum-Labs-HQ", now "continuum-unify"
#   [MEDIUM]   Filters out __init__.py and non-pipeline utility files
#   [MINOR]    Added validation and dry-run option
# =============================================================================

set -e

# Configuration
GITHUB_ORG="continuum-unify"                # [FIXED] Was "Continuum-Labs-HQ"
GITHUB_REPO="unify-pipelines"
GITHUB_BRANCH="main"
PIPELINES_DIR="pipelines"
NAMESPACE="open-webui"
DEPLOYMENT="open-webui-pipelines"

# Files to exclude (not actual pipelines)
EXCLUDE_FILES="__init__.py|customer_database.py"

# Parse arguments
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - no changes will be applied"
    echo ""
fi

# GitHub API to get all files in pipelines directory
GITHUB_API="https://api.github.com/repos/${GITHUB_ORG}/${GITHUB_REPO}/contents/${PIPELINES_DIR}?ref=${GITHUB_BRANCH}"

echo "🔍 Fetching pipeline list from GitHub..."
echo "   Repository: ${GITHUB_ORG}/${GITHUB_REPO} (${GITHUB_BRANCH})"
echo ""

# Fetch file list from GitHub API
RESPONSE=$(curl -s "$GITHUB_API")

# Check for API errors
if echo "$RESPONSE" | grep -q '"message"'; then
    echo "❌ GitHub API error:"
    echo "$RESPONSE" | grep '"message"' | head -1
    exit 1
fi

# Extract .py filenames, excluding non-pipeline files
PIPELINE_FILES=$(echo "$RESPONSE" \
    | grep -o '"name": *"[^"]*\.py"' \
    | grep -o '[^"]*\.py' \
    | grep -vE "$EXCLUDE_FILES")

if [ -z "$PIPELINE_FILES" ]; then
    echo "❌ No pipeline files found in repository"
    exit 1
fi

echo "📋 Found pipelines:"
echo "$PIPELINE_FILES" | sed 's/^/   ✓ /'
echo ""

EXCLUDED=$(echo "$RESPONSE" \
    | grep -o '"name": *"[^"]*\.py"' \
    | grep -o '[^"]*\.py' \
    | grep -E "$EXCLUDE_FILES" || true)

if [ -n "$EXCLUDED" ]; then
    echo "⏭️  Excluded (not pipelines):"
    echo "$EXCLUDED" | sed 's/^/   ✗ /'
    echo ""
fi

# Build PIPELINES_URLS string
BASE_URL="https://raw.githubusercontent.com/${GITHUB_ORG}/${GITHUB_REPO}/${GITHUB_BRANCH}/${PIPELINES_DIR}"
URLS=""

for file in $PIPELINE_FILES; do
    if [ -z "$URLS" ]; then
        URLS="${BASE_URL}/${file}"
    else
        URLS="${URLS};${BASE_URL}/${file}"
    fi
done

echo "🔗 Generated PIPELINES_URLS:"
echo "================================"
echo "$URLS" | tr ';' '\n' | sed 's/^/   /'
echo ""

PIPELINE_COUNT=$(echo "$PIPELINE_FILES" | wc -l | tr -d ' ')
echo "📊 Total: ${PIPELINE_COUNT} pipelines"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "🔍 DRY RUN complete - no changes applied"
    echo "   Run without --dry-run to apply changes"
    exit 0
fi

# Update Kubernetes deployment
echo "🚀 Updating Kubernetes deployment..."
kubectl set env deployment/${DEPLOYMENT} \
    PIPELINES_URLS="$URLS" \
    -n ${NAMESPACE}

if [ $? -eq 0 ]; then
    echo "✅ Deployment updated successfully!"
    echo ""
    echo "📊 Watching rollout status..."
    kubectl rollout status deployment/${DEPLOYMENT} -n ${NAMESPACE}
    echo ""
    echo "🎉 All pipelines loaded!"
    echo ""
    echo "📝 Loaded pipelines:"
    echo "$PIPELINE_FILES" | sed 's/^/   ✓ /'
else
    echo "❌ Failed to update deployment"
    exit 1
fi