#!/usr/bin/env bash
# Save your Hugging Face token for `hf` and Git operations against hf.co.
# Usage (pick one):
#   export HF_TOKEN=hf_xxxxxxxx
#   ./scripts/hf_set_token.sh
# Or pass the token as the first argument (still ends up in shell history):
#   ./scripts/hf_set_token.sh hf_xxxxxxxx
#
# Create a token with write access: https://huggingface.co/settings/tokens
# Do not commit real tokens.

set -euo pipefail

if [[ -n "${1:-}" ]]; then
  export HF_TOKEN="$1"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: HF_TOKEN is not set."
  echo ""
  echo "In this same terminal, run (paste your real hf_... token):"
  echo "  export HF_TOKEN=hf_your_token_here"
  echo "  ./scripts/hf_set_token.sh"
  echo ""
  echo "Or in one line:"
  echo "  ./scripts/hf_set_token.sh hf_your_token_here"
  echo ""
  echo "Create tokens: https://huggingface.co/settings/tokens"
  exit 1
fi

if ! command -v hf &>/dev/null; then
  echo "Installing Hugging Face CLI (hf)..."
  curl -LsSf https://hf.co/cli/install.sh | bash
  echo "Open a new shell or add ~/.local/bin to PATH, then re-run this script."
  exit 1
fi

hf auth login --token "${HF_TOKEN}" --add-to-git-credential

echo "OK: hf auth configured. Test with:  hf whoami"
