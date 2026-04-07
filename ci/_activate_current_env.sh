# This script should never be called directly, only sourced:

#     source _activate_current_env.sh

# For robustness, try all possible activate commands.
conda activate "${ENV_NAME}" 2>/dev/null
