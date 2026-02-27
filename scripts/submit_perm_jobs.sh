#!/bin/bash
# Submit all permutation jobs (shuffle + model fitting) to SLURM.
#
# Usage:
#   bash submit_perm_jobs.sh          # submit all jobs
#   bash submit_perm_jobs.sh --dry    # print sbatch commands without submitting
#
# Prerequisites:
#   - conda env 'sfp' available on HPC
#   - Input CSVs already exist in OUTPUT_DIR

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
REPO_DIR="/home/jh7685/spatial-frequency-preferences_NSDsyn"
OUTPUT_DIR="/scratch/jh7685/projects/sfp_nsd/derivatives"
SCRIPT="${REPO_DIR}/scripts/run_perm_job.py"
LOG_DIR="${OUTPUT_DIR}/logs/slurm/perm"

# Singularity settings
OVERLAY="/scratch/jh7685/overlay/sfp/overlay-25GB-500K.ext3"
SIF="/share/apps/images/ubuntu-22.04.4.sif"
SING="singularity exec --overlay ${OVERLAY}:ro ${SIF}"

SUBJECTS=(subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08)
ROIS=(V1)
SHUFFLE_TYPES=(orientation eccentricity)
PERM_RANGE="0-99"

DRY_RUN=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — no jobs will be submitted ==="
fi

# ── Helper ───────────────────────────────────────────────────────────────
submit_or_print() {
    # Submits an sbatch command and captures the job ID, or prints it in dry-run mode.
    # Returns the job ID via stdout (caller captures with $(...)).
    local cmd="$1"
    if $DRY_RUN; then
        echo "$cmd" >&2
        echo "DRY_RUN_JOB_ID"
    else
        local out
        out=$(eval "$cmd")
        # sbatch output: "Submitted batch job 12345678"
        local job_id
        job_id=$(echo "$out" | grep -oE '[0-9]+$')
        echo "$job_id"
    fi
}

# ── Submit jobs ──────────────────────────────────────────────────────────
echo "Submitting permutation jobs..."
echo "Subjects: ${SUBJECTS[*]}"
echo "ROIs:     ${ROIS[*]}"
echo "Shuffles: ${SHUFFLE_TYPES[*]}"
echo "Perms:    ${PERM_RANGE}"
echo ""

for SHUF in "${SHUFFLE_TYPES[@]}"; do
    for SUBJ in "${SUBJECTS[@]}"; do
        for ROI in "${ROIS[@]}"; do
            # Log directories
            SHUF_LOG="${LOG_DIR}/shuf-${SHUF}/${SUBJ}"
            MODEL_LOG="${LOG_DIR}/model-shuf-${SHUF}/${SUBJ}"
            mkdir -p "${SHUF_LOG}" "${MODEL_LOG}" 2>/dev/null || true

            # ── Shuffle array job ────────────────────────────────────
            SHUF_CMD="sbatch \
                --job-name=shuf_${SHUF}_${SUBJ}_${ROI} \
                --array=${PERM_RANGE} \
                --nodes=1 --ntasks=1 --cpus-per-task=1 \
                --mem=2G --time=01:00:00 \
                --output=${SHUF_LOG}/${ROI}_%a.out \
                --error=${SHUF_LOG}/${ROI}_%a.err \
                --wrap=\"${SING} /bin/bash -c 'source /ext3/env.sh && python ${SCRIPT} shuffle \
                    --shuffle-type ${SHUF} --subj ${SUBJ} --roi ${ROI} \
                    --perm \\\$SLURM_ARRAY_TASK_ID --output-dir ${OUTPUT_DIR}'\""

            SHUF_JOB_ID=$(submit_or_print "$SHUF_CMD")
            echo "  [shuffle] ${SHUF} subj=${SUBJ} roi=${ROI} -> job ${SHUF_JOB_ID}"

            # ── Model array job (depends on shuffle) ─────────────────
            DEP_FLAG=""
            if [[ "$SHUF_JOB_ID" != "DRY_RUN_JOB_ID" ]]; then
                DEP_FLAG="--dependency=afterok:${SHUF_JOB_ID}"
            fi

            MODEL_CMD="sbatch \
                --job-name=model_${SHUF}_${SUBJ}_${ROI} \
                --array=${PERM_RANGE} \
                --nodes=1 --ntasks=1 --cpus-per-task=1 \
                --mem=4G --time=08:00:00 \
                ${DEP_FLAG} \
                --output=${MODEL_LOG}/${ROI}_%a.out \
                --error=${MODEL_LOG}/${ROI}_%a.err \
                --wrap=\"${SING} /bin/bash -c 'source /ext3/env.sh && python ${SCRIPT} model \
                    --shuffle-type ${SHUF} --subj ${SUBJ} --roi ${ROI} \
                    --perm \\\$SLURM_ARRAY_TASK_ID --output-dir ${OUTPUT_DIR}'\""

            MODEL_JOB_ID=$(submit_or_print "$MODEL_CMD")
            echo "  [model]   ${SHUF} subj=${SUBJ} roi=${ROI} -> job ${MODEL_JOB_ID} (after ${SHUF_JOB_ID})"
        done
    done
done

echo ""
echo "Done. Total: $((${#SHUFFLE_TYPES[@]} * ${#SUBJECTS[@]} * ${#ROIS[@]} * 2)) array submissions."
echo "Use 'squeue -u \$USER' to monitor jobs."
