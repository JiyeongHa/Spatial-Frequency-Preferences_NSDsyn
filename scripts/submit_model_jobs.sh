#!/bin/bash
# Submit model-fitting permutation jobs to SLURM. Skips perms whose output already exists.
# Run AFTER shuffle jobs are complete.
#
# Usage:
#   bash submit_model_jobs.sh                          # perms 0-99 (default)
#   bash submit_model_jobs.sh --perm-range 100-199
#   bash submit_model_jobs.sh --dry                    # dry-run (no submission)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
REPO_DIR="/home/jh7685/Spatial-Frequency-Preferences_NSDsyn"
OUTPUT_DIR="/scratch/jh7685/projects/sfp_nsd/derivatives"
SCRIPT="${REPO_DIR}/scripts/run_perm_job.py"
LOG_DIR="${OUTPUT_DIR}/logs/slurm/perm"

ACCOUNT="torch_pr_506_general"
OVERLAY="/scratch/jh7685/overlay/sfp/overlay-25GB-500K.ext3"
SIF="/share/apps/images/ubuntu-22.04.4.sif"
SING="singularity exec --overlay ${OVERLAY}:ro ${SIF}"

SUBJECTS=(subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08)
ROIS=(V1)
SHUFFLE_TYPES=(orientation eccentricity local_sf)
DSET="nsdsyn"
VS="pRFsize"
LR="0.0005"
EPOCH="30000"

# ── Parse CLI arguments ───────────────────────────────────────────────────
PERM_RANGE="0-99"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --perm-range) PERM_RANGE="$2"; shift 2 ;;
        --dry)        DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

START="${PERM_RANGE%-*}"
END="${PERM_RANGE#*-}"

if $DRY_RUN; then echo "=== DRY RUN — no jobs will be submitted ==="; fi

# ── Helper ───────────────────────────────────────────────────────────────
submit_or_print() {
    local cmd="$1"
    if $DRY_RUN; then
        echo "$cmd" >&2
        echo "DRY_RUN_JOB_ID"
    else
        local out
        out=$(eval "$cmd")
        echo "$out" | grep -oE '[0-9]+$'
    fi
}

# ── Submit jobs ──────────────────────────────────────────────────────────
echo "Submitting model fitting jobs..."
echo "Subjects: ${SUBJECTS[*]}"
echo "ROIs:     ${ROIS[*]}"
echo "Shuffles: ${SHUFFLE_TYPES[*]}"
echo "Perms:    ${PERM_RANGE}"
echo ""

TOTAL_SUBMITTED=0

for SHUF in "${SHUFFLE_TYPES[@]}"; do
    for SUBJ in "${SUBJECTS[@]}"; do
        for ROI in "${ROIS[@]}"; do
            MODEL_LOG="${LOG_DIR}/model-shuf-${SHUF}/${SUBJ}"
            mkdir -p "${MODEL_LOG}" 2>/dev/null || true

            # Find perms whose model output does not yet exist
            MISSING=()
            for perm in $(seq "$START" "$END"); do
                BATCH="batch_$(printf '%03d' $(( (perm / 100) * 100 )))"
                OUT="${OUTPUT_DIR}/sfp_model/results_2D/${DSET}/perm/shuf-${SHUF}/${SUBJ}/${BATCH}/perm-${perm}_model-params_lr-${LR}_eph-${EPOCH}_sub-${SUBJ}_roi-${ROI}_vs-${VS}.pt"
                [[ ! -f "$OUT" ]] && MISSING+=("$perm")
            done

            if [[ ${#MISSING[@]} -eq 0 ]]; then
                echo "  [model] ${SHUF} ${SUBJ} ${ROI} -> all done, skipping"
                continue
            fi

            ARRAY_STR=$(IFS=,; echo "${MISSING[*]}")
            CMD="sbatch \
                --account=${ACCOUNT} \
                --job-name=model_${SHUF}_${SUBJ}_${ROI} \
                --array=${ARRAY_STR} \
                --nodes=1 --ntasks=1 --cpus-per-task=1 \
                --mem=4G --time=08:00:00 \
                --output=${MODEL_LOG}/${ROI}_%a.out \
                --error=${MODEL_LOG}/${ROI}_%a.err \
                --wrap=\"module purge; ${SING} /bin/bash -c 'source /ext3/env.sh; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ${SCRIPT} model \
                    --shuffle-type ${SHUF} --subj ${SUBJ} --roi ${ROI} \
                    --perm \\\$SLURM_ARRAY_TASK_ID --output-dir ${OUTPUT_DIR}'\""

            JOB_ID=$(submit_or_print "$CMD")
            echo "  [model] ${SHUF} ${SUBJ} ${ROI} -> job ${JOB_ID} (${#MISSING[@]}/${END-START+1} perms missing)"
            TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
        done
    done
done

echo ""
echo "Done. Submitted ${TOTAL_SUBMITTED} array jobs."
echo "Use 'squeue -u \$USER' to monitor."
