#!/usr/bin/env bash
# run_test_all_models.sh
# ======================
# Roda os 6 modelos (3 frontier + 3 econômicos) em paralelo no split TEST (780).
# Dashboard de progresso atualizado a cada 3s (contagem correta via grep '^MV_').
#
# Uso:  bash run_test_all_models.sh
#
# Saídas:
#   responses/{tag}_test.csv  — output de cada modelo
#   logs/{tag}_test.log       — stdout/stderr de cada job
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv não encontrado. Rode 'bash setup.sh' primeiro."
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

mkdir -p responses logs

INPUT="data/outputs/reports_master.csv"
PROMPT="prompts/cxr_prompt_v4.yaml"
TOTAL=780

TAGS=(    "sonnet-4-6"        "haiku-4-5"        "gpt-4-1"  "gpt-4-1-mini"  "gemini-2-5-pro"  "gemini-2-5-flash" )
VENDORS=( "claude"            "claude"           "gpt4o"    "gpt4o"         "gemini"          "gemini"           )
MODELS=(  "claude-sonnet-4-6" "claude-haiku-4-5" "gpt-4.1"  "gpt-4.1-mini"  "gemini-2.5-pro"  "gemini-2.5-flash" )

N=${#TAGS[@]}
PIDS=()
OUTS=()

echo "==> Disparando $N modelos em paralelo no split TEST ($TOTAL laudos cada)"
for i in $(seq 0 $((N-1))); do
    tag="${TAGS[$i]}"
    vendor="${VENDORS[$i]}"
    model="${MODELS[$i]}"
    out="responses/${tag}_test.csv"
    log="logs/${tag}_test.log"
    rm -f "$out"
    python pipeline/api_inference.py \
        --vendor "$vendor" --model-id "$model" \
        --input "$INPUT" --prompt "$PROMPT" \
        --only-split test --output "$out" \
        > "$log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    OUTS+=("$out")
    echo "  [$pid] $tag  ($vendor / $model)"
done

START=$(date +%s)

trap 'echo; echo "Interrompido. Matando jobs..."; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; exit 130' INT

while true; do
    clear
    elapsed=$(( $(date +%s) - START ))
    mm=$((elapsed/60)); ss=$((elapsed%60))
    printf "Stanford CXR LLM — TEST split (%d laudos) — decorrido: %02d:%02d\n\n" "$TOTAL" "$mm" "$ss"
    printf "%-22s %8s %-52s %-8s %s\n" "MODEL" "DONE" "PROGRESS" "STATUS" "ETA"
    printf "%-22s %8s %-52s %-8s %s\n" "----------" "------" "----------------------------------------------------" "------" "----"

    all_done=true
    for i in $(seq 0 $((N-1))); do
        tag="${TAGS[$i]}"
        pid="${PIDS[$i]}"
        out="${OUTS[$i]}"

        if [ -f "$out" ]; then
            # grep -c retorna "0" em stdout E exit 1 quando não acha — o `|| true`
            # ignora o exit code mantendo só o "0" do stdout. Sem isso, `|| echo 0`
            # empilha um "0\n0" e quebra a aritmética.
            done=$(grep -c '^MV_' "$out" 2>/dev/null || true)
            done=${done:-0}
        else
            done=0
        fi

        pct=$(( done * 100 / TOTAL ))
        (( pct > 100 )) && pct=100
        bar_len=$(( pct / 2 ))
        bar=$(printf '%*s' "$bar_len" '' | tr ' ' '#')
        bar=$(printf '%-50s' "$bar")

        if kill -0 "$pid" 2>/dev/null; then
            status="rodando"
            all_done=false
            if [ "$done" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
                rate=$(awk "BEGIN{print $done/$elapsed}")
                remaining=$(( TOTAL - done ))
                eta_s=$(awk "BEGIN{printf \"%d\",$remaining/$rate}")
                eta=$(printf '%02d:%02d' $((eta_s/60)) $((eta_s%60)))
            else
                eta="--:--"
            fi
        else
            if [ "$done" -ge "$TOTAL" ]; then
                status="OK"
            else
                status="FALHOU"
            fi
            eta="-----"
        fi

        printf "%-22s %4d/%-3d %3d%% [%s] %-8s %s\n" "$tag" "$done" "$TOTAL" "$pct" "$bar" "$status" "$eta"
    done

    echo
    if $all_done; then
        echo "Todos os jobs terminaram."
        break
    fi
    sleep 3
done

ALL_OK=true
for i in $(seq 0 $((N-1))); do
    if ! wait "${PIDS[$i]}"; then
        ALL_OK=false
        echo "❌ ${TAGS[$i]} falhou (ver logs/${TAGS[$i]}_test.log)"
    fi
done

if $ALL_OK; then
    echo
    echo "✅ Todos os 6 modelos completaram com sucesso."
    echo
    echo "Próximo passo: rodar compare_predictions.py em cada um:"
    for tag in "${TAGS[@]}"; do
        echo "  python pipeline/compare_predictions.py --preds responses/${tag}_test.csv --vendor ${tag}_test"
    done
fi
