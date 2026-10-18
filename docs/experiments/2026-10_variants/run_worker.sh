#!/usr/bin/env bash
# 사용: run_worker.sh <작업자 번호> <작업 목록 파일>
# 작업 목록 한 줄: <이름> <시드> <추가 인자...>
set -u
SP="${SP:-$(pwd)}"   # 작업 폴더: repo_w1, repo_w2, results, cca_run/shim 이 있는 곳
W="$1"; JOBS="$2"
REPO="$SP/cca_exp/repo_w$W"
OUT="$SP/cca_exp/results"
mkdir -p "$OUT"
export PYTHONPATH="$SP/cca_run/shim" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
BASE="--dataset gowalla --model lgn --recdim 32 --layer 1 --epochs 10 --bpr_batch 4096 --testbatch 512 --num_clusters 64 --recluster_interval 128 --cluster_warmup 1024"
while read -r name seed rest; do
  [ -z "$name" ] && continue
  [ -f "$OUT/${name}_s${seed}.csv" ] && { echo "skip $name $seed"; continue; }
  cd "$REPO/code" || exit 1
  rm -f runs/*.csv
  start=$(date +%s)
  python -u main.py $BASE --seed "$seed" $rest > "$OUT/${name}_s${seed}.log" 2>&1
  code=$?
  csv=$(ls -t runs/*.csv 2>/dev/null | head -1)
  [ -n "$csv" ] && cp "$csv" "$OUT/${name}_s${seed}.csv"
  echo "$(date +%H:%M) w$W $name seed=$seed exit=$code $(( ($(date +%s)-start)/60 ))min"
done < "$JOBS"
