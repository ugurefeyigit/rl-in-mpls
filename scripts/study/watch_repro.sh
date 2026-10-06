#!/bin/bash
# Evaluate each reproduced run as soon as its training finishes.
cd /home/user/rl-in-mpls
R=.worktrees/repro/runs/repro
while true; do
  for d in $R/seed*/; do
    [ -f $d/manifest.json ] || continue
    grep -q '"status": "completed"' $d/manifest.json || continue
    [ -f $d/eval/selection_study.json ] && continue
    [ -f $d/.eval_started ] && continue
    touch $d/.eval_started
    nice -n 2 python scripts/study/eval_repro.py $d > logs/eval_repro_$(basename $d).log 2>&1 &
  done
  n=$(ls -d $R/seed*/ 2>/dev/null | wc -l); m=$(ls $R/seed*/eval/selection_study.json 2>/dev/null | wc -l)
  [ "$n" -ge 6 ] && [ "$m" -ge 6 ] && break
  sleep 60
done
