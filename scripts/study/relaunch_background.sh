#!/bin/bash
# (Re)launch every background job of the study. Safe to run repeatedly: all
# jobs skip completed work (finished runs, cached evaluations, per-episode files).
cd /home/user/rl-in-mpls
echo 0 > /proc/sys/kernel/sched_autogroup_enabled 2>/dev/null
mkdir -p logs runs/study
running() { ps -eo args | grep -v grep | grep -q -- "$1"; }
running "scheduler.py --slots" || nohup python scripts/study/scheduler.py --slots 4 --ids E4_delay E1_main E2_horizon__ppog0__r42 E2_horizon__ppog0__r314159 E2_horizon__ppog0__r271828 E2_horizon__qg09__r42 E2_horizon__qg09__r314159 E2_horizon__qg09__r271828 E4_delay E7_time E2_horizon__ppog09__r42 E2_horizon__ppog09__r314159 E2_horizon__ppog09__r271828 E3_ppo_tuning E2_horizon E6_shaping E5_budget >> runs/study/scheduler.log 2>&1 &
running "run_mask_audit.py" || [ -f experiments/raw/mask_audit/summary.json ] || nohup python scripts/study/run_mask_audit.py > logs/mask_audit2.log 2>&1 &
running "run_seqdiag.py" || nohup bash -c "python scripts/study/run_seqdiag.py --reference greedy --out experiments/raw/seqdiag_greedy >> logs/seqdiag_greedy.log 2>&1; python scripts/study/run_seqdiag.py --reference greedy --frozen --out experiments/raw/seqdiag_greedy_frozen >> logs/seqdiag_greedy_frozen.log 2>&1" > /dev/null 2>&1 &
running "seqdiag_greedy_delay1" || nohup bash -c 'nice -n 15 python scripts/study/run_seqdiag.py --reference greedy --env "{\"variant\": \"delayed\", \"delay_steps\": 1}" --frozen --out experiments/raw/seqdiag_greedy_delay1_frozen >> logs/seqdiag_delay1_frozen.log 2>&1; nice -n 15 python scripts/study/run_seqdiag.py --reference greedy --env "{\"variant\": \"delayed\", \"delay_steps\": 1}" --out experiments/raw/seqdiag_greedy_delay1 >> logs/seqdiag_delay1.log 2>&1' > /dev/null 2>&1 &
running "policies oracle_h" || nohup bash -c "python scripts/study/run_oracles.py --policies oracle_h1 --seedset test --out experiments/raw/references >> logs/oracle_h1.log 2>&1; python scripts/study/run_oracles.py --policies oracle_h3 oracle_h6 --seeds 3001 3002 3003 3004 3005 --out experiments/raw/references >> logs/oracle_h36.log 2>&1" > /dev/null 2>&1 &
running "model_fidelity.py" || nohup python scripts/study/model_fidelity.py --runs E0_repro__bandit__r42 E0_repro__ppo__r42 E0_repro__bandit__r314159 E0_repro__ppo__r314159 E0_repro__bandit__r271828 E0_repro__ppo__r271828 >> logs/model_fidelity.log 2>&1 &
sleep 2; ps -eo args | grep -E "python scripts" | grep -v grep | cut -c1-110
running "milp_grid" || nohup nice -n 10 python scripts/study/run_oracles.py --policies milp_g000_largest milp_g020_largest milp_g050_largest milp_g100_largest milp_g000_smallest milp_g020_smallest milp_g050_smallest milp_g100_smallest --seedset validation --out experiments/raw/references_validation/milp_grid >> logs/milp_grid.log 2>&1 &
