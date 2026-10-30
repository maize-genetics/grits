---
tags: [skill-creator]
max_turns: 60
timeout_seconds: 3600
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

I have low-pass (~0.1x) whole-genome reads from the maize inbred Ia453 at $GRITS_WORKDIR/scratch/lift_s200_local20k/eval__OUT-INBRED__Ia453__0.1x/reads.fastq . Can you run GRITS on it and tell me which founders it looks like along each chromosome? Please use at most 8 CPU threads and only GPU 0 (CUDA_VISIBLE_DEVICES=0); put outputs under $GRITS_WORKDIR/scratch/evalruns/$RUN/ where $RUN is your run label.
