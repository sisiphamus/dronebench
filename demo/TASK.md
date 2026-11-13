# Demo

The brief an agent reads is generated per workspace (`TASK.md` inside it), with the real baseline score and commands filled in.

```bash
dronebench init runs/demo --fixture falcon_v2 --steps 12
dronebench serve runs/demo --open          # terminal 1: live dashboard
dronebench run runs/demo --agent claude    # terminal 2: Claude Code iterates
# comparison line: dronebench init runs/rand --force && dronebench run runs/rand --agent random --steps 12
```

Template: `TASK_TEMPLATE` in `dronebench/workspace.py`.
