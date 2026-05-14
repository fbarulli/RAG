---
id: 10316840ce
question: 'Docker: The input device is not a TTY when running docker in interactive
  mode (Running Docker on Windows in GitBash)'
sort_order: 19
---

```bash
docker exec -it 1e5a1b663052 bash
```

Error:
```
the input device is not a TTY. If you are using mintty, try prefixing the command with 'winpty'
```

Fix:

```bash
winpty docker exec -it 1e5a1b663052 bash
```

A TTY is a terminal interface that supports escape sequences, moving the cursor around, etc. Winpty is a Windows software package providing an interface similar to a Unix pty-master for communicating with Windows console programs.

For more information on terminal, shell, and console applications:

[https://conemu.github.io/en/TerminalVsShell.html](https://conemu.github.io/en/TerminalVsShell.html)