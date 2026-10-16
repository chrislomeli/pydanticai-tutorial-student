---
title: "Install the tools: Windows"
description: "Set up WSL on Windows, then install git and uv inside it, for the setup lesson."
---

# Install the tools: Windows

This series runs on macOS and Linux. On Windows, you work inside WSL (the
Windows Subsystem for Linux), which gives you a Linux system alongside
Windows. Later lessons give the agent a workspace that only works on macOS
and Linux, so a native Windows setup would stop working partway through the
series.

> We have not run these steps. They are the standard commands, but if one
> fails for you, the lesson has a gap: please report it.

## Install WSL

In PowerShell, run as administrator:

```powershell
wsl --install
```

Restart when it asks, and open **Ubuntu** from the Start menu. Installing
WSL can take longer than the lesson itself, and is not counted in its 15
minutes.

## Then follow the Linux guide

From here on, everything happens inside Ubuntu: follow
[Install the tools: Linux](install-linux.md), then go back to the lesson
and run every command there inside Ubuntu too.
