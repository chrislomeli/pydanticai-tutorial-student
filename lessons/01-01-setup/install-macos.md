---
title: "Install the tools: macOS"
description: "Install or upgrade git and uv on macOS, for the setup lesson."
---

# Install the tools: macOS

The setup lesson needs `git`, and `uv` 0.13.1 or newer. You do not need to
install Python: `uv` downloads the version the project asks for. When both
checks below pass, go back to the lesson.

> Tested on macOS (Apple silicon) with `uv` 0.13.1.

## Check `git`

Run:

```bash
git --version
```

You should see a version, such as:

```text
git version 2.50.1 (Apple Git-155)
```

Any recent git version works. If the command is not found, or macOS says
no developer tools were found and offers to install them, install Apple's
command-line developer tools, which include `git`:

```bash
xcode-select --install
```

git also needs to know your name and email, to record them in your
commits. If you have never set them on this machine, do it now:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## Install or upgrade `uv`

[`uv`](https://docs.astral.sh/uv/) is the tool that manages the project:
it downloads Python, creates the virtual environment, installs packages and
runs your commands inside the environment. Check whether you have it, and
which version.

Run:

```bash
uv --version
```

If you have it, you see its version:

```text
uv 0.13.1 (62faed335 2026-10-13 aarch64-apple-darwin)
```

The lesson was tested with 0.13.1. What to do next depends on what you
saw.

**`command not found`.** Install it with the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You should see:

```text
downloading uv 0.13.1 aarch64-apple-darwin
installing to /Users/you/.local/bin
  uv
  uvx
everything's installed!

To add $HOME/.local/bin to your PATH, either restart your shell or run:

    source $HOME/.local/bin/env (sh, bash, zsh)
    source $HOME/.local/bin/env.fish (fish)
```

Open a new terminal, and run `uv --version` again.

**A version older than 0.13.1.** Ask `uv` to update itself:

```bash
uv self update
```

If `uv` was installed with the official installer, this upgrades it, or
tells you it is already current:

```text
info: Checking for updates...
success: You're already on version v0.13.1 of uv (the latest version).
```

If it was installed some other way, it refuses, and says so:

```text
error: Self-update is only available for uv binaries installed via the standalone installation scripts.

If you installed uv with pip, brew, or another package manager, update uv with `pip install --upgrade`, `brew upgrade`, or similar.
```

Then upgrade it the way it was installed: `brew upgrade uv` for Homebrew,
`pipx upgrade uv` for pipx, or `pip install --upgrade uv` in the
environment you installed it into. Run `uv --version` again afterwards.

**0.13.1 or newer.** You are ready. A newer `uv` may write files that
differ slightly from the ones shown in the lesson, such as a version number
in `pyproject.toml`; that is fine.
