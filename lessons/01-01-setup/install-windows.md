---
title: "Install the tools: Windows"
description: "Set up WSL 2 on Windows, check it works, connect VS Code to it, then install git and uv inside it, for the setup lesson."
---

# Install the tools: Windows

This series runs on macOS and Linux. On Windows, you work inside **WSL**,
the Windows Subsystem for Linux: a real Linux system running alongside
Windows. Later lessons give the agent a workspace that only works on macOS
and Linux, so a native Windows setup would stop working partway through the
series. With WSL, every lesson works the same as on Linux.

> We have not run these steps on Windows: everything here comes from
> Microsoft's and VS Code's documentation, linked below. If a step does not
> do what it says, the lesson has a gap: please report it.

Installing WSL can take longer than the lesson itself, and is not counted in
its 15 minutes. You do it once.

## How WSL works

In Microsoft's words, WSL "allows you to run a Linux environment on your
Windows machine". WSL 2, the version installed by default, runs a real Linux
kernel inside a lightweight virtual machine that Windows starts and stops for
you. You do not manage the virtual machine: you open a Linux terminal, and
it is there.

The two systems share files:

- From Linux, your Windows drives appear under `/mnt/`: `C:\Users` is
  `/mnt/c/Users`.
- From Windows, your Linux files appear in File Explorer: type `\\wsl$` in
  the address bar.

**Keep your projects in Linux, in your Linux home folder (`~`), not under
`/mnt/c`.** Microsoft: "For the fastest performance speed, store your files
in the WSL file system if you are working in a Linux command line." Every
command in this series runs in the Linux terminal, so the project belongs
there.

Sources: [What is WSL?](https://learn.microsoft.com/en-us/windows/wsl/about),
[Working across file systems](https://learn.microsoft.com/en-us/windows/wsl/filesystems).

## Install WSL

You need Windows 11, or Windows 10 version 2004 or newer (build 19041 or
newer).

1. Open PowerShell **as administrator**: right-click it in the Start menu
   and choose **Run as administrator**.
2. Run:

   ```powershell
   wsl --install
   ```

   This turns on the features WSL needs, sets WSL 2 as the default, and
   installs the Ubuntu distribution of Linux.
3. Restart Windows when it asks.
4. Open **Ubuntu** from the Start menu. The first time, it finishes
   installing and asks you to create a Linux user name and password:
   - They are for Linux only, and have nothing to do with your Windows
     account.
   - Nothing appears on screen while you type the password. That is normal.
   - You will need the password whenever a command starts with `sudo`.

Source: [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install),
[Set up a WSL development environment](https://learn.microsoft.com/en-us/windows/wsl/setup/environment).

## Check that it works

**From PowerShell** (it does not need to be administrator), check that
Ubuntu runs on WSL 2:

```powershell
wsl -l -v
```

You should see `Ubuntu` listed, with `2` in the `VERSION` column. If it says
`1`, switch it with `wsl --set-version Ubuntu 2`.

**In the Ubuntu terminal**, bring Ubuntu's packages up to date, entering
your Linux password when asked:

```bash
sudo apt update && sudo apt upgrade
```

Then check where you are:

```bash
cd ~
pwd
```

You should see `/home/` followed by your Linux user name. That is your Linux
home folder: make the lesson's project here.

Microsoft recommends [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/)
for working with WSL: each Linux distribution you install gets its own entry
in it. The Ubuntu app from the Start menu works too.

## Connect VS Code to WSL

If you use VS Code, it runs on Windows and works on your Linux files through
an extension. Do not install VS Code inside Linux.

1. Install [VS Code](https://code.visualstudio.com/) **on Windows**. In the
   installer, keep **Add to PATH** checked, so you can open it from the
   Ubuntu terminal.
2. In VS Code, open the Extensions view, search for **WSL**, and install
   the extension by Microsoft (identifier `ms-vscode-remote.remote-wsl`).
   It is also part of the **Remote Development** extension pack.
3. In the Ubuntu terminal, go to a folder and open it in VS Code:

   ```bash
   cd ~
   code .
   ```

   The first time, VS Code installs its server inside WSL, which takes a
   moment. Then a VS Code window opens on your Linux folder, with a **WSL**
   indicator in its bottom-left corner. A terminal opened in that window is
   an Ubuntu terminal.
4. Extensions that work on your code, such as the Python extension, must be
   installed again for WSL: with the WSL window open, install them from its
   Extensions view. Themes need installing only once.

If `code .` is not found, VS Code was installed without **Add to PATH**:
reinstall it with that box checked, then open a new Ubuntu terminal.

Sources: [Get started using VS Code with WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode),
[Developing in WSL](https://code.visualstudio.com/docs/remote/wsl).

## Then follow the Linux guide

From here on, everything happens inside Ubuntu: follow
[Install the tools: Linux](install-linux.md), then go back to the lesson
and run every command there in the Ubuntu terminal, starting in your Linux
home folder (`cd ~`).
