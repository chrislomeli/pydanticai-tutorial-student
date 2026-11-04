# Pick your operating system

*A very short page. No tools, no code — just a signpost.*

The next few pages install the tools the series needs. But this series
reaches every kind of machine, so there is one page per operating
system. You read **only the one that matches yours**, then move on.

Here's the picture:

- **macOS and Linux.** Everything in this series runs the same on any
  Unix-like system, and these are the two we test on. If either one is
  your machine, you have the easy path: install two tools and you're
  done.
- **Windows.** You can absolutely follow the series on Windows — you
  just do it inside **WSL 2**, the Windows Subsystem for Linux, which
  runs a real Linux system alongside Windows. It's an excellent piece
  of engineering, and once it's set up every lesson works exactly as it
  does on Linux. It does take a little setup first, so the Windows page
  is the longest of the three.

Why not native Windows? Later in the series your agent gets a workspace
of its own, and that only works on Unix-like systems. WSL keeps every
lesson working, start to finish — it isn't a workaround, it's the
recommended way.

Pick your page:

- [Install the tools: macOS](../00-04-macos-installs/)
- [Install the tools: Linux](../00-05-linux-installation/)
- [Install the tools: Windows](../00-06-windows-installation/)

## Next

When your install page's checks pass, skip the rest of this chapter and go
straight to [Setting up the environment](../01-01-setup/).
The last page of Chapter 0, [the example you'll work
with](../00-07-the-tinyjobs-example/), is a short reading you can take
any time — before or after setup.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
