# Chapter 3: The application and a workspace

In Chapter 2, you turned an agent into a chat. It can carry a conversation, stream its answers, and search the web. It works—but so far it has made most of its decisions in Python code, and its only way to help is to answer you.

In this chapter, we'll start treating that chat as an **application**. First, we'll decide how it gets its configuration: which model to use, which features are available, and where credentials come from. We'll assemble the pieces at startup instead of leaving important decisions until the first message.

Then we'll give the chat a place to work. A *workspace* identifies a folder; file tools let the model request reads and changes; a shell lets it request commands. Each addition gives the chat more reach. Each also raises a question: **who decides what it may do?**

This is a little different from learning another agent feature. We're beginning to make choices that matter in real applications: keeping configuration separate from code, constructing components in a predictable place, and distinguishing a convenient restriction from a real security boundary.

You'll keep working on the **same chat application**. Each lesson adds one piece and gives you a way to see what changed. You don't need to know application architecture before you start—we'll build the picture as we go.

Start with [03-01: Settings and a composition root](../03-01-settings/).
