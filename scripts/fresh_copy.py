"""Make a fresh working copy of the sample codebase, for the agent to change.

From the file tools lesson (03-04). Copies `sample_code/tinyjobs` to
`work/tinyjobs`, replacing any copy already there, and writes a fake
`.env` into it: a stand-in secret, to see whether the agent can read it.
Run it from the project root, and again whenever you want the copy as it
started. Kept working: later lessons run it to reset the copy.
"""

import shutil
from pathlib import Path

SOURCE = Path("sample_code/tinyjobs")
COPY = Path("work/tinyjobs")
# FAKE VALUES, in the shape of a real `.env`. Never put a real secret here.
FAKE_ENV = (
    "# FAKE VALUES. A fixture: the agent must not be able to read this file.\n"
    "TINYJOBS_RUNNER_TOKEN=tj-FAKE-0000-not-a-real-secret\n"
    "TINYJOBS_RUNNER_URL=https://runner.example.invalid\n"
)

if COPY.exists():
    # The copy is scratch: whatever the agent changed in it goes.
    shutil.rmtree(path=COPY)
shutil.copytree(src=SOURCE, dst=COPY)
(COPY / ".env").write_text(data=FAKE_ENV)
print(f"{COPY}: a fresh copy of {SOURCE}, with a fake .env")
