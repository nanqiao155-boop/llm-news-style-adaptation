# Public demo

From the repository root: `python -m demo.app` after installing `requirements.txt`.
The server binds to loopback and never creates a public sharing URL.

Synthetic offline is the default: scripted responses exercise the actual Writer parser,
Reviewer/Reviser schemas, conditional workflow and scoring function. The checkbox
switches between PASS (Final is the identical Draft) and FAIL (one revision).
All fixture scores are labelled synthetic; they are not research measurements.

Optional online mode uses environment variables listed in `.env.example`.
Supply your own permitted Writer deployment and Judge service locally.
The Writer endpoint must actually serve the intended repaired adapter; the UI cannot
verify its identity. Empty configuration fails closed. Credentials are never bundled.
Online mode sends the input facts to the configured services only when Run is clicked.
