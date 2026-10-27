# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `AntSentry/context-mode-hermes`.
Use the `gh` CLI with `--repo AntSentry/context-mode-hermes`.

## Conventions

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --comments`
- List: `gh issue list --state open --json number,title,body,labels,comments`
- Comment: `gh issue comment <number> --body "..."`
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Close: `gh issue close <number> --comment "..."`

Append the repository flag above to these commands. Use the label mapping in
`docs/agents/triage-labels.md`.

When a skill says "publish to the issue tracker", create a GitHub issue.
When it says "fetch the relevant ticket", read the issue with its comments.

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub issues and PRs share a number space. Resolve an ambiguous reference with
`gh pr view <number>` before falling back to `gh issue view <number>`.

## Wayfinding operations

- Map: one issue labelled `wayfinder:map`, containing Notes, Decisions-so-far and Fog.
- Child tickets: GitHub sub-issues labelled `wayfinder:<type>`; use a map task list
  and `Part of #<map>` in child bodies when sub-issues are unavailable.
- Blocking: use native issue dependencies when available; otherwise record
  `Blocked by: #<number>` in the child body. Every blocker must be closed.
- Frontier: choose the first open, unblocked, unassigned child in map order.
- Claim: assign the ticket to the driving developer before beginning work.
- Resolve: comment with the answer, close the child, and add a gist plus link to
  the map's Decisions-so-far.
