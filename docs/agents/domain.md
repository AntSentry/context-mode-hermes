# Domain docs

## Layout

This is a single-context repository:

- `GLOSSARY.md` at the repository root defines domain vocabulary.
- `docs/adr/` holds architecture decision records.

## Before exploring

Read the glossary, if present, and ADRs relevant to the area being explored.
If these documents do not exist, proceed silently without suggesting that they
be created upfront. Domain-modeling work creates them when concrete terms or
decisions need recording.

## Vocabulary and decisions

Use glossary terms in issue titles, hypotheses, tests and refactor proposals.
If a needed concept is absent, reconsider the term or note the gap for
domain-modeling work.

Explicitly flag proposals that contradict an ADR, naming the decision and
explaining why it should be reopened.
