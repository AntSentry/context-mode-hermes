"""Controlled, reproducible benchmarks for the context-mode Hermes adapter.

Everything here drives the real pinned Hermes agent + native plugin/MCP dispatch
against a scripted OpenAI-compatible fake provider on loopback. It measures wire
request bytes, rough token estimates and tool plumbing — it is NOT an LLM
quality or speed benchmark and never talks to a real model.
"""
