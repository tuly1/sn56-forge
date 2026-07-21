# Axolotl chat-template snapshot

The `.jinja` files in this directory are an unmodified snapshot of Axolotl's
named chat-template registry at commit
`0bda5a13e4d52ceec58104f44fabb7bd314f9c02`, the OCI revision corresponding to
the current G.O.D evaluator image (`axolotlai/axolotl:main-20260701`). They are
vendored because the trainer runs offline and does not otherwise depend on
Axolotl.

The templates are byte-identical in the later Axolotl source-audit snapshot
`f3be6690490ced965bfbadcbc86462ee36e00201`.

Source path:
`src/axolotl/utils/chat_templates/templates/`

Axolotl is licensed under the Apache License 2.0. The project-level `NOTICE`
records this attribution.
