# IGN Zelda Walkthrough Special Set

This directory contains a Formal4 special set based on the previously evaluated Online-Mind2Web task `b320c68bffc1f3c7f2a8dc9d5478fb27`.

- `tasks.jsonl` is the runnable dataset and contains only fields from the current task schema.
- The static variant preserves the original Online-Mind2Web instruction, and the additive variant preserves the existing Formal3 design.
- `override_section` changes from the BOTW Shrines guide to the full Walkthrough.
- `override_game` corrects the game from Tears of the Kingdom to Breath of the Wild.
- `conditional` keeps the main BOTW Guide unless it exposes a dedicated full-game Walkthrough section.
- `design_metadata.json` records temporary grouping, rationale, and validation status; the runner does not read it.

The three newly designed dynamic variants are candidates, not validated benchmark items. This set remains isolated under `special/` until each task passes live execution and evaluator checks.
