---
name: coding-standards
description: Use when writing or editing code.
---

# General

- when writing a piece of code, look around to see if there are similar code-snippets.
If there are, extract common logic into a helper function/class/module and reuse it. That is,
proactively follow the DRY principle, even if it involves refactoring existing code.

- when introducing a new entity (function/variable/class/module), make sure the existing neighboring
entities align with the new one. Modify the old ones if necessary to maintain consistency. Example:

```python
# before:
data = calc_load_data()

# after (adding a new entity):
# wrong:
data = calc_load_data()
weather = calc_weather_data()

# better (old entity is renamed to align with the new one):
load_data = calc_load_data()
weather_data = calc_weather_data()
```

- if a function/method occupies more than 30 lines, extract helper functions to break it down
into smaller, manageable pieces.

## Comments

- only add comments when they bring new information that is not easily inferred from the code itself.
Avoid redundant comments that just restate what the code does.

- never hard-code dynamic values in comments (numbers, variable names, file paths, etc.)
