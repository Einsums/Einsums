---
name: New version
about: Template with a checklist for minting a new version.
title: 'VERSION:'

---

## Checklist:

- [ ] Run the pre-commit hook.
- [ ] Make sure the change logs are up to date.
- [ ] Set the version numbers and date string in `CMakeLists.txt`.
- [ ] Ensure that the builds are passing in the most recent PR, especially the `Consume Installation` steps.

If no other changes have been made, this pull request may be merged and a new release generated.