# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project does not yet use formal version numbers; changes accumulate under
`[Unreleased]` until a versioning scheme is adopted.

## [Unreleased]

### Changed
- Renamed the wire-format dataclass `ProfileData` → `ProfileDataRaw` in
  `udpCapturing.py` to remove the name collision with the analysis `profileData`
  (`profilePointsClass.py`). All references within the module were updated;
  behaviour and the HDF5 schema are unchanged.

### Added
- Project documentation and scaffolding (no runtime code changed):
  - `README.md` — project overview, quickstart, data flow, documentation map.
  - `ARCHITECTURE.md` — components, module dependency graph, execution flows,
    coordinate convention, HDF5 schema, and a known-technical-debt register.
  - `CLAUDE.md` — coding standards and conventions for future AI/human sessions.
  - `TODO.md` — prioritised backlog of suggested (not-yet-applied) improvements.
  - `CHANGELOG.md` — this file.
  - `requirements.txt` — runtime dependencies derived from the source imports.
  - `.gitignore` — Python, editor, and project data/output exclusions.

### Notes
- This pass is documentation-only by design; all identified code issues
  (e.g. the `.y`/`.z` mismatch in dormant registration code, wildcard imports,
  magic constants) are catalogued in `TODO.md` rather than changed.
