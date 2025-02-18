# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-02-18
### Added
- new checkpoint with support for cystine residues

### Fixed
- fix UB bug that arises with consecutive cystine residues

## [1.0.0] - 2025-02-07
### Added
- option to exclude solvent in output (`--[no-]write-solvent`)
- check number of residues in input file; fail gracefully with helpful message

### Changed
- avoid usage of deprecated function (move socket file for async servers)
- sanitise logging; `--verbose` controls all output

## [0.1.0] - 2024-11-07
### Added
- initial public release
