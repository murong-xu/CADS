# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).




## [2.0.0] - 2026-01-29
### Changed
- Switched to use a `pyproject.toml` with `pixi` as the primary environment
  manager instead of the previous `setup.py` file.
- A number of package versionings and workspaces.
- Moved `./cads` folder to be under `./src/cads` for proper package managment.
- Updated README with instructions for the new project structure and environment management.

### Removed
- License files and `README.md` within the `./cads` folder as they were
  redundant and empty, respectively.
- `setup.py` for creating the 


### Added 
- `pyproject.toml`, `pixi.lock`, `CHANGELOG.md` files.


## [1.0.0] - 2026-01-29
### Changed
- All previous commits and features that were not previously tracked in the changelog history.