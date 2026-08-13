# Repository Setup Documentation

This document describes the extraction and setup of daz-script-server-examples.

## Extraction Process

1. Created new repository: `bluemoonfoundry/daz-script-server-examples`
2. Copied examples from `daz-script-server/docs/examples/`
3. Restructured single-file examples to folder-per-example
4. Generated individual READMEs using standard template
5. Created requirements.txt for examples with external dependencies
6. Generated comprehensive root README with examples table
7. Added LICENSE (AGPL v3) and CONTRIBUTING guidelines
8. Set GitHub topics and description

## Repository Structure

```
daz-script-server-examples/
├── README.md                    # Comprehensive table of all examples
├── LICENSE                      # AGPL v3
├── CONTRIBUTING.md              # Contribution guidelines
├── docs/
│   ├── SETUP.md                # This file
│   └── OLD_README.md           # Archived original README
├── tools/                      # Generation scripts
│   ├── level_map.json          # Level classifications
│   ├── extract_readme_sections.py
│   ├── generate_example_readmes.py
│   └── generate_table.py
├── fundamentals/               # 6 examples
├── character/                  # 4 examples
├── animation/                  # 3 examples
├── geometry/                   # 2 examples
├── export/                     # 1 example
├── rendering/                  # 8 examples
├── ml_data/                    # 1 example
├── ai_vision/                  # 2 examples
└── bvh/                        # 3 examples (in development)
```

## Examples Count

- Total: 30 examples
- Beginner: 6
- Intermediate: 12
- Advanced: 12

## Links

- Parent repository: https://github.com/bluemoonfoundry/daz-script-server
- dazpy PyPI: https://pypi.org/project/dazpy/
- Documentation: https://bluemoonfoundry.github.io/daz-script-server/
