# Contributing Examples

## New Example Checklist

- [ ] Example follows standard README template
- [ ] Code includes docstrings for key functions/classes
- [ ] requirements.txt included if dependencies needed beyond dazpy
- [ ] Tested against current dazpy stable release
- [ ] Fits existing category or proposes new category with rationale
- [ ] Skill level classification with justification
- [ ] No hard-coded paths (use arguments or environment variables)
- [ ] Error handling for common failure cases
- [ ] Example is self-contained (no external file dependencies except documented)

## Example Requirements

### 1. README must include:

- Level (Beginner/Intermediate/Advanced) with rationale
- Category
- Overview (2-3 sentences)
- What You'll Learn section
- Prerequisites
- Dependencies
- Usage with argument table
- How It Works walkthrough
- Output description
- SDK Features Demonstrated
- Related Examples (if applicable)

### 2. Code must:

- Use argparse for command-line arguments
- Include docstrings
- Handle errors gracefully with clear messages
- Follow PEP 8 style
- Use type hints where helpful (not required for Beginner examples)

### 3. Dependencies:

- Minimize external dependencies
- Pin major version only (e.g., `package>=1.0`)
- Document why each dependency is needed in requirements.txt comments

## Proposing New Categories

New categories require:
- Minimum 3 examples demonstrating the category's scope
- Clear differentiation from existing categories
- Rationale for why examples don't fit existing categories

## Review Process

1. Fork repo and create feature branch
2. Add example following template
3. Test against current dazpy stable release
4. Submit PR with example checklist completed
5. Maintainer review for:
   - Documentation completeness
   - Code quality
   - Level classification accuracy
   - Category fit

## Questions?

Open an issue in [daz-script-server repository](https://github.com/bluemoonfoundry/daz-script-server/issues)
