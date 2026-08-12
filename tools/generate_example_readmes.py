"""Generate individual example READMEs from master README and level map"""
import re
import json
from pathlib import Path
from extract_readme_sections import parse_readme, extract_description, extract_usage

# Load level classifications
with open('tools/level_map.json') as f:
    LEVEL_MAP = json.load(f)

# Category mapping
CATEGORY_MAP = {
    'fundamentals': 'Fundamentals',
    'character': 'Character',
    'animation': 'Animation',
    'geometry': 'Geometry',
    'export': 'Export',
    'rendering': 'Rendering',
    'ml_data': 'ML/Data',
    'ai_vision': 'AI/Vision',
    'bvh': 'BVH'
}

def find_example_path(example_name):
    """Find which category folder the example lives in"""
    # Handle multi-file examples like comfyui_enhance/main -> rendering/comfyui_enhance
    if '/' in example_name:
        # Extract base example name (comfyui_enhance from comfyui_enhance/main)
        parts = example_name.split('/')
        base_example = parts[0]

        # Try each category directory
        for category_key in CATEGORY_MAP.keys():
            path = Path(category_key) / base_example
            if path.exists() and path.is_dir():
                return path, category_key

    # Try each category directory
    for category_key, category_display in CATEGORY_MAP.items():
        path = Path(category_key) / example_name
        if path.exists() and path.is_dir():
            return path, category_key

    return None, None

def extract_sdk_features(section_text):
    """Extract SDK features demonstrated"""
    features = []

    # Look for "SDK features demonstrated:" section
    sdk_match = re.search(r'\*\*SDK features demonstrated:\*\*(.*?)(?=\n\n|\Z)', section_text, re.DOTALL)
    if sdk_match:
        feature_text = sdk_match.group(1)
        # Parse backtick-delimited features
        feature_lines = re.findall(r'`([^`]+)`', feature_text)
        features = feature_lines

    return features

def generate_readme(example_path, category, example_name, section_text):
    """Generate README for one example"""

    # Get level
    lookup_key = f"{category}/{example_name}"
    level = LEVEL_MAP.get(lookup_key, "Intermediate")

    # Extract components
    description = extract_description(section_text)
    usage = extract_usage(section_text)
    sdk_features = extract_sdk_features(section_text)
    category_display = CATEGORY_MAP.get(category, category)

    # Generate title from example name
    title = example_name.replace('_', ' ').title()

    readme = f"""# {title}

**Level:** {level}
**Category:** {category_display}

## Overview

{description}

## What You'll Learn

- Practical implementation of {example_name} workflow
- SDK patterns for remote DAZ Studio control
- Production-ready code structure
"""

    if sdk_features:
        readme += "\n**SDK features used:**\n"
        for feature in sdk_features:
            readme += f"- `{feature}`\n"

    readme += """
## Prerequisites

- DAZ Studio with DazScriptServer plugin running
- `dazpy` installed (`pip install dazpy`)
- Basic Python knowledge
"""

    # Check for requirements.txt
    req_file = example_path / 'requirements.txt'
    if req_file.exists():
        readme += """
## Dependencies

Install additional dependencies:
```bash
pip install -r requirements.txt
```
"""
    else:
        readme += """
## Dependencies

No additional dependencies beyond `dazpy`.
"""

    # Add usage section
    if usage:
        readme += f"""
## Usage

{usage}
"""

    readme += """
## How It Works

[See code comments in the script for detailed implementation walkthrough]

## Output

[Example-specific output description]

## Related Examples

- See main repository [README](../../README.md) for related examples
"""

    return readme

def main():
    sections = parse_readme()
    print(f"Found {len(sections)} example sections in README.md\n")

    generated = 0
    skipped = 0

    for example_name, section_text in sections.items():
        example_path, category = find_example_path(example_name)

        if not example_path:
            print(f"⚠️  Could not find path for: {example_name}")
            skipped += 1
            continue

        readme_path = example_path / 'README.md'

        # Extract base example name for multi-file examples (comfyui_enhance/main -> comfyui_enhance)
        base_example_name = example_name.split('/')[0]

        # Generate README content
        readme_content = generate_readme(example_path, category, base_example_name, section_text)

        # Write README
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        print(f"✓ Generated: {readme_path}")
        generated += 1

    print(f"\n✓ Generated {generated} READMEs")
    if skipped:
        print(f"⚠️  Skipped {skipped} (path not found)")

if __name__ == '__main__':
    main()
