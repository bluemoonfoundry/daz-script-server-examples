"""Generate comprehensive examples table for root README"""
import json
from pathlib import Path

# Load level map
with open('tools/level_map.json') as f:
    LEVEL_MAP = json.load(f)

def find_all_examples():
    """Find all example directories"""
    examples = []

    categories = ['fundamentals', 'character', 'animation', 'geometry',
                  'export', 'rendering', 'ml_data', 'ai_vision', 'bvh']

    for category in categories:
        cat_path = Path(category)
        if not cat_path.exists():
            continue

        for example_dir in sorted(cat_path.iterdir()):
            if example_dir.is_dir() and (example_dir / 'README.md').exists():
                example_name = example_dir.name
                lookup_key = f"{category}/{example_name}"
                level = LEVEL_MAP.get(lookup_key, "Intermediate")

                # Extract description from README
                readme_path = example_dir / 'README.md'
                with open(readme_path) as f:
                    readme_content = f.read()
                    # Extract Overview section
                    import re
                    desc_match = re.search(r'## Overview\n\n(.*?)(?=\n##|\Z)', readme_content, re.DOTALL)
                    description = desc_match.group(1).strip()[:100] + "..." if desc_match else ""

                # Check for requirements.txt
                has_deps = (example_dir / 'requirements.txt').exists()
                deps = "Yes" if has_deps else "None"

                examples.append({
                    'name': example_name,
                    'category': category,
                    'level': level,
                    'description': description,
                    'deps': deps,
                    'path': f"{category}/{example_name}"
                })

    return examples

def generate_table(examples):
    """Generate markdown table"""
    table = ["| Example | Category | Level | Description | Dependencies |",
             "|---------|----------|-------|-------------|--------------|"]

    for ex in examples:
        name_link = f"[{ex['name']}]({ex['path']}/)"
        desc = ex['description'].replace('\n', ' ')
        row = f"| {name_link} | {ex['category']} | {ex['level']} | {desc} | {ex['deps']} |"
        table.append(row)

    return '\n'.join(table)

if __name__ == '__main__':
    examples = find_all_examples()
    table = generate_table(examples)
    print(table)
