"""Extract individual example sections from master README.md"""
import re
import json
from pathlib import Path

# Load level classifications
with open('tools/level_map.json') as f:
    LEVEL_MAP = json.load(f)

def parse_readme():
    """Parse README.md and extract sections by example"""
    with open('README.md') as f:
        content = f.read()

    sections = {}

    # Match sections like ### raw_script.py or ### scene_event_monitor.py
    # Section ends at next ### or ---
    pattern = r'### ([a-z_]+(?:/[a-z_]+)?\.py)\n\n(.*?)(?=\n###|\n---|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    for script_name, section_content in matches:
        # Extract example name without .py
        name = script_name.replace('.py', '')
        sections[name] = section_content.strip()

    return sections

def extract_description(section_text):
    """Extract first paragraph as description"""
    lines = section_text.split('\n\n')
    if lines:
        # Clean up markdown formatting
        desc = lines[0].strip()
        # Remove bold/italic markers for table display
        desc = re.sub(r'\*\*', '', desc)
        desc = re.sub(r'\*', '', desc)
        # Remove newlines for single-line descriptions
        desc = ' '.join(desc.split('\n'))
        return desc
    return ""

def extract_usage(section_text):
    """Extract usage examples and argument tables"""
    # Find ```bash blocks and argument tables
    usage_parts = []

    # Find code blocks
    code_blocks = re.findall(r'```bash\n(.*?)\n```', section_text, re.DOTALL)
    if code_blocks:
        usage_parts.append("```bash\n" + code_blocks[0] + "\n```")

    # Find argument tables
    table_pattern = r'\| Argument.*?\n.*?\n((?:\|.*?\n)+)'
    table_match = re.search(table_pattern, section_text, re.DOTALL)
    if table_match:
        usage_parts.append("\n### Arguments\n\n" + table_match.group(0))

    return '\n\n'.join(usage_parts) if usage_parts else ""

def extract_dependencies(section_text):
    """Extract dependency installation instructions"""
    # Look for pip install commands in code blocks or dependency sections
    deps = []

    # Find pip install commands in bash blocks
    dep_matches = re.findall(r'```bash\npip install (.*?)\n```', section_text, re.DOTALL)
    for dep_match in dep_matches:
        dep_line = dep_match.strip()
        if dep_line and dep_line not in deps:
            deps.append(dep_line)

    return deps if deps else None

if __name__ == '__main__':
    sections = parse_readme()
    print(f"Extracted {len(sections)} sections from README.md")
    for name in sorted(sections.keys()):
        print(f"  - {name}")
