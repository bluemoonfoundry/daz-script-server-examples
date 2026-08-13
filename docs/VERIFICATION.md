# Repository Verification Checklist

Run these checks to verify repository setup:

## Structure Verification

- [ ] All 9 category directories exist
- [ ] Each example has its own folder
- [ ] Each example folder contains Python script(s)
- [ ] Each example has README.md
- [ ] Examples with external deps have requirements.txt

```bash
# Count examples per category
for cat in fundamentals character animation geometry export rendering ml_data ai_vision bvh; do
  count=$(find $cat -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  echo "$cat: $count examples"
done

# Verify all have READMEs
find . -mindepth 2 -maxdepth 2 -type d ! -path "./docs/*" ! -path "./tools/*" | while read dir; do
  if [ ! -f "$dir/README.md" ]; then
    echo "Missing README: $dir"
  fi
done
```

## Documentation Verification

- [ ] Root README has comprehensive table
- [ ] Root README has categories section
- [ ] Root README has skill level guide
- [ ] LICENSE file present (AGPL v3)
- [ ] CONTRIBUTING.md present
- [ ] All example READMEs follow template

```bash
# Check root README structure
grep -c "## Overview" README.md
grep -c "## Categories" README.md
grep -c "## Skill Level Guide" README.md

# Check example READMEs have required sections
for readme in */*/README.md; do
  if ! grep -q "## Overview" "$readme"; then
    echo "Missing Overview: $readme"
  fi
done
```

## Metadata Verification

- [ ] GitHub topics set
- [ ] Repository description set
- [ ] .gitignore configured
- [ ] All commits pushed

```bash
# Check remote status
git status
git log --oneline | head -10

# Verify topics (requires gh CLI)
gh repo view bluemoonfoundry/daz-script-server-examples --json repositoryTopics
```

## Parent Repository Updates

- [ ] MOVED.md created in parent docs/examples/
- [ ] Parent README updated with link to new repo
- [ ] Parent repo changes committed

```bash
cd ../daz-script-server
git log --oneline | head -3
cat docs/examples/MOVED.md
```
