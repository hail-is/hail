#!/usr/bin/env python3
"""Verify that every dependsOn entry in build.yaml refers to a step defined earlier in the file.

Steps are resolved in declaration order at CI startup — a forward reference is silently
dropped, producing a broken DAG where the dependency is never enforced.
"""
import sys

import yaml


def main():
    with open('build.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    seen: set = set()
    errors: list = []

    for step in config.get('steps', []):
        name = step.get('name', '<unnamed>')
        for dep in step.get('dependsOn', []):
            if dep not in seen:
                errors.append(f"  '{name}' dependsOn '{dep}', which is defined later or missing")
        seen.add(name)

    if errors:
        print(f"build.yaml ordering errors ({len(errors)} found):")
        for e in errors:
            print(e)
        sys.exit(1)

    print(f"OK: all dependsOn entries in build.yaml refer to earlier-defined steps ({len(seen)} steps checked)")


if __name__ == '__main__':
    main()
