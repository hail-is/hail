#!/usr/bin/env python3
"""
Verify that no VM with a public IP in the project has ALLOW ingress rules
that permit traffic from public (non-private) source ranges.

Usage:
    python3 devbin/check_public_ip_no_ingress.py --project <gcp-project-id>

Requires: google-api-python-client (already a hailtop dependency)
Credentials: application default credentials (run `gcloud auth application-default login`)
"""

import argparse
import ipaddress
import sys
from typing import Iterator

from googleapiclient import discovery


_PRIVATE_NETWORKS = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('169.254.0.0/16'),
]


def is_private_cidr(cidr: str) -> bool:
    try:
        net = ipaddress.ip_network(cidr, strict=False)
        return any(net.subnet_of(p) for p in _PRIVATE_NETWORKS)
    except ValueError:
        return False


def iter_instances_with_public_ip(compute, project: str) -> Iterator[dict]:
    """Yield all VM instances that have at least one external IP."""
    request = compute.instances().aggregatedList(project=project)
    while request is not None:
        response = request.execute()
        for zone_data in response.get('items', {}).values():
            for instance in zone_data.get('instances', []):
                for iface in instance.get('networkInterfaces', []):
                    for ac in iface.get('accessConfigs', []):
                        if ac.get('natIP'):
                            yield instance
                            break
                    else:
                        continue
                    break
        request = compute.instances().aggregatedList_next(
            previous_request=request, previous_response=response
        )


def get_firewall_rules(compute, project: str) -> list[dict]:
    rules = []
    request = compute.firewalls().list(project=project)
    while request is not None:
        response = request.execute()
        rules.extend(response.get('items', []))
        request = compute.firewalls().list_next(previous_request=request, previous_response=response)
    return rules


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', required=True, help='GCP project ID to scan')
    args = parser.parse_args()

    compute = discovery.build('compute', 'v1')
    project = args.project

    firewall_rules = get_firewall_rules(compute, project)
    public_ip_vms = list(iter_instances_with_public_ip(compute, project))

    ingress_allow_rules = [
        r for r in firewall_rules
        if 'allowed' in r
        and r.get('direction', 'INGRESS') == 'INGRESS'
        and not r.get('disabled', False)
    ]
    private_firewall_rules = {
        r['name'] for r in ingress_allow_rules
        if all(is_private_cidr(cidr) for cidr in r.get('sourceRanges', []))
    }

    violations = []

    for instance in public_ip_vms:
        name = instance['name']
        zone = instance['zone'].split('/')[-1]
        external_ips = [
            ac['natIP']
            for iface in instance.get('networkInterfaces', [])
            for ac in iface.get('accessConfigs', [])
            if ac.get('natIP')
        ]
        vm_tags = set(instance.get('tags', {}).get('items', []))
        applicable_rules = [
            r for r in ingress_allow_rules
            if not set(r.get('targetTags', [])) or set(r.get('targetTags', [])) & vm_tags
        ]
        open_rules = [r['name'] for r in applicable_rules if r['name'] not in private_firewall_rules]
        if open_rules:
            violations.append(
                f'  VIOLATION  {name} ({zone}) public IP(s) {external_ips} '
                f'is reachable from public internet via rule(s): {open_rules}'
            )
        else:
            print(f'  OK         {name} ({zone}) public IP(s) {external_ips}')

    if violations:
        print()
        print(f'Found {len(violations)} violation(s):')
        for v in violations:
            print(v)
        sys.exit(1)
    else:
        print()
        print(f'All {len(public_ip_vms)} public-IP VM(s) have no public-source ALLOW ingress rules. No violations.')


if __name__ == '__main__':
    main()
