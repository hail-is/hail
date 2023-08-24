import abc
import json
import yaml

from typing import List, Optional


class CleanupPolicy(abc.ABC):
    @abc.abstractmethod
    def to_dict(self):
        pass


class DeletePolicy(CleanupPolicy):
    def __init__(self,
                 name: str,
                 tag_state: str,
                 *,
                 tag_prefixes: Optional[List[str]] = None,
                 version_name_prefixes: Optional[List[str]] = None,
                 package_name_prefixes: Optional[List[str]] = None,
                 older_than: Optional[str] = None,
                 newer_than: Optional[str] = None):
        self.name = name
        self.tag_state = tag_state
        self.tag_prefixes = tag_prefixes
        self.version_name_prefixes = version_name_prefixes
        self.package_name_prefixes = package_name_prefixes
        self.older_than = older_than
        self.newer_than = newer_than

    def to_dict(self):
        data = {'name': self.name, 'action': {'type': 'Delete'}, 'condition': {'tagState': self.tag_state}}
        condition = data['condition']
        if self.tag_prefixes is not None:
            condition['tagPrefixes'] = self.tag_prefixes
        if self.version_name_prefixes is not None:
            condition['versionNamePrefixes'] = self.version_name_prefixes
        if self.package_name_prefixes is not None:
            condition['packageNamePrefixes'] = self.package_name_prefixes
        if self.older_than:
            condition['olderThan'] = self.older_than
        if self.newer_than:
            condition['newerThan'] = self.newer_than
        return data


class ConditionalKeepPolicy(CleanupPolicy):
    def __init__(self,
                 name: str,
                 tag_state: str,
                 *,
                 tag_prefixes: Optional[List[str]] = None,
                 version_name_prefixes: Optional[List[str]] = None,
                 package_name_prefixes: Optional[List[str]] = None,
                 older_than: Optional[str] = None,
                 newer_than: Optional[str] = None):
        self.name = name
        self.tag_state = tag_state
        self.tag_prefixes = tag_prefixes
        self.version_name_prefixes = version_name_prefixes
        self.package_name_prefixes = package_name_prefixes
        self.older_than = older_than
        self.newer_than = newer_than

    def to_dict(self):
        data = {'name': self.name, 'action': {'type': 'Keep'}, 'condition': {'tagState': self.tag_state}}
        condition = data['condition']
        if self.tag_prefixes is not None:
            condition['tagPrefixes'] = self.tag_prefixes
        if self.version_name_prefixes is not None:
            condition['versionNamePrefixes'] = self.version_name_prefixes
        if self.package_name_prefixes is not None:
            condition['packageNamePrefixes'] = self.package_name_prefixes
        if self.older_than:
            condition['olderThan'] = self.older_than
        if self.newer_than:
            condition['newerThan'] = self.newer_than
        return data


class MostRecentVersionKeepPolicy(CleanupPolicy):
    def __init__(self,
                 name: str,
                 package_name_prefixes: List[str],
                 keep_count: int):
        self.name = name
        self.package_name_prefixes = package_name_prefixes
        self.keep_count = keep_count

    def to_dict(self):
        data = {
            'name': self.name,
            'action': {'type': 'Keep'},
            'mostRecentVersions': {
                'packageNamePrefixes': self.package_name_prefixes,
                'keepCount': self.keep_count
            }
        }
        return data


third_party_images_fp = 'docker/third-party/images.txt'
third_party_packages = []
third_party_tags = []
with open(third_party_images_fp, 'r') as f:
    for image in f:
        image = image.strip()
        package, tag = image.split(':')
        if package not in third_party_packages:
            third_party_packages.append(package)
        if tag not in third_party_tags:
            third_party_tags.append(tag)

deploy_packages = []


def scrape_build_yaml(file_path: str):
    found_packages = []
    with open(file_path, 'r') as f:
        config_str = f.read().strip()
        build_config = yaml.safe_load(config_str)
        for step in build_config['steps']:
            if step['kind'] == 'buildImage2':
                image = step['publishAs']
                if image not in found_packages:
                    found_packages.append(image)
    return found_packages


deploy_packages.extend(scrape_build_yaml('build.yaml'))
deploy_packages.extend(scrape_build_yaml('ci/test/resources/build.yaml'))

deploy_packages = list(set(deploy_packages))

third_party_packages.sort()
third_party_tags.sort()
deploy_packages.sort()

policies = [
    DeletePolicy('delete_untagged', 'untagged'),
    DeletePolicy('delete_dev', 'tagged', tag_prefixes=['dev-'], older_than='3d'),
    DeletePolicy('delete_test_pr', 'tagged', tag_prefixes=['test-pr-'], older_than='3d'),
    DeletePolicy('delete_test_deploy', 'tagged', tag_prefixes=['test-deploy-'], older_than='3d'),
    DeletePolicy('delete_pr_cache', 'tagged', tag_prefixes=['cache-pr-'], older_than='7d'),
    DeletePolicy('delete_cache', 'tagged', tag_prefixes=['cache-'], older_than='30d'),
    ConditionalKeepPolicy('keep_third_party', 'any', package_name_prefixes=third_party_packages, tag_prefixes=third_party_tags),
    MostRecentVersionKeepPolicy('keep_most_recent_deploy', package_name_prefixes=deploy_packages, keep_count=10),
]

policies = [p.to_dict() for p in policies]

print(json.dumps(policies, indent=4))
