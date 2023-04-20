import asyncio
import base64
import itertools
import json
import sys
import warnings
from datetime import datetime, timedelta
from enum import Enum
from typing import Generator, List, Optional, TextIO, Tuple, Any, TypedDict, Set

import kubernetes_asyncio.client
import kubernetes_asyncio.config
import pytz
from dateutil.parser import isoparse

from hailtop.aiocloud.aiogoogle import GoogleIAmClient
from hailtop.utils import retry_transient_errors

warnings.simplefilter('always', UserWarning)

OK_BLUE = '\033[94m'
OK_GREEN = '\033[92m'
WARNING_YELLOW = '\033[93m'
ERROR_RED = '\033[91m'
GREY = '\033[37m'
ENDC = '\033[0m'


class ServiceAccountDict(TypedDict):
    name: str
    projectId: str
    uniqueId:  str
    email: str
    displayName: str
    etag: str
    oauth2ClientId: str
    disabled: bool


class RotationState(Enum):
    UP_TO_DATE = f'{OK_GREEN}UP TO DATE{ENDC}'
    IN_PROGRESS = f'{OK_BLUE}IN PROGRESS{ENDC}'
    READY_FOR_DELETE = f'{WARNING_YELLOW}READY FOR DELETE{ENDC}'
    SOON_TO_EXPIRE = f'{WARNING_YELLOW}SOON TO EXPIRE{ENDC}'
    EXPIRED = f'{ERROR_RED}EXPIRED{ENDC}'


class GSAKeySecret:
    def __init__(self, raw_secret):
        self.name = raw_secret.metadata.name
        self.namespace = raw_secret.metadata.namespace
        self.key_data = json.loads(base64.b64decode(raw_secret.data['key.json']))

    def service_account_email(self):
        return self.key_data['client_email']

    def private_key_id(self):
        return self.key_data['private_key_id']

    def matches_iam_key(self, k: 'IAMKey'):
        return self.private_key_id() == k.id

    def __str__(self):
        return f'{self.name} ({self.namespace})'


class KubeSecretManager:
    def __init__(self, kube_client):
        self.kube_client = kube_client

    async def get_gsa_key_secrets(self) -> List[GSAKeySecret]:
        secrets = (await retry_transient_errors(self.kube_client.list_secret_for_all_namespaces)).items
        return [GSAKeySecret(s) for s in secrets if s.data is not None and 'key.json' in s.data]

    async def get_secrets(self) -> Tuple[Any, List[GSAKeySecret]]:
        secrets = (await retry_transient_errors(self.kube_client.list_secret_for_all_namespaces)).items
        return (
            [s for s in secrets
             if (s.data is None or 'key.json' not in s.data)
             and not s.metadata.name.endswith('-tokens')
             and not s.metadata.name.startswith('ssl-config-')
             and not (s.metadata.name.startswith('sql-') and s.metadata.name.endswith('-config'))
             and s.metadata.name not in ('database-server-config', 'ci-config', 'deploy-config', 'gce-deploy-config')
             and not s.metadata.namespace == 'kube-system'
             and not s.type == 'kubernetes.io/service-account-token'],
            [GSAKeySecret(s) for s in secrets if s.data is not None and 'key.json' in s.data]
        )

    async def update_gsa_key_secret(self, secret: GSAKeySecret, key_data: str) -> GSAKeySecret:
        data = {'key.json': key_data}
        await self.update_secret(secret.name, secret.namespace, data)
        print(f'Updated secret {secret}')
        return GSAKeySecret(await self.get_secret(secret.name, secret.namespace))

    async def update_secret(self, name, namespace, data):
        await retry_transient_errors(
            self.kube_client.replace_namespaced_secret,
            name=name,
            namespace=namespace,
            body=kubernetes_asyncio.client.V1Secret(  # type: ignore
                metadata=kubernetes_asyncio.client.V1ObjectMeta(name=name),  # type: ignore
                data={k: base64.b64encode(v.encode('utf-8')).decode('utf-8') for k, v in data.items()},
            ),
        )

    async def get_secret(self, name, namespace):
        return await retry_transient_errors(
            self.kube_client.read_namespaced_secret, name, namespace, _request_timeout=5
        )


class IAMKey:
    def __init__(self, key_json):
        self.raw_key = key_json
        self.id = key_json['name'].split('/')[-1]
        self.created_readable = key_json['validAfterTime']
        self.expiration_readable = key_json['validBeforeTime']
        self.created = isoparse(self.created_readable)
        assert key_json['keyType'] in ('USER_MANAGED', 'SYSTEM_MANAGED')
        self.user_managed = key_json['keyType'] == 'USER_MANAGED'

    def expired(self) -> bool:
        return self.older_than(90)

    def soon_to_expire(self) -> bool:
        return self.older_than(60)

    def recently_created(self) -> bool:
        return not self.older_than(30)

    def older_than(self, days: int) -> bool:
        return self.created < datetime.now(pytz.utc) - timedelta(days=days)


class ServiceAccount:
    def __init__(self, email: str, disabled: bool, keys: List[IAMKey]):
        self.email: str = email
        self.disabled: bool = disabled
        self.keys: List[IAMKey] = keys

        self.kube_secrets: List[GSAKeySecret] = []

    def username(self):
        return self.email.split('@')[0]

    def add_new_key(self, k: IAMKey):
        self.keys.insert(0, k)

    def list_keys(self, s: TextIO) -> None:
        def color_id(k: IAMKey) -> str:
            if not k.user_managed:
                color = GREY
            elif k.expired():
                color = ERROR_RED
            elif not k.recently_created():
                color = WARNING_YELLOW
            else:
                color = OK_GREEN
            return f'{color}{k.id}{ENDC}'

        s.write(f'{self}\n')
        for k in self.keys:
            s.write(f'{color_id(k)} \t Created: {k.created_readable} \t Expires: {k.expiration_readable}')
            matching_secrets = [str(s) for s in self.kube_secrets if s.matches_iam_key(k)]
            if len(matching_secrets) > 0:
                s.write(f'\t <== {" ".join(matching_secrets)}')
            s.write('\n')

    def rotation_state(self) -> RotationState:
        active_key = self.active_user_key()

        # Solely system-managed accounts are always up to date
        if not active_key:
            return RotationState.UP_TO_DATE

        old_user_keys = self.redundant_user_keys()
        if active_key.expired():
            return RotationState.EXPIRED
        elif active_key.soon_to_expire():
            return RotationState.SOON_TO_EXPIRE
        elif active_key.recently_created() and len(old_user_keys) > 0:
            return RotationState.IN_PROGRESS
        elif not active_key.recently_created() and len(old_user_keys) > 0:
            return RotationState.READY_FOR_DELETE
        else:
            assert len(old_user_keys) == 0
            return RotationState.UP_TO_DATE

    def active_user_key(self) -> Optional[IAMKey]:
        if len(self.kube_secrets) > 0:
            try:
                kube_key = next(k for k in self.keys if all(s.matches_iam_key(k) for s in self.kube_secrets))
            except StopIteration:
                keys_to_k8s_secret = defaultdict(list)
                for s in self.kube_secrets:
                    keys_to_k8s_secret[s.private_key_id()].append((s.name, s.namespace))
                keys_to_k8s_secret_str = "\n".join(
                    f'{k} ' + ", ".join(str(s) for s in secrets)
                    for k, secrets in keys_to_k8s_secret.items()
                )
                known_iam_keys_str = "\n".join(
                    f'{k.id} Created: {k.created_readable} Expires: {k.expiration_readable}' for k in self.keys
                )
                kube_key = sorted(
                    [k for k in self.keys
                     if k.id in keys_to_k8s_secret],
                    key=lambda k: -k.created.timestamp())[0]
                print(f'''Found a user ({self.username()}) without a unique active key in Kubernetes.
The known IAM keys are:
{known_iam_keys_str}

These keys are present in Kubernetes:
{keys_to_k8s_secret_str}

We will assume {kube_key.id} is the active key.
''')
            assert kube_key is not None
            assert kube_key.user_managed
            return kube_key
        # We assume that if the key is not used in kubernetes, the most recently
        # created user-managed key should be considered "active".
        return next((k for k in self.keys if k.user_managed), None)

    def redundant_user_keys(self) -> List[IAMKey]:
        active_key = self.active_user_key()
        return [k for k in self.keys if k is not active_key and k.user_managed]

    def __str__(self):
        s = self.rotation_state().value + ' ' + self.email
        if self.disabled:
            return f'{WARNING_YELLOW}DISABLED{ENDC} ' + s
        return s


class IAMManager:
    def __init__(self, iam_client: GoogleIAmClient):
        self.iam_client = iam_client

    async def create_new_key(self, sa: ServiceAccount) -> Tuple[IAMKey, str]:
        key_json = await self.iam_client.post(f'/serviceAccounts/{sa.email}/keys')
        encoded_key_data = key_json['privateKeyData']
        return IAMKey(key_json), base64.b64decode(encoded_key_data.encode('utf-8')).decode('utf-8')

    async def delete_key(self, sa_email: str, key: IAMKey):
        # SYSTEM_MANAGED keys are rotated by google and we cannot delete them
        assert key.user_managed
        await self.iam_client.delete(f'/serviceAccounts/{sa_email}/keys/{key.id}')

    async def get_all_service_accounts(self) -> List[ServiceAccount]:
        all_accounts = list(
            await asyncio.gather(
                *[asyncio.create_task(self.service_account_from_dict(d)) async for d in self.all_sa_dicts()]
            )
        )
        all_accounts.sort(key=lambda sa: sa.email)
        return all_accounts

    async def service_account_from_dict(self, d: ServiceAccountDict) -> ServiceAccount:
        return ServiceAccount(d['email'],
                              d.get('disabled', False),
                              await self.get_sa_keys(d['email']))

    async def get_sa_keys(self, sa_email: str) -> List[IAMKey]:
        keys_json = (await self.iam_client.get(f'/serviceAccounts/{sa_email}/keys'))['keys']
        keys = [IAMKey(k) for k in keys_json]
        keys.sort(key=lambda k: k.created)
        keys.reverse()
        return keys

    async def all_sa_dicts(self) -> Generator[ServiceAccountDict, None, None]:
        res = await self.iam_client.get('/serviceAccounts')
        next_page_token = res.get('nextPageToken')
        for acc in res['accounts']:
            yield acc
        while next_page_token is not None:
            res = await self.iam_client.get('/serviceAccounts', params={'pageToken': next_page_token})
            next_page_token = res.get('nextPageToken')
            for acc in res['accounts']:
                yield acc


async def add_new_keys(service_accounts: List[ServiceAccount],
                       iam_manager: IAMManager,
                       k8s_manager: KubeSecretManager,
                       *,
                       exclude: Optional[Set[RotationState]] = None,
                       interactive: bool):
    exclude = exclude or {}
    service_accounts_under_consideration = [
        sa
        for sa in service_accounts
        if not sa.disabled and sa.rotation_state() not in exclude
    ]
    if not interactive:
        accounts_str = '\n'.join(str(sa) for sa in service_accounts_under_consideration)
        print(accounts_str)
        if input('New keys will be created for the above accounts. Proceed? Only yes will be accepted: ') != 'yes':
            return

    for sa in service_accounts_under_consideration:
        sa.list_keys(sys.stdout)
        if interactive:
            if input('Create new key?\nOnly yes will be accepted: ') != 'yes':
                continue

        new_key, key_data = await iam_manager.create_new_key(sa)
        sa.add_new_key(new_key)
        print(f'Created new key: {new_key.id}')
        new_secrets = await asyncio.gather(
            *[k8s_manager.update_gsa_key_secret(s, key_data) for s in sa.kube_secrets]
        )
        sa.kube_secrets = list(new_secrets)
        sa.list_keys(sys.stdout)


async def delete_old_keys(service_accounts: List[ServiceAccount], iam_manager: IAMManager, focus: Optional[RotationState] = None):
    async def delete_old_and_refresh(sa: ServiceAccount):
        to_delete = sa.redundant_user_keys()
        await asyncio.gather(*[iam_manager.delete_key(sa.email, k) for k in to_delete])
        print(f'Deleted keys:')
        for k in to_delete:
            print(f'\t{k.id}')
        sa.keys = await iam_manager.get_sa_keys(sa.email)
        sa.list_keys(sys.stdout)

    for sa in service_accounts:
        rotation_state = sa.rotation_state()
        if sa.disabled or focus is not None and rotation_state != focus:
            continue
        sa.list_keys(sys.stdout)
        if input('Delete all but the newest key?\nOnly yes will be accepted: ') == 'yes':
            if rotation_state == RotationState.READY_FOR_DELETE:
                await delete_old_and_refresh(sa)
            elif rotation_state == RotationState.IN_PROGRESS:
                warnings.warn(
                    'The most recent key was generated less than '
                    'thirty days ago. Old keys should not be deleted '
                    'as they might still be in use.',
                    stacklevel=2,
                )
                if input('Are you sure you want to delete old keys? ') == 'yes':
                    await delete_old_and_refresh(sa)
            else:
                warnings.warn(
                    f'Cannot delete keys in rotation state: {rotation_state}',
                    stacklevel=2,
                )


async def main():
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} <PROJECT>')
        sys.exit(1)
    project = sys.argv[1]

    iam_client = GoogleIAmClient(project)
    iam_manager = IAMManager(iam_client)

    await kubernetes_asyncio.config.load_kube_config()
    k8s_client = kubernetes_asyncio.client.CoreV1Api()
    k8s_manager = KubeSecretManager(k8s_client)

    try:
        service_accounts = [x for x in await iam_manager.get_all_service_accounts()]
        other_secrets, gsa_key_secrets = await k8s_manager.get_secrets()
        seen_secrets = set()
        for sa in service_accounts:
            for secret in gsa_key_secrets:
                if any(secret.matches_iam_key(k) for k in sa.keys):
                    sa.kube_secrets.append(secret)
                    seen_secrets.add(secret.name)

        for sa in service_accounts:
            secrets = sorted(sa.kube_secrets, key=lambda s: s.namespace)
            secrets_by_namespace = [list(g) for _, g in itertools.groupby(secrets, key=lambda s: s.namespace)]
            dup_secrets = [secrets for secrets in secrets_by_namespace if len(secrets) > 1]
            # There should only be one k8s key secret per service account per namespace
            if len(dup_secrets):
                new_line = "\n"
                warnings.warn(
                    f'Service account {sa.email} represented by multiple secrets in the same namespace:\n'
                    f'{new_line.join(", ".join(str(s) for s in dups) for dups in dup_secrets)}',
                    stacklevel=2,
                )

        print('Discovered the following matching service accounts and k8s secrets')
        for sa in service_accounts:
            if len(sa.kube_secrets) > 0:
                print(f'\t{sa}: {", ".join(str(s) for s in sa.kube_secrets)}')

        print('Discovered the following key secrets with no matching service account')
        unmatched_secrets = set(k.name for k in gsa_key_secrets).difference(seen_secrets)
        for s in unmatched_secrets:
            print(f'\t{s}')

        print('Discovered the following service accounts with no k8s key secrets')
        for sa in service_accounts:
            if len(sa.kube_secrets) == 0:
                print(f'\t{sa}')

        print('Discovered the secrets which we ASSUME have no keys')
        for secret in other_secrets:
            print(f'\t{secret.metadata.name} ({secret.metadata.namespace})')

        action = input('What action would you like to take?[update/update-all/delete/delete-ready-only/delete-in-progress-only]: ')
        if action == 'update':
            await add_new_keys(service_accounts, iam_manager, k8s_manager,
                               exclude={RotationState.UP_TO_DATE,
                                        RotationState.IN_PROGRESS,
                                        RotationState.READY_FOR_DELETE},
                               interactive=False)
        if action == 'interactive-update':
            await add_new_keys(service_accounts, iam_manager, k8s_manager, interactive=True)
        elif action == 'delete':
            await delete_old_keys(service_accounts, iam_manager)
        elif action == 'delete-ready-only':
            await delete_old_keys(service_accounts, iam_manager, focus=RotationState.READY_FOR_DELETE)
        elif action == 'delete-in-progress-only':
            await delete_old_keys(service_accounts, iam_manager, focus=RotationState.IN_PROGRESS)
        else:
            print('Doing nothing')
    finally:
        try:
            await iam_client.close()
        finally:
            await k8s_client.api_client.rest_client.pool_manager.close()


asyncio.get_event_loop().run_until_complete(main())
