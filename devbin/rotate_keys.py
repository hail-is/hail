import aiohttp
import asyncio
import base64
import itertools
from dateutil.parser import isoparse
from datetime import datetime, timedelta
import pytz
import json
from typing import List, Optional, Tuple, Generator
import warnings
import sys
import kubernetes_asyncio as kube
from hailtop.aiocloud.aiogoogle import GoogleIAmClient
from hailtop.utils import retry_transient_errors

warnings.simplefilter('always', UserWarning)


class GSAKeySecret:
    def __init__(self, raw_secret):
        self.name = raw_secret.metadata.name
        self.namespace = raw_secret.metadata.namespace
        self.key_data = json.loads(base64.b64decode(raw_secret.data['key.json']))

    def service_account_email(self):
        return self.key_data['client_email']

    def private_key_id(self):
        return self.key_data['private_key_id']

    def __str__(self):
        return f'{self.name} ({self.namespace})'


class KubeSecretManager:
    def __init__(self, kube_client):
        self.kube_client = kube_client

    async def get_gsa_key_secrets(self) -> List[GSAKeySecret]:
        secrets = (await retry_transient_errors(self.kube_client.list_secret_for_all_namespaces)).items
        return [GSAKeySecret(s) for s in secrets if s.data is not None and 'key.json' in s.data]

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
            body=kube.client.V1Secret(  # type: ignore
                metadata=kube.client.V1ObjectMeta(name=name),  # type: ignore
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
        self.expiration = isoparse(self.expiration_readable)


class ServiceAccount:
    def __init__(self, email: str, keys: List[IAMKey]):
        self.email: str = email
        self.keys: List[IAMKey] = keys

        self.kube_secrets: List[GSAKeySecret] = []

    def username(self):
        return self.email.split('@')[0]

    def add_new_key(self, k: IAMKey):
        self.keys.insert(0, k)

    def list_keys(self) -> str:
        msg = f'GSA Keys for {self.email}\n'
        for k in self.keys:
            msg += f'{k.id} \t Created: {k.created_readable} \t Expires: {k.expiration_readable}'
            matching_secrets = [str(s) for s in self.kube_secrets if s.private_key_id() == k.id]
            if len(matching_secrets) > 0:
                msg += f'\t <== {" ".join(matching_secrets)}'
            msg += '\n'
        return msg


class IAMManager:
    def __init__(self, iam_client: GoogleIAmClient):
        self.iam_client = iam_client

    async def create_new_key(self, sa: ServiceAccount) -> Tuple[IAMKey, str]:
        key_json = await self.iam_client.post(f'/serviceAccounts/{sa.email}/keys')
        encoded_key_data = key_json['privateKeyData']
        return IAMKey(key_json), base64.b64decode(encoded_key_data.encode('utf-8')).decode('utf-8')

    async def delete_key(self, sa_email: str, key: IAMKey) -> Optional[str]:
        try:
            await self.iam_client.delete(f'/serviceAccounts/{sa_email}/keys/{key.id}')
            return key.id
        except aiohttp.ClientResponseError as e:
            warnings.warn(str(e), stacklevel=2)
            return None

    async def get_all_service_accounts(self) -> List[ServiceAccount]:
        all_accounts = list(
            await asyncio.gather(
                *[asyncio.create_task(self.service_account_from_email(email)) async for email in self.all_sa_emails()]
            )
        )
        all_accounts.sort(key=lambda sa: sa.email)
        return all_accounts

    async def service_account_from_email(self, email: str) -> ServiceAccount:
        return ServiceAccount(email, await self.get_sa_keys(email))

    async def get_sa_keys(self, sa_email: str) -> List[IAMKey]:
        keys_json = (await self.iam_client.get(f'/serviceAccounts/{sa_email}/keys'))['keys']
        keys = [IAMKey(k) for k in keys_json]
        keys.sort(key=lambda k: k.created)
        keys.reverse()
        return keys

    async def all_sa_emails(self) -> Generator[str, None, None]:
        res = await self.iam_client.get('/serviceAccounts')
        next_page_token = res.get('nextPageToken')
        for acc in res['accounts']:
            yield acc['email']
        while next_page_token is not None:
            res = await self.iam_client.get('/serviceAccounts', params={'pageToken': next_page_token})
            next_page_token = res.get('nextPageToken')
            for acc in res['accounts']:
                yield acc['email']


async def add_new_keys(service_accounts: List[ServiceAccount], iam_manager: IAMManager, k8s_manager: KubeSecretManager):
    for sa in service_accounts:
        print(sa.list_keys())
        if input('Create new key?\nOnly yes will be accepted: ') == 'yes':
            new_key, key_data = await iam_manager.create_new_key(sa)
            sa.add_new_key(new_key)
            print(f'Created new key: {new_key.id}')
            new_secrets = await asyncio.gather(
                *[k8s_manager.update_gsa_key_secret(s, key_data) for s in sa.kube_secrets]
            )
            sa.kube_secrets = list(new_secrets)
            print(sa.list_keys())
            if input('Continue?[Yes/no]') == 'no':
                break


async def delete_old_keys(service_accounts: List[ServiceAccount], iam_manager: IAMManager):
    for sa in service_accounts:
        print(sa.list_keys())
        if input('Delete all but the newest key?\nOnly yes will be accepted: ') == 'yes':
            newest_key, *to_delete = sa.keys
            thirty_days_ago = datetime.now(pytz.utc) - timedelta(days=30)
            if newest_key.created > thirty_days_ago:
                warnings.warn(
                    "The most recent key was generated less than "
                    "thirty days ago. Old keys cannot yet be deleted "
                    "as they might still be in use.",
                    stacklevel=2,
                )
            else:
                deleted_ids = await asyncio.gather(*[iam_manager.delete_key(sa.email, k) for k in to_delete])
                deleted_ids = [kid for kid in deleted_ids if kid is not None]
                print(f'Deleted keys:')
                for kid in deleted_ids:
                    print(f'\t{kid}')
                sa.keys = await iam_manager.get_sa_keys(sa.email)
                print(sa.list_keys())
            if input('Continue?[Yes/no] ') == 'no':
                break


async def main():
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} <PROJECT>')
        sys.exit(1)
    project = sys.argv[1]

    iam_client = GoogleIAmClient(project)
    iam_manager = IAMManager(iam_client)

    await kube.config.load_kube_config()  # type: ignore
    k8s_manager = KubeSecretManager(kube.client.CoreV1Api())  # type: ignore

    try:
        service_accounts = await iam_manager.get_all_service_accounts()
        gsa_key_secrets = await k8s_manager.get_gsa_key_secrets()
        seen_secrets = set()
        for sa in service_accounts:
            for secret in gsa_key_secrets:
                if any(k.id == secret.private_key_id() for k in sa.keys):
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
                print(f'\t{sa.username()}: {", ".join(str(s) for s in sa.kube_secrets)}')

        print('Discovered the following key secrets with no matching service account')
        unmatched_secrets = set(k.name for k in gsa_key_secrets).difference(seen_secrets)
        for s in unmatched_secrets:
            print(f'\t{s}')

        print('Discovered the following service accounts with no k8s key secrets')
        for sa in service_accounts:
            if len(sa.kube_secrets) == 0:
                print(f'\t{sa.email}')

        action = input('What action would you like to take?[update/delete]: ')
        if action == 'update':
            await add_new_keys(service_accounts, iam_manager, k8s_manager)
        elif action == 'delete':
            await delete_old_keys(service_accounts, iam_manager)
        else:
            print('Doing nothing')
    finally:
        await iam_client.close()


asyncio.get_event_loop().run_until_complete(main())
