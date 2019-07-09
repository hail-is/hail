from hailtop.gear.auth import get_userinfo

GITHUB_CLONE_URL = 'https://github.com/'

userinfo = get_userinfo()
BUCKET = f'gs://{userinfo["bucket_name"]}'

AUTHORIZED_USERS = {
    'danking',
    'cseed',
    'konradjk',
    'jigold',
    'jbloom22',
    'patrick-schultz',
    'lfrancioli',
    'akotlar',
    'tpoterba',
    'chrisvittal',
    'catoverdrive',
    'daniel-goldstein',
    'ahiduchick',
    'GreatBrando',
    'johnc1231',
    'iitalics'
}
