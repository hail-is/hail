from hailtop.auth import get_userinfo

GITHUB_CLONE_URL = 'https://github.com/'
GITHUB_STATUS_CONTEXT = 'ci-test'

userinfo = get_userinfo()
BUCKET = f'gs://{userinfo["bucket_name"]}'

AUTHORIZED_USERS = {
    'danking',
    'cseed',
    'konradjk',
    'jigold',
    'patrick-schultz',
    'lfrancioli',
    'akotlar',
    'tpoterba',
    'chrisvittal',
    'catoverdrive',
    'GreatBrando',
    'johnc1231',
    'gsarma',
    'mkveerapen'
}
