import os
import hailtop.gear.auth as hj

GITHUB_CLONE_URL = 'https://github.com/'

userdata = hj.JWTClient.find_userdata()
BUCKET = f'gs://{userdata["bucket_name"]}'

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
