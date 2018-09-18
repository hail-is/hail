import os
from flask import Flask, render_template, request, jsonify, abort, url_for
from github import Github
import random

GITHUB_TOKEN_PATH = os.environ.get('GITHUB_TOKEN_PATH',
                                   '/secrets/scorecard-github-access-token.txt')
with open(GITHUB_TOKEN_PATH, 'r') as f:
    token = f.read().strip()
github = Github(token)

users = ['danking', 'cseed', 'tpoterba', 'jigold', 'jbloom22', 'catoverdrive', 'patrick-schultz', 'rcownie', 'chrisvittal']

default_repo = 'hail'
repos = {
    'hail': 'hail-is/hail',
    'batch': 'hail-is/batch',
    'ci': 'hail-is/hail',
    'scorecard': 'hail-is/scorecard',
    'cloudtools': 'Nealelab/cloudtools'
}

app = Flask('scorecard')

def get_id(repo_name, number):
    if repo_name == default_repo:
        return f'{number}'
    else:
        return f'{repo_name}/{number}'

def pr_template_data(repo_name, pr):
    return {
        'id': get_id(repo_name, pr.number),
        'html_url': pr.html_url
    }

def issue_template_data(repo_name, pr):
    return {
        'id': get_id(repo_name, pr.number),
        'html_url': pr.html_url
    }

@app.route('/')
def index():
    unassigned = []
    user_data = {}

    def get_user_data(user):
        if user not in user_data:
            d = {'CHANGES_REQUESTED': [],
                 'NEEDS_REVIEW': [],
                 'ISSUES': []}
            user_data[user] = d
        else:
            d = user_data[user]
        return d

    def add_pr_to(repo_name, pr, col):
        pr_data = pr_template_data(repo_name, pr)
        for user in pr.assignees:
            d = get_user_data(user.login)
            d[col].append(pr_data)

    def add_issue(repo_name, issue):
        issue_data = issue_template_data(repo_name, issue)
        for user in issue.assignees:
            d = get_user_data(user.login)
            d['ISSUES'].append(issue_data)

    for repo_name, fq_repo in repos.items():
        repo = github.get_repo(fq_repo)

        for pr in repo.get_pulls(state='open'):
            if len(pr.assignees) == 0:
                unassigned.append(pr_template_data(repo_name, pr))
                continue

            state = 'NEEDS_REVIEW'
            for review in pr.get_reviews():
                if review.state == 'CHANGES_REQUESTED':
                    state = review.state
                    break
                elif review.state == 'DISMISSED':
                    break
                elif review.state == 'APPROVED':
                    state = 'APPROVED'
                    break
                else:
                    assert review.state == 'COMMENTED'

            if state != 'APPROVED':
                add_pr_to(repo_name, pr, state)

        for issue in repo.get_issues(state='open'):
            add_issue(repo_name, issue)

    print(unassigned, user_data)

    random_user = random.choice(users)

    return render_template('index.html', unassigned=unassigned,
                           user_data=user_data, random_user=random_user)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
