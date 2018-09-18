from flask import Flask, render_template, request, jsonify, abort, url_for
from github import Github
import random

github = Github('3a1d3bea64788b0694a1b038a4d98325af3a536f')

users = ['danking', 'cseed', 'tpoterba', 'jigold', 'jbloom22', 'catoverdrive', 'patrick-schultz', 'maccum', 'rcownie', 'chrisvittal']

hail_repo = github.get_repo('hail-is/hail')

app = Flask('scorecard')

def pr_template_data(pr):
    return {
        'id': pr.number,
        'html_url': pr.html_url
    }

def issue_template_data(pr):
    return {
        'id': pr.number,
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

    def add_pr_to(pr, col):
        pr_data = pr_template_data(pr)
        for user in pr.assignees:
            d = get_user_data(user.login)
            d[col].append(pr_data)

    def add_issue(issue):
        issue_data = issue_template_data(issue)
        for user in issue.assignees:
            d = get_user_data(user.login)
            d['ISSUES'].append(issue_data)

    for pr in hail_repo.get_pulls(state='open'):
        if len(pr.assignees) == 0:
            unassigned.append(pr_template_data(pr))
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
                print(review.state)
                assert review.state == 'COMMENTED'

        if state != 'APPROVED':
            add_pr_to(pr, state)

    for issue in hail_repo.get_issues(state='open'):
        add_issue(issue)

    print(unassigned, user_data)

    random_user = random.choice(users)

    return render_template('index.html', unassigned=unassigned,
                           user_data=user_data, random_user=random_user)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
