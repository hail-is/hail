from batch.client import Job
from batch_helper import try_to_cancel_job, job_ordering
from build_state import build_state_from_gh_json
from ci_logging import log
from constants import BUILD_JOB_TYPE, GCS_BUCKET, DEPLOY_JOB_TYPE
from environment import \
    batch_client, \
    WATCHED_TARGETS, \
    REFRESH_INTERVAL_IN_SECONDS
from flask import Flask, request, jsonify
from git_state import Repo, FQRef, FQSHA
from github import open_pulls, overall_review_state, latest_sha_for_ref
from google_storage import \
    upload_public_gs_file_from_filename, \
    upload_public_gs_file_from_string
from http_helper import BadStatus
from http_helper import get_repo
from pr import review_status, GitHubPR
from prs import PRS
import collections
import json
import logging
import requests
import threading
import time

prs = PRS({k: v for [k, v] in WATCHED_TARGETS})

app = Flask(__name__)


@app.errorhandler(BadStatus)
def handle_invalid_usage(error):
    log.exception('bad status found when making request')
    return jsonify(error.data), error.status_code


@app.route('/status')
def status():
    return jsonify(prs.to_json())


@app.route('/push', methods=['POST'])
def github_push():
    d = request.json
    ref = d['ref']
    if ref.startswith('refs/heads'):
        target_ref = FQRef(Repo.from_gh_json(d['repository']), ref[11:])
        target = FQSHA(target_ref, d['after'])
        prs.push(target)
    else:
        log.info(
            f'ignoring ref push {ref} because it does not start with '
            '"refs/heads/"'
        )
    return '', 200


@app.route('/pull_request', methods=['POST'])
def github_pull_request():
    d = request.json
    assert 'action' in d, d
    assert 'pull_request' in d, d
    action = d['action']
    if action in ('opened', 'synchronize'):
        target_sha = FQSHA.from_gh_json(d['pull_request']['base']).sha
        gh_pr = GitHubPR.from_gh_json(d['pull_request'], target_sha)
        prs.pr_push(gh_pr)
    elif action == 'closed':
        gh_pr = GitHubPR.from_gh_json(d['pull_request'])
        log.info(f'forgetting closed pr {gh_pr.short_str()}')
        prs.forget(gh_pr.source.ref, gh_pr.target_ref)
    else:
        log.info(f'ignoring pull_request with action {action}')
    return '', 200


@app.route('/pull_request_review', methods=['POST'])
def github_pull_request_review():
    d = request.json
    action = d['action']
    gh_pr = GitHubPR.from_gh_json(d['pull_request'])
    if action == 'submitted':
        state = d['review']['state'].lower()
        if state == 'changes_requested':
            prs.review(gh_pr, state)
        else:
            # FIXME: track all reviewers, then we don't need to talk to github
            prs.review(
                gh_pr,
                review_status(
                    get_reviews(gh_pr.target_ref.repo,
                                gh_pr.number)))
    elif action == 'dismissed':
        # FIXME: track all reviewers, then we don't need to talk to github
        prs.review(
            gh_pr,
            review_status(get_reviews(gh_pr.target_ref.repo,
                                      gh_pr.number)))
    else:
        log.info(f'ignoring pull_request_review with action {action}')
    return '', 200


@app.route('/ci_build_done', methods=['POST'])
def ci_build_done():
    d = request.json
    attributes = d['attributes']
    source = FQSHA.from_json(json.loads(attributes['source']))
    target = FQSHA.from_json(json.loads(attributes['target']))
    job = Job(batch_client, d['id'], attributes=attributes, _status=d)
    receive_ci_job(source, target, job)
    return '', 200


@app.route('/deploy_build_done', methods=['POST'])
def deploy_build_done():
    d = request.json
    attributes = d['attributes']
    target = FQSHA.from_json(json.loads(attributes['target']))
    job = Job(batch_client, d['id'], attributes=attributes, _status=d)
    receive_deploy_job(target, job)
    return '', 200


@app.route('/refresh_batch_state', methods=['POST'])
def refresh_batch_state():
    jobs = batch_client.list_jobs()
    build_jobs = [
        job for job in jobs
        if job.attributes and job.attributes.get('type', None) == BUILD_JOB_TYPE
    ]
    refresh_ci_build_jobs(build_jobs)
    deploy_jobs = [
        job for job in jobs
        if job.attributes and job.attributes.get('type', None) == DEPLOY_JOB_TYPE
    ]
    refresh_deploy_jobs(deploy_jobs)
    return '', 200


def refresh_ci_build_jobs(jobs):
    jobs = [
        (FQSHA.from_json(json.loads(job.attributes['source'])),
         FQSHA.from_json(json.loads(job.attributes['target'])),
         job)
        for job in jobs
    ]
    jobs = [(s, t, j) for (s, t, j) in jobs if prs.exists(s, t)]
    latest_jobs = {}
    for (source, target, job) in jobs:
        key = (source, target)
        job2 = latest_jobs.get(key, None)
        if job2 is None:
            latest_jobs[key] = job
        else:
            if job_ordering(job, job2) > 0:
                log.info(
                    f'cancelling {job2.id}, preferring {job.id}'
                )
                try_to_cancel_job(job2)
                latest_jobs[key] = job
            else:
                log.info(
                    f'cancelling {job.id}, preferring {job2.id}'
                )
                try_to_cancel_job(job)
    for ((source, target), job) in latest_jobs.items():
        prs.refresh_from_ci_job(source, target, job)


def refresh_deploy_jobs(jobs):
    jobs = [
        (FQSHA.from_json(json.loads(job.attributes['target'])),
         job)
        for job in jobs
        if 'target' in job.attributes
    ]
    jobs = [
        (target, job)
        for (target, job) in jobs
        if target in prs.deploy_jobs
    ]
    latest_jobs = {}
    for (target, job) in jobs:
        job2 = latest_jobs.get(target, None)
        if job2 is None:
            latest_jobs[target] = job
        else:
            if job_ordering(job, job2) > 0:
                log.info(
                    f'cancelling {job2.id}, preferring {job.id}'
                )
                try_to_cancel_job(job2)
                latest_jobs[target] = job
            else:
                log.info(
                    f'cancelling {job.id}, preferring {job2.id}'
                )
                try_to_cancel_job(job)
    for (target, job) in latest_jobs.items():
        prs.refresh_from_deploy_job(target, job)


@app.route('/force_retest', methods=['POST'])
def force_retest():
    d = request.json
    source = FQRef.from_json(d['source'])
    target = FQRef.from_json(d['target'])
    prs.build(source, target)
    return '', 200


@app.route('/force_redeploy', methods=['POST'])
def force_redeploy():
    d = request.json
    target = FQRef.from_json(d)
    if target in prs.watched_target_refs():
        prs.try_deploy(target)
        return '', 200
    else:
        return f'{target.short_str()} not in {[ref.short_str() for ref in prs.watched_target_refs()]}', 400


@app.route('/refresh_github_state', methods=['POST'])
def refresh_github_state():
    for target_repo in prs.watched_repos():
        try:
            pulls = open_pulls(target_repo)
            pulls_by_target = collections.defaultdict(list)
            latest_target_shas = {}
            for pull in pulls:
                gh_pr = GitHubPR.from_gh_json(pull)
                if gh_pr.target_ref not in latest_target_shas:
                    latest_target_shas[gh_pr.target_ref] = latest_sha_for_ref(gh_pr.target_ref)
                sha = latest_target_shas[gh_pr.target_ref]
                gh_pr.target_sha = sha
                pulls_by_target[gh_pr.target_ref].append(gh_pr)
            refresh_pulls(target_repo, pulls_by_target)
            refresh_reviews(pulls_by_target)
            # FIXME: I can't fit build state json in the status description
            # refresh_statuses(pulls_by_target)
        except Exception as e:
            log.exception(
                f'could not refresh state for {target_repo.short_str()} due to {e}')
    return '', 200


def refresh_pulls(target_repo, pulls_by_target):
    dead_targets = (
        set(prs.live_target_refs_for_repo(target_repo)) -
        {x for x in pulls_by_target.keys()}
    )
    for dead_target_ref in dead_targets:
        prs.forget_target(dead_target_ref)
    for (target_ref, pulls) in pulls_by_target.items():
        for gh_pr in pulls:
            prs.pr_push(gh_pr)
        dead_prs = ({x.source.ref for x in prs.for_target(target_ref)} -
                    {x.source.ref for x in pulls})
        log.info(f'for {target_ref.short_str()}, forgetting {[x.short_str() for x in dead_prs]}')
        for source_ref in dead_prs:
            prs.forget(source_ref, target_ref)
    return pulls_by_target


def refresh_reviews(pulls_by_target):
    for (_, pulls) in pulls_by_target.items():
        for gh_pr in pulls:
            reviews = get_repo(
                gh_pr.target_ref.repo.qname,
                'pulls/' + gh_pr.number + '/reviews',
                status_code=200)
            state = overall_review_state(reviews)['state']
            prs.review(gh_pr, state)


def refresh_statuses(pulls_by_target):
    for pulls in pulls_by_target.values():
        for gh_pr in pulls:
            statuses = get_repo(
                gh_pr.target_ref.repo.qname,
                'commits/' + gh_pr.source.sha + '/statuses',
                status_code=200)
            prs.refresh_from_github_build_status(
                gh_pr,
                build_state_from_gh_json(statuses))


@app.route('/heal', methods=['POST'])
def heal():
    prs.heal()
    return '', 200


@app.route('/healthcheck')
def healthcheck():
    return '', 200


@app.route('/watched_repo', methods=['POST'])
def set_deployable():
    d = request.json
    target_ref = FQRef.from_json(d['target_ref'])
    action = d['action']
    assert action in ('unwatch', 'watch', 'deploy')
    prs.update_watch_state(target_ref, action)
    return '', 200


###############################################################################


def receive_ci_job(source, target, job):
    upload_public_gs_file_from_string(GCS_BUCKET,
                                      f'ci/{source.sha}/{target.sha}/job-log',
                                      job.cached_status()['log'])
    upload_public_gs_file_from_filename(
        GCS_BUCKET,
        f'ci/{source.sha}/{target.sha}/index.html',
        'index.html')
    prs.ci_build_finished(source, target, job)


def receive_deploy_job(target, job):
    upload_public_gs_file_from_string(GCS_BUCKET,
                                      f'deploy/{target.sha}/job-log',
                                      job.cached_status()['log'])
    upload_public_gs_file_from_filename(
        GCS_BUCKET,
        f'deploy/{target.sha}/index.html',
        'deploy-index.html')
    prs.deploy_build_finished(target, job)


def get_reviews(repo, pr_number):
    return get_repo(
        repo.qname,
        'pulls/' + pr_number + '/reviews',
        status_code=200)


def polling_event_loop():
    time.sleep(1)
    while True:
        try:
            r = requests.post(
                'http://127.0.0.1:5000/refresh_github_state',
                timeout=360)
            r.raise_for_status()
            r = requests.post(
                'http://127.0.0.1:5000/refresh_batch_state',
                timeout=360)
            r.raise_for_status()
            r = requests.post('http://127.0.0.1:5000/heal', timeout=360)
            r.raise_for_status()
        except Exception as e:
            log.error(f'Could not poll due to exception: {e}')
        time.sleep(REFRESH_INTERVAL_IN_SECONDS)


def fix_werkzeug_logs():
    # https://github.com/pallets/flask/issues/1359#issuecomment-291749259
    werkzeug_logger = logging.getLogger('werkzeug')
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.log = lambda self, type, message, *args: \
        getattr(werkzeug_logger, type)('%s %s' % (self.address_string(), message % args))


if __name__ == '__main__':
    fix_werkzeug_logs()
    threading.Thread(target=polling_event_loop).start()
    app.run(host='0.0.0.0', threaded=False)
