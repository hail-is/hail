import json
import os

import subprocess as sp
from batch.client import Job

from .batch_helper import short_str_build_job
from .build_state import \
    Failure, Mergeable, Unknown, NoImage, Building, Buildable, Merged, \
    build_state_from_json
from .ci_logging import log
from .constants import BUILD_JOB_TYPE, VERSION, GCS_BUCKET, SHA_LENGTH, \
    GCS_BUCKET_PREFIX
from .environment import PR_BUILD_SCRIPT, SELF_HOSTNAME, batch_client, CONTEXT
from .git_state import FQSHA, FQRef
from .github import latest_sha_for_ref
from .http_helper import post_repo, BadStatus
from .sentinel import Sentinel
from .shell_helper import shell


def try_new_build(source, target):
    img = maybe_get_image(target, source)
    if img:
        attributes = {
            'target': json.dumps(target.to_dict()),
            'source': json.dumps(source.to_dict()),
            'image': img,
            'type': BUILD_JOB_TYPE
        }
        try:
            job = batch_client.create_job(
                img,
                command=['/bin/bash',
                         '-c',
                         PR_BUILD_SCRIPT],
                env={
                    'SOURCE_REPO_URL': source.ref.repo.url,
                    'SOURCE_BRANCH': source.ref.name,
                    'SOURCE_SHA': source.sha,
                    'TARGET_REPO_URL': target.ref.repo.url,
                    'TARGET_BRANCH': target.ref.name,
                    'TARGET_SHA': target.sha
                },
                resources={'requests': {
                    'cpu': '3.7',
                    'memory': '4G'
                }},
                tolerations=[{
                    'key': 'preemptible',
                    'value': 'true'
                }],
                service_account_name='test-svc',
                callback=SELF_HOSTNAME + '/ci_build_done',
                attributes=attributes,
                volumes=[{
                    'volume': {
                        'name': f'hail-ci-{VERSION}-service-account-key',
                        'secret': {
                            'optional': False,
                            'secretName':
                            f'hail-ci-{VERSION}-service-account-key'
                        }
                    },
                    'volume_mount': {
                        'mountPath': '/secrets',
                        'name': f'hail-ci-{VERSION}-service-account-key',
                        'readOnly': True
                    }
                }])
            return Building(job, img, target.sha)
        except Exception as e:
            log.exception(f'could not start batch job due to {e}')
            return Buildable(img, target.sha)
    else:
        return NoImage(target.sha)


def determine_buildability(source, target):
    img = maybe_get_image(source, target)
    if img:
        return Buildable(img, target.sha)
    else:
        return NoImage(target.sha)


def get_image_for_target(target):
    import requests
    assert isinstance(target, FQRef), target
    url = f'https://github.com/{target.repo.qname}/raw/{target.name}/hail-ci-build-image'
    r = requests.get(url, timeout=5)
    if r.status_code != 200:
        raise BadStatus(f'could not get raw hail-ci-build-image for {target.short_str()}',
                        r.status_code)
    return r.text.strip()


def maybe_get_image(source, target):
    assert isinstance(source, FQSHA)
    assert isinstance(target, FQSHA)
    d = os.getcwd()
    try:
        srepo = source.ref.repo
        trepo = target.ref.repo
        if not os.path.isdir(trepo.qname):
            os.makedirs(trepo.qname, exist_ok=True)
            os.chdir(trepo.qname)
            shell('git', 'clone', trepo.url, '.')
        else:
            os.chdir(trepo.qname)
        if sp.run(['/bin/sh', '-c', f'git remote | grep -q {srepo.qname}']).returncode != 0:
            shell('git', 'remote', 'add', srepo.qname, srepo.url)
        shell('git', 'fetch', 'origin')
        shell('git', 'fetch', srepo.qname)
        shell('git', 'checkout', target.sha)
        shell('git', 'config', 'user.email', 'hail-ci-leader@example.com')
        shell('git', 'config', 'user.name', 'hail-ci-leader')
        shell('git', 'merge', source.sha, '-m', 'foo')
        # a force push that removes refs could fail us... not sure what we
        # should do in that case. maybe 500'ing is OK?
        with open('hail-ci-build-image', 'r') as f:
            return f.read().strip()
    except (sp.CalledProcessError, FileNotFoundError) as e:
        log.exception(f'could not get hail-ci-build-image due to {e}')
        return None
    finally:
        shell('git', 'reset', '--merge')
        os.chdir(d)


class GitHubPR(object):
    def __init__(self, state, number, title, source, target_ref, target_sha=None):
        assert state in ['closed', 'open']
        assert isinstance(number, str), f'{type(number)} {number}'
        assert isinstance(title, str), f'{type(title)} {title}'
        assert isinstance(source, FQSHA), f'{type(source)} {source}'
        assert isinstance(target_ref, FQRef), f'{type(target_ref)} {target_ref}'
        assert target_sha is None or isinstance(target_sha, str), f'{type(target_sha)} {target_sha}'
        self.state = state
        self.number = number
        self.title = title
        self.source = source
        self.target_ref = target_ref
        self.target_sha = target_sha

    @staticmethod
    def from_gh_json(d, target_sha=None):
        assert 'state' in d, d
        assert 'number' in d, d
        assert 'title' in d, d
        assert 'head' in d, d
        assert 'base' in d, d
        return GitHubPR(d['state'],
                        str(d['number']),
                        str(d['title']),
                        FQSHA.from_gh_json(d['head']),
                        FQSHA.from_gh_json(d['base']).ref,
                        target_sha)

    def __str__(self):
        return json.dumps(self.to_dict())

    def short_str(self):
        tsha = self.target_sha
        if tsha:
            tsha = tsha[:SHA_LENGTH]
        return (
            f'[GHPR {self.number}]{self.target_ref.short_str()}:{tsha}..{self.source.short_str()};'
            f'{self.state};'
        )

    def to_dict(self):
        return {
            'state': self.state,
            'number': self.number,
            'title': self.title,
            'source': self.source.to_dict(),
            'target_ref': self.target_ref.to_dict(),
            'target_sha': self.target_sha
        }

    def to_PR(self, start_build=False):
        if self.target_sha is None:
            target_sha = latest_sha_for_ref(self.target_ref)
        else:
            target_sha = self.target_sha
        target = FQSHA(self.target_ref, target_sha)
        pr = PR.fresh(self.source, target, self.number, self.title)
        if start_build:
            return pr.build_it()
        else:
            return pr


class PR(object):
    def __init__(self, source, target, review, build, number, title):
        assert isinstance(target, FQSHA), target
        assert isinstance(source, FQSHA), source
        assert number is None or isinstance(number, str)
        assert title is None or isinstance(title, str)
        assert review in ['pending', 'approved', 'changes_requested']
        self.source = source
        self.target = target
        self.review = review
        self.build = build
        self.number = number
        self.title = title

    keep = Sentinel()

    def copy(self,
             source=keep,
             target=keep,
             review=keep,
             build=keep,
             number=keep,
             title=keep):
        return PR(
            source=self.source if source is PR.keep else source,
            target=self.target if target is PR.keep else target,
            review=self.review if review is PR.keep else review,
            build=self.build if build is PR.keep else build,
            number=self.number if number is PR.keep else number,
            title=self.title if title is PR.keep else title)

    def _maybe_new_shas(self, new_source=None, new_target=None):
        assert new_source is not None or new_target is not None
        assert new_source is None or isinstance(new_source, FQSHA)
        assert new_target is None or isinstance(new_target, FQSHA)
        if new_source and self.source != new_source:
            assert not self.is_merged()
            if new_target and self.target != new_target:
                log.info(
                    f'new source and target sha {new_target.short_str()} {new_source.short_str()} {self.short_str()}'
                )
                return self._new_target_and_source(new_target, new_source)
            else:
                log.info(f'new source sha {new_source.short_str()} {self.short_str()}')
                return self._new_source(new_source)
        else:
            if new_target and self.target != new_target:
                if self.is_merged():
                    log.info(f'ignoring new target sha for merged PR {self.short_str()}')
                    return self
                else:
                    log.info(f'new target sha {new_target.short_str()} {self.short_str()}')
                    return self._new_target(new_target)
            else:
                return self

    def _new_target_and_source(self, new_target, new_source):
        return self.copy(
            source=new_source,
            target=new_target,
            review='pending'
        )._new_build(
            # FIXME: if I already have an image, just use it
            try_new_build(new_source, new_target)
        )

    def _new_target(self, new_target):
        return self.copy(
            target=new_target
        )._new_build(
            determine_buildability(self.source, new_target)
        )

    def _new_source(self, new_source):
        return self.copy(
            source=new_source,
            review='pending'
        )._new_build(
            # FIXME: if I already have an image, just use it
            try_new_build(new_source, self.target)
        )

    def _new_build(self, new_build):
        if self.build != new_build:
            self.notify_github(new_build)
            return self.copy(build=self.build.transition(new_build))
        else:
            return self

    def build_it(self):
        # FIXME: if I already have an image, just use it
        return self._new_build(try_new_build(self.source, self.target))

    # FIXME: this should be a verb
    def merged(self):
        return self._new_build(Merged(self.target.sha))

    def notify_github(self, build, status_sha=None):
        log.info(f'notifying github of {build} for {self.short_str()}')
        json = {
            'state': build.gh_state(),
            'description': str(build),
            'context': CONTEXT
        }
        if isinstance(build, Failure) or isinstance(build, Mergeable):
            json['target_url'] = \
                f'https://storage.googleapis.com/{GCS_BUCKET}/{GCS_BUCKET_PREFIX}ci/{self.source.sha}/{self.target.sha}/index.html'
        try:
            post_repo(
                self.target.ref.repo.qname,
                'statuses/' + (status_sha if status_sha is not None else self.source.sha),
                json=json,
                status_code=201)
        except BadStatus as e:
            if e.status_code == 422:
                log.exception(
                    f'Too many statuses applied to {self.source.sha}! This is a '
                    f'dangerous situation because I can no longer block merging '
                    f'of failing PRs.')
            else:
                raise e

    @staticmethod
    def fresh(source, target, number=None, title=None):
        return PR(source, target, 'pending', Unknown(), number, title)

    def __str__(self):
        return json.dumps(self.to_dict())

    def short_str(self):
        return (
            f'[PR {self.number}]{self.target.short_str()}..{self.source.short_str()};'
            f'{self.review};{self.build};'
        )

    @staticmethod
    def from_json(d):
        assert 'target' in d, d
        assert 'source' in d, d
        assert 'review' in d, d
        assert 'build' in d, d
        assert 'number' in d, d
        assert 'title' in d, d
        return PR(
            FQSHA.from_json(d['source']),
            FQSHA.from_json(d['target']),
            d['review'],
            build_state_from_json(d['build']),
            d['number'],
            d['title'],
        )

    def to_dict(self):
        return {
            'target': self.target.to_dict(),
            'source': self.source.to_dict(),
            'review': self.review,
            'build': self.build.to_dict(),
            'number': self.number,
            'title': self.title
        }

    def is_mergeable(self):
        return (isinstance(self.build, Mergeable) and
                self.review == 'approved')

    def is_approved(self):
        return self.review == 'approved'

    def is_building(self):
        return isinstance(self.build, Building)

    def is_pending_build(self):
        return isinstance(self.build, Buildable)

    def is_merged(self):
        return isinstance(self.build, Merged)

    def update_from_github_push(self, push):
        assert isinstance(push, FQSHA)
        assert self.target.ref == push.ref, f'{push} {self.short_str()}'
        return self._maybe_new_shas(new_target=push)

    def update_from_github_pr(self, gh_pr):
        assert isinstance(gh_pr, GitHubPR)
        assert self.target.ref == gh_pr.target_ref
        assert self.source.ref == gh_pr.source.ref
        # this will build new PRs when the server restarts
        if gh_pr.target_sha:
            result = self._maybe_new_shas(
                new_source=gh_pr.source,
                new_target=FQSHA(gh_pr.target_ref, gh_pr.target_sha))
        else:
            result = self._maybe_new_shas(new_source=gh_pr.source)
        if self.title != gh_pr.title:
            log.info(f'found new title from github {gh_pr.title} {self.short_str()}')
            result = result.copy(title=gh_pr.title)
        if self.number != gh_pr.number:
            log.info(f'found new PR number from github {gh_pr.title} {self.short_str()}')
            result = result.copy(number=gh_pr.number)
        return result

    def update_from_github_review_state(self, review):
        if self.review != review:
            log.info(f'review state changing from {self.review} to {review} {self.short_str()}')
            return self.copy(review=review)
        else:
            return self

    def update_from_github_status(self, build):
        if isinstance(self.build, Unknown):
            if self.target.sha == build.target_sha:
                log.info(
                    f'recovering from unknown build state via github. {build} {self.short_str()}'
                )
                return self.copy(build=build)
            else:
                log.info('ignoring github build state for wrong target. '
                         f'{build} {self.short_str()}')
                return self
        else:
            log.info(f'ignoring github build state. {build} {self.short_str()}')
            return self

    def refresh_from_batch_job(self, job):
        state = job.cached_status()['state']
        if state == 'Complete':
            return self.update_from_completed_batch_job(job)
        elif state == 'Cancelled':
            log.error(
                f'a job for me was cancelled {short_str_build_job(job)} {self.short_str()}')
            return self._new_build(try_new_build(self.source, self.target))
        else:
            assert state == 'Created', f'{state} {job.id} {job.attributes} {self.short_str()}'
            assert 'target' in job.attributes, job.attributes
            assert 'image' in job.attributes, job.attributes
            target = FQSHA.from_json(json.loads(job.attributes['target']))
            image = job.attributes['image']
            if target == self.target:
                return self._new_build(Building(job, image, target.sha))
            else:
                log.info(f'found deploy job {job.id} for wrong target {target}, should be {self.target}')
                job.cancel()
                return self

    def refresh_from_missing_job(self):
        assert isinstance(self.build, Building)
        return self.build_it()

    def update_from_completed_batch_job(self, job):
        assert isinstance(job, Job)
        job_status = job.cached_status()
        exit_code = job_status['exit_code']
        job_source = FQSHA.from_json(json.loads(job.attributes['source']))
        job_target = FQSHA.from_json(json.loads(job.attributes['target']))
        assert job_source.ref == self.source.ref
        assert job_target.ref == self.target.ref

        if job_target.sha != self.target.sha:
            log.info(
                f'notified of job for old target {job.id}'
                # too noisy: f' {job.attributes} {self.short_str()}'
            )
            x = self
        elif job_source.sha != self.source.sha:
            log.info(
                f'notified of job for old source {job.id}'
                # too noisy: f' {job.attributes} {self.short_str()}'
            )
            x = self
        elif exit_code == 0:
            log.info(f'job finished success {short_str_build_job(job)} {self.short_str()}')
            x = self._new_build(Mergeable(self.target.sha, job))
        else:
            log.info(f'job finished failure {short_str_build_job(job)} {self.short_str()}')
            x = self._new_build(
                Failure(exit_code,
                        job,
                        job.attributes['image'],
                        self.target.sha))
        job.cancel()
        return x
