# Hail Development Process

The lifecycle of a new contribution to the Hail code base consists of the
following steps:  designing the feature, implementing changes, creating a PR,
reviewing a PR, approving and merging the PR, deploying the changes, and then
making periodic releases for users.


## Design

New features can either be bug fixes that users run into, small feature
improvements, or larger, more complicated features. For larger projects, we have
found having the developer write a formal proposal in a Google Doc or Dev
Discuss (dev.hail.is) post is very helpful. We use this process as a chance to
refine the design as well as educate the rest of the team on proposed
changes. It helps to have multiple eyes thinking about what the implications of
the changes are to the rest of the system. In addition, we use this time to
think about how to break down the feature into smaller, more manageable
chunks. Ideally, branches should contain up to 200 lines of changes to make the
process easier on the reviewer. It may not always be possible to break up a
feature into smaller components.


## Implementation

### Environment / Tooling
Before you can write code, there are some setup steps that will allow you to
develop effectively.

Hail currently supports Python version 3.7 or greater.

```
make install-dev-requirements
```

to install the full set of python dependencies across the Hail repo.

To make sure that certain formatting requirements are caught early, run

```
pre-commit install --install-hooks
```

This creates git hooks that run certain linting checks and auto-formatting on
changed files every commit. For example, services code uses the
[Black python formatter](https://black.readthedocs.io/en/stable/)
to enforce PEP8 compliance.

Sometimes large formatting or refactoring commits can muddle the git history
for a file. If your change is one of these, follow up by adding the commit SHA to
`.git-blame-ignore-revs`. To configure `git blame` to ignore these commits, run

```
git config blame.ignoreRevsFile $HAIL/.git-blame-ignore-revs
```

#### Services

Install and configure tools necessary for working on the Hail Services:

1. Install [Docker](https://docker.com)
2. Install [`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl/), if not already installed. (To test installation, run `kubectl` in a terminal window)
3. Install [`gcloud`](https://cloud.google.com/sdk/docs/install)
4. Configure gcloud and Docker for Hail:
```
gcloud auth login
gcloud config set project hail-vdc
gcloud container clusters get-credentials vdc --zone=us-central1-a
gcloud auth -q configure-docker us-docker.pkg.dev
```

5. Add these lines to `~/.zshrc` or `~/.bashrc` to configure your shell and environment for Hail:
```
# BuildKit, a fast docker backend
export DOCKER_BUILDKIT=1
# Shell utilities for managing the Hail kubernetes cluster
source /path/to/hail-repository/devbin/functions.sh
```
6. Run `brew install fswatch`

### Testing / Debugging
There are different strategies for debugging depending on whether you are
working on a compiler project or a services project.

#### Compiler:

For a compiler project, you can build and run the tests locally on your
computer. To build hail for development purposes, you should run the following
command in the hail/hail directory:

```
make install-editable
```

There are tests written in Python and tests written in Scala. For the python
tests, we use pytest.


#### Services:

For a services project, you can push your branch to GitHub and then run what we
call “dev deploy”. The command to invoke this is

```
hailctl dev deploy -b <github_user_name>/hail:<branch_name> -s <step1>,<step2>,...
```

Dev deploy creates a batch that deploys the build steps specified by the `-s` in
your own Kubernetes namespace, MySQL database. For example, if we want to test
whether the Batch tests still pass, we would specify -s test_batch. This will
run all the dependent steps for testing Batch such as creating credentials,
a live Auth service, a MySQL database for Batch, and a live Batch deployment.
Your namespace name is the same as your username.
Submitting a dev deploy with hailctl will give you the link to a UI
where you can monitor the progress of everything deploying and get the logs for
any steps that fail. You can also see a recent history of your dev deploys at
[ci.hail.is/me](https://ci.hail.is/me).


If the tests fail, you can then examine the Kubernetes logs for the service
using something like

```
kubectl -n <my_namespace> logs -l app=batch-driver --tail=999999 | less
```

To check the MySQL database, you first need to find the name of the specific pod
running an “admin-pod” and then log into that pod:

```
kubectl -n <my_namespace> get pods -l app=admin-pod
kubectl -n <my_namespace> exec -it <admin_pod_name> /bin/bash
$ mysql
```

## PR

Once you have a branch that you are happy with, then you create a Pull Request
on the GitHub UI.

You’ll want to add an appropriate reviewer in the "Reviewers" box on the
right hand side of the page. If you are an outside contributor and cannot
request reviews, you can have CI automatically assign a reviewer. By writing
`#assign services` or `#assign compiler` in the PR body, CI will randomly select
a collaborator on the relevant team and assign them for you.

You can also give the PR a set of labels. The important ones are “WIP” to make
sure the pull request doesn’t get merged accidentally until you are ready,
“migration” to warn everyone that the changes will shut down the Batch
deployment if it requires a database migration, “bug” for bug fixes, “breaking
change” for any user breaking changes for Hail Query, and “prio:high” to make
this PR the first one in line to merge. There’s also “stacked PR” to indicate
that the changes in the PR are dependent on the changes in another PR. You
should reference that PR in your commit message with “Stacked on #9883”. Most
PRs will not have any labels.

For the PR title, start the title with the name of the service(s) the changes
impact. For example, if it’s a Benchmark change, then you’d write
`[benchmark]`. If it’s a Hail Query change, then it would be `[query]`. We also want
the title to be descriptive enough to know what the change is without being too
verbose. An example is “`[batch]` Added read_only option for gcsfuse”.

For the PR commit message, we want the message to be descriptive of the complete
set of changes that occurred, especially if it’s a complicated set of
changes. If it’s a smaller, obvious change like a one-liner, then it’s okay to
omit the commit message. For UI changes, it’s helpful to paste a screenshot of
the changes. It’s also a good idea to comment on how you tested the changes and
whether there are any implications of your changes. If the PR fixes a bug that
is a GitHub issue, then you can say “Fixes #8900” to make sure the issue gets
automatically closed by GitHub when your PR is merged. You can also tag a
specific member of the team in the message to get their attention with “@user”.

If the changes are user-facing, then add a line in your commit message that
starts with “CHANGELOG: description.”. This should be one line with one sentence
that ends in a period.

Once you are done with writing up all the details of the Pull Request, you can
then submit it. Our continuous integration (CI) system watches for new PRs. When
it sees a new PR, it creates a new batch that will test everything defined in
the build.yaml file in the root of the repository. This will create a temporary
namespace in kubernetes for your PR and deploy all the services into it. It will
also run all the tests such as for query, batch, and ci after merging it with
the latest version of main. You can view the progress of the build and the
logs for your PR at [ci.hail.is](https://ci.hail.is).


## Review

Once the PR has been created, it is the responsibility of the reviewer(s) to
review the PR. Our goal as a team is to give comments within 24 hours. To review
someone else’s changes, click on “Files changed”. This will show the diff
between the old code and the new proposed changes. You can make comments on
specific lines of the code. Feel free to ask questions here, especially if you
don’t understand something! It’s a good idea to think critically about the
changes. There should also be tests either added or existing to make sure the
code changes do not break any existing functionality and actually implement what
was intended. For example, a change to test whether Batch doesn’t crash when a
user gives a bad input should have a test with bad inputs. It’s okay to spend a
lot of time reviewing PRs! This is a critical part of our development process to
avoid bugs and unintentional breaking changes. If there are items for the
developer to address, then submit your review with “Request Changes”. Otherwise,
once you are happy with the changes and all comments have been addressed, you
can “Approve” the PR.

If you are the person whose code is being reviewed and your PR is in the Request
Changes state, then you’ll need to address their comments by pushing new commit
changes or answering questions. Once you are done, then you can re-request a review
in the "Reviewers" box.

If your review is requested on a PR submitted by an outside contributor, you should
"assign" yourself or the appropriate team member to the PR. The assignee is
responsible for ensuring that the PR does not go stale and is eventually
merged or closed.

![](dismiss_review.png)


## Merge / Deploy

Once a PR has been approved, our continuous integration service (CI) will squash
merge the commits in the PR into the main branch of the repository. This will
then trigger a deploy batch. The deploy batch will first deploy all of the new
Docker images, redeploy the running services in the default namespace with the
latest changes, and rerun all of the tests with the new version of main
incorporating your changes.

If this batch fails, a Zulip message will be sent to the entire team linking to
the UI for the failing batch. We should never ignore this message and figure out
what component broke. If it’s a transient error, then we need to harden our
retry strategy.

If a Batch database migration is involved in the PR, then we’ll need to wait for
the database to be migrated and then redeploy the Batch service by hand using

```
make -c batch deploy
```

Be aware that the deploy process impacts all running services immediately, but
the changes to the documentation, website, and the Spark version of Hail Query
are not user-visible.


## Release

To actually expose changes to our users for the hailctl, hailtop, and hail
Python libraries and create a new JAR file and Wheel to the cloud, we create a
PR that bumps up the version number of the PIP package and adds the changes in
the CHANGELOG to the appropriate places. Once this PR merges, then CI will
redeploy and test everything as above, but it will also update the website,
documentation, and release a new version of the Hail Python library on
PyPi. Once this is successful, then we make an announcement post on Zulip and
the Discuss Forum.

It is relatively easy for us to make a new release. We try and do this at least
once every other week, but can also do it for bug fixes and user-facing changes
that people have requested immediate access to.
