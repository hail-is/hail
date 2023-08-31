# Forks and PR workflows

Changes to the Hail codebase are made through pull requests directly to the
`main` branch of the `hail-is/hail` repository. Here are the steps to
take when contributing for the first time and when making a pull request.


## First time actions

In order to keep the `hail-is/hail` repo clean, we ask that contributors develop
features in branches on a personal fork. The first step then is to [fork
the repository](https://github.com/hail-is/hail/fork).

Once you've done so, clone the repository from your fork. This will set up a
local copy of your fork with a single remote called `origin`. In this document,
we will refer to `origin` as your developer fork, and `upstream` as `hail-is/hail`.
You can check which origins you have configured by running `git remote -v`.

While feature branches will live on your fork, those branches are still going to want
to be based on the latest changes in `hail-is/hail:main`. So we will add the
upstream repository as another remote so we can pull in those changes.

```bash
git remote add upstream https://github.com/hail-is/hail.git
```

If you run `git remote -v` again, you should see something like the following:

```
origin	https://github.com/<your-github-username>/hail.git (fetch)
origin	https://github.com/<your-github-username>/hail.git (push)
upstream	https://github.com/hail-is/hail.git (fetch)
upstream	https://github.com/hail-is/hail.git (push)
```

When starting a new feature branch, retrieve the latest changes from
upstream and checkout a branch based on those changes:

```bash
git fetch upstream
git checkout -b <feature_branch> upstream/main
```

## While developing a feature

`hail-is/hail:main` moves quickly, and it is likely that it will have progressed
significantly while you work on a feature. This is not in itself a problem,
but can cause headaches if changes in upstream `main` conflict with the code you
are working on. To cope with this, we recommend [rebasing](https://git-scm.com/docs/git-rebase#_description)
regularly and often. Rebasing will incoporate the new changes from main into your
branch and allow you to resolve any conflicts that might have developed along the
way.

To rebase your feature branch on the latest upstream main, run

```bash
git fetch upstream
git rebase upstream/main <feature_branch>
```

The `-i` here is optional but extremely informative, and can help you understand
what the rebase is doing. You can leave all your commits as `pick`, and keep
all your commits while perfoming the rebase, or you can change some commits to
`squash` to collapse your changes into a smaller number of commits. When opening
your change for review, it can be helpful for the reviewers if you squash your
branch into a small number of self-contained commits. For example, if your change
requires upgrading a dependency, it is helpful to put the dependency upgrades
into a separate commmit from the code changes.


## Making a PR

When a feature branch is ready for PR, push the branch to your fork:

```bash
git push origin <feature_branch>
```

If you have already done this before but have since rebased, you may get
an error because the history on your fork is no longer a prefix of your local
branch's history. Force push your branch to overwrite the history on GitHub
with that of your local branch.

```bash
git push --force-with-lease origin <feature_branch>
```

You can then make a PR on GitHub from the branch on your fork to `hail-is/hail:main`.


## While a feature is in PR

If a reviewer requests changes on a PR, you can make those changes on
your feature branch and `git push origin <feature_branch>` to reflect those
changes in your PR.

However, once the review process has begun it is best not to `rebase` the branch
any further. Doing so rewrites the commit history of the PR and causes GitHub to lose
when and where review comments were made. It also removes the PR reviewer's ability
to use the GitHub feature "see changes since last review", which can be very
helpful for long PRs and review processes.

If an existing PR runs into merge conflicts, you can instead merge main *into* your
feature branch.

```bash
git fetch upstream
git checkout <feature_branch>
git merge upstream/main

... resolve any conflicts and `git add` any resolved files ...

git commit
git push origin <feature_branch>
```

Instead of rewriting the history and losing the state of the review, this will
add a single merge commit that can be ignored by the reviewer.

Another reason not to force push once a change is in PR is if another collaborator
adds changes to the branch. If someone else has made a change to your PR, pull
those changes into your local branch before adding new changes by running

```bash
git pull origin <feature_branch>
```
