This directory must contain two files. These files must be named `user1` and
`user2`. Each one must contain a valid OAuth token. Each token must be for a
distinct user. The test system will use these users to create PRs and approve
PRs.

The tokens need:

 - `admin:repo_hook`, used to set up CI webhooks for test repos
 - `delete_repo`, used to delete the repo after tests are finished,
 - `repo`, used to commit, update status, and review PRs on a repo
