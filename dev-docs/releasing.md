# Releasing

## Version

As far as I can tell, we don't really subscribe to any particular church when it
comes to versioning. Our team is light on process and this is reflected on how
we version our software.

Konrad argues that we follow [numpy](https://numpy.org/doc/1.22/user/depending_on_numpy.html#understanding-numpy-s-versioning-and-api-abi-stability),
"but `s/minor/patch/ | s/major/minor/`".

### Format

Released versions of hail follow the format MAJOR.MINOR.PATCH. Development
versions use the current release version with the short SHA appended.

### Major

For now, we've never felt like decaring hail "`1.0` ready", so for now the
major version is always `0`.

### Minor

Hail 0.1 started life as a command line tool called `k3`, allowing users could
perform simple tasks like convert VCFs to VDS and run sampleqc. The team quickly
realised that this had some major shortcomings and so added a python interface
for hail 0.2 known as [PyHail](https://github.com/hail-is/hail/commit/b9a51e969928c5c6bf0fe303036450b9f54fe759).
The name "PyHail" was dropped for "Hail" and this is where we are today.

The minor version of `2` communicates that this is the second revision of hail,
and programs that use a 0.2 release of hail will work for all subseqent 0.2
releases, barring deprecations etc.

### Patch

We favour linear history - it's the reason we use squash commits on top of main.
In the same way, our patch version records a linear sequence of releases. For
now, we increment the patch version for all changes. We avoid breaking changes
and provide deprecation notices for changes of behaviour.

### Summary

For the forseeable future, hail's version is `0.2.X`.
Increment `X` for each new release.
Avoid major breaking changes - try to maintain backwards compatibily.
Issue deprecations notices and release note anything that may require users to
change their code.

## Preparing a New Release


### Release Checklist

[ ] bump patch version in `hail/build.sc` and `hail/version.mk`
[ ] add query release notes in `hail/python/hail/docs/change_log.md`
[ ] add batch release notes in `hail/python/hailtop/batch/docs/change_log.rst`
[ ] open a PR titled "[release] 0.2.X", replacing X with the new patch version.

## FAQ

### Commits were merged after a broken release

If the release process broke and new commits have merged since we modified the change log, which
commit should we release?

If a commit has been tagged, release that commit. If fixes are necessary, create a branch from the
tagged commit, add commits as necessary, modify the tag, and hand release that.

### Failure due to "active Temporary Hold"

The release build.yaml job fails due to "'hail-common/dataproc/0.2.XXX/vep-GRCh37.sh' is under
active Temporary Hold". What do I do?

There are four files uploaded by Hail for use by Dataproc clusters (a wheel, two VEP scripts, and a
notebook initialization script). We place a temporary hold on these files to prevent them from being
inadvertently deleted. If all four files were successfully uploaded, you can continue the release
from this point by directly executing release.sh for the particular commit used to generate the
uploaded wheel. You can check the commit by downloading the wheel, installing it, and running

    python3 -c 'import hail; print(hail.version())'

Checkout this commit locally. Then find the corresponding release batch by searching for:

    sha = THE_FULL_SHA

Change directories into the Hail directory:

    cd /PATH/TO/REPO/hail

You can download all the necessary files to execute release.sh by downloading them from the
hail-ci-bpk3h bucket. The necessary files are listed under "Sources: " in the "Inputs" log of the
"release" build step. They should look something like:

    Sources:
      gs://hail-ci-bpk3h/build/9cabeeb4ba047d1722e6f8da0383ab97/hail_version: 1 files, 21 Bytes
      gs://hail-ci-bpk3h/build/9cabeeb4ba047d1722e6f8da0383ab97/hail_pip_version: 1 files, 8 Bytes
      gs://hail-ci-bpk3h/build/9cabeeb4ba047d1722e6f8da0383ab97/git_version: 1 files, 41 Bytes
      gs://hail-ci-bpk3h/build/9cabeeb4ba047d1722e6f8da0383ab97/repo: 6272 files, 205.1 MB
      gs://hail-ci-bpk3h/build/9cabeeb4ba047d1722e6f8da0383ab97/azure-wheel: 1 files, 144.5 MB
      gs://hail-ci-bpk3h/build/9cabeeb4ba047d1722e6f8da0383ab97/www.tar.gz: 1 files, 43.5 MB

Download all these files except the repo (which you do not need, because you checked out the commit):

    BUILD_TOKEN=9cabeeb4ba047d1722e6f8da0383ab97
    mkdir $BUILD_TOKEN
	RELEASE_ARTIFACTS_DIR=$(realpath $BUILD_TOKEN)
    gcloud storage cp -r \
      gs://hail-ci-bpk3h/build/$BUILD_TOKEN/hail_version \
      gs://hail-ci-bpk3h/build/$BUILD_TOKEN/hail_pip_version \
      gs://hail-ci-bpk3h/build/$BUILD_TOKEN/git_version \
      gs://hail-ci-bpk3h/build/$BUILD_TOKEN/azure-wheel \
      gs://hail-ci-bpk3h/build/$BUILD_TOKEN/www.tar.gz \
      $BUILD_TOKEN

Note that the `-r` is necessary because some of these things like `azure-wheel` are folders.

Next we need to authenticate with DockerHub. Download the secret and authenticate skopeo with
it. `download-secret` is a function stored in `devbin/functions.sh`.

    download-secret docker-hub-hailgenetics
    cat contents/password | skopeo login --username hailgenetics --password-stdin docker.io
	popd

Next we need a valid pypirc:

    download-secret pypi-credentials
    cp contents/pypirc $HOME/.pypirc
	popd

Next we need a valid github-oauth token (for creating GitHub releases):

    download-secret hail-ci-0-1-github-oauth-token
    printf 'Authorization: token ' > $RELEASE_ARTIFACTS_DIR/github-oauth
    cat contents/oauth-token >>$RELEASE_ARTIFACTS_DIR/github-oauth

We use those same credentials to automatically create releases against DSP's repositories:

    printf '#!/bin/bash\necho ' > $RELEASE_ARTIFACTS_DIR/git-askpass
    cat contents/oauth-token >>$RELEASE_ARTIFACTS_DIR/git-askpass
    chmod 755 git-askpass
    export GIT_ASKPASS=$RELEASE_ARTIFACTS_DIR/git-askpass
	popd

Ensure you have returned to the `hail` sub-folder of the Hail git repository.

Now we can construct a `release.sh` invocation. Find the invocation in the "command" part of the
"Job Specification" table. It should look like:

    bash scripts/release.sh $(cat /io/hail_pip_version) \
                            $(cat /io/hail_version) \
                            $(cat /io/git_version) \
                            origin \
                            /io/repo/hail/build/deploy/dist/hail-*-py3-none-any.whl \
                            /io/github-oauth \
                            docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-syrodsx1m9j7 \
                            docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hailtop:deploy-a3opsijrtgir \
                            docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-tmdcpjx6zbvh \
                            docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-w1ehxyfzy2jl \
                            docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch37-85:deploy-f51bxmvgmwsb \
                            docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch38-95:deploy-dv77x7gtm8ns \
                            /io/azure-wheel/hail-*-py3-none-any.whl \
                            /io/www.tar.gz'

We need to make two replacements:

1. Replace the path to the wheel with the path to the wheel we downloaded from hail-common.

2. Replace `/io` with `$RELEASE_ARTIFACTS_DIR`.

It should look something like this:

    bash scripts/release.sh \
        $(cat $RELEASE_ARTIFACTS_DIR/hail_pip_version) \
        $(cat $RELEASE_ARTIFACTS_DIR/hail_version) \
        $(cat $RELEASE_ARTIFACTS_DIR/git_version) \
        origin \
        /PATH/TO/DOWNLOADED/HAIL-COMMON/WHEEL/hail-0.2.XXX-py3-none-any.whl \
        $RELEASE_ARTIFACTS_DIR/github-oauth \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-syrodsx1m9j7 \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hailtop:deploy-a3opsijrtgir \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-tmdcpjx6zbvh \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-w1ehxyfzy2jl \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch37-85:deploy-f51bxmvgmwsb \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch38-95:deploy-dv77x7gtm8ns \
        $RELEASE_ARTIFACTS_DIR/azure-wheel/hail-*-py3-none-any.whl \
        $RELEASE_ARTIFACTS_DIR/www.tar.gz'

When you are complete, delete all the credentials:

    rm $RELEASE_ARTIFACTS_DIR/git-askpass
    rm $RELEASE_ARTIFACTS_DIR/github-oauth
    rm $HOME/.pypirc
	skopeo logout docker.io

You should also delete the temporary directories used to download the credentials. On Mac OS X,
those directories are all under $TMPDIR which looks like
`/var/folders/x1/601098gx0v11qjx2l_7qfw2c0000gq/T/`. If you're comfortable deleting all of $TMPDIR,
just run:

    rm -rf $TMPDIR

### Failure due to a tag or a release already existing

If you are hand releasing and the release script exits because the tag or release already exists,
you can safely comment out the lines that check for that and the lines that create those
things. Then you may execute the script to continue with the rest of the release.
