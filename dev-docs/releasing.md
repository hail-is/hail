# Releasing FAQ

## Commits were merged after a broken release

If the release process broken and new commits have merged since we modified the change log, which
commit should we release?

If a commit has been tagged, release that commit. If fixes are necessary, create a branch from the
tagged commit, add commits as necessary, modify the tag, and hand release that.

## Failure due to "active Temporary Hold"

The release build.yaml job fails due to "'hail-common/dataproc/0.2.XXX/vep-GRCh37.sh' is under
active Temporary Hold". What do I do?

There are four files uploaded by Hail for use by Dataproc clusters (a wheel, to VEP scripts, and a
notebook initialization script). We place a temporary hold on these files to prevent them from being
inadvertently deleted. If all four files were successfully uploaded, you can continue the release
from this point by directly executing release.sh for the particular commit used to generate the
uploaded wheel. You can check the commit by downloading the wheel, installing it, and running

    python3 -c 'import hail; print(hail.version())'

Checkout this commit locally. Then find the corresponding release batch by searching for:

    sha = THE_FULL_SHA

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
    gcloud storage cp -r \
	  gs://hail-ci-bpk3h/build/$BUILD_TOKEN/hail_version \
      gs://hail-ci-bpk3h/build/$BUILD_TOKEN/hail_pip_version \
      gs://hail-ci-bpk3h/build/$BUILD_TOKEN/git_version \
      gs://hail-ci-bpk3h/build/$BUILD_TOKEN/azure-wheel \
      gs://hail-ci-bpk3h/build/$BUILD_TOKEN/www.tar.gz \
	  $BUILD_TOKEN

Note that the `-r` is necessary because some of these things like, `azure-wheel` are folder.

Next we need to authenticate with DockerHub. Download the secret and authenticate skopeo with it.

    download-secret docker-hub-hailgenetics
    cat contents/password | skopeo login --username hailgenetics --password-stdin docker.io

Next we need to cook up a valid pypirc:

    download-secret hail-ci-0-1-github-oauth-token
    cp contents/pypirc $HOME/.pypirc

Next we need to cook up a valid github-oauth token (for creating GitHub releases):

    printf 'Authorization: token ' > /PATH/TO/WORKING_DIR/$BUILD_TOKEN/github-oauth
    cat contents/oauth-token >>/PATH/TO/WORKING_DIR/$BUILD_TOKEN/github-oauth

We use those same credentials to automatically create releases against DSP's repositories:

    printf '#!/bin/bash\necho ' > git-askpass
    cat contents/oauth-token >>git-askpass
    chmod 755 git-askpass
    export GIT_ASKPASS=$(pwd)/git-askpass

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

2. Replace `/io` with `$BUILD_TOKEN`.

It should look something like this:

    bash /PATH/TO/YOUR/HAIL/REPOSITORY/hail/scripts/release.sh \
	    $(cat $BUILD_TOKEN/hail_pip_version) \
        $(cat $BUILD_TOKEN/hail_version) \
        $(cat $BUILD_TOKEN/git_version) \
        origin \
        /PATH/TO/DOWNLOADED/HAIL-COMMON/WHEEL/hail-0.2.XXX-py3-none-any.whl \
        $BUILD_TOKEN/github-oauth \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-syrodsx1m9j7 \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hailtop:deploy-a3opsijrtgir \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-tmdcpjx6zbvh \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-w1ehxyfzy2jl \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch37-85:deploy-f51bxmvgmwsb \
        docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch38-95:deploy-dv77x7gtm8ns \
        $BUILD_TOKEN/azure-wheel/hail-*-py3-none-any.whl \
        $BUILD_TOKEN/www.tar.gz'

## Failure due to a tag or a release already existing

If you are hand releasing and the release script exits because the tag or release already exists,
you can safely comment out the lines that check for that and the lines that create those
things. Then you may execute the script to continue with the rest of the release.
