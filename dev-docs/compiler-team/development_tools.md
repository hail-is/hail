# Hail development tools

This document describes and links tools used by the Hail compiler team.
The document is written for the most common operating system used by
the team, macOS.

## General tools

##### Homebrew - macOS package manager

Homebrew is hard to live without. Use it to install many of the other tools
used by the team.

https://brew.sh/

##### git - version control

It's nice to have a relatively recent version of git. Install this with
brew:

    brew install git

It will probably be necessary to change system paths so that the
installed git is available before system git, as [described here](https://ajahne.github.io/blog/tools/2018/06/11/how-to-upgrade-git-mac.html).

Once this is working, you should fork the hail-is/hail repository into
your own user space, then clone the repository locally:

    git clone https://github.com/username/hail.git

Then add a remote for the main repository to pull in changes:

    git remote add hi https://github.com/hail-is/hail.git


##### Zulip - dev / user chat

We use Zulip for development discussion and conversations with users
(though not typically for user support).

Get it here:

https://zulip.com/

Our Zulip server is https://hail.zulipchat.com

##### Anaconda - manage Python installations and packages

https://www.anaconda.com/download/#macos

After installing Anaconda, you should create a new dev environment
for Hail with:

    conda create --name hail python=3.9

and

    conda activate hail

(put the latter in a shell .rc file so this is done on shell startup)

##### IntelliJ IDEA - IDE for java/scala/python

https://www.jetbrains.com/idea/

Configuration is hard to document here, get help by asking the team.

##### iTerm2 - terminal replacement

iTerm2 is (subjectively) nicer to use and objectively more customizable
than the built-in macOS terminal.

https://iterm2.com/

##### Google cloud utilities

We primarily use Google Cloud for development. Get the SDK here:

https://cloud.google.com/sdk/docs/install
