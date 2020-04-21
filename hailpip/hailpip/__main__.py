from hailtop.utils import sync_retry_transient_errors
import pip._internal
import pip._internal.commands
import pip._internal.commands.install
import sys


def main():
    def install():
        pip._internal.commands.install.InstallCommand('x', 'y').main(sys.argv[1:])

    sync_retry_transient_errors(install)


if __name__ == '__main__':
    main()
