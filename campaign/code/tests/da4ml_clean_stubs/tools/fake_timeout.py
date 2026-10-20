"""G72 STAND-IN for GNU coreutils `timeout` (dry run only).

Models the part that shapes a line's process tree: without --foreground, `timeout` puts itself in
a new process group (setpgid(0, 0)), runs the command as its child in that group, forwards
SIGTERM, SIGINT and SIGHUP to the child, and returns the child's status. The duration and the
--signal value are parsed but no timer is armed (the dry run never reaches 1800 or 7200 s), and
coreutils' own signal forwarding to its whole group is not modelled.
"""
import os
import signal
import sys


def main():
    args = sys.argv[1:]
    foreground = False
    while args and args[0].startswith('-'):
        a = args.pop(0)
        if a == '--foreground':
            foreground = True
        elif a in ('-s', '--signal', '-k', '--kill-after'):
            args.pop(0)
        elif a == '--':
            break
    args.pop(0)                      # duration
    if not foreground:
        os.setpgid(0, 0)
    pid = os.fork()
    if pid == 0:
        os.execvp(args[0], args)

    def forward(signum, _frame):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, forward)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    return 128 + os.WTERMSIG(status)


if __name__ == '__main__':
    sys.exit(main())
