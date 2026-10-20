"""Open-file audit for gates G07 and G08 (Linux only; PREFLIGHT-PLAN section 3 rows G07, G08).

HDF5 opens files in C, so Python's audit hook (`sys.addaudithook`, event `open`) does not see an
h5py open. This module asks the kernel instead:

- inotify (`inotify_init1` and `inotify_add_watch` from libc through ctypes), mask IN_OPEN. A watch
  on a file reports every open of that inode, through any path, symbolic link or hard link; a
  watch on a directory reports opens of the directory itself and of every entry reached through
  it, by name. Events come from every process on the machine, so a gate lists every event of its
  window and fails on any forbidden one, whoever caused it. The verdict rests on these events.
- `/proc/<pid>/fd` of the audited child, read every millisecond while it runs, as an inventory of
  the files it held open. An open shorter than the sampling interval can be missed, so the
  inventory supports the verdict but does not decide it.

Standard library only. Every reading is engineering, not quotable.
"""
import ctypes
import ctypes.util
import os
import struct
import subprocess
import threading
import time

from syncroot import require_unsynced

IN_OPEN = 0x00000020
IN_Q_OVERFLOW = 0x00004000
IN_ISDIR = 0x40000000
IN_NONBLOCK = 0o4000
IN_CLOEXEC = 0o2000000
_EVENT = struct.Struct('iIII')


class Inotify:
    def __init__(self):
        self.libc = ctypes.CDLL(ctypes.util.find_library('c') or 'libc.so.6', use_errno=True)
        self.fd = self.libc.inotify_init1(IN_NONBLOCK | IN_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1 failed')
        self.watches = {}

    def watch(self, path, label):
        wd = self.libc.inotify_add_watch(self.fd, os.fsencode(str(path)), IN_OPEN)
        if wd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_add_watch failed for %s' % path)
        self.watches[wd] = (str(path), label)
        return wd

    def drain(self):
        out = []
        while True:
            try:
                buf = os.read(self.fd, 1 << 16)
            except BlockingIOError:
                break
            if not buf:
                break
            i = 0
            while i + _EVENT.size <= len(buf):
                wd, mask, _cookie, ln = _EVENT.unpack_from(buf, i)
                name = buf[i + _EVENT.size:i + _EVENT.size + ln].split(b'\0', 1)[0].decode('utf-8', 'replace')
                i += _EVENT.size + ln
                path, label = self.watches.get(wd, ('?', '?'))
                out.append({'label': label, 'watch': path, 'name': name, 'isdir': bool(mask & IN_ISDIR),
                            'overflow': bool(mask & IN_Q_OVERFLOW)})
        return out

    def close(self):
        os.close(self.fd)


def run_audited(cmd, *, watches, env=None, cwd=None, timeout=3600, sample_s=0.001):
    """Run `cmd` with inotify watches [(path, label), ...] active from before it starts until
    after it ends, sampling its /proc fd table. Returns returncode, stdout, stderr, the events of
    the window, events seen before it started (must be none), and the fd inventory."""
    watches = tuple(watches)
    require_unsynced(*(path for path, _ in watches), cwd if cwd is not None else os.getcwd())
    ino = Inotify()
    try:
        for path, label in watches:
            ino.watch(path, label)
        before = ino.drain()
        seen = set()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd, text=True)

        def sample():
            fd_dir = '/proc/%d/fd' % p.pid
            while p.poll() is None:
                try:
                    for e in os.listdir(fd_dir):
                        try:
                            seen.add(os.readlink(os.path.join(fd_dir, e)))
                        except OSError:
                            pass
                except OSError:
                    pass
                time.sleep(sample_s)

        t = threading.Thread(target=sample, daemon=True)
        t.start()
        out, err = p.communicate(timeout=timeout)
        t.join(timeout=10)
        time.sleep(0.1)
        events = ino.drain()
    finally:
        ino.close()
    return {'returncode': p.returncode, 'stdout': out, 'stderr': err, 'events': events,
            'events_before_start': before, 'fd_paths': sorted(seen)}


def events_with(events, labels, name=None):
    """Events whose label is in `labels` (and, if given, whose entry name equals `name`)."""
    return [e for e in events if e['label'] in labels and (name is None or e['name'] == name)]
