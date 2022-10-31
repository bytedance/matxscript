# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: This file originates from py-filelock
# https://github.com/tox-dev/py-filelock
# with changes applied:
#  - refactor type hint for compatible with py3.6
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# fmt: off
import contextlib
import logging
import os
import sys
import stat
import time
import warnings
from abc import ABC, abstractmethod
from threading import Lock
from types import TracebackType
from typing import Type, Optional
from errno import EACCES, EEXIST, ENOENT
from typing import cast

_LOGGER = logging.getLogger("filelock")


def raise_on_exist_ro_file(filename: str) -> None:
    try:
        file_stat = os.stat(filename)  # use stat to do exists + can write to check without race condition
    except OSError:
        return None  # swallow does not exist or other errors

    if file_stat.st_mtime != 0:  # if os.stat returns but modification is zero that's an invalid os.stat - ignore it
        if not (file_stat.st_mode & stat.S_IWUSR):
            raise PermissionError(f"Permission denied: {filename!r}")


class FileLockTimeout(TimeoutError):
    """Raised when the lock could not be acquired in *timeout* seconds."""

    def __init__(self, lock_file: str) -> None:
        #: The path of the file lock.
        self.lock_file = lock_file

    def __str__(self) -> str:
        return f"The file lock '{self.lock_file}' could not be acquired."


# This is a helper class which is returned by :meth:`BaseFileLock.acquire` and wraps the lock to make sure __enter__
# is not called twice when entering the with statement. If we would simply return *self*, the lock would be acquired
# again in the *__enter__* method of the BaseFileLock, but not released again automatically. issue #37 (memory leak)
class AcquireReturnProxy:
    """A context aware object that will release the lock file when exiting."""

    def __init__(self, lock) -> None:
        self.lock = lock

    def __enter__(self):
        return self.lock

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[TracebackType],
    ) -> None:
        self.lock.release()


class BaseFileLock(ABC, contextlib.ContextDecorator):
    """Abstract base class for a file lock object."""

    def __init__(self, lock_file: str, timeout: float = -1) -> None:
        """Create a new lock object.

        Parameters
        ----------
        lock_file: str
            path to the file

        timeout: float
            default timeout when acquiring the lock, in seconds. It will be used as fallback value in
            the acquire method, if no timeout value (``None``) is given. If you want to disable the timeout, set it
            to a negative value. A timeout of 0 means, that there is exactly one attempt to acquire the file lock.
        """
        # The path to the lock file.
        self._lock_file: str = os.fspath(lock_file)

        # The file descriptor for the *_lock_file* as it is returned by the os.open() function.
        # This file lock is only NOT None, if the object currently holds the lock.
        self._lock_file_fd: Optional[int] = None

        # The default timeout value.
        self.timeout: float = timeout

        # We use this lock primarily for the lock counter.
        self._thread_lock: Lock = Lock()

        # The lock counter is used for implementing the nested locking mechanism. Whenever the lock is acquired, the
        # counter is increased and the lock is only released, when this value is 0 again.
        self._lock_counter: int = 0

    @property
    def lock_file(self) -> str:
        """
        Returns
        -------
        result: str
            path to the lock file
        """
        return self._lock_file

    @property
    def timeout(self) -> float:
        """

        Returns
        -------
        result: float
            the default timeout value, in seconds
        """
        return self._timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        """Change the default timeout value.

        Parameters
        ----------
        value: float
            the new value, in seconds

        Returns
        -------

        """
        self._timeout = float(value)

    @abstractmethod
    def _acquire(self) -> None:
        """If the file lock could be acquired, self._lock_file_fd holds the file descriptor of the lock file."""
        raise NotImplementedError

    @abstractmethod
    def _release(self) -> None:
        """Releases the lock and sets self._lock_file_fd to None."""
        raise NotImplementedError

    @property
    def is_locked(self) -> bool:
        """

        Returns
        -------
        state: bool
            A boolean indicating if the lock file is holding the lock currently.
        """
        return self._lock_file_fd is not None

    def acquire(
            self,
            timeout: Optional[float] = None,
            poll_interval: float = 0.05,
            *,
            blocking: bool = True,
    ) -> AcquireReturnProxy:
        """Try to acquire the file lock.

        Parameters
        ----------
        timeout: Optional[float]
            maximum wait time for acquiring the lock, ``None`` means use the default :attr:`~timeout` is and
            if ``timeout < 0``, there is no timeout and this method will block until the lock could be acquired

        poll_interval: float
            interval of trying to acquire the lock file

        blocking: bool
            defaults to True. If False, function will return immediately if it cannot obtain a lock on the
            first attempt. Otherwise this method will block until the timeout expires or the lock is acquired.

        Returns
        -------
        result: AcquireReturnProxy
            raise Timeout if fails to acquire lock within the timeout period
            a context object that will unlock the file when the context is exited

        Examples
        --------
        >>> # You can use this method in the context manager (recommended)
        >>> lock = FileLock("test.lock")
        >>> with lock.acquire():
        ...    pass
        >>> # Or use an equivalent try-finally construct:
        >>> lock.acquire()
        >>> try:
        ...    pass
        ... finally:
        ...    lock.release()
        """
        # Use the default timeout, if no timeout is provided.
        if timeout is None:
            timeout = self.timeout

        # Increment the number right at the beginning. We can still undo it, if something fails.
        with self._thread_lock:
            self._lock_counter += 1

        lock_id = id(self)
        lock_filename = self._lock_file
        start_time = time.monotonic()
        try:
            while True:
                with self._thread_lock:
                    if not self.is_locked:
                        _LOGGER.debug("Attempting to acquire lock %s on %s", lock_id, lock_filename)
                        self._acquire()

                if self.is_locked:
                    _LOGGER.debug("Lock %s acquired on %s", lock_id, lock_filename)
                    break
                elif blocking is False:
                    _LOGGER.debug("Failed to immediately acquire lock %s on %s", lock_id, lock_filename)
                    raise FileLockTimeout(self._lock_file)
                elif 0 <= timeout < time.monotonic() - start_time:
                    _LOGGER.debug("Timeout on acquiring lock %s on %s", lock_id, lock_filename)
                    raise FileLockTimeout(self._lock_file)
                else:
                    msg = "Lock %s not acquired on %s, waiting %s seconds ..."
                    _LOGGER.debug(msg, lock_id, lock_filename, poll_interval)
                    time.sleep(poll_interval)
        except BaseException:  # Something did go wrong, so decrement the counter.
            with self._thread_lock:
                self._lock_counter = max(0, self._lock_counter - 1)
            raise
        return AcquireReturnProxy(lock=self)

    def release(self, force: bool = False) -> None:
        """Releases the file lock. Please note, that the lock is only completely released, if the lock counter is 0. Also
        note, that the lock file itself is not automatically deleted.

        Parameters
        ----------
        force: bool
            If true, the lock counter is ignored and the lock is released in every case/

        Returns
        -------
        """
        with self._thread_lock:

            if self.is_locked:
                self._lock_counter -= 1

                if self._lock_counter == 0 or force:
                    lock_id, lock_filename = id(self), self._lock_file

                    _LOGGER.debug("Attempting to release lock %s on %s", lock_id, lock_filename)
                    self._release()
                    self._lock_counter = 0
                    _LOGGER.debug("Lock %s released on %s", lock_id, lock_filename)

    def __enter__(self):
        """Acquire the lock.

        Returns
        -------
        lock: BaseFileLock
            the lock object
        """
        self.acquire()
        return self

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[TracebackType],
    ) -> None:
        """Release the lock.

        Parameters
        ----------
        exc_type: type of BaseException, optional
            the exception type if raised

        exc_value: BaseException, optional
            the exception value if raised

        traceback: TracebackType, optional
            the exception traceback if raised

        Returns
        -------

        """
        self.release()

    def __del__(self) -> None:
        """Called when the lock object is deleted."""
        self.release(force=True)


class SoftFileLock(BaseFileLock):
    """Simply watches the existence of the lock file."""

    def _acquire(self) -> None:
        raise_on_exist_ro_file(self._lock_file)
        # first check for exists and read-only mode as the open will mask this case as EEXIST
        mode = (
                os.O_WRONLY  # open for writing only
                | os.O_CREAT
                | os.O_EXCL  # together with above raise EEXIST if the file specified by filename exists
                | os.O_TRUNC  # truncate the file to zero byte
        )
        try:
            fd = os.open(self._lock_file, mode)
        except OSError as exception:
            if exception.errno == EEXIST:  # expected if cannot lock
                pass
            elif exception.errno == ENOENT:  # No such file or directory - parent directory is missing
                raise
            elif exception.errno == EACCES and sys.platform != "win32":  # pragma: win32 no cover
                # Permission denied - parent dir is R/O
                raise  # note windows does not allow you to make a folder r/o only files
        else:
            self._lock_file_fd = fd

    def _release(self) -> None:
        os.close(self._lock_file_fd)  # type: ignore # the lock file is definitely not None
        self._lock_file_fd = None
        try:
            os.remove(self._lock_file)
        except OSError:  # the file is already deleted and that's what we want
            pass


if sys.platform == "win32":  # pragma: win32 cover
    import msvcrt


    class WindowsFileLock(BaseFileLock):
        """Uses the :func:`msvcrt.locking` function to hard lock the lock file on windows systems."""

        def _acquire(self) -> None:
            raise_on_exist_ro_file(self._lock_file)
            mode = (
                    os.O_RDWR  # open for read and write
                    | os.O_CREAT  # create file if not exists
                    | os.O_TRUNC  # truncate file  if not empty
            )
            try:
                fd = os.open(self._lock_file, mode)
            except OSError as exception:
                if exception.errno == ENOENT:  # No such file or directory
                    raise
            else:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError:
                    os.close(fd)
                else:
                    self._lock_file_fd = fd

        def _release(self) -> None:
            fd = cast(int, self._lock_file_fd)
            self._lock_file_fd = None
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)

            try:
                os.remove(self._lock_file)
            # Probably another instance of the application hat acquired the file lock.
            except OSError:
                pass


    class UnixFileLock(BaseFileLock):
        def _acquire(self) -> None:
            raise NotImplementedError

        def _release(self) -> None:
            raise NotImplementedError


    FileLock = WindowsFileLock

else:  # pragma: win32 no cover
    class WindowsFileLock(BaseFileLock):

        def _acquire(self) -> None:
            raise NotImplementedError

        def _release(self) -> None:
            raise


    has_fcntl = False
    try:
        import fcntl
    except ImportError:
        pass
    else:
        has_fcntl = True


    class UnixFileLock(BaseFileLock):
        """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""

        def _acquire(self) -> None:
            open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
            fd = os.open(self._lock_file, open_mode)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
            else:
                self._lock_file_fd = fd

        def _release(self) -> None:
            # Do not remove the lockfile:
            #   https://github.com/tox-dev/py-filelock/issues/31
            #   https://stackoverflow.com/questions/17708885/flock-removing-locked-file-without-race-condition
            fd = cast(int, self._lock_file_fd)
            self._lock_file_fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


    if has_fcntl:
        FileLock = UnixFileLock
    else:
        FileLock = SoftFileLock
        if warnings is not None:
            warnings.warn("only soft file lock is available")

__all__ = [
    "FileLock",
    "SoftFileLock",
    "FileLockTimeout",
    "UnixFileLock",
    "WindowsFileLock",
    "BaseFileLock",
    "AcquireReturnProxy",
]
# fmt: on
