from typing import Optional, Dict, Union, Iterable
from typing_extensions import Literal
from shlex import quote as shq
import abc
import dill
import urllib
import base64
import re
import os

# I need to restore the jobs types without pulling in a ton of dependencies. The dependencies screw
# with dill serialization.
from .exceptions import BatchException


_last_used_uid = 0


def generate_uid() -> int:
    global _last_used_uid
    _last_used_uid += 1
    return _last_used_uid


def encode_uid(uid: int) -> str:
    return base64.b64encode(uid.to_bytes(8, byteorder='big')).decode('utf-8')


def decode_uid(uid_str: str) -> int:
    try:
        return int.from_bytes(base64.b64decode(uid_str.encode('utf-8')), byteorder='big')
    except Exception as err:
        raise ValueError(f'bad uid: {uid_str}') from err


class Resource(abc.ABC):
    """
    Abstract class for resources.
    """

    LOCAL_PREFIX = '/__HAIL_1kh7ah_/L'
    REMOTE_PREFIX = '/__HAIL_1kh7ah_/R'
    # FIXME: should use base63 (no /)
    REGEXP = re.compile('/__HAIL_1kh7ah_/([LR])([+/0-9A-Za-z]+=*)')

    @abc.abstractmethod
    def _io_directory(self) -> str:
        pass

    def uid_remote_path_needle(self):
        return Resource.REMOTE_PREFIX + encode_uid(self.uid())

    def uid_local_path_needle(self):
        return Resource.LOCAL_PREFIX + encode_uid(self.uid())

    def local_prefix_from_uid(self) -> str:
        io = self._io_directory()
        assert io.endswith('/')
        return io[:-1] + self.uid_local_path_needle()

    # FIXME: do I need this
    @abc.abstractmethod
    def uses_remote_tmpdir(self) -> bool:
        pass

    def as_py(self) -> Union[str, Dict[str, str]]:
        return self.local_location()

    def is_remote(self) -> bool:
        return False

    def defining_resource(self) -> 'Resource':
        return self

    @abc.abstractmethod
    def uid(self) -> int:
        pass

    @abc.abstractmethod
    def humane_name(self) -> str:
        pass

    @abc.abstractmethod
    def source(self):
        pass

    @abc.abstractmethod
    def remote_location(self) -> str:
        pass

    @abc.abstractmethod
    def local_location(self) -> str:
        pass

    @abc.abstractmethod
    def group(self) -> Optional[Union['ResourceGroup', 'ExternalResourceGroup']]:
        pass

    @abc.abstractmethod
    def members(self) -> Optional[Iterable[Union['ResourceGroupMember', 'ExternalResourceGroupMember']]]:
        pass

    def __str__(self) -> str:
        return shq(self.local_location())

    def __repr__(self) -> str:
        return repr(shq(self.local_location()))


class RemoteResource(Resource):
    def __init__(self, resource: Resource, io_directory: str):
        self.resource = resource
        self._deserialized_remote_location: Optional[str] = None
        self.__io_directory = io_directory
        if isinstance(self.resource, PythonResult):
            raise ValueError('cannot remote resource a PythonResult')

    def __getitem__(self, name: str) -> Resource:
        assert isinstance(self.resource, (ResourceGroup, ExternalResourceGroup))
        return self.resource.members_by_name[name]

    def __getattr__(self, name: str) -> Resource:
        return self.__getitem__(name)

    def _io_directory(self) -> str:
        return self.__io_directory

    def uses_remote_tmpdir(self) -> bool:
        return False

    def uid(self) -> int:
        return self.resource.uid()

    def as_py(self) -> Union[str, Dict[str, str]]:
        if isinstance(self.resource, (ResourceGroup, ExternalResourceGroup)):
            return {name: m.remote_location() for name, m in self.resource.members_by_name.items()}
        assert not isinstance(self.resource, PythonResult)
        return self.resource.remote_location()

    def is_remote(self) -> bool:
        return True

    def defining_resource(self) -> 'Resource':
        return self.resource

    def humane_name(self) -> str:
        return self.resource.humane_name()

    def source(self):
        return self.resource.source()

    def remote_location(self) -> str:
        return self._deserialized_remote_location or self.resource.remote_location()

    def local_location(self) -> str:
        return self._deserialized_remote_location or self.resource.remote_location()

    def group(self) -> Optional[Union['ResourceGroup', 'ExternalResourceGroup']]:
        return self.resource.group()

    def members(self) -> Optional[Iterable[Union['ResourceGroupMember', 'ExternalResourceGroupMember']]]:
        return None


def remote(resource: Resource) -> RemoteResource:
    return RemoteResource(resource, resource._io_directory())


class ExternalResource(Resource):
    def __init__(self, remote_location: str, humane_name: str, io_directory: str):
        self.__io_directory = io_directory
        self._uid = generate_uid()
        self._remote_location = remote_location
        self._local_location = self.local_prefix_from_uid() + '/' + os.path.basename(self._remote_location)
        self._humane_name = humane_name

    def _io_directory(self) -> str:
        return self.__io_directory

    def uid(self) -> int:
        return self._uid

    def uses_remote_tmpdir(self) -> bool:
        return False

    def humane_name(self) -> str:
        return self._humane_name

    def source(self) -> Literal[None]:
        return None

    def remote_location(self) -> str:
        return self._remote_location

    def local_location(self) -> str:
        return self._local_location

    def group(self) -> Literal[None]:
        return None

    def members(self) -> Optional[Iterable[Union['ResourceGroupMember', 'ExternalResourceGroupMember']]]:
        return None


class ExternalResourceGroupMember(Resource):
    def __init__(self, group: 'ExternalResourceGroup', remote_location: str, suffix: str, humane_name: str, io_directory: str):
        self.__io_directory = io_directory
        self._group = group
        self._remote_location = remote_location
        self._suffix = suffix
        self._humane_name = humane_name

    def _io_directory(self) -> str:
        return self.__io_directory

    def uid(self) -> int:
        return self._group.uid()

    def uses_remote_tmpdir(self) -> bool:
        return False

    def humane_name(self) -> str:
        return self._humane_name + ' in ' + self._group.humane_name()

    def source(self):
        return self._group.source()

    def remote_location(self) -> str:
        return self._remote_location

    def local_location(self) -> str:
        return self._group.local_location() + '/' + self._suffix

    def group(self) -> Optional['ExternalResourceGroup']:
        return self._group

    def members(self) -> Optional[Iterable[Union['ResourceGroupMember', 'ExternalResourceGroupMember']]]:
        return None


class ExternalResourceGroup(Resource):
    def __init__(self, named_members: Dict[str, str], humane_name: str, io_directory: str):
        self.__io_directory = io_directory
        self._uid = generate_uid()
        parsed = [urllib.parse.urlparse(url) for url in named_members.values()]
        common_path_prefix = os.path.commonpath([url.path for url in parsed])
        n = len(common_path_prefix)
        suffices = [url.path[n:] for url in parsed]
        self._local_location = self.local_prefix_from_uid() + '/' + common_path_prefix
        self.members_by_name = {
            name: ExternalResourceGroupMember(self, remote_location, suffix, name, io_directory)
            for (name, remote_location), suffix in zip(named_members.items(), suffices)
        }
        self._humane_name = humane_name

    def _io_directory(self) -> str:
        return self.__io_directory

    def __getitem__(self, name: str) -> ExternalResourceGroupMember:
        return self.members_by_name[name]

    def __getattr__(self, name: str) -> ExternalResourceGroupMember:
        return self.__getitem__(name)

    def as_py(self) -> Dict[str, str]:
        return {name: m.local_location() for name, m in self.members_by_name.items()}

    def uid(self) -> int:
        return self._uid

    def uses_remote_tmpdir(self) -> bool:
        return False

    def humane_name(self) -> str:
        return self._humane_name

    def source(self) -> Literal[None]:
        return None

    def remote_location(self) -> str:
        raise ValueError('ExternalResourceGroup has no sensible remote location')

    def local_location(self) -> str:
        return self._local_location

    def group(self) -> Optional['ResourceGroup']:
        return None

    def members(self) -> Optional[Iterable[Union['ResourceGroupMember', 'ExternalResourceGroupMember']]]:
        return self.members_by_name.values()


class JobResource(Resource):
    def __init__(self, remote_dir: str, j, humane_name: str, extension: Optional[str], io_directory: str):
        self.__io_directory = io_directory
        self._uid = generate_uid()
        self._remote_location = remote_dir + self.uid_remote_path_needle()
        self._extension: Optional[str] = extension
        self._local_location: Optional[str] = None
        self._job = j
        self._humane_name = humane_name

    def _io_directory(self) -> str:
        return self.__io_directory

    def uid(self) -> int:
        return self._uid

    def uses_remote_tmpdir(self) -> bool:
        return True

    def humane_name(self) -> str:
        return self._humane_name

    def source(self):
        return self._job

    def remote_location(self) -> str:
        return self._remote_location

    def local_location(self) -> str:
        if self._local_location is None:
            self._local_location = self.local_prefix_from_uid()
            if self._extension is not None:
                self._local_location += self._extension
        return self._local_location

    def group(self) -> Literal[None]:
        return None

    def members(self) -> Optional[Iterable[Union['ResourceGroupMember', 'ExternalResourceGroupMember']]]:
        return None

    def set_extension(self, extension: str) -> 'JobResource':
        """
        Specify the file extension to use.

        Examples
        --------

        >>> b = Batch()
        >>> j = b.new_job()
        >>> j.command(f'echo "hello" > {j.ofile}')
        >>> j.ofile.set_extension('.txt')
        >>> b.run()

        Parameters
        ----------
        extension: :obj:`str`
            File extension to use.

        Returns
        -------
        :class:`.JobResourceFile`
            Same resource file with the extension specified
        """
        if self._extension is not None:
            raise BatchException("Resource already has a file extension added.")
        assert self._local_location is None
        self._extension = extension
        return self


class ResourceGroupMember(Resource):
    def __init__(self, group: 'ResourceGroup', suffix: str, humane_name: str, io_directory: str):
        self.__io_directory = io_directory
        self._group = group
        self._suffix = suffix
        self._humane_name = humane_name

    def _io_directory(self) -> str:
        return self.__io_directory

    def uid(self) -> int:
        return self._group.uid()

    def uses_remote_tmpdir(self) -> bool:
        return True

    def humane_name(self) -> str:
        return self._humane_name + ' in ' + self._group.humane_name()

    def source(self):
        return self._group.source()

    def remote_location(self) -> str:
        return self._group.remote_location() + '/' + self._suffix

    def local_location(self) -> str:
        return self._group.local_location() + '/' + self._suffix

    def group(self):
        return self._group

    def members(self) -> Optional[Iterable[Union['ResourceGroupMember', 'ExternalResourceGroupMember']]]:
        return None


class ResourceGroup(Resource):
    def __init__(self, remote_prefix: str, named_format_strings: Dict[str, str], j, humane_name: str, io_directory: str):
        self.__io_directory = io_directory
        self._uid = generate_uid()
        assert remote_prefix.endswith('/')
        self._remote_location = remote_prefix[:-1] + self.uid_remote_path_needle()
        self._local_location = self.local_prefix_from_uid() + '/' + os.path.basename(self._remote_location)
        self._job = j
        self.members_by_name = {
            name: ResourceGroupMember(self, format_string.format(root='root'), name, io_directory)
            for name, format_string in named_format_strings.items()
        }
        self._humane_name = humane_name

    def _io_directory(self) -> str:
        return self.__io_directory

    def __getitem__(self, name: str) -> ResourceGroupMember:
        return self.members_by_name[name]

    def __getattr__(self, name: str) -> ResourceGroupMember:
        return self.__getitem__(name)

    def as_py(self) -> Dict[str, str]:
        return {name: m.local_location() for name, m in self.members_by_name.items()}

    def uid(self) -> int:
        return self._uid

    def uses_remote_tmpdir(self) -> bool:
        return True

    def humane_name(self) -> str:
        return self._humane_name

    def source(self):
        return self._job

    def remote_location(self) -> str:
        return self._remote_location

    def local_location(self) -> str:
        return self._local_location

    def group(self) -> Optional['ResourceGroup']:
        return None

    def members(self) -> Optional[Iterable[Union['ResourceGroupMember', 'ExternalResourceGroupMember']]]:
        return self.members_by_name.values()

class PythonResult(Resource):
    def __init__(self, remote_dir: str, j, humane_name: str, io_directory: str):
        self.__io_directory = io_directory
        self._uid = generate_uid()
        self._remote_location = remote_dir + '/' + encode_uid(self._uid)
        self._extension: Optional[str] = None
        self._local_location: Optional[str] = None
        self._job = j
        self._humane_name = humane_name

        self._remote_dir = remote_dir
        self._as_json = None
        self._as_str = None
        self._as_repr = None

    def __getstate__(self):
        return self._local_location

    def __setstate__(self):
        raise ValueError('PythonResults should be deserialized by the special hail Unpickler')

    def _io_directory(self) -> str:
        return self.__io_directory

    def as_py(self) -> Dict[str, str]:
        raise ValueError('hmm')
        # FIXME: Hmm. how do we do this with RemoteResource
        return dill.load(open(self.local_location()))

    def uid(self) -> int:
        return self._uid

    def uses_remote_tmpdir(self) -> bool:
        return True

    def humane_name(self) -> str:
        return self._humane_name

    def source(self):
        return self._job

    def remote_location(self) -> str:
        return self._remote_location

    def local_location(self) -> str:
        if self._local_location is None:
            self._local_location = self.local_prefix_from_uid() + '/' + os.path.basename(self._remote_location)
            if self._extension is not None:
                self._local_location += self._extension
        return self._local_location

    def group(self) -> Literal[None]:
        return None

    def members(self) -> Optional[Iterable[Union['ResourceGroupMember', 'ExternalResourceGroupMember']]]:
        return None

    def as_json(self) -> JobResource:
        """
        Convert a Python result to a file with a JSON representation of the object.

        Examples
        --------

        .. code-block:: python

            def add(x, y):
                return {'result': x + y}


            b = Batch()
            j = b.new_python_job(name='add')
            result = j.call(add, 3, 2)
            b.write_output(result.as_json(), 'output/add.json')
            b.run()

        Returns
        -------
        :class:`.JobResourceFile`
            A new resource file where the contents are a Python object
            that has been converted to JSON.
        """
        if self._as_json is None:
            self._as_json = JobResource(self._remote_dir, self._job, self._humane_name + '-json', '.json', self.__io_directory)
        return self._as_json

    def as_str(self) -> Resource:
        """
        Convert a Python result to a file with the str representation of the object.

        Examples
        --------

        .. code-block:: python

            def add(x, y):
                return x + y


            b = Batch()
            j = b.new_python_job(name='add')
            result = j.call(add, 3, 2)
            b.write_output(result.as_str(), 'output/add.txt')
            b.run()

        Returns
        -------
        :class:`.Resource`
            A new resource file where the contents are the str representation
            of a Python object.
        """
        if self._as_str is None:
            self._as_str = JobResource(self._remote_dir, self._job, self._humane_name + '-str', '.txt', self.__io_directory)
        return self._as_str

    def as_repr(self) -> Resource:
        """
        Convert a Python result to a file with the repr representation of the object.

        Examples
        --------

        .. code-block:: python

            def add(x, y):
                return x + y


            b = Batch()
            j = b.new_python_job(name='add')
            result = j.call(add, 3, 2)
            b.write_output(result.as_repr(), 'output/add.txt')
            b.run()

        Returns
        -------
        :class:`.Resource`
            A new resource file where the contents are the repr representation
            of a Python object.
        """
        if self._as_repr is None:
            self._as_repr = JobResource(self._remote_dir, self._job, self._humane_name + '-repr', '.repr', self.__io_directory)
        return self._as_repr


ALL_RESOURCE_CLASSES = [
    RemoteResource,
    ExternalResourceGroupMember,
    ExternalResourceGroup,
    JobResource,
    ResourceGroupMember,
    ResourceGroup,
    PythonResult
]
