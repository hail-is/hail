from ci.build import BuildConfiguration, Code

_YAML = """
steps:
  - kind: runImage
    name: merge_code
    image: ubuntu:22.04
    script: "true"
  - kind: runImage
    name: default_ns
    image: ubuntu:22.04
    script: "true"
    dependsOn:
      - merge_code
  - kind: runImage
    name: ci_utils_image
    image: ubuntu:22.04
    script: "true"
    dependsOn:
      - merge_code
  - kind: runImage
    name: build_artifact
    image: ubuntu:22.04
    script: "true"
    scopes:
      - test
      - dev
    dependsOn:
      - merge_code
  - kind: runImage
    name: forced_step
    image: ubuntu:22.04
    script: "true"
    scopes:
      - deploy
      - dev
    forceIfLabeled:
      - run-dataproc-tests
    dependsOn:
      - ci_utils_image
      - default_ns
      - build_artifact
  - kind: runImage
    name: unrelated_deploy_step
    image: ubuntu:22.04
    script: "true"
    scopes:
      - deploy
  - kind: runImage
    name: downstream_of_forced_step
    image: ubuntu:22.04
    script: "true"
    scopes:
      - deploy
    dependsOn:
      - forced_step
  - kind: runImage
    name: downstream_of_pulled_in_ancestor
    image: ubuntu:22.04
    script: "true"
    scopes:
      - deploy
    dependsOn:
      - ci_utils_image
"""


class _FakeCode(Code):
    def short_str(self):
        return 'fake-code'

    def config(self):
        return {}

    def repo_dir(self):
        return '/repo'

    def checkout_script(self):
        return ''


class _FakeJob:
    def __init__(self, name):
        self.name = name


class _FakeJobGroup:
    def __init__(self, batch, attributes):
        self._batch = batch
        self.attributes = attributes

    def create_job(self, image, *, attributes, **kwargs):  # pylint: disable=unused-argument
        self._batch.built_step_names.append(attributes['name'])
        return _FakeJob(attributes['name'])


class _FakeBatch:
    def __init__(self):
        self.attributes = {'token': 'test-token'}
        self.built_step_names = []
        self.job_groups_created = []  # one attributes dict per create_job_group call

    def create_job(self, image, *, attributes, **kwargs):  # pylint: disable=unused-argument
        self.built_step_names.append(attributes['name'])
        return _FakeJob(attributes['name'])

    def create_job_group(self, *, attributes, **kwargs):  # pylint: disable=unused-argument
        self.job_groups_created.append(attributes)
        return _FakeJobGroup(self, attributes)


def _configuration(pr_labels=frozenset()):
    return BuildConfiguration(_FakeCode(), _YAML, scope='test', requested_step_names=[], pr_labels=pr_labels)


_ALL_STEP_NAMES = {
    'merge_code',
    'default_ns',
    'ci_utils_image',
    'build_artifact',
    'forced_step',
    'unrelated_deploy_step',
    'downstream_of_forced_step',
    'downstream_of_pulled_in_ancestor',
}
_OUT_OF_SCOPE_NON_FORCED_STEPS = {
    'unrelated_deploy_step',
    'downstream_of_forced_step',
    'downstream_of_pulled_in_ancestor',
}


def test_unforced_out_of_scope_step_is_not_selected():
    config = _configuration()
    names = {step.name for step in config.steps}
    assert 'forced_step' not in names
    assert _OUT_OF_SCOPE_NON_FORCED_STEPS <= (_ALL_STEP_NAMES - names)


def test_forced_step_pulls_in_its_out_of_scope_ancestors():
    config = _configuration(pr_labels=frozenset({'run-dataproc-tests'}))
    names = {step.name for step in config.steps}

    assert {'forced_step', 'ci_utils_image', 'default_ns', 'build_artifact'} <= names
    assert _OUT_OF_SCOPE_NON_FORCED_STEPS.isdisjoint(names)


def test_forced_step_ancestors_are_actually_built():
    config = _configuration(pr_labels=frozenset({'run-dataproc-tests'}))
    batch = _FakeBatch()

    config.build(batch, _FakeCode(), 'test')

    assert {'merge_code', 'ci_utils_image', 'default_ns', 'build_artifact', 'forced_step'} <= set(
        batch.built_step_names
    )
    assert _OUT_OF_SCOPE_NON_FORCED_STEPS.isdisjoint(batch.built_step_names)


_SPLIT_YAML = """
steps:
  - kind: runImage
    name: sharded_step
    image: ubuntu:22.04
    script: "true"
    numSplits: 3
"""


def test_split_step_shards_are_created_in_a_job_group():
    config = BuildConfiguration(
        _FakeCode(), _SPLIT_YAML, scope='test', requested_step_names=['sharded_step'], pr_labels=frozenset()
    )
    batch = _FakeBatch()

    config.build(batch, _FakeCode(), 'test')

    assert batch.built_step_names == ['sharded_step_0', 'sharded_step_1', 'sharded_step_2']
    assert batch.job_groups_created == [{'name': 'sharded_step'}]


def test_unsplit_step_is_not_wrapped_in_a_job_group():
    config = _configuration(pr_labels=frozenset({'run-dataproc-tests'}))
    batch = _FakeBatch()

    config.build(batch, _FakeCode(), 'test')

    assert 'merge_code' in batch.built_step_names
    assert not batch.job_groups_created
