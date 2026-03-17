import unittest

from hailtop.config.deploy_config import DeployConfig


class Test(unittest.TestCase):
    def test_deploy_external_default(self):
        deploy_config = DeployConfig('external', 'default', 'organization.tld', None)

        self.assertEqual(deploy_config.location(), 'external')
        self.assertEqual(deploy_config.default_namespace(), 'default')
        self.assertEqual(deploy_config.scheme(), 'https')
        self.assertEqual(deploy_config.auth_session_cookie_name(), 'session')

        self.assertEqual(deploy_config.domain('quam'), 'quam.organization.tld')
        self.assertEqual(deploy_config.base_path('quam'), '')
        self.assertEqual(deploy_config.base_url('quam'), 'https://quam.organization.tld')
        self.assertEqual(deploy_config.url('quam', '/moo'), 'https://quam.organization.tld/moo')
        self.assertEqual(deploy_config.external_url('quam', '/moo'), 'https://quam.organization.tld/moo')

    def test_deploy_external_bar(self):
        deploy_config = DeployConfig('external', 'bar', 'internal.organization.tld', '/bar')

        self.assertEqual(deploy_config.location(), 'external')
        self.assertEqual(deploy_config.default_namespace(), 'bar')
        self.assertEqual(deploy_config.scheme(), 'https')
        self.assertEqual(deploy_config.auth_session_cookie_name(), 'sesh')

        self.assertEqual(deploy_config.base_path('foo'), '/bar/foo')
        self.assertEqual(deploy_config.base_url('foo'), 'https://internal.organization.tld/bar/foo')
        self.assertEqual(deploy_config.url('foo', '/moo'), 'https://internal.organization.tld/bar/foo/moo')
        self.assertEqual(deploy_config.external_url('foo', '/moo'), 'https://internal.organization.tld/bar/foo/moo')

    def test_deploy_k8s_default(self):
        deploy_config = DeployConfig('k8s', 'default', 'organization.tld', None)

        self.assertEqual(deploy_config.location(), 'k8s')
        self.assertEqual(deploy_config.default_namespace(), 'default')
        self.assertEqual(deploy_config.scheme(), 'https')
        self.assertEqual(deploy_config.auth_session_cookie_name(), 'session')

        self.assertEqual(deploy_config.domain('quam'), 'quam.default')
        self.assertEqual(deploy_config.base_path('quam'), '')
        self.assertEqual(deploy_config.base_url('quam'), 'https://quam.default')
        self.assertEqual(deploy_config.url('quam', '/moo'), 'https://quam.default/moo')
        self.assertEqual(deploy_config.external_url('quam', '/moo'), 'https://quam.organization.tld/moo')

    def test_deploy_k8s_bar(self):
        deploy_config = DeployConfig('k8s', 'bar', 'internal.organization.tld', '/bar')

        self.assertEqual(deploy_config.location(), 'k8s')
        self.assertEqual(deploy_config.default_namespace(), 'bar')
        self.assertEqual(deploy_config.scheme(), 'https')
        self.assertEqual(deploy_config.auth_session_cookie_name(), 'sesh')

        self.assertEqual(deploy_config.base_path('foo'), '/bar/foo')
        self.assertEqual(deploy_config.base_url('foo'), 'https://foo.bar/bar/foo')
        self.assertEqual(deploy_config.url('foo', '/moo'), 'https://foo.bar/bar/foo/moo')
        self.assertEqual(deploy_config.external_url('foo', '/moo'), 'https://internal.organization.tld/bar/foo/moo')

    def test_deploy_batch_job_default(self):
        deploy_config = DeployConfig('gce', 'default', 'organization.tld', None)

        self.assertEqual(deploy_config.location(), 'gce')
        self.assertEqual(deploy_config.default_namespace(), 'default')
        self.assertEqual(deploy_config.scheme(), 'http')
        self.assertEqual(deploy_config.auth_session_cookie_name(), 'session')

        self.assertEqual(deploy_config.domain('quam'), 'quam.hail')
        self.assertEqual(deploy_config.base_path('quam'), '')
        self.assertEqual(deploy_config.base_url('quam'), 'http://quam.hail')
        self.assertEqual(deploy_config.url('quam', '/moo'), 'http://quam.hail/moo')
        self.assertEqual(deploy_config.external_url('quam', '/moo'), 'https://quam.organization.tld/moo')

    def test_deploy_batch_job_bar(self):
        deploy_config = DeployConfig('gce', 'bar', 'internal.organization.tld', '/bar')

        self.assertEqual(deploy_config.location(), 'gce')
        self.assertEqual(deploy_config.default_namespace(), 'bar')
        self.assertEqual(deploy_config.scheme(), 'http')
        self.assertEqual(deploy_config.auth_session_cookie_name(), 'sesh')

        self.assertEqual(deploy_config.base_path('foo'), '/bar/foo')
        self.assertEqual(deploy_config.base_url('foo'), 'http://internal.hail/bar/foo')
        self.assertEqual(deploy_config.url('foo', '/moo'), 'http://internal.hail/bar/foo/moo')
        self.assertEqual(deploy_config.external_url('foo', '/moo'), 'https://internal.organization.tld/bar/foo/moo')

    def test_with_default_namespace_from_default(self):
        config = DeployConfig('external', 'default', 'organization.tld', None)
        result = config.with_default_namespace('foo')
        self.assertEqual(result.default_namespace(), 'foo')
        self.assertEqual(result.url('svc', '/path'), 'https://internal.organization.tld/foo/svc/path')

    def test_with_default_namespace_to_default(self):
        config = DeployConfig('external', 'foo', 'internal.organization.tld', '/foo')
        result = config.with_default_namespace('default')
        self.assertEqual(result.default_namespace(), 'default')
        self.assertEqual(result.url('svc', '/path'), 'https://svc.organization.tld/path')

    def test_with_default_namespace_switch_non_default(self):
        config = DeployConfig('external', 'foo', 'internal.organization.tld', '/foo')
        result = config.with_default_namespace('bar')
        self.assertEqual(result.default_namespace(), 'bar')
        self.assertEqual(result.url('svc', '/path'), 'https://internal.organization.tld/bar/svc/path')

    def test_namespace_domain_and_base_path_default(self):
        domain, base_path = DeployConfig._namespace_domain_and_base_path('organization.tld', 'default')
        self.assertEqual(domain, 'organization.tld')
        self.assertIsNone(base_path)

    def test_namespace_domain_and_base_path_non_default(self):
        domain, base_path = DeployConfig._namespace_domain_and_base_path('organization.tld', 'foo')
        self.assertEqual(domain, 'internal.organization.tld')
        self.assertEqual(base_path, '/foo')

    def test_namespace_domain_and_base_path_switch_namespaces(self):
        # Handles 'internal.' prefix itself so callers pass domain as-is
        domain, base_path = DeployConfig._namespace_domain_and_base_path('internal.organization.tld', 'bar')
        self.assertEqual(domain, 'internal.organization.tld')
        self.assertEqual(base_path, '/bar')
