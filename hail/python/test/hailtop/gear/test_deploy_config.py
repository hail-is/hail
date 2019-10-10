import unittest
from hailtop.config.deploy_config import DeployConfig

class Test(unittest.TestCase):
    def test_deploy_external_default(self):
        deploy_config = DeployConfig('external', 'default', {'foo': 'bar'})

        self.assertEqual(deploy_config.location(), 'external')
        self.assertEqual(deploy_config.service_ns('quam'), 'default')
        self.assertEqual(deploy_config.service_ns('foo'), 'bar')
        self.assertEqual(deploy_config.scheme(), 'https')
        self.assertEqual(deploy_config.auth_session_cookie_name(), 'session')

        self.assertEqual(deploy_config.domain('quam'), 'quam.hail.is')
        self.assertEqual(deploy_config.base_path('quam'), '')
        self.assertEqual(deploy_config.base_url('quam'), 'https://quam.hail.is')
        self.assertEqual(deploy_config.url('quam', '/moo'), 'https://quam.hail.is/moo')
        self.assertEqual(deploy_config.external_url('quam', '/moo'), 'https://quam.hail.is/moo')

        self.assertEqual(deploy_config.base_path('foo'), '/bar/foo')
        self.assertEqual(deploy_config.base_url('foo'), 'https://internal.hail.is/bar/foo')
        self.assertEqual(deploy_config.url('foo', '/moo'), 'https://internal.hail.is/bar/foo/moo')
        self.assertEqual(deploy_config.external_url('foo', '/moo'), 'https://internal.hail.is/bar/foo/moo')

    def test_deploy_k8s_default(self):
        deploy_config = DeployConfig('k8s', 'default', {'foo': 'bar'})

        self.assertEqual(deploy_config.location(), 'k8s')
        self.assertEqual(deploy_config.service_ns('quam'), 'default')
        self.assertEqual(deploy_config.service_ns('foo'), 'bar')
        self.assertEqual(deploy_config.scheme(), 'http')
        self.assertEqual(deploy_config.auth_session_cookie_name(), 'session')

        self.assertEqual(deploy_config.domain('quam'), 'quam.default')
        self.assertEqual(deploy_config.base_path('quam'), '')
        self.assertEqual(deploy_config.base_url('quam'), 'http://quam.default')
        self.assertEqual(deploy_config.url('quam', '/moo'), 'http://quam.default/moo')
        self.assertEqual(deploy_config.external_url('quam', '/moo'), 'https://quam.hail.is/moo')

        self.assertEqual(deploy_config.base_path('foo'), '/bar/foo')
        self.assertEqual(deploy_config.base_url('foo'), 'http://foo.bar/bar/foo')
        self.assertEqual(deploy_config.url('foo', '/moo'), 'http://foo.bar/bar/foo/moo')
        self.assertEqual(deploy_config.external_url('foo', '/moo'), 'https://internal.hail.is/bar/foo/moo')
