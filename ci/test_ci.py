import requests
import unittest

CI_URL='http://localhost:5000'

def post(endpoint, json=None, status_code=None, json_response=True):
    r = requests.post(CI_URL + endpoint, json = json)
    if status_code and r.status_code != status_code:
        raise ValueError(
            'bad status_code from pull_request: {status_code}\n{message}'.format(
                endpoint=endpoint, status_code=r.status_code, message=r.text))
    if json_response:
        return r.json()
    else:
        return r.text

def get(endpoint, status_code=None, json_response=True):
    r = requests.get(CI_URL + endpoint)
    if status_code and r.status_code != status_code:
        raise ValueError(
            'bad status_code from {endpoint}: {status_code}\n{message}'.format(
                endpoint=endpoint, status_code=r.status_code, message=r.text))
    if json_response:
        return r.json()
    else:
        return r.text

synchronize_pr_2_hash_1 = {
  "action": "synchronize",
  "number": 2,
  "pull_request": {
    "number": 2,
    "head": {
      "label": "danking:bar",
      "ref": "bar",
      "sha": "4affff998e12377a36436d9cfbc965834865e1f7",
      "repo": {
        "clone_url": "https://github.com/danking/docker-build-test.git"
      }
    },
    "base": {
      "label": "danking:master",
      "ref": "master",
      "sha": "da332ac1b3d1a5f2a670b0ceb964caeb8f1e36fe",
      "repo": {
        "clone_url": "https://github.com/danking/docker-build-test.git"
      }
    }
  }
}

synchronize_pr_2_hash_2 = {
  "action": "synchronize",
  "number": 2,
  "pull_request": {
    "number": 2,
    "head": {
      "label": "danking:bar",
      "ref": "bar",
      "sha": "9cfbc965834865e1f74affff998e12377a36436d",
      "repo": {
        "clone_url": "https://github.com/danking/docker-build-test.git"
      }
    },
    "base": {
      "label": "danking:master",
      "ref": "master",
      "sha": "da332ac1b3d1a5f2a670b0ceb964caeb8f1e36fe",
      "repo": {
        "clone_url": "https://github.com/danking/docker-build-test.git"
      }
    }
  }
}

class TestStringMethods(unittest.TestCase):

    def test_pull_request_adds_pr_to_status(self):
        status = get('/status', status_code=200)
        self.assertEqual(status, {}) # start in empty state
        self.assertEqual('', post('/pull_request', json = synchronize_pr_2_hash_1, status_code=200, json_response=False))
        status = get('/status', status_code=200)
        self.assertTrue(len(status) == 1)
        self.assertIn('2', status)
        pr = status['2']
        self.assertIn('attributes', pr)
        attributes = pr['attributes']
        self.assertIn('pr_number', attributes)
        self.assertEqual(attributes['pr_number'], '2')
        self.assertIn('source_branch', attributes)
        self.assertEqual(attributes['source_branch'], 'bar')
        self.assertIn('source_hash', attributes)
        self.assertEqual(attributes['source_hash'], '4affff998e12377a36436d9cfbc965834865e1f7')
        self.assertIn('target_branch', attributes)
        self.assertEqual(attributes['target_branch'], 'master')
        # do it again, new hash, should replace old pr
        # self.assertEqual('', post('/pull_request', json = synchronize_pr_2_hash_2, status_code=200, json_response=False))
        # status = get('/status', status_code=200)
        # self.assertTrue(len(status) == 1)
        # self.assertIn('2', status)
        # pr = status['2']
        # self.assertIn('attributes', pr)
        # attributes = pr['attributes']
        # self.assertIn('pr_number', attributes)
        # self.assertEqual(attributes['pr_number'], '2')
        # self.assertIn('source_branch', attributes)
        # self.assertEqual(attributes['source_branch'], 'bar')
        # self.assertIn('source_hash', attributes)
        # self.assertEqual(attributes['source_hash'], '9cfbc965834865e1f74affff998e12377a36436d')
        # self.assertIn('target_branch', attributes)
        # self.assertEqual(attributes['target_branch'], 'master')

if __name__ == '__main__':
    unittest.main()

