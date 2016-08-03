node {
  stage 'Checkout'
  checkout scm
  
  stage 'Build'
  def gradleHome = tool 'gradle'
  
  sh """
#!/bin/bash

set -o pipefail

${gradleHome}/bin/gradle --stacktrace --info -Dscan clean test | tee test.log
"""
  
  // update build description with build scan
  def output = readFile('test.log')
  def match = (output =~ /https:\/\/gradle\.com\/s\/[0-9a-zA-Z]*/)
  if (match.size() == 1) {
    def url = match[0]
    currentBuild.description = "Build scan: <a href=\"${url}\">${url}</a>"
  }
}
