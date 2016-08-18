package org.broadinstitute.hail.driver


object GroupTest extends SuperCommand {
  def name = "grouptest"

  def description = "Group-based tests"

  register(GroupTestSKATO)
}
