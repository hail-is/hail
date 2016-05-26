package org.broadinstitute.hail.driver


object GroupTest extends SuperCommand {
  def name = "grouptest"

  def description = "Calculate p-values for previously calculated groups"

  register(GroupTestLinReg)
  register(GroupTestFET)

}
