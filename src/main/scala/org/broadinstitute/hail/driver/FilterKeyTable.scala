package org.broadinstitute.hail.driver

object FilterKeyTable extends SuperCommand {
  def name = "filterkeytable"

  def description = "Filter key tables"

  register(FilterKeyTableExpr)
}
