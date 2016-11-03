package org.broadinstitute.hail.driver

object AnnotateKeyTable extends SuperCommand {
  def name = "annotatekeytable"

  def description = "Annotate key tables"

  register(AnnotateKeyTableExpr)
}
