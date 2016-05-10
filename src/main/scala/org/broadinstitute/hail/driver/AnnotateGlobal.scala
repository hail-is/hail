package org.broadinstitute.hail.driver

object AnnotateGlobal extends SuperCommand {
  def name = "annotateglobal"

  def description = "Annotate the global table"

  override def run(state: State, args: Array[String]): State = {
    val (c, newArgs) = lookup(args)
    c.run(state, newArgs)
  }

  register(AnnotateGlobalExpr)
  register(AnnotateGlobalList)
  register(AnnotateGlobalTable)
}
