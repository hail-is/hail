package org.broadinstitute.hail.driver

object AnnotateAlleles extends SuperCommand {
  def name = "annotatealleles"

  def description = "Annotate alleles in current dataset"

  override def run(state: State, args: Array[String]): State = {
    val (c, newArgs) = lookup(args)
    c.run(state, newArgs)
  }

  register(AnnotateAllelesExpr)
}