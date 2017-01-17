package is.hail.driver

object AnnotateGlobal extends SuperCommand {
  def name = "annotateglobal"

  def description = "Annotate the global table"

  override def run(state: State, args: Array[String]): State = {
    val (c, newArgs) = lookup(args)
    c.run(state, newArgs)
  }

  register(AnnotateGlobalExprByVariant)
  register(AnnotateGlobalExprBySample)
  register(AnnotateGlobalList)
  register(AnnotateGlobalTable)
}
