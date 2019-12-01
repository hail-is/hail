package is.hail.expr.types.physical

trait PCollection {
  def nMissingBytes: Int
  def loadLength: Int
  def lengthHeaderBytes: Long
}
