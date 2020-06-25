package is.hail.lir

class Locals(
  val locals: Array[Local]
) extends IndexedSeq[Local] {
  private val localIdx = locals.iterator.zipWithIndex.toMap

  def nLocals: Int = locals.length

  def length: Int = locals.length

  def apply(i: Int): Local = locals(i)

  def index(l: Local): Int = {
    l match {
      case p: Parameter =>
        p.i
      case _ =>
        localIdx(l)
    }
  }
}
