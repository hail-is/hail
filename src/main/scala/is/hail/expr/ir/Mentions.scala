package is.hail.expr.ir

object Mentions {
  def apply(x: IR, v: String): Boolean = CountMentions(x, v) > 0
}

object CountMentions {
  def apply(x: IR, v: String): Int = {
    x match {
      case Ref(n, _) =>
        if (v == n)
          1
        else
          0
      case _ =>
        if (Binds(x).contains(v))
          0
        else
          Children(x).iterator.map {
            case c: IR => CountMentions(c, v)
            case _ => 0
          }
            .sum
    }
  }
}
