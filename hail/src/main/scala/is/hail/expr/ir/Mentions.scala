package is.hail.expr.ir

object Mentions {
  def apply(x: IR, v: Sym): Boolean = CountMentions(x, v) > 0
}

object CountMentions {
  def apply(x: IR, v: Sym): Int = {
    x match {
      case Ref(n, _) =>
        if (v == n)
          1
        else
          0
      case _ =>
        Children(x).iterator.zipWithIndex.map {
          case (c: IR, i) =>
            if (Binds(x, v, i))
              0
            else
              CountMentions(c, v)
          case _ => 0
        }
          .sum
    }
  }
}
