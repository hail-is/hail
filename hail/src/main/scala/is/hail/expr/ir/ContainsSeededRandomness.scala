package is.hail.expr.ir

object ContainsSeededRandomness {
  /**
    * This analysis pass computes for each value IR a boolean which is true if
    * there is a seeded random function (e.g. ApplySeeded) as a transitive child
    * in the same scope.
    *
    * The scope is broken by valueIR nodes with relational children,
    * like TableAggregate, TableCount, etc.
    */
  def analyze(ir: BaseIR): Memo[Boolean] = {

    val m = Memo.empty[Boolean]
    def mark(node: IR, flag: Boolean): Unit = m.bind(node, flag)
    VisitIR.bottomUp(ir) {
      case _: TableIR =>
      case _: BlockMatrixIR =>
      case _: MatrixIR =>
      case x: IR if !Compilable(x) =>
        // catches all relational value IRs like TableAggregate/TableCount/etc
        mark(x, false)
      case x: ApplySeeded => mark(x, true)
      case x: IR => mark(x, x.children.exists(m.lookup))
    }

    m
  }
}
