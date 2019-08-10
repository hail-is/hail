package is.hail.backend.local

import is.hail.expr.ir._
import is.hail.cxx
import is.hail.expr.types.virtual._
import is.hail.utils._
import org.json4s.jackson.JsonMethods

case class LocalBinding(name: String, value: IR)

case class LocalTableIR(
  globals: List[LocalBinding],
  key: IndexedSeq[String],
  rows: IR) {

  def close(x: IR, globals: List[LocalBinding]): IR = globals match {
    case LocalBinding(g, gv) :: rest =>
      close(Let(g, gv, x), rest)
    case Nil => x
  }

  def close(x: IR): IR = close(x, globals)

  def closedGlobals(): IR = (globals: @unchecked) match {
    case LocalBinding(g, gv) :: rest =>
      close(gv, rest)
  }

  def closedRows(): IR = close(rows, globals)
}

object LowerTableIR {
  def apply(ir0: IR, timer: Option[ExecutionTimer], optimize: Boolean = true): IR = {

    def opt(context: String, ir: IR): IR =
      Optimize(ir, noisy = true, canGenerateLiterals = true,
        Some(timer.map(t => s"${ t.context }: $context")
          .getOrElse(context)))

    def time(context: String, ir: (String) => IR): IR =
      timer.map(t => t.time(ir(context), context))
        .getOrElse(ir(context))

    var ir = ir0

    ir = ir.unwrap
    if (optimize) { ir = time( "first pass", opt(_, ir)) }

    ir = time("lowering MatrixIR", _ => LowerMatrixIR(ir))

    if (optimize) { ir = time("after MatrixIR lowering", opt(_, ir)) }

    ir = time("lowering TableIR", _ => LowerTableIR.lower(ir))

    if (optimize) { ir = time("after TableIR lowering", opt(_, ir)) }
    ir
  }

  def lower(ir: BaseIR): BaseIR = ir match {
    case ir: IR => lower(ir)
    case tir: TableIR => lower(tir)
    case mir: MatrixIR => lower(mir)
  }

  def lower(ir: IR): IR = ir match {
    case TableGetGlobals(child) =>
      lower(child).closedGlobals()
    case TableCollect(child) =>
      val p = lower(child)
      val LocalBinding(g, gv) = p.globals.head
      p.close(
        MakeStruct(Seq(
          "rows" -> p.rows,
          "global" -> Ref(g, gv.typ))))
    case TableCount(child) =>
      ArrayLen(lower(child).closedRows())

    case _ =>
      ir.copy(ir.children.map(lower))
  }

  def lower(tir: TableIR): LocalTableIR = tir match {
    case TableRange(n, nPartitions) =>
      val g = genUID()
      val i = genUID()
      LocalTableIR(
        List(LocalBinding(g, MakeStruct(FastIndexedSeq()))),
        Array("idx"),
        ArrayMap(
          ArrayRange(I32(0), I32(n), I32(1)),
          i,
          MakeStruct(Seq("idx" -> Ref(i, TInt32())))))

    case TableParallelize(rowsAndGlobal, nPartitions) =>
      val rowsAndGlobalType = rowsAndGlobal.typ.asInstanceOf[TStruct]

      val rg = genUID()
      val g = genUID()
      LocalTableIR(
        List(
          LocalBinding(g, GetField(Ref(rg, rowsAndGlobalType), "global")),
          LocalBinding(rg, rowsAndGlobal)),
        FastIndexedSeq(),
        GetField(Ref(rg, rowsAndGlobalType), "rows"))

    case TableMapRows(child, newRow) =>
      val p = lower(child)
      val LocalBinding(g, gv) = p.globals.head
      p.copy(
        rows = ArrayMap(p.rows, "row",
          Let("global", Ref(g, gv.typ), newRow)))

    case TableMapGlobals(child, newGlobals) =>
      val p = lower(child)
      val newG = genUID()
      val LocalBinding(g, gv) = p.globals.head
      p.copy(
        globals = LocalBinding(newG,
          Let("global", Ref(g, gv.typ),
            newGlobals)) :: p.globals)

    case TableFilter(child, pred) =>
      val row = genUID()
      val p = lower(child)
      val LocalBinding(g, gv) = p.globals.head
      p.copy(
        rows = ArrayFilter(p.rows, "row",
          Let("global", Ref(g, gv.typ),
            pred)))

    case TableUnion(children) =>
      val ps = children.map(lower)
      val p0 = ps(0)
      val a = genUID()
      p0.copy(
        rows = ArrayFlatMap(
          MakeArray(p0.rows +: ps.tail.map(_.closedRows()), TArray(p0.rows.typ)),
          a,
          Ref(a, p0.rows.typ)))

    case TableRepartition(child, n, strategy) =>
      lower(child)
    case TableHead(child, n) =>
      val rows = genUID()
      val i = genUID()
      val p = lower(child)
      p.copy(
        rows = Let(rows, p.rows,
          If(ApplyComparisonOp(LTEQ(TInt32()),
            ArrayLen(Ref(rows, p.rows.typ)),
            I32(n.toInt)),
            Ref(rows, p.rows.typ),
            ArrayMap(
              ArrayRange(I32(0), I32(n.toInt), I32(1)),
              i,
              ArrayRef(Ref(rows, p.rows.typ), Ref(i, TInt32()))))))
    case _ => throw new cxx.CXXUnsupportedOperation(tir.toString)
  }

  def lower(mir: MatrixIR): MatrixIR = ???
}
