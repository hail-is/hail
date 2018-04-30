package is.hail.expr.ir

import is.hail.utils._
import is.hail.expr.{BaseIR, FilterColsIR, FilterRowsIR, MatrixRead, TableFilter, TableRead, TableUnion}

object Simplify {
  private[this] def isStrict(x: IR): Boolean = {
    x match {
      case _: Apply |
        _: ApplyUnaryPrimOp |
        _: ApplyBinaryPrimOp |
        _: ArrayRange |
        _: ArrayRef |
        _: ArrayLen |
        _: GetField |
        _: GetTupleElement => true
      case _ => false
    }
  }

  private[this] def isDefinitelyDefined(x: IR): Boolean = {
    x match {
      case _: MakeArray |
           _: MakeStruct |
           _: MakeTuple |
           _: IsNA |
           _: I32 | _: I64 | _: F32 | _: F64 | True() | False() => true
      case _ => false
    }
  }

  def apply(ir: BaseIR): BaseIR = {
    RewriteBottomUp(ir, matchErrorToNone {
      // optimize IR

      // propagate NA
      case x: IR if isStrict(x) && Children(x).exists(_.isInstanceOf[NA]) =>
        NA(x.typ)

      case x@If(NA(_), _, _) => NA(x.typ)

      case x@ArrayMap(NA(_), _, _) => NA(x.typ)

      case x@ArrayFlatMap(NA(_), _, _) => NA(x.typ)

      case x@ArrayFilter(NA(_), _, _) => NA(x.typ)

      case x@ArrayFold(NA(_), _, _, _, _) => NA(x.typ)

      case IsNA(NA(_)) => True()

      case IsNA(x) if isDefinitelyDefined(x) => False()

      case Let(n1, v, Ref(n2, _)) if n1 == n2 => v

      case If(True(), x, _) => x
      case If(False(), _, x) => x

      case If(c, cnsq, altr) if cnsq == altr =>
        If(IsNA(c), NA(cnsq.typ), cnsq)

      case Cast(x, t) if x.typ == t => x

      case ArrayLen(MakeArray(args, _)) => I32(args.length)

      case ArrayRef(MakeArray(args, _), I32(i)) => args(i)

      case ArrayFilter(a, _, True()) => a

      case AggFilter(a, _, True()) => a
        
      case Let(n, v, b) if !Mentions(b, n) => b

      case AggFilter(AggMap(a, n1, b), n2, p) if !Mentions(p, n2) =>
        AggMap(AggFilter(a, n2, p), n1, b)

      case AggMap(AggMap(a, n1, b1), n2, b2) =>
        AggMap(a, n1, Let(n2, b1, b2))

      case ArrayMap(ArrayMap(a, n1, b1), n2, b2) =>
        ArrayMap(a, n1, Let(n2, b1, b2))

      case GetField(MakeStruct(fields), name) =>
        val (_, x) = fields.find { case (n, _) => n == name }.get
        x

      case GetField(InsertFields(old, fields), name) =>
        fields.find { case (n, _) => n == name } match {
          case Some((_, x)) => x
          case None => GetField(old, name)
        }

      case InsertFields(InsertFields(base, fields1), fields2) =>
        val fields1Set = fields1.map(_._1).toSet
        val fields2Map = fields2.toMap

        val finalFields = fields1.map { case (name, fieldIR) => name -> fields2Map.getOrElse(name, fieldIR) } ++
          fields2.filter { case (name, _) => !fields1Set.contains(name) }
        InsertFields(base, finalFields)

      case InsertFields(MakeStruct(fields1), fields2) =>
        val fields1Set = fields1.map(_._1).toSet
        val fields2Map = fields2.toMap

        val finalFields = fields1.map { case (name, fieldIR) => name -> fields2Map.getOrElse(name, fieldIR) } ++
          fields2.filter { case (name, _) => !fields1Set.contains(name) }
        MakeStruct(finalFields)

      case GetTupleElement(MakeTuple(xs), idx) => xs(idx)

      // optimize TableIR
      case TableFilter(t, True()) => t

      case TableFilter(TableRead(path, spec, _), False() | NA(_)) =>
        TableRead(path, spec, dropRows = true)

      case TableFilter(TableFilter(t, p1), p2) =>
        TableFilter(t,
          ApplySpecial("&&", Array(p1, p2)))

        // flatten unions
      case TableUnion(children) if children.exists(_.isInstanceOf[TableUnion]) =>
        TableUnion(children.flatMap {
          case u: TableUnion => u.children
          case c => Some(c)
        })

      // optimize MatrixIR

      // Equivalent rewrites for the new Filter{Cols,Rows}IR
      case FilterRowsIR(MatrixRead(path, spec, dropCols,  _), False() | NA(_)) =>
        MatrixRead(path, spec, dropCols, dropRows = true)

      case FilterColsIR(MatrixRead(path, spec, _, dropRows), False() | NA(_)) =>
        MatrixRead(path, spec, dropCols = true, dropRows)

      // Keep all rows/cols = do nothing
      case FilterRowsIR(m, True()) => m

      case FilterColsIR(m, True()) => m
    })
  }
}
