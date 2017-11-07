package is.hail.expr.ir

import is.hail.expr.BaseIR

object Copy {
  def apply(x: IR, newChildren: IndexedSeq[BaseIR]): BaseIR = {
    val children = newChildren.asInstanceOf[IndexedSeq[IR]]
    lazy val same = {
      assert(children.isEmpty)
      x
    }
    x match {
      case I32(_) => same
      case I64(_) => same
      case F32(_) => same
      case F64(_) => same
      case True() => same
      case False() => same
      case NA(_) => same
      case MapNA(name, _, _, typ) =>
        val IndexedSeq(value, body) = children
        MapNA(name, value, body, typ)
      case IsNA(value) =>
        val IndexedSeq(value) = children
        IsNA(value)
      case If(_, _, _, typ) =>
        val IndexedSeq(cond, cnsq, altr) = children
        If(cond, cnsq, altr, typ)
      case Let(name, _, _, typ) =>
        val IndexedSeq(value, body) = children
        Let(name, value, body, typ)
      case Ref(_, _) =>
        same
      case ApplyBinaryPrimOp(op, _, _, typ) =>
        val IndexedSeq(l, r) = children
        ApplyBinaryPrimOp(op, l, r, typ)
      case ApplyUnaryPrimOp(op, _, typ) =>
        val IndexedSeq(x) = children
        ApplyUnaryPrimOp(op, x, typ)
      case MakeArray(args, typ) =>
        assert(args.length == children.length)
        MakeArray(children.toArray, typ)
      case MakeArrayN(_, elementType) =>
        val IndexedSeq(len) = children
        MakeArrayN(len, elementType)
      case ArrayRef(_, _, typ) =>
        val IndexedSeq(a, i) = children
        ArrayRef(a, i, typ)
      case ArrayMissingnessRef(_, _) =>
        val IndexedSeq(a, i) = children
        ArrayMissingnessRef(a, i)
      case ArrayLen(_) =>
        val IndexedSeq(a) = children
        ArrayLen(a)
      case ArrayMap(_, name, _, elementTyp) =>
        val IndexedSeq(a, body) = children
        ArrayMap(a, name, body, elementTyp)
      case ArrayFold(_, _, accumName, valueName, _, typ) =>
        val IndexedSeq(a, zero, body) = children
        ArrayFold(a, zero, accumName, valueName, body, typ)
      case MakeStruct(fields) =>
        assert(fields.length == children.length)
        MakeStruct(fields.zip(children).map { case ((n, t, _), v) => (n, t, v) })
      case GetField(_, name, typ) =>
        val IndexedSeq(o) = children
        GetField(o, name, typ)
      case GetFieldMissingness(_, name) =>
        val IndexedSeq(o) = children
        GetFieldMissingness(o, name)
      case In(_, _) =>
        same
      case InMissingness(_) =>
        same
      case Die(message) =>
        same
    }
  }
}
