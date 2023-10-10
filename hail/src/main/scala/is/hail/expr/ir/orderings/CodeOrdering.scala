package is.hail.expr.ir.orderings

import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.virtual._
import is.hail.utils.FastSeq

object CodeOrdering {

  sealed trait Op {
    type ReturnType
    val rtti: TypeInfo[ReturnType]
    val missingEqual: Boolean
  }

  final case class Compare(missingEqual: Boolean = true) extends Op {
    type ReturnType = Int
    val rtti = typeInfo[Int]
  }

  sealed trait BooleanOp extends Op {
    type ReturnType = Boolean
    val rtti = typeInfo[Boolean]
  }

  final case class Equiv(missingEqual: Boolean = true) extends BooleanOp

  final case class Lt(missingEqual: Boolean = true) extends BooleanOp

  final case class Lteq(missingEqual: Boolean = true) extends BooleanOp

  final case class Gt(missingEqual: Boolean = true) extends BooleanOp

  final case class Gteq(missingEqual: Boolean = true) extends BooleanOp

  final case class Neq(missingEqual: Boolean = true) extends BooleanOp

  final case class StructLt(missingEqual: Boolean = true) extends BooleanOp

  final case class StructLteq(missingEqual: Boolean = true) extends BooleanOp

  final case class StructGt(missingEqual: Boolean = true) extends BooleanOp

  final case class StructCompare(missingEqual: Boolean = true) extends BooleanOp

  type F[R] = (EmitCodeBuilder, EmitValue, EmitValue) => Value[R]

  def makeOrdering(t1: SType, t2: SType, ecb: EmitClassBuilder[_]): CodeOrdering = {
    val canCompare = (t1.virtualType, t2.virtualType) match {
      case (t1: TStruct, t2: TStruct) => t1.isIsomorphicTo(t2)
      case (t1, t2) if t1 == t2 => t1 == t2
    }
    if (!canCompare) {
      throw new RuntimeException(s"ordering: type mismatch:\n  left: ${ t1.virtualType }\n right: ${ t2.virtualType }")
    }

    t1.virtualType match {
      case TInt32 => Int32Ordering.make(t1.asInstanceOf[SInt32.type], t2.asInstanceOf[SInt32.type], ecb)
      case TInt64 => Int64Ordering.make(t1.asInstanceOf[SInt64.type], t2.asInstanceOf[SInt64.type], ecb)
      case TFloat32 => Float32Ordering.make(t1.asInstanceOf[SFloat32.type], t2.asInstanceOf[SFloat32.type], ecb)
      case TFloat64 => Float64Ordering.make(t1.asInstanceOf[SFloat64.type], t2.asInstanceOf[SFloat64.type], ecb)
      case TBoolean => BooleanOrdering.make(t1.asInstanceOf[SBoolean.type], t2.asInstanceOf[SBoolean.type], ecb)
      case TCall => CallOrdering.make(t1.asInstanceOf[SCall], t2.asInstanceOf[SCall], ecb)
      case TString => StringOrdering.make(t1.asInstanceOf[SString], t2.asInstanceOf[SString], ecb)
      case TBinary => BinaryOrdering.make(t1.asInstanceOf[SBinary], t2.asInstanceOf[SBinary], ecb)
      case _: TBaseStruct => StructOrdering.make(t1.asInstanceOf[SBaseStruct], t2.asInstanceOf[SBaseStruct], ecb)
      case _: TLocus => LocusOrdering.make(t1.asInstanceOf[SLocus], t2.asInstanceOf[SLocus], ecb)
      case _: TInterval => IntervalOrdering.make(t1.asInstanceOf[SInterval], t2.asInstanceOf[SInterval], ecb)
      case _: TSet | _: TArray | _: TDict =>
        IterableOrdering.make(t1.asInstanceOf[SContainer], t2.asInstanceOf[SContainer], ecb)
    }
  }
}

abstract class CodeOrdering {
  outer =>

  val type1: SType
  val type2: SType

  def reversed: Boolean = false

  final def checkedSCode[T](cb: EmitCodeBuilder, arg1: SValue, arg2: SValue, context: String,
    f: (EmitCodeBuilder, SValue, SValue) => Value[T])(implicit ti: TypeInfo[T]): Value[T] = {
    if (arg1.st != type1)
      throw new RuntimeException(s"CodeOrdering: $context: type mismatch (left)\n  generated: $type1\n  argument:  ${ arg1.st }")
    if (arg2.st != type2)
      throw new RuntimeException(s"CodeOrdering: $context: type mismatch (right)\n  generated: $type2\n  argument:  ${ arg2.st }")

    val cacheKey = ("ordering", reversed, type1, type2, context)
    val mb = cb.emb.ecb.getOrGenEmitMethod(s"ord_$context", cacheKey,
      FastSeq(arg1.st.paramType, arg2.st.paramType), ti) { mb =>

      mb.emitWithBuilder[T] { cb =>
        val arg1 = mb.getSCodeParam(1)
        val arg2 = mb.getSCodeParam(2)
        f(cb, arg1, arg2)
      }
    }
    cb.memoize(cb.invokeCode[T](mb, arg1, arg2))
  }

  final def checkedEmitCode[T](cb: EmitCodeBuilder, arg1: EmitValue, arg2: EmitValue, missingEqual: Boolean, context: String,
    f: (EmitCodeBuilder, EmitValue, EmitValue, Boolean) => Value[T])(implicit ti: TypeInfo[T]): Value[T] = {
    if (arg1.st != type1)
      throw new RuntimeException(s"CodeOrdering: $context: type mismatch (left)\n  generated: $type1\n  argument:  ${ arg1.st }")
    if (arg2.st != type2)
      throw new RuntimeException(s"CodeOrdering: $context: type mismatch (right)\n  generated: $type2\n  argument:  ${ arg2.st }")

    val cacheKey = ("ordering", reversed, arg1.emitType, arg2.emitType, context, missingEqual)
    val mb = cb.emb.ecb.getOrGenEmitMethod(s"ord_$context", cacheKey,
      FastSeq(arg1.emitParamType, arg2.emitParamType), ti) { mb =>

      mb.emitWithBuilder[T] { cb =>
        val arg1 = mb.getEmitParam(cb, 1)
        val arg2 = mb.getEmitParam(cb, 2)
        f(cb, arg1, arg2, missingEqual)
      }
    }
    cb.memoize(cb.invokeCode[T](mb, arg1, arg2))
  }


  final def compareNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Int] = {
    checkedSCode(cb, x, y, "compareNonnull", _compareNonnull)
  }

  final def ltNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
    checkedSCode(cb, x, y, "ltNonnull", _ltNonnull)
  }

  final def lteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
    checkedSCode(cb, x, y, "lteqNonnull", _lteqNonnull)
  }

  final def gtNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
    checkedSCode(cb, x, y, "gtNonnull", _gtNonnull)
  }

  final def gteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
    checkedSCode(cb, x, y, "gteqNonnull", _gteqNonnull)
  }

  final def equivNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
    checkedSCode(cb, x, y, "equivNonnull", _equivNonnull)
  }

  final def lt(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "lt", _lt)
  }

  final def lteq(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "lteq", _lteq)
  }

  final def gt(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "gt", _gt)
  }

  final def gteq(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "gteq", _gteq)
  }

  final def equiv(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "equiv", _equiv)
  }

  final def compare(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Int] = {
    checkedEmitCode(cb, x, y, missingEqual, "compare", _compare)
  }

  def _compareNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Int]

  def _ltNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean]

  def _lteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean]

  def _gtNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean]

  def _gteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean]

  def _equivNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean]

  def _compare(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean = true): Value[Int] = {
    val cmp = cb.newLocal[Int]("cmp")
    cb.ifx(x.m,
      cb.ifx(y.m, cb.assign(cmp, if (missingEqual) 0 else -1), cb.assign(cmp, 1)),
      cb.ifx(y.m, cb.assign(cmp, -1), cb.assign(cmp, compareNonnull(cb, x.v, y.v))))
    cmp
  }

  def _lt(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    val ret = cb.newLocal[Boolean]("lt")
    if (missingEqual) {
      cb.ifx(x.m,
        cb.assign(ret, false),
        cb.ifx(y.m,
          cb.assign(ret, true),
          cb.assign(ret, ltNonnull(cb, x.v, y.v))))
    } else {
      cb.ifx(y.m,
        cb.assign(ret, true),
        cb.ifx(x.m,
          cb.assign(ret, false),
          cb.assign(ret, ltNonnull(cb, x.v, y.v))))
    }
    ret
  }

  def _lteq(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    val ret = cb.newLocal[Boolean]("lteq")
    cb.ifx(y.m,
      cb.assign(ret, true),
      cb.ifx(x.m,
        cb.assign(ret, false),
        cb.assign(ret, lteqNonnull(cb, x.v, y.v))))
    ret
  }

  def _gt(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    val ret = cb.newLocal[Boolean]("gt")
    cb.ifx(y.m,
      cb.assign(ret, false),
      cb.ifx(x.m,
        cb.assign(ret, true),
        cb.assign(ret, gtNonnull(cb, x.v, y.v))))
    ret
  }

  def _gteq(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    val ret = cb.newLocal[Boolean]("gteq")
    if (missingEqual) {
      cb.ifx(x.m,
        cb.assign(ret, true),
        cb.ifx(y.m,
          cb.assign(ret, false),
          cb.assign(ret, gteqNonnull(cb, x.v, y.v))))
    } else {
      cb.ifx(y.m,
        cb.assign(ret, false),
        cb.ifx(x.m,
          cb.assign(ret, true),
          cb.assign(ret, gteqNonnull(cb, x.v, y.v))))
    }
    ret
  }

  def _equiv(cb: EmitCodeBuilder, x: EmitValue, y: EmitValue, missingEqual: Boolean): Value[Boolean] = {
    val ret = cb.newLocal[Boolean]("eq")
    if (missingEqual) {
      cb.ifx(x.m && y.m,
        cb.assign(ret, true),
        cb.ifx(!x.m && !y.m,
          cb.assign(ret, equivNonnull(cb, x.v, y.v)),
          cb.assign(ret, false)))
    } else {
      cb.ifx(!x.m && !y.m, cb.assign(ret, equivNonnull(cb, x.v, y.v)), cb.assign(ret, false))
    }
    ret
  }

  // reverses the sense of the non-null comparison only
  def reverse: CodeOrdering = new CodeOrdering() {
    override def reverse: CodeOrdering = outer

    override val type1: SType = outer.type1
    override val type2: SType = outer.type2

    override def reversed: Boolean = true

    override def _compareNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Int] =
      outer._compareNonnull(cb, y, x)

    override def _ltNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
      outer._ltNonnull(cb, y, x)

    override def _lteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
      outer._lteqNonnull(cb, y, x)

    override def _gtNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
      outer._gtNonnull(cb, y, x)

    override def _gteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
      outer._gteqNonnull(cb, y, x)

    override def _equivNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
      outer._equivNonnull(cb, y, x)
  }
}

abstract class CodeOrderingCompareConsistentWithOthers extends CodeOrdering {
  override def _ltNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
    cb.memoize(compareNonnull(cb, x, y) < 0)

  override def _lteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
    cb.memoize(compareNonnull(cb, x, y) <= 0)

  override def _gtNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
    cb.memoize(compareNonnull(cb, x, y) > 0)

  override def _gteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
    cb.memoize(compareNonnull(cb, x, y) >= 0)

  override def _equivNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] =
    cb.memoize(compareNonnull(cb, x, y).ceq(0))
}
