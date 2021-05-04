package is.hail.expr.ir.orderings

import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.types.physical._
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.utils.FastIndexedSeq

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

  type F[R] = (EmitCodeBuilder, EmitCode, EmitCode) => Code[R]

  def makeOrdering(t1: SType, t2: SType, ecb: EmitClassBuilder[_]): CodeOrdering = {
    val canCompare = (t1.virtualType, t2.virtualType) match {
      case (t1: TStruct, t2: TStruct) => t1.isIsomorphicTo(t2)
      case (t1, t2) if t1 == t2 => t1 == t2
    }
    if (!canCompare) {
      throw new RuntimeException(s"ordering: type mismatch:\n  left: ${ t1.virtualType }\n right: ${ t2.virtualType }")
    }

    t1.virtualType match {
      case TInt32 => Int32Ordering.make(t1.asInstanceOf[SInt32], t2.asInstanceOf[SInt32], ecb)
      case TInt64 => Int64Ordering.make(t1.asInstanceOf[SInt64], t2.asInstanceOf[SInt64], ecb)
      case TFloat32 => Float32Ordering.make(t1.asInstanceOf[SFloat32], t2.asInstanceOf[SFloat32], ecb)
      case TFloat64 => Float64Ordering.make(t1.asInstanceOf[SFloat64], t2.asInstanceOf[SFloat64], ecb)
      case TBoolean => BooleanOrdering.make(t1.asInstanceOf[SBoolean], t2.asInstanceOf[SBoolean], ecb)
      case TCall => CallOrdering.make(t1.asInstanceOf[SCall], t2.asInstanceOf[SCall], ecb)
      case TString => StringOrdering.make(t1.asInstanceOf[SString], t2.asInstanceOf[SString], ecb)
      case TBinary => BinaryOrdering.make(t1.asInstanceOf[SBinary], t2.asInstanceOf[SBinary], ecb)
      case _: TBaseStruct => StructOrdering.make(t1.asInstanceOf[SBaseStruct], t2.asInstanceOf[SBaseStruct], ecb)
      case _: TShuffle => ShuffleOrdering.make(t1.asInstanceOf[SShuffle], t2.asInstanceOf[SShuffle], ecb)
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

  final def checkedPCode[T](cb: EmitCodeBuilder, arg1: PCode, arg2: PCode, context: String,
    f: (EmitCodeBuilder, PCode, PCode) => Code[T])(implicit ti: TypeInfo[T]): Code[T] = {
    if (!arg1.st.equalsExceptTopLevelRequiredness(type1))
      throw new RuntimeException(s"CodeOrdering: $context: type mismatch (left)\n  generated: $type1\n  argument:  ${ arg1.st }")
    if (!arg2.st.equalsExceptTopLevelRequiredness(type2))
      throw new RuntimeException(s"CodeOrdering: $context: type mismatch (right)\n  generated: $type2\n  argument:  ${ arg2.st }")

    val cacheKey = ("ordering", reversed, type1, type2, context)
    val mb = cb.emb.ecb.getOrGenEmitMethod(s"ord_$context", cacheKey,
      FastIndexedSeq(arg1.st.paramType, arg2.st.paramType), ti) { mb =>

      mb.emitWithBuilder[T] { cb =>
        val arg1 = mb.getPCodeParam(1)
        val arg2 = mb.getPCodeParam(2)
        f(cb, arg1, arg2)
      }
    }
    cb.invokeCode[T](mb, arg1, arg2)
  }

  final def checkedEmitCode[T](cb: EmitCodeBuilder, arg1: EmitCode, arg2: EmitCode, missingEqual: Boolean, context: String,
    f: (EmitCodeBuilder, EmitCode, EmitCode, Boolean) => Code[T])(implicit ti: TypeInfo[T]): Code[T] = {
    if (!arg1.st.equalsExceptTopLevelRequiredness(type1))
      throw new RuntimeException(s"CodeOrdering: $context: type mismatch (left)\n  generated: $type1\n  argument:  ${ arg1.st }")
    if (!arg2.st.equalsExceptTopLevelRequiredness(type2))
      throw new RuntimeException(s"CodeOrdering: $context: type mismatch (right)\n  generated: $type2\n  argument:  ${ arg2.st }")

    val cacheKey = ("ordering", reversed, type1, type2, context, missingEqual)
    val mb = cb.emb.ecb.getOrGenEmitMethod(s"ord_$context", cacheKey,
      FastIndexedSeq(arg1.emitParamType, arg2.emitParamType), ti) { mb =>

      mb.emitWithBuilder[T] { cb =>
        val arg1 = mb.getEmitParam(1, null) // can't contain streams
        val arg2 = mb.getEmitParam(2, null) // can't contain streams
        f(cb, arg1, arg2, missingEqual)
      }
    }
    cb.invokeCode[T](mb, arg1, arg2)
  }


  final def compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = {
    checkedPCode(cb, x, y, "compareNonnull", _compareNonnull)
  }

  final def ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
    checkedPCode(cb, x, y, "ltNonnull", _ltNonnull)
  }

  final def lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
    checkedPCode(cb, x, y, "lteqNonnull", _lteqNonnull)
  }

  final def gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
    checkedPCode(cb, x, y, "gtNonnull", _gtNonnull)
  }

  final def gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
    checkedPCode(cb, x, y, "gteqNonnull", _gteqNonnull)
  }

  final def equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
    checkedPCode(cb, x, y, "equivNonnull", _equivNonnull)
  }

  final def lt(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "lt", _lt)
  }

  final def lteq(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "lteq", _lteq)
  }

  final def gt(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "gt", _gt)
  }

  final def gteq(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "gteq", _gteq)
  }

  final def equiv(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    checkedEmitCode(cb, x, y, missingEqual, "equiv", _equiv)
  }

  final def compare(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Int] = {
    checkedEmitCode(cb, x, y, missingEqual, "compare", _compare)
  }

  def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int]

  def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def _compare(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean = true): Code[Int] = {
    cb += x.setup
    cb += y.setup
    val xm = cb.newLocal("cord_compare_xm", x.m)
    val ym = cb.newLocal("cord_compare_ym", y.m)
    val cmp = cb.newLocal[Int]("cmp")
    cb.ifx(xm,
      cb.ifx(ym, cb.assign(cmp, if (missingEqual) 0 else -1), cb.assign(cmp, 1)),
      cb.ifx(ym, cb.assign(cmp, -1), cb.assign(cmp, compareNonnull(cb, x.pv, y.pv))))
    cmp
  }

  def _lt(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("lt")
    cb += x.setup
    cb += y.setup
    if (missingEqual) {
      cb.ifx(x.m,
        cb.assign(ret, false),
        cb.ifx(y.m,
          cb.assign(ret, true),
          cb.assign(ret, ltNonnull(cb, x.pv, y.pv))))
    } else {
      cb.ifx(y.m,
        cb.assign(ret, true),
        cb.ifx(x.m,
          cb.assign(ret, false),
          cb.assign(ret, ltNonnull(cb, x.pv, y.pv))))
    }
    ret
  }

  def _lteq(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("lteq")
    cb += x.setup
    cb += y.setup
    cb.ifx(y.m,
      cb.assign(ret, true),
      cb.ifx(x.m,
        cb.assign(ret, false),
        cb.assign(ret, lteqNonnull(cb, x.pv, y.pv))))
    ret
  }

  def _gt(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("gt")
    cb += x.setup
    cb += y.setup
    cb.ifx(y.m,
      cb.assign(ret, false),
      cb.ifx(x.m,
        cb.assign(ret, true),
        cb.assign(ret, gtNonnull(cb, x.pv, y.pv))))
    ret
  }

  def _gteq(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("gteq")
    cb += x.setup
    cb += y.setup
    if (missingEqual) {
      cb.ifx(x.m,
        cb.assign(ret, true),
        cb.ifx(y.m,
          cb.assign(ret, false),
          cb.assign(ret, gteqNonnull(cb, x.pv, y.pv))))
    } else {
      cb.ifx(y.m,
        cb.assign(ret, false),
        cb.ifx(x.m,
          cb.assign(ret, true),
          cb.assign(ret, gteqNonnull(cb, x.pv, y.pv))))
    }
    ret
  }

  def _equiv(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("eq")
    cb += x.setup
    cb += y.setup
    if (missingEqual) {
      val xm = cb.newLocal("cord_equiv_xm", x.m)
      val ym = cb.newLocal("cord_equiv_ym", y.m)
      cb.ifx(xm && ym,
        cb.assign(ret, true),
        cb.ifx(!xm && !ym,
          cb.assign(ret, equivNonnull(cb, x.pv, y.pv)),
          cb.assign(ret, false)))
    } else {
      cb.ifx(!x.m && !y.m, cb.assign(ret, equivNonnull(cb, x.pv, y.pv)), cb.assign(ret, false))
    }
    ret
  }

  // reverses the sense of the non-null comparison only
  def reverse: CodeOrdering = new CodeOrdering() {
    override def reverse: CodeOrdering = outer

    val type1: SType = outer.type1
    val type2: SType = outer.type2

    override def reversed: Boolean = true

    override def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = outer._compareNonnull(cb, y, x)

    override def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = outer._ltNonnull(cb, y, x)

    override def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = outer._lteqNonnull(cb, y, x)

    override def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = outer._gtNonnull(cb, y, x)

    override def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = outer._gteqNonnull(cb, y, x)

    override def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = outer._equivNonnull(cb, y, x)
  }
}

abstract class CodeOrderingCompareConsistentWithOthers extends CodeOrdering {
  def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y) < 0

  def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y) <= 0

  def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y) > 0

  def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y) >= 0

  def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y).ceq(0)
}
