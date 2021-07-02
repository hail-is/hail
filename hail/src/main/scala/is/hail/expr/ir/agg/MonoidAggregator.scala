package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.functions.UtilFunctions
import is.hail.expr.ir.{coerce => _, _}
import is.hail.types.physical.{PCode, PType, typeToTypeInfo}
import is.hail.types.virtual._

import scala.language.existentials
import scala.reflect.ClassTag

trait StagedMonoidSpec {
  val typ: Type

  def neutral: Option[Code[_]]

  def apply(v1: Code[_], v2: Code[_]): Code[_]
}

class MonoidAggregator(monoid: StagedMonoidSpec) extends StagedAggregator {
  type State = PrimitiveRVAState
  val typ: PType = PType.canonical(monoid.typ, required = monoid.neutral.isDefined)

  def resultType: PType = typ

  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type](monoid.typ)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    val stateRequired = state.vtypes.head.r.required
    val ev = state.fields(0)
    if (!ev.pt.required) {
        assert(!stateRequired, s"monoid=$monoid, stateRequired=$stateRequired")
        cb.assign(ev, EmitCode.missing(cb.emb, ev.pt))
    } else {
        assert(stateRequired, s"monoid=$monoid, stateRequired=$stateRequired")
        cb.assign(ev, EmitCode.present(cb.emb, PCode(ev.pt, monoid.neutral.get)))
    }
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt) = seq
    val ev = state.fields(0)
    val update = cb.memoizeField(elt, "monoid_elt")
    combine(cb, ev, update)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    val ev1 = state.fields(0)
    val ev2 = other.fields(0)
    combine(cb, ev1, ev2)
  }

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    state.fields(0).toI(cb).consume(cb,
      ifMissing(cb),
      sc => pt.storeAtAddress(cb, addr, region, sc, deepCopy = true))
  }

  private def combine(
    cb: EmitCodeBuilder,
    ev1: EmitSettable,
    ev2: EmitValue
  ): Unit = {
    cb.ifx(ev1.m,
      cb.ifx(!ev2.m, cb.assign(ev1, ev2)),
      cb.ifx(!ev2.m,
        cb.assign(ev1, EmitCode.present(cb.emb, PCode(ev1.pt, monoid(ev1.v, ev2.v))))))
  }
}

class ComparisonMonoid(val typ: Type, val functionName: String) extends StagedMonoidSpec {

  def neutral: Option[Code[_]] = None

  private def cmp[T](v1: Code[T], v2: Code[T])(implicit tct: ClassTag[T]): Code[T] =
    Code.invokeStatic2[Math, T, T, T](functionName, v1, v2)

  private def nancmp[T](v1: Code[T], v2: Code[T])(implicit tct: ClassTag[T]): Code[T] =
    Code.invokeScalaObject2[T, T, T](UtilFunctions.getClass, "nan" + functionName, v1, v2)

  def apply(v1: Code[_], v2: Code[_]): Code[_] = typ match {
    case TInt32 => cmp[Int](coerce(v1), coerce(v2))
    case TInt64 => cmp[Long](coerce(v1), coerce(v2))
    case TFloat32 => nancmp[Float](coerce(v1), coerce(v2))
    case TFloat64 => nancmp[Double](coerce(v1), coerce(v2))
    case _ => throw new UnsupportedOperationException(s"can't $functionName over type $typ")
  }
}

class SumMonoid(val typ: Type) extends StagedMonoidSpec {

  def neutral: Option[Code[_]] = Some(typ match {
    case TInt64 => const(0L)
    case TFloat64 => const(0.0d)
    case _ => throw new UnsupportedOperationException(s"can't sum over type $typ")
  })

  def apply(v1: Code[_], v2: Code[_]): Code[_] = typ match {
    case TInt64 => coerce[Long](v1) + coerce[Long](v2)
    case TFloat64 => coerce[Double](v1) + coerce[Double](v2)
    case _ => throw new UnsupportedOperationException(s"can't sum over type $typ")
  }
}

class ProductMonoid(val typ: Type) extends StagedMonoidSpec {

  def neutral: Option[Code[_]] = Some(typ match {
    case TInt64 => const(1L)
    case TFloat64 => const(1.0d)
    case _ => throw new UnsupportedOperationException(s"can't product over type $typ")
  })

  def apply(v1: Code[_], v2: Code[_]): Code[_] = typ match {
    case TInt64 => coerce[Long](v1) * coerce[Long](v2)
    case TFloat64 => coerce[Double](v1) * coerce[Double](v2)
    case _ => throw new UnsupportedOperationException(s"can't product over type $typ")
  }
}

class MinAggregator(typ: Type) extends MonoidAggregator(new ComparisonMonoid(typ, "min"))

class MaxAggregator(typ: Type) extends MonoidAggregator(new ComparisonMonoid(typ, "max"))

class SumAggregator(typ: Type) extends MonoidAggregator(new SumMonoid(typ))

class ProductAggregator(typ: Type) extends MonoidAggregator(new ProductMonoid(typ))
