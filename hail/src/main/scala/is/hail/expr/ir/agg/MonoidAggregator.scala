package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.UtilFunctions
import is.hail.expr.ir._
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.{PType, typeToTypeInfo}
import is.hail.types.virtual._

import scala.language.existentials
import scala.reflect.ClassTag

trait StagedMonoidSpec {
  val typ: Type

  def neutral: Option[Value[_]]

  def apply(cb: EmitCodeBuilder, v1: Value[_], v2: Value[_]): Value[_]
}

class MonoidAggregator(monoid: StagedMonoidSpec) extends StagedAggregator {
  type State = PrimitiveRVAState
  val sType = SType.canonical(monoid.typ)
  def resultEmitType = EmitType(sType, monoid.neutral.isDefined)

  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type](monoid.typ)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    val stateRequired = state.vtypes.head.r.required
    val ev = state.fields(0)
    if (!ev.required) {
        assert(!stateRequired, s"monoid=$monoid, stateRequired=$stateRequired")
        cb.assign(ev, EmitCode.missing(cb.emb, ev.st))
    } else {
        assert(stateRequired, s"monoid=$monoid, stateRequired=$stateRequired")
        cb.assign(ev, EmitCode.present(cb.emb, primitive(ev.st.virtualType, monoid.neutral.get)))
    }
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt) = seq
    val ev = state.fields(0)
    val update = cb.memoizeField(elt, "monoid_elt")
    combine(cb, ev, update)
  }

  protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, state: PrimitiveRVAState, other: PrimitiveRVAState): Unit = {
    val ev1 = state.fields(0)
    val ev2 = other.fields(0)
    combine(cb, ev1, ev2)
  }

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    state.fields(0).toI(cb)
  }

  private def combine(
    cb: EmitCodeBuilder,
    ev1: EmitSettable,
    ev2: EmitValue
  ): Unit = {
    val combined = primitive(monoid.typ, monoid(cb, ev1.pv.asPrimitive.primitiveValue, ev2.pv.asPrimitive.primitiveValue))
    cb.if_(ev1.m,
      cb.if_(!ev2.m, cb.assign(ev1, ev2)),
      cb.if_(!ev2.m,
        cb.assign(ev1, EmitCode.present(cb.emb, combined))))
  }
}

class ComparisonMonoid(val typ: Type, val functionName: String) extends StagedMonoidSpec {

  def neutral: Option[Value[_]] = None

  private def cmp[T](v1: Code[T], v2: Code[T])(implicit tct: ClassTag[T]): Code[T] =
    Code.invokeStatic2[Math, T, T, T](functionName, v1, v2)

  private def nancmp[T](v1: Code[T], v2: Code[T])(implicit tct: ClassTag[T]): Code[T] =
    Code.invokeScalaObject2[T, T, T](UtilFunctions.getClass, "nan" + functionName, v1, v2)

  def apply(cb: EmitCodeBuilder, v1: Value[_], v2: Value[_]): Value[_] = typ match {
    case TInt32 => cb.memoize(cmp[Int](coerce(v1), coerce(v2)))
    case TInt64 => cb.memoize(cmp[Long](coerce(v1), coerce(v2)))
    case TFloat32 => cb.memoize(nancmp[Float](coerce(v1), coerce(v2)))
    case TFloat64 => cb.memoize(nancmp[Double](coerce(v1), coerce(v2)))
    case _ => throw new UnsupportedOperationException(s"can't $functionName over type $typ")
  }
}

class SumMonoid(val typ: Type) extends StagedMonoidSpec {

  def neutral: Option[Value[_]] = Some(typ match {
    case TInt64 => const(0L)
    case TFloat64 => const(0.0d)
    case _ => throw new UnsupportedOperationException(s"can't sum over type $typ")
  })

  def apply(cb: EmitCodeBuilder, v1: Value[_], v2: Value[_]): Value[_] = typ match {
    case TInt64 => cb.memoize(coerce[Long](v1) + coerce[Long](v2))
    case TFloat64 => cb.memoize(coerce[Double](v1) + coerce[Double](v2))
    case _ => throw new UnsupportedOperationException(s"can't sum over type $typ")
  }
}

class ProductMonoid(val typ: Type) extends StagedMonoidSpec {

  def neutral: Option[Value[_]] = Some(typ match {
    case TInt64 => const(1L)
    case TFloat64 => const(1.0d)
    case _ => throw new UnsupportedOperationException(s"can't product over type $typ")
  })

  def apply(cb: EmitCodeBuilder, v1: Value[_], v2: Value[_]): Value[_] = typ match {
    case TInt64 => cb.memoize(coerce[Long](v1) * coerce[Long](v2))
    case TFloat64 => cb.memoize(coerce[Double](v1) * coerce[Double](v2))
    case _ => throw new UnsupportedOperationException(s"can't product over type $typ")
  }
}

class MinAggregator(typ: Type) extends MonoidAggregator(new ComparisonMonoid(typ, "min"))

class MaxAggregator(typ: Type) extends MonoidAggregator(new ComparisonMonoid(typ, "max"))

class SumAggregator(typ: Type) extends MonoidAggregator(new SumMonoid(typ))

class ProductAggregator(typ: Type) extends MonoidAggregator(new ProductMonoid(typ))
