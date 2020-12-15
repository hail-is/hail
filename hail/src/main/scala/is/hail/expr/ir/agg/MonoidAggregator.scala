package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{coerce => _, _}
import is.hail.expr.ir.functions.UtilFunctions
import is.hail.types.physical.{PFloat32, PFloat64, PInt32, PInt64, PType, typeToTypeInfo}
import is.hail.types.virtual.{TFloat32, TFloat64, TInt32, TInt64, Type}

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
    val (mOpt, v, _) = state.fields(0)
    cb += { (mOpt, monoid.neutral) match {
      case (Some(m), _)  => m.store(true)
      case (_, Some(v0)) => v.storeAny(v0)
    }}
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt) = seq
    val (mOpt, v, _) = state.fields(0)
    val eltm = state.kb.genFieldThisRef[Boolean]()
    val eltv = state.kb.genFieldThisRef()(typeToTypeInfo(typ))
    cb.assign(eltm, elt.m)
    cb.ifx(eltm,
      {},
      cb.assign(eltv, elt.value))
    combine(cb, mOpt, v, Some(eltm), eltv)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    val (m1, v1, _) = state.fields(0)
    val (m2, v2, _) = other.fields(0)
    combine(cb, m1, v1, m2.map(_.load), v2.load)
  }

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    val (mOpt, v, _) = state.fields(0)
    cb += { mOpt match {
      case None => srvb.addIRIntermediate(typ)(v)
      case Some(m) =>
        m.mux(
          srvb.setMissing(),
          srvb.addIRIntermediate(typ)(v))
    }}
  }

  private def combine(
    cb: EmitCodeBuilder,
    m1Opt: Option[Settable[Boolean]],
    v1: Settable[_],
    m2Opt: Option[Code[Boolean]],
    v2: Code[_]
  ): Unit = {
    val ti = typeToTypeInfo(typ)
    ti match {
      case ti: TypeInfo[t] =>
        (m1Opt, m2Opt) match {
          case (None, None) =>
            cb.assignAny(v1, monoid(v1, v2))
          case (None, Some(m2)) =>
            // only update if the element is not missing
            cb.ifx(m2, cb.assignAny(v1, monoid(v1, v2)))
          case (Some(m1), None) =>
            val v2var = cb.newLocalAny("mon_agg_combine_v2", v2)(ti)
            cb.ifx(m1,
            {
              cb.assign(m1, false)
              cb.assignAny(v1, v2var)
            },
            {
             cb.assignAny(v1, monoid(v1, v2))
            })
          case (Some(m1), Some(m2)) =>
            val m2var = cb.newLocal[Boolean]("mon_agg_combine_m2", m2)
            val v2var = cb.newLocalAny("mon_agg_combine_v2", v2)(ti)
            cb.ifx(m1,
              {
                cb.assign(m1, m2var)
                cb.assignAny(v1, v2var)
              },
              {
                cb.ifx(m2var,
                  {},
                  cb.assignAny(v1, monoid(v1, v2var)))
              })
        }
    }
  }
}

class ComparisonMonoid(val typ: Type, val functionName: String) extends StagedMonoidSpec {

  def neutral: Option[Code[_]] = None

  private def cmp[T](v1: Code[T], v2: Code[T])(implicit tct: ClassTag[T]): Code[T] =
    Code.invokeStatic2[Math,T,T,T](functionName, v1, v2)

  private def nancmp[T](v1: Code[T], v2: Code[T])(implicit tct: ClassTag[T]): Code[T] =
    Code.invokeScalaObject2[T,T,T](UtilFunctions.getClass, "nan" + functionName, v1, v2)

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
