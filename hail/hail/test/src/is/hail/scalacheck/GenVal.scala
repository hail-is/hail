package is.hail.scalacheck

import is.hail.annotations.An
import is.hail.backend.ExecuteContext
import is.hail.types.physical.{PBaseStruct, PContainer, PDict, PType}
import is.hail.types.virtual._
import is.hail.utils.Interval

import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen
import org.scalacheck.Gen.{const, containerOf, mapOf, sequence, zip}

private[scalacheck] trait GenVal {

  private[this] def build(ctx: ExecuteContext, t: Type, elem: => Gen[An]): Gen[An] =
    t match {
      case _: TArray => containerOf[IndexedSeq, An](elem)
      case TBoolean => arbitrary[Boolean]
      case TBinary => containerOf[Array, Byte](arbitrary[Byte])
      case TCall => genCall
      case _: TDict => elem match { case kvs: Gen[(An, An)] => mapOf(kvs) }
      case TFloat32 => arbitrary[Float]
      case TFloat64 => arbitrary[Double]
      case TInt32 => arbitrary[Int]
      case TInt64 => arbitrary[Long]
      case TLocus(rgName) => genLocus(ctx.references(rgName))
      case _: TSet => containerOf[Set, An](elem)
      case TString => arbitrary[String]
      case _: TBaseStruct =>
        elem match { case ts: Gen[Seq[Gen[An]]] => ts flatMap sequence[Seq[An], An] }
      case TInterval(point) =>
        val ord = point.mkOrdering(ctx.stateManager)
        for {
          (a, b, s, e) <- zip(elem, elem, arbitrary[Boolean], arbitrary[Boolean])
          if ord.compare(a, b) != 0 || (s && e)
        } yield Interval(ord.min(a, b), ord.max(a, b), s, e)
      case TVoid => const(null)
    }

  def genWith(genE: (ExecuteContext, Type) => Gen[An])(ctx: ExecuteContext, typ: Type): Gen[An] =
    build(
      ctx,
      typ,
      typ match {
        case TDict(k, v) => zip(genNonMissing(ctx, k), genE(ctx, v))
        case t: TContainer => genE(ctx, t.elementType)
        case t: TBaseStruct => const(t.types.map(genE(ctx, _)))
      },
    )

  def genNonMissing(ctx: ExecuteContext, typ: Type): Gen[An] =
    genWith(genNonMissing)(ctx, typ)

  def genNullable(ctx: ExecuteContext, typ: Type): Gen[An] =
    nullable {
      genWith(genNullable)(ctx, typ)
    }

  def genVal(ctx: ExecuteContext, pt: PType): Gen[An] = {

    def wrap(g: Gen[An]): Gen[An] =
      if (pt.required) g else nullable(g)

    wrap {
      build(
        ctx,
        pt.virtualType,
        wrap {
          pt match {
            case p: PDict =>
              zip(genNonMissing(ctx, p.keyType.virtualType), genVal(ctx, p.valueType))
            case t: PContainer => genVal(ctx, t.elementType)
            case t: PBaseStruct => const(t.types.map(genVal(ctx, _)))
          }
        },
      )
    }
  }

  def genNullableT[A <: Null](ctx: ExecuteContext, typ: Type): Gen[A] =
    genNullable(ctx, typ).asInstanceOf[Gen[A]]

  def genNonMissingT[A](ctx: ExecuteContext, typ: Type): Gen[A] =
    genNonMissing(ctx, typ).asInstanceOf[Gen[A]]

}
