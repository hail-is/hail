package is.hail.scalacheck

import is.hail.annotations.{Annotation => An, SafeNDArray}
import is.hail.backend.ExecuteContext
import is.hail.scalacheck
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.Interval

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen._

private[scalacheck] trait GenVal {

  def genNonMissing(ctx: ExecuteContext, typ: Type): Gen[An] =
    genVal(ctx, PType.canonical(typ, required = true, innerRequired = true))

  def genNullable(ctx: ExecuteContext, typ: Type): Gen[An] =
    genVal(ctx, PType.canonical(typ))

  def genVal(ctx: ExecuteContext, pt: PType): Gen[An] = {

    def wrap(g: Gen[An]): Gen[An] =
      if (pt.required) g else nullable(g)

    def smaller(ctx: ExecuteContext, t: PType): Gen[An] =
      scalacheck.smaller(genVal(ctx, t))

    wrap {
      pt match {
        case p: PArray =>
          containerOf[IndexedSeq, An](smaller(ctx, p.elementType))
        case t: PBaseStruct =>
          for {
            sizes <- partition(t.types.length)
            values <- sequence((sizes, t.types).zipped.map { case (s, t) =>
              resize(s, genVal(ctx, t))
            })
          } yield new GenericRow(values)
        case _: PBoolean =>
          arbitrary[Boolean]
        case _: PBinary =>
          containerOf[Array, Byte](arbitrary[Byte])
        case _: PCall =>
          genCall
        case t: PDict =>
          mapOf(smaller(ctx, t.elementType).map { case Row(k, v) => (k, v) })
        case _: PFloat32 =>
          arbitrary[Float]
        case _: PFloat64 =>
          arbitrary[Double]
        case _: PInt32 =>
          arbitrary[Int]
        case _: PInt64 =>
          arbitrary[Long]
        case t: PInterval =>
          distribute(2, genVal(ctx, t.pointType)) flatMap { case Array(a, b) =>
            t.pointType.virtualType.mkOrdering(ctx.stateManager).compare(a, b) match {
              case 0 => const(Interval(a, b, true, true))
              case n =>
                val (start, end) = if (n < 0) (a, b) else (b, a)
                resultOf[Boolean, Boolean, Interval](Interval(start, end, _, _))
            }
          }
        case p: PLocus =>
          genLocus(ctx.references(p.rg))
        case t: PNDArray =>
          for {
            shape <- partition(t.nDims)
            data <- containerOfN[Array, An](shape.sum, smaller(ctx, t.elementType))
          } yield SafeNDArray(shape.map(_.toLong), data)
        case p: PSet =>
          containerOf[Set, An](smaller(ctx, p.elementType))
        case _: PString =>
          asciiStr
        case PVoid =>
          null
      }
    }
  }

  def genNullableT[A <: Null](ctx: ExecuteContext, typ: Type): Gen[A] =
    genNullable(ctx, typ).asInstanceOf[Gen[A]]

  def genNonMissingT[A](ctx: ExecuteContext, typ: Type): Gen[A] =
    genNonMissing(ctx, typ).asInstanceOf[Gen[A]]

  def genTypeVal[T <: Type: Arbitrary](ctx: ExecuteContext): Gen[(T, An)] =
    genTypeValImpl[T](genNullable(ctx, _))

  def genPTypeVal[P <: PType: Arbitrary](ctx: ExecuteContext): Gen[(P, An)] =
    genTypeValImpl[P](genVal(ctx, _))

  private[this] def genTypeValImpl[A: Arbitrary](an: A => Gen[An]): Gen[(A, An)] =
    for {
      factor <- beta(1, 8)
      t <- scale(factor, arbitrary[A])
      v <- scale(1 - factor, an(t))
      if v != null
    } yield (t, v)
}
