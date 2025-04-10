package is.hail.scalacheck

import is.hail.annotations.{Annotation => An, SafeNDArray}
import is.hail.backend.ExecuteContext
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.Interval

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen._

private[scalacheck] trait GenVal {

  def genNonMissing(typ: Type): Gen[Any] =
    genVal(PType.canonical(typ, required = true, innerRequired = true))

  def genNullable(typ: Type): Gen[Any] =
    genVal(PType.canonical(typ))

  def genVal(pt: PType): Gen[Any] = {

    def wrap(g: Gen[An]): Gen[An] =
      if (pt.required) g else nullable(g)

    wrap {
      pt match {
        case p: PArray =>
          containerOf[IndexedSeq, An](genVal(p.elementType))
        case t: PBaseStruct =>
          sequence(t.types.map(genVal)).map(arr => new GenericRow(arr))
        case _: PBoolean =>
          arbitrary[Boolean]
        case _: PBinary =>
          containerOf[Array, Byte](arbitrary[Byte])
        case _: PCall =>
          genCall
        case t: PDict =>
          mapOf(genVal(t.elementType).map { case Row(k, v) => (k, v) })
        case _: PFloat32 =>
          arbitrary[Float]
        case _: PFloat64 =>
          arbitrary[Double]
        case _: PInt32 =>
          arbitrary[Int]
        case _: PInt64 =>
          arbitrary[Long]
        case t: PInterval =>
          val ord = t.pointType.virtualType.ordering
          val elem = genVal(t.pointType)
          for {
            (a, b, s, e) <- zip(elem, elem, arbitrary[Boolean], arbitrary[Boolean])
            if ord.compare(a, b) != 0 || (s && e)
          } yield Interval(ord.min(a, b), ord.max(a, b), s, e)
        case p: PLocus =>
          genLocus(p.rg)
        case t: PNDArray =>
          for {
            len <- size
            scale <- dirichlet(Array.fill(t.nDims)(1d))
            shape = scale map { factor => (factor * len).ceil.toLong }
            data <- containerOfN[Array, An](shape.sum.toInt, genVal(t.elementType))
          } yield SafeNDArray(shape, data)
        case p: PSet =>
          containerOf[Set, An](genVal(p.elementType))
        case _: PString =>
          asciiStr
        case PVoid =>
          null
      }
    }
  }

  def genNullableT[A <: Null](typ: Type): Gen[Any] =
    genNullable(typ).asInstanceOf[Gen[A]]

  def genNonMissingT[A](typ: Type): Gen[Any] =
    genNonMissing(typ).asInstanceOf[Gen[A]]

  def genTypeVal[T <: Type: Arbitrary]: Gen[(T, Any)] =
    genTypeValImpl[T](genNullable(_).filter(_ != null))

  def genPTypeVal[P <: PType: Arbitrary]: Gen[(P, Any)] =
    genTypeValImpl[P](genVal)

  private[this] def genTypeValImpl[A: Arbitrary](an: A => Gen[An]): Gen[(A, An)] =
    for {
      factor <- beta(1, 8)
      t <- scale(factor, arbitrary[A])
      v <- scale(1 - factor, an(t))
    } yield (t, v)
}
