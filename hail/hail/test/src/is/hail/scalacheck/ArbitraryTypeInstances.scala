package is.hail.scalacheck

import is.hail.expr.Nat
import is.hail.types.virtual._
import is.hail.variant.ReferenceGenome.hailReferences

import scala.collection.compat._

import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen._

private[scalacheck] trait ArbitraryTypeInstances {

  implicit lazy val arbNumericType: Arbitrary[TNumeric] =
    oneOf(
      TInt32,
      TInt64,
      TFloat32,
      TFloat64,
    )

  lazy val genPrimitiveType: Gen[Type] =
    frequency(
      4 -> arbitrary[TNumeric],
      1 -> TBoolean,
      1 -> TCall,
    )

  implicit lazy val arbTLocus: Arbitrary[TLocus] =
    oneOf(hailReferences).map(TLocus(_))

  lazy val genScalarType: Gen[Type] =
    frequency(
      6 -> genPrimitiveType,
      1 -> TString,
      1 -> arbitrary[TLocus],
    )

  implicit lazy val arbTStruct: Arbitrary[TStruct] =
    for {
      len <- size
      names <- distinctContainerOfN[Array, String](len, identifier)
      types <- distribute(len, arbitrary[Type])
    } yield TStruct((names lazyZip types lazyZip (0 until len)).map(Field))

  implicit lazy val arbTTuple: Arbitrary[TTuple] =
    sized(n => distribute(n, arbitrary[Type]) map { TTuple(_: _*) })

  implicit lazy val arbTArray: Arbitrary[TArray] =
    resultOf(TArray)(smaller[Type])

  implicit lazy val arbTNDArray: Arbitrary[TNDArray] =
    liftA2(TNDArray(_, _), genPrimitiveType, oneOf(0 until 6) map Nat)

  implicit lazy val arbTDict: Arbitrary[TDict] =
    resultOf(TDict)(smaller[Type], smaller[Type])

  implicit lazy val arbTInterval: Arbitrary[TInterval] =
    resultOf(TInterval)(smaller[Type])

  implicit lazy val arbTSet: Arbitrary[TSet] =
    resultOf(TSet)(smaller[Type])

  implicit lazy val arbTContainer: Arbitrary[TContainer] =
    oneOf(
      arbitrary[TArray],
      arbitrary[TDict],
      arbitrary[TSet],
    )

  implicit lazy val arbType: Arbitrary[Type] =
    sized { x =>
      frequency(
        1 + 8 * x -> genScalarType,
        x -> arbitrary[TArray],
        x -> arbitrary[TDict],
        x -> arbitrary[TInterval],
        x -> arbitrary[TSet],
        x -> arbitrary[TStruct],
        x -> arbitrary[TTuple],
      )
    }
}
