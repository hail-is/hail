package is.hail.scalacheck

import is.hail.types.virtual._
import is.hail.variant.ReferenceGenome.hailReferences

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

  implicit lazy val arbTLocus: Arbitrary[TLocus] =
    oneOf(hailReferences).map(TLocus(_))

  lazy val genScalarType: Gen[Type] =
    oneOf(
      arbitrary[TNumeric],
      arbitrary[TLocus],
      oneOf(TBoolean, TCall, TString),
    )

  implicit lazy val arbTStruct: Arbitrary[TStruct] =
    limit(16) {
      for {
        len <- size
        names <- distinctContainerOfN[Array, String](len, identifier)
        types <- distinctContainerOfN[Array, Type](len, smaller[Type])
      } yield TStruct((names, types, 0 until len).zipped.map(Field))
    }

  implicit lazy val arbTTuple: Arbitrary[TTuple] =
    limit(16) {
      distinctNonEmptyContainer[Array, Type](smaller[Type]).map(TTuple(_: _*))
    }

  implicit lazy val arbTArray: Arbitrary[TArray] =
    resultOf(TArray)(smaller[Type])

  implicit lazy val arbTDict: Arbitrary[TDict] =
    resultOf(TDict)(smaller[Type], smaller[Type])

  implicit lazy val arbTInterval: Arbitrary[TInterval] =
    resultOf(TInterval)(smaller[Type])

  implicit lazy val arbTSet: Arbitrary[TSet] =
    resultOf(TSet)(smaller[Type])

  implicit lazy val arbType: Arbitrary[Type] =
    sized { x =>
      frequency(
        1 + 4 * x -> genScalarType,
        x -> arbitrary[TArray],
        x -> arbitrary[TDict],
        x -> arbitrary[TInterval],
        x -> arbitrary[TSet],
        x -> arbitrary[TStruct],
        x -> arbitrary[TTuple],
      )
    }
}
