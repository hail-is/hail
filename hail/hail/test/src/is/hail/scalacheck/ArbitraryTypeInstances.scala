package is.hail.scalacheck

import is.hail.types.virtual._
import is.hail.variant.ReferenceGenome.hailReferences

import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen.{const, identifier, oneOf, resultOf, size}

private[scalacheck] trait ArbitraryTypeInstances {

  implicit lazy val arbNumericType: Arbitrary[TNumeric] =
    Gen.oneOf(
      TInt32,
      TInt64,
      TFloat32,
      TFloat64,
    )

  implicit lazy val arbTLocus: Arbitrary[TLocus] =
    oneOf(hailReferences).map(TLocus(_))

  lazy val genScalarType: Gen[Type] =
    Gen.oneOf(
      arbitrary[TNumeric],
      arbitrary[TLocus],
      oneOf(TBoolean, TCall, TString),
    )

  implicit lazy val arbTStruct: Arbitrary[TStruct] =
    for {
      len <- size
      names <- distinctContainerOfN[Array, String](len, identifier)
      types <- distinctContainerOfN[Array, Type](len, arbitrary[Type])
    } yield TStruct((names, types, 0 until len).zipped.map(Field))

  implicit lazy val arbTTuple: Arbitrary[TTuple] =
    distinctNonEmptyContainer[Array, Type](arbitrary[Type]).map(TTuple(_: _*))

  implicit lazy val arbTArray: Arbitrary[TArray] =
    resultOf(TArray)

  implicit lazy val arbTDict: Arbitrary[TDict] =
    resultOf(TDict)

  implicit lazy val arbTSet: Arbitrary[TSet] =
    resultOf(TSet)

  implicit lazy val arbTInterval: Arbitrary[TInterval] =
    arbitrary[Type].map(TInterval)

  implicit lazy val arbType: Arbitrary[Type] =
    oneOf(
      arbitrary[TArray],
      arbitrary[TDict],
      arbitrary[TInterval],
      arbitrary[TStruct],
      arbitrary[TTuple],
      genScalarType,
      const(TVoid),
    )
}
