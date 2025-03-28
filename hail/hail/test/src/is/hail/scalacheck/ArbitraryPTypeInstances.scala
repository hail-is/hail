package is.hail.scalacheck

import is.hail.scalacheck.ArbitraryPTypeInstances.DefaultRequiredGenRatio
import is.hail.types.physical._
import is.hail.variant.ReferenceGenome

import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen._

private[scalacheck] object ArbitraryPTypeInstances {
  val DefaultRequiredGenRatio = 0.2
}

private[scalacheck] trait ArbitraryPTypeInstances {

  lazy val genIsRequired: Gen[Boolean] =
    prob(DefaultRequiredGenRatio)

  implicit lazy val arbPNumeric: Arbitrary[PNumeric] =
    oneOf[Boolean => PNumeric](PInt32(_), PInt64(_), PFloat32(_), PFloat64(_)) ap genIsRequired

  lazy val genScalarPType: Gen[PType] =
    oneOf(
      arbitrary[PNumeric],
      oneOf[Boolean => PType](PBoolean(_), PCanonicalCall(_), PCanonicalString(_)) ap genIsRequired,
    )

  implicit lazy val arbPCanonicalLocus: Arbitrary[PCanonicalLocus] =
    liftA2(PCanonicalLocus(_, _), arbitrary[ReferenceGenome].map(_.name), genIsRequired)

  implicit lazy val arbPCanonicalCall: Arbitrary[PCanonicalCall] =
    genIsRequired.map(PCanonicalCall)

  implicit lazy val arbPCanonicalStruct: Arbitrary[PCanonicalStruct] =
    for {
      len <- size
      names <- distinctContainerOfN[Array, String](len, identifier)
      types <- distinctContainerOfN[Array, PType](len, arbitrary[PType])
      required <- genIsRequired
    } yield PCanonicalStruct((names, types, Array.range(0, len)).zipped.map(PField), required)

  implicit lazy val arbPCanonicalTuple: Arbitrary[PCanonicalTuple] =
    for {
      len <- size
      types <- distinctContainerOfN[Array, PType](len, arbitrary[PType])
      required <- genIsRequired
    } yield PCanonicalTuple((Array.range(0, len), types).zipped.map(PTupleField), required)

  implicit lazy val arbPCanonicalArray: Arbitrary[PCanonicalArray] =
    liftA2(PCanonicalArray(_, _), arbitrary[PType], genIsRequired)

  implicit lazy val arbPCanonicalDict: Arbitrary[PCanonicalDict] =
    liftA3(
      PCanonicalDict(_, _, _),
      arbitrary[PType] suchThat (_.required),
      arbitrary[PType],
      genIsRequired,
    )

  implicit lazy val arbPCanonicalSet: Arbitrary[PCanonicalSet] =
    liftA2(PCanonicalSet(_, _), arbitrary[PType], genIsRequired)

  implicit lazy val arbPCanonicalInterval: Arbitrary[PCanonicalInterval] =
    liftA2(PCanonicalInterval(_, _), arbitrary[PType], genIsRequired)

  implicit lazy val arbPType: Arbitrary[PType] =
    oneOf(
      arbitrary[PCanonicalArray],
      arbitrary[PCanonicalCall],
      arbitrary[PCanonicalDict],
      arbitrary[PCanonicalInterval],
      arbitrary[PCanonicalLocus],
      arbitrary[PCanonicalSet],
      arbitrary[PCanonicalStruct],
      arbitrary[PCanonicalTuple],
      genScalarPType,
    )

}
