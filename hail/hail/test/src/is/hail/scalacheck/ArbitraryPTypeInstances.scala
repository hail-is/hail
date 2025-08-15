package is.hail.scalacheck

import is.hail.scalacheck.ArbitraryPTypeInstances.DefaultRequiredGenRatio
import is.hail.types.physical._
import is.hail.variant.ReferenceGenome

import scala.collection.compat._

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

  lazy val genPrimitivePType: Gen[PType] =
    frequency(
      4 -> arbitrary[PNumeric],
      1 -> (genIsRequired map { PBoolean(_) }),
      1 -> (genIsRequired map PCanonicalCall),
    )

  lazy val genScalarPType: Gen[PType] =
    frequency(
      6 -> genPrimitivePType,
      1 -> (genIsRequired map { PCanonicalString(_) }),
      1 -> arbitrary[PCanonicalLocus],
    )

  implicit lazy val arbPCanonicalLocus: Arbitrary[PCanonicalLocus] =
    liftA2(PCanonicalLocus(_, _), oneOf(ReferenceGenome.hailReferences), genIsRequired)

  implicit lazy val arbPCanonicalStruct: Arbitrary[PCanonicalStruct] =
    for {
      len <- size
      names <- distinctContainerOfN[Array, String](len, identifier)
      types <- distribute(len, arbitrary[PType])
      required <- genIsRequired
    } yield PCanonicalStruct(
      (names lazyZip types lazyZip Array.range(0, len)).map(PField),
      required,
    )

  implicit lazy val arbPCanonicalTuple: Arbitrary[PCanonicalTuple] =
    for {
      len <- size
      types <- distribute(len, arbitrary[PType])
      required <- genIsRequired
    } yield PCanonicalTuple(Array.range(0, len).lazyZip(types).map(PTupleField), required)

  implicit lazy val arbPCanonicalArray: Arbitrary[PCanonicalArray] =
    liftA2(PCanonicalArray(_, _), smaller[PType], genIsRequired)

  implicit lazy val genPCanonicalNDArray: Arbitrary[PCanonicalNDArray] =
    liftA3(PCanonicalNDArray(_, _, _), genPrimitivePType, oneOf(0 until 6), genIsRequired)

  implicit lazy val arbPCanonicalDict: Arbitrary[PCanonicalDict] =
    liftA3(
      PCanonicalDict(_, _, _),
      smaller[PType] map { _.setRequired(true) },
      smaller[PType],
      genIsRequired,
    )

  implicit lazy val arbPCanonicalSet: Arbitrary[PCanonicalSet] =
    liftA2(PCanonicalSet(_, _), smaller[PType], genIsRequired)

  implicit lazy val arbPCanonicalInterval: Arbitrary[PCanonicalInterval] =
    liftA2(PCanonicalInterval(_, _), smaller[PType], genIsRequired)

  implicit lazy val arbPContainer: Arbitrary[PContainer] =
    oneOf(
      arbitrary[PCanonicalArray],
      arbitrary[PCanonicalDict],
      arbitrary[PCanonicalSet],
    )

  implicit lazy val arbPType: Arbitrary[PType] =
    sized { n =>
      frequency(
        1 + 8 * n -> genScalarPType,
        n -> arbitrary[PCanonicalArray],
        n -> arbitrary[PCanonicalDict],
        n -> arbitrary[PCanonicalInterval],
        n -> arbitrary[PCanonicalLocus],
        n -> arbitrary[PCanonicalSet],
        n -> arbitrary[PCanonicalStruct],
        n -> arbitrary[PCanonicalTuple],
      )
    }

}
