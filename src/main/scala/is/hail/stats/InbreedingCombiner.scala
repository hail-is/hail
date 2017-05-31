package is.hail.stats


import is.hail.annotations.Annotation
import is.hail.expr.{Field, TDouble, TLong, TStruct}
import is.hail.utils._
import is.hail.variant.Genotype

object InbreedingCombiner {
  def signature = TStruct(Array(("Fstat", TDouble, "Inbreeding coefficient"),
    ("nTotal", TLong, "Number of genotypes analyzed"),
    ("nCalled", TLong, "number of genotypes with non-missing calls"),
    ("expectedHoms", TDouble, "Expected number of homozygote calls"),
    ("observedHoms", TLong, "Total number of homozygote calls observed")
  ).zipWithIndex.map { case ((n, t, d), i) => Field(n, t, i, Map(("desc", d))) })
}

class InbreedingCombiner extends Serializable {
  var nCalled = 0L
  var expectedHoms = 0d
  var observedHoms = 0L
  var total = 0L

  def merge(g: Genotype, af: Double): InbreedingCombiner = {
    total += 1
    if (Genotype.isCalled(g)) {
      nCalled += 1
      expectedHoms += 1 - (2 * af * (1 - af))

      if (Genotype.isHomRef(g) || Genotype.isHomVar(g))
        observedHoms += 1
    }
    this
  }

  def merge(other: InbreedingCombiner): InbreedingCombiner = {
    nCalled += other.nCalled
    expectedHoms += other.expectedHoms
    observedHoms += other.observedHoms
    total += other.total
    this
  }

  def Fstat: Option[Double] = divOption(observedHoms - expectedHoms, nCalled - expectedHoms)

  def asAnnotation: Annotation = Annotation(Fstat.orNull, total, nCalled, expectedHoms, observedHoms)
}
