package is.hail.io.bgen

import is.hail.annotations._
import is.hail.io.KeySerializedValueRecord
import is.hail.io.gen.GenReport._
import is.hail.utils._
import is.hail.variant.{DosageGenotype, Genotype, Variant}

abstract class BgenRecord extends KeySerializedValueRecord[Variant, Iterable[Genotype]] {
  var ann: Annotation = _

  def setAnnotation(ann: Annotation) {
    this.ann = ann
  }

  def getAnnotation: Annotation = ann

  override def getValue: Iterable[Genotype]
}

class BgenRecordV11(compressed: Boolean,
  nSamples: Int,
  tolerance: Double) extends BgenRecord {

  override def getValue: Iterable[Genotype] = {
    require(input != null, "called getValue before serialized value was set")

    val bytes = if (compressed) decompress(input, nSamples * 6) else input

    assert(bytes.length == nSamples * 6)

    resetWarnings()

    val lowerTol = (32768 * (1.0 - tolerance) + 0.5).toInt
    val upperTol = (32768 * (1.0 + tolerance) + 0.5).toInt
    assert(lowerTol > 0)

    val noCall: Genotype = new DosageGenotype(-1, null)

    new Iterable[Genotype] {
      def iterator = new Iterator[Genotype] {
        var i = 0

        def hasNext: Boolean = i < bytes.length

        def next(): Genotype = {
          val d0 = (bytes(i) & 0xff) | ((bytes(i + 1) & 0xff) << 8)
          val d1 = (bytes(i + 2) & 0xff) | ((bytes(i + 3) & 0xff) << 8)
          val d2 = (bytes(i + 4) & 0xff) | ((bytes(i + 5) & 0xff) << 8)

          i += 6

          val dsum = d0 + d1 + d2
          if (dsum >= lowerTol) {
            if (dsum <= upperTol) {
              val px =
                if (dsum == 32768)
                  Array(d0, d1, d2)
                else
                  Genotype.weightsToLinear(d0, d1, d2)
              val gt = Genotype.unboxedGTFromLinear(px)
              new DosageGenotype(gt, px)
            } else {
              setWarning(dosageGreaterThanTolerance)
              noCall
            }
          } else {
            if (dsum == 0)
              setWarning(dosageNoCall)
            else
              setWarning(dosageLessThanTolerance)

            noCall
          }
        }
      }
    }
  }
}

class BgenRecordV12(compressed: Boolean, nSamples: Int, tolerance: Double) extends BgenRecord {
  var expectedDataSize: Int = _
  var expectedNumAlleles: Int = _

  def setExpectedDataSize(size: Int) {
    this.expectedDataSize = size
  }

  def setExpectedNumAlleles(n: Int) {
    this.expectedNumAlleles = n
  }

  override def getValue: Iterable[Genotype] = {
    require(input != null, "called getValue before serialized value was set")

    val bytes = if (compressed) decompress(input, expectedDataSize) else input
    val nRow = (bytes(0) & 0xff) | ((bytes(1) & 0xff) << 8) | ((bytes(2) & 0xff) << 16) | ((bytes(3) & 0xff) << 24)
    assert(nRow == nSamples, "row nSamples is not equal to header nSamples")

    val nAlleles = (bytes(4) & 0xff) | ((bytes(5) & 0xff) << 8)
    assert(nAlleles == expectedNumAlleles, s"Value for `nAlleles' in genotype probability data storage is not equal to value in variant identifying data. Expected $expectedNumAlleles but found $nAlleles.")

    val minPloidy = bytes(6) & 0xff
    val maxPloidy = bytes(7) & 0xff

    if (minPloidy != 2 || maxPloidy != 2)
      fatal(s"Hail only supports diploid genotypes. Found min ploidy equals `$minPloidy' and max ploidy equals `$maxPloidy'.")

    val samplePloidy = Array.ofDim[Int](nSamples)
    var i = 0
    while (i < nSamples) {
      val b = bytes(i + 8)
      val isMissing = ((b & 0xffL) >>> 7) == 1
      val ploidy = if (isMissing) -1 else b & 0x3f
      assert(ploidy >= -1 && ploidy <= 63, s"Ploidy value must be in range [0, 63]. Found $ploidy.")
      samplePloidy(i) = ploidy
      i += 1
    }
    assert(i == nSamples, s"Number of ploidy values `$i' does not equal the number of samples `$nSamples'.")

    val phase = bytes(nSamples + 8) & 0xff
    assert(phase == 0 || phase == 1, s"Value for phase must be 0 or 1. Found $phase.")
    val isPhased = phase == 1

    if (isPhased)
      fatal("Hail does not support phased genotypes.")

    val nBitsPerProb = bytes(nSamples + 9) & 0xff
    assert(nBitsPerProb >= 1 && nBitsPerProb <= 32, s"Value for nBits must be between 1 and 32 inclusive. Found $nBitsPerProb.")

    val probabilityIterator = new BgenProbabilityIterator(bytes, nBitsPerProb, nSamples + 10)

    val nGenotypes = triangle(nAlleles)
    val sampleProbs = Array.ofDim[Int](nGenotypes)
    val totalProbInt = (~0L >>> (64 - nBitsPerProb)).toInt
    val noCall: Genotype = new DosageGenotype(-1, null)

    val nExpectedBytesProbs = math.ceil((nSamples * (nGenotypes - 1) * 10).toDouble / 8).toInt
    assert(bytes.length == nExpectedBytesProbs + nSamples + 10, s"Number of uncompressed bytes `${ bytes.length }' does not match the expected size `$nExpectedBytesProbs'.")

    new Iterable[Genotype] {
      def iterator = new Iterator[Genotype] {
        var sampleIndex = 0

        def hasNext: Boolean = sampleIndex < nSamples

        def next(): Genotype = {
          var i = 0
          var sumProbInt = 0
          while (i < nGenotypes - 1) {
            assert(probabilityIterator.hasNext, "Did not decode bytes correctly. Ran out of probabilities.")
            val p = probabilityIterator.next()
            sampleProbs(i) = p
            i += 1
            sumProbInt += p
          }

          assert(sumProbInt <= totalProbInt, "Sum of probabilities is greater than 1.")
          sampleProbs(i) = totalProbInt - sumProbInt

          val gt = if (samplePloidy(sampleIndex) == -1)
            noCall
          else {
            val px = Genotype.weightsToLinear(sampleProbs)
            val gt = Genotype.unboxedGTFromLinear(px)
            new DosageGenotype(gt, px)
          }

          sampleIndex += 1
          gt
        }
      }
    }
  }
}

