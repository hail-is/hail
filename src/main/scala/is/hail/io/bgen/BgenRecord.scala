package is.hail.io.bgen

import is.hail.annotations._
import is.hail.io.{ByteArrayReader, KeySerializedValueRecord}
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

    val byteReader = new ByteArrayReader(if (compressed) decompress(input, nSamples * 6) else input)

    assert(byteReader.length == nSamples * 6)

    resetWarnings()

    val lowerTol = (32768 * (1.0 - tolerance) + 0.5).toInt
    val upperTol = (32768 * (1.0 + tolerance) + 0.5).toInt
    assert(lowerTol > 0)

    val noCall: Genotype = new DosageGenotype(-1, null)

    new Iterable[Genotype] {
      def iterator = new Iterator[Genotype] {
        var i = 0

        def hasNext: Boolean = i < byteReader.length

        def next(): Genotype = {
          val d0 = byteReader.readShort()
          val d1 = byteReader.readShort()
          val d2 = byteReader.readShort()

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
              setWarning(gpSumGreaterThanTolerance)
              noCall
            }
          } else {
            if (dsum == 0)
              setWarning(gpNoCall)
            else
              setWarning(gpSumLessThanTolerance)

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

    val reader = new ByteArrayReader(if (compressed) decompress(input, expectedDataSize) else input)

    val nRow = reader.readInt()
    assert(nRow == nSamples, "row nSamples is not equal to header nSamples")

    val nAlleles = reader.readShort()
    assert(nAlleles == expectedNumAlleles, s"Value for `nAlleles' in genotype probability data storage is not equal to value in variant identifying data. Expected $expectedNumAlleles but found $nAlleles.")

    val minPloidy = reader.read()
    val maxPloidy = reader.read()

    if (minPloidy != 2 || maxPloidy != 2)
      fatal(s"Hail only supports diploid genotypes. Found min ploidy equals `$minPloidy' and max ploidy equals `$maxPloidy'.")

    val samplePloidy = new Array[Int](nSamples)
    var i = 0
    while (i < nSamples) {
      val ploidy = reader.read()
      assert(ploidy == 2 || ploidy < 0, s"Ploidy value must be either 2 or missing. Found $ploidy.")
      samplePloidy(i) = ploidy
      i += 1
    }
    assert(i == nSamples, s"Number of ploidy values `$i' does not equal the number of samples `$nSamples'.")

    val phase = reader.read()
    assert(phase == 0 || phase == 1, s"Value for phase must be 0 or 1. Found $phase.")
    val isPhased = phase == 1

    if (isPhased)
      fatal("Hail does not support phased genotypes.")

    val nBitsPerProb = reader.read()
    assert(nBitsPerProb >= 1 && nBitsPerProb <= 32, s"Value for nBits must be between 1 and 32 inclusive. Found $nBitsPerProb.")

    val nGenotypes = triangle(nAlleles)
    val nExpectedBytesProbs = (nSamples * (nGenotypes - 1) * nBitsPerProb + 7) / 8
    assert(reader.length == nExpectedBytesProbs + nSamples + 10, s"Number of uncompressed bytes `${ reader.length }' does not match the expected size `$nExpectedBytesProbs'.")

    val probabilityIterator = new BgenProbabilityIterator(reader, nBitsPerProb)

    val sampleProbs = new Array[UInt](nGenotypes)

    val totalProbInt = ((1L << nBitsPerProb) - 1).toUInt
    val noCall = new DosageGenotype(-1, null)

    new Iterable[Genotype] {
      def iterator = new Iterator[Genotype] {
        var sampleIndex = 0

        def hasNext: Boolean = sampleIndex < nSamples

        def next(): Genotype = {
          var i = 0
          var sumProbInt = UInt(0)
          while (i < nGenotypes - 1) {
            assert(probabilityIterator.hasNext, "Did not decode bytes correctly. Ran out of probabilities.")
            val p = probabilityIterator.next()
            sampleProbs(i) = p
            i += 1
            sumProbInt += p
          }

          assert(sumProbInt <= totalProbInt, "Sum of probabilities is greater than 1.")
          sampleProbs(i) = totalProbInt - sumProbInt

          val gt = if (samplePloidy(sampleIndex) < 0)
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

