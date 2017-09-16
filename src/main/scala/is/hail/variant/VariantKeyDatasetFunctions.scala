package is.hail.variant

import is.hail.io.vcf.ExportVCF
import is.hail.methods.VEP
import is.hail.utils._

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

class VariantKeyDatasetFunctions[T >: Null](private val vsm: VariantSampleMatrix[Locus, Variant, T]) {
  implicit val tct: ClassTag[T] = vsm.tct

  /**
    *
    * @param path     output path
    * @param append   append file to header
    * @param parallel export VCF in parallel using the path argument as a directory
    */
  def exportVCF(path: String, append: Option[String] = None, parallel: Boolean = false) {
    ExportVCF(vsm, path, append, parallel)
  }

  def minRep(maxShift: Int = 100): VariantSampleMatrix[Locus, Variant, T] = {
    require(maxShift > 0, s"invalid value for maxShift: $maxShift. Parameter must be a positive integer.")
    val minrepped = vsm.rdd.map { case (v, (va, gs)) =>
      (v.minRep, (va, gs))
    }
    vsm.copy(rdd = minrepped.smartShuffleAndSort(vsm.rdd.orderedPartitioner, maxShift))
  }

  /**
    *
    * @param config    VEP configuration file
    * @param root      Variant annotation path to store VEP output
    * @param csq       Annotates with the VCF CSQ field as a string, rather than the full nested struct schema
    * @param blockSize Variants per VEP invocation
    */
  def vep(config: String, root: String = "va.vep", csq: Boolean = false,
    blockSize: Int = 1000): VariantSampleMatrix[Locus, Variant, T] = {
    VEP.annotate(vsm, config, root, csq, blockSize)
  }

  def filterIntervals(intervals: java.util.ArrayList[Interval[Locus]], keep: Boolean): VariantSampleMatrix[Locus, Variant, T] = {
    val iList = IntervalTree[Locus](intervals.asScala.toArray)
    filterIntervals(iList, keep)
  }

  def filterIntervals(iList: IntervalTree[Locus, _], keep: Boolean): VariantSampleMatrix[Locus, Variant, T] = {
    if (keep)
      vsm.copy(rdd = vsm.rdd.filterIntervals(iList))
    else {
      val iListBc = vsm.sparkContext.broadcast(iList)
      vsm.filterVariants { (v, va, gs) => !iListBc.value.contains(v.locus) }
    }
  }

  /**
    * Remove multiallelic variants from this dataset.
    *
    * Useful for running methods that require biallelic variants without calling the more expensive split_multi step.
    */
  def filterMulti(): VariantSampleMatrix[Locus, Variant, T] = {
    if (vsm.wasSplit) {
      warn("called redundant `filter_multi' on an already split or multiallelic-filtered VDS")
      vsm
    } else {
      vsm.filterVariants {
        case (v, va, gs) => v.isBiallelic
      }.copy2(wasSplit = true)
    }
  }

  def verifyBiallelic(): VariantSampleMatrix[Locus, Variant, T] =
    verifyBiallelic("verifyBialellic")

  def verifyBiallelic(method: String): VariantSampleMatrix[Locus, Variant, T] = {
    if (vsm.wasSplit) {
      assert(method == "verify_biallelic")
      warn("called redundant `verify_biallelic' on biallelic VDS")
      vsm
    } else {
      vsm.copy(
        rdd = vsm.rdd.map { case vvags@(v, _) =>
          if (!v.isBiallelic)
            fatal("in $method: found non-biallelic variant: $v")
          vvags
        }.toOrderedRDD,
        wasSplit = true)
    }
  }
}
