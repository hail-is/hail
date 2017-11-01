package is.hail.variant

import is.hail.annotations._
import is.hail.expr.{TArray, TCall, TGenotype, TInt32, TString, TStruct, Type}
import is.hail.io.vcf.ExportVCF
import is.hail.methods.{SplitMulti, VEP}
import is.hail.sparkextras.OrderedRDD2
import is.hail.utils._
import org.apache.spark.rdd.RDD

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

  def minRep(leftAligned: Boolean = false): VariantSampleMatrix[Locus, Variant, T] = {
    val localRowType = vsm.rowType

    def minRep1(removeLeftAligned: Boolean, removeMoving: Boolean, verifyLeftAligned: Boolean): RDD[RegionValue] = {
        vsm.rdd2.mapPartitions { it =>
          var prevLocus: Locus = null
          val rvb = new RegionValueBuilder()
          val rv2 = RegionValue()

          it.flatMap { rv =>
            val ur = new UnsafeRow(localRowType, rv.region, rv.offset)
            val v = ur.getAs[Variant](1)
            val minv = v.minRep

            var isLeftAligned = (prevLocus == null || prevLocus != v.locus) &&
              (v.locus == minv.locus)

            if (isLeftAligned && removeLeftAligned)
              None
            else if (!isLeftAligned && removeMoving)
              None
            else if (!isLeftAligned && verifyLeftAligned)
              fatal(s"found non-left aligned variant $v")
            else {

              rvb.set(rv.region)
              rvb.start(localRowType)
              rvb.startStruct()
              rvb.addAnnotation(localRowType.fieldType(0), minv.locus)
              rvb.addAnnotation(localRowType.fieldType(1), minv)
              rvb.addField(localRowType, rv, 2)
              rvb.addField(localRowType, rv, 3)
              rvb.endStruct()
              rv2.set(rv.region, rvb.end())
              Some(rv2)
            }
          }
        }
    }

    val newRDD2 =
      if (leftAligned)
        vsm.rdd2.copy(
          rdd = minRep1(removeLeftAligned = false, removeMoving = false, verifyLeftAligned = true))
      else {
        val leftAlignedVariants = vsm.rdd2.copy(
          rdd = minRep1(removeLeftAligned = false, removeMoving = true, verifyLeftAligned = false))
        val movedVariants =
          OrderedRDD2.shuffle(
            vsm.matrixType.orderedRDD2Type,
            vsm.rdd2.orderedPartitioner,
            minRep1(removeLeftAligned = true, removeMoving = false, verifyLeftAligned = false))

        leftAlignedVariants.partitionSortedUnion(movedVariants)
      }

    vsm.copy2(rdd2 = newRDD2)
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
    verifyBiallelic("verifyBiallelic")

  def verifyBiallelic(method: String): VariantSampleMatrix[Locus, Variant, T] = {
    if (vsm.wasSplit) {
      warn("called redundant `$method' on biallelic VDS")
      vsm
    } else {
      val localRowType = vsm.rowType
      vsm.copy2(
        rdd2 = vsm.rdd2.mapPreservesPartitioning { rv =>
          val ur = new UnsafeRow(localRowType, rv.region, rv.offset)
          val v = ur.getAs[Variant](1)
          if (!v.isBiallelic)
            fatal("in $method: found non-biallelic variant: $v")
          rv
        },
        wasSplit = true)
    }
  }

  def exportGen(path: String, precision: Int = 4) {
    require(vsm.wasSplit)

    def writeSampleFile() {
      // FIXME: should output all relevant sample annotations such as phenotype, gender, ...
      vsm.hc.hadoopConf.writeTable(path + ".sample",
        "ID_1 ID_2 missing" :: "0 0 0" :: vsm.sampleIds.map(s => s"$s $s 0").toList)
    }

    def writeGenFile() {
      val varidSignature = vsm.vaSignature.getOption("varid")
      val varidQuery: Querier = varidSignature match {
        case Some(_) =>
          val (t, q) = vsm.queryVA("va.varid")
          t match {
            case TString => q
            case _ => a => null
          }
        case None => a => null
      }

      val rsidSignature = vsm.vaSignature.getOption("rsid")
      val rsidQuery: Querier = rsidSignature match {
        case Some(_) =>
          val (t, q) = vsm.queryVA("va.rsid")
          t match {
            case TString => q
            case _ => a => null
          }
        case None => a => null
      }

      val localNSamples = vsm.nSamples
      val localRowType = vsm.rowType
      vsm.rdd2.mapPartitions { it =>
        val sb = new StringBuilder
        val view = new ArrayGenotypeView(localRowType)
        it.map { rv =>
          view.setRegion(rv)
          val ur = new UnsafeRow(localRowType, rv)

          val v = ur.getAs[Variant](1)
          val va = ur.get(2)

          sb.clear()
          sb.append(v.contig)
          sb += ' '
          sb.append(Option(varidQuery(va)).getOrElse(v.toString))
          sb += ' '
          sb.append(Option(rsidQuery(va)).getOrElse("."))
          sb += ' '
          sb.append(v.start)
          sb += ' '
          sb.append(v.ref)
          sb += ' '
          sb.append(v.alt)

          var i = 0
          while (i < localNSamples) {
            view.setGenotype(i)
            if (view.hasGP) {
              sb += ' '
              sb.append(formatDouble(view.getGP(0), precision))
              sb += ' '
              sb.append(formatDouble(view.getGP(1), precision))
              sb += ' '
              sb.append(formatDouble(view.getGP(2), precision))
            } else
              sb.append(" 0 0 0")
            i += 1
          }
          sb.result()
        }
      }.writeTable(path + ".gen", vsm.hc.tmpDir, None)
    }

    writeSampleFile()
    writeGenFile()
  }

  def splitMulti(propagateGQ: Boolean = false, keepStar: Boolean = false, leftAligned: Boolean = false): VariantSampleMatrix[Locus, Variant, T] = {
    if (vsm.genotypeSignature == TGenotype) {
      vsm.splitMulti("va.aIndex = aIndex, va.wasSplit = wasSplit", s"""
g = let
    newgt = downcode(Call(g.gt), aIndex) and
    newad = if (isDefined(g.ad))
        let sum = g.ad.sum() and adi = g.ad[aIndex] in [sum - adi, adi]
      else
        NA: Array[Int] and
    newpl = if (isDefined(g.pl))
        range(3).map(i => range(g.pl.length).filter(j => downcode(Call(j), aIndex) == Call(i)).map(j => g.pl[j]).min())
      else
        NA: Array[Int] and
    newgq = ${ if (propagateGQ) "g.gq" else "gqFromPL(newpl)" }
  in Genotype(v, newgt, newad, g.dp, newgq, newpl)
    """,
        keepStar, leftAligned)
    } else {
      def hasGenotypeFieldOfType(name: String, expected: Type): Boolean = {
        vsm.genotypeSignature match {
          case t: TStruct =>
            t.selfField(name) match {
              case Some(f) => f.typ == expected
              case None => false
            }
          case _ => false
        }
      }

      val b = new ArrayBuilder[String]()
      if (hasGenotypeFieldOfType("GT", TCall))
        b += "g.GT = downcode(g.GT, aIndex)"
      if (hasGenotypeFieldOfType("AD", TArray(TInt32)))
        b += """
g.AD = if (isDefined(g.AD))
  let sum = g.AD.sum() and adi = g.AD[aIndex] in [sum - adi, adi]
else
  NA: Array[Int]
        """
      if (hasGenotypeFieldOfType("PL", TArray(TInt32)))
        b += """
g.PL = if (isDefined(g.PL))
  range(3).map(i => range(g.PL.length).filter(j => downcode(Call(j), aIndex) == Call(i)).map(j => g.PL[j]).min())
else
  NA: Array[Int]
        """

      var vsm2 = vsm.splitMulti("va.aIndex = aIndex, va.wasSplit = wasSplit", b.result().mkString(","), keepStar, leftAligned)

      if (!propagateGQ && hasGenotypeFieldOfType("GQ", TInt32)) {
        val vsm3 = vsm2.annotateGenotypesExpr("g.GQ = gqFromPL(g.PL)")
        vsm3.mapValues(vsm3.genotypeSignature, g => g.asInstanceOf[T])
      } else
        vsm2
    }
  }

  def splitMulti(variantExpr: String, genotypeExpr: String, keepStar: Boolean, leftAligned: Boolean): VariantSampleMatrix[Locus, Variant, T] = {
    val splitmulti = new SplitMulti(vsm, variantExpr, genotypeExpr, keepStar, leftAligned)
    splitmulti.split()
  }
}
