package is.hail.variant

import is.hail.annotations.{Annotation, _}
import is.hail.expr.{EvalContext, Parser, TAggregable, TString, TStruct, Type, _}
import is.hail.table.Table
import is.hail.methods._
import is.hail.rvd.OrderedRVD
import is.hail.stats.ComputeRRM
import is.hail.utils._
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._
import scala.language.implicitConversions
import scala.language.existentials

object VariantDataset {

  def fromKeyTable(kt: Table): MatrixTable = {
    val vType: Type = kt.keyFields.map(_.typ) match {
      case Array(t@TVariant(_, _)) => t
      case arr => fatal("Require one key column of type Variant to produce a variant dataset, " +
        s"but found [ ${ arr.mkString(", ") } ]")
    }

    val rdd = kt.keyedRDD()
      .map { case (k, v) => (k.asInstanceOf[Row].get(0), v) }
      .filter(_._1 != null)
      .mapValues(a => (a: Annotation, Iterable.empty[Annotation]))

    val metadata = VSMMetadata(
      saSignature = TStruct.empty(),
      vSignature = vType,
      vaSignature = kt.valueSignature,
      globalSignature = TStruct.empty())

    MatrixTable.fromLegacy(kt.hc, metadata,
      VSMLocalValue(Annotation.empty, Array.empty[Annotation], Array.empty[Annotation]), rdd)
  }
}

class VariantDatasetFunctions(private val vsm: MatrixTable) extends AnyVal {

  def concordance(other: MatrixTable): (IndexedSeq[IndexedSeq[Long]], Table, Table) = {
    CalculateConcordance(vsm, other)
  }

  def filterAlleles(filterExpr: String, variantExpr: String = "",
    keep: Boolean = true, subset: Boolean = true, leftAligned: Boolean = false, keepStar: Boolean = false): MatrixTable = {
    if (!vsm.genotypeSignature.isOfType(Genotype.htsGenotypeType))
      fatal(s"filter_alleles: genotype_schema must be the HTS genotype schema, found: ${ vsm.genotypeSignature }")

    val genotypeExpr = if (subset) {
      """
g = let newpl = if (isDefined(g.PL))
        let unnorm = range(newV.nGenotypes).map(newi =>
            let oldi = gtIndex(newToOld[gtj(newi)], newToOld[gtk(newi)])
             in g.PL[oldi]) and
            minpl = unnorm.min()
         in unnorm - minpl
      else
        NA: Array[Int] and
    newgt = gtFromPL(newpl) and
    newad = if (isDefined(g.AD))
        range(newV.nAlleles).map(newi => g.AD[newToOld[newi]])
      else
        NA: Array[Int] and
    newgq = gqFromPL(newpl) and
    newdp = g.DP
 in { GT: Call(newgt), AD: newad, DP: newdp, GQ: newgq, PL: newpl }
        """
    } else {
      // downcode
      s"""
g = let newgt = gtIndex(oldToNew[gtj(g.GT)], oldToNew[gtk(g.GT)]) and
    newad = if (isDefined(g.AD))
        range(newV.nAlleles).map(i => range(v.nAlleles).filter(j => oldToNew[j] == i).map(j => g.AD[j]).sum())
      else
        NA: Array[Int] and
    newdp = g.DP and
    newpl = if (isDefined(g.PL))
        range(newV.nGenotypes).map(gi => range(v.nGenotypes).filter(gj => gtIndex(oldToNew[gtj(gj)], oldToNew[gtk(gj)]) == gi).map(gj => g.PL[gj]).min())
      else
        NA: Array[Int] and
    newgq = gqFromPL(newpl)
 in { GT: Call(newgt), AD: newad, DP: newdp, GQ: newgq, PL: newpl }
        """
    }

    FilterAlleles(vsm, filterExpr, variantExpr, genotypeExpr,
      keep = keep, leftAligned = leftAligned, keepStar = keepStar)
  }

  def hardCalls(): MatrixTable = {
    vsm.annotateGenotypesExpr("g = {GT: g.GT}")
  }

  /**
    *
    * @param mafThreshold     Minimum minor allele frequency threshold
    * @param includePAR       Include pseudoautosomal regions
    * @param fFemaleThreshold Samples are called females if F < femaleThreshold
    * @param fMaleThreshold   Samples are called males if F > maleThreshold
    * @param popFreqExpr      Use an annotation expression for estimate of MAF rather than computing from the data
    */
  def imputeSex(mafThreshold: Double = 0.0, includePAR: Boolean = false, fFemaleThreshold: Double = 0.2,
    fMaleThreshold: Double = 0.8, popFreqExpr: Option[String] = None): MatrixTable = {
    ImputeSexPlink(vsm,
      mafThreshold,
      includePAR,
      fMaleThreshold,
      fFemaleThreshold,
      popFreqExpr)
  }

  def ldMatrix(forceLocal: Boolean = false): LDMatrix = {
    LDMatrix(vsm, Some(forceLocal))
  }

  def nirvana(config: String, blockSize: Int = 500000, root: String): MatrixTable = {
    Nirvana.annotate(vsm, config, blockSize, root)
  }

  /**
    *
    * @param config    VEP configuration file
    * @param root      Variant annotation path to store VEP output
    * @param csq       Annotates with the VCF CSQ field as a string, rather than the full nested struct schema
    * @param blockSize Variants per VEP invocation
    */
  def vep(config: String, root: String = "va.vep", csq: Boolean = false,
    blockSize: Int = 1000): MatrixTable = {
    VEP.annotate(vsm, config, root, csq, blockSize)
  }

  def filterIntervals(intervals: java.util.ArrayList[Interval[Locus]], keep: Boolean): MatrixTable = {
    implicit val locusOrd = vsm.genomeReference.locusOrdering
    val iList = IntervalTree[Locus](intervals.asScala.toArray)
    filterIntervals(iList, keep)
  }

  def filterIntervals[T, U](iList: IntervalTree[Locus, U], keep: Boolean): MatrixTable = {
    implicit val locusOrd = vsm.matrixType.locusType.ordering(missingGreatest = true)

    val ab = new ArrayBuilder[(Interval[Annotation], Annotation)]()
    iList.foreach { case (i, v) =>
      ab += (Interval[Annotation](i.start, i.end), v)
    }

    val iList2 = IntervalTree.annotationTree(ab.result())

    if (keep)
      vsm.copy(rdd = vsm.rdd.filterIntervals(iList2))
    else {
      val iListBc = vsm.sparkContext.broadcast(iList)
      vsm.filterVariants { (v, va, gs) => !iListBc.value.contains(v.asInstanceOf[Variant].locus) }
    }
  }
}
