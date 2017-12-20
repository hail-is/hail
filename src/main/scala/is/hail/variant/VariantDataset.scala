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

  def annotateAllelesExpr(expr: String): MatrixTable = {
    if (!vsm.genotypeSignature.isOfType(Genotype.htsGenotypeType))
      fatal(s"annotate_alleles: genotype_schema must be the HTS genotype schema, found: ${ vsm.genotypeSignature }")

    val splitVariantExpr = "va.aIndex = aIndex, va.wasSplit = wasSplit"
    val splitGenotypeExpr =
        """
          g = let
            newgt = downcode(g.GT, aIndex) and
            newad = if (isDefined(g.AD))
                let sum = g.AD.sum() and adi = g.AD[aIndex] in [sum - adi, adi]
              else
                NA: Array[Int] and
            newpl = if (isDefined(g.PL))
                range(3).map(i => range(g.PL.length).filter(j => downcode(Call(j), aIndex) == Call(i)).map(j => g.PL[j]).min())
              else
                NA: Array[Int] and
            newgq = gqFromPL(newpl)
          in { GT: newgt, AD: newad, DP: g.DP, GQ: newgq, PL: newpl }"""

    annotateAllelesExprGeneric(splitVariantExpr, splitGenotypeExpr, variantExpr = expr)
  }

  def annotateAllelesExprGeneric(splitVariantExpr: String, splitGenotypeExpr: String, variantExpr: String): MatrixTable = {

    val splitmulti = new SplitMulti(vsm, splitVariantExpr, splitGenotypeExpr,
      keepStar = true, leftAligned = false)

    val splitMatrixType = splitmulti.newMatrixType

    val aggregationST = Map(
      "global" -> (0, vsm.globalSignature),
      "v" -> (1, vsm.vSignature),
      "va" -> (2, splitMatrixType.vaType),
      "g" -> (3, splitMatrixType.genotypeType),
      "s" -> (4, TString()),
      "sa" -> (5, vsm.saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, vsm.globalSignature),
      "v" -> (1, vsm.vSignature),
      "va" -> (2, splitMatrixType.vaType),
      "gs" -> (3, TAggregable(splitMatrixType.genotypeType, aggregationST))))

    val (paths, types, f) = Parser.parseAnnotationExprs(variantExpr, ec, Some(Annotation.VARIANT_HEAD))

    val inserterBuilder = new ArrayBuilder[Inserter]()
    val newType = (paths, types).zipped.foldLeft(vsm.vaSignature) { case (vas, (ids, signature)) =>
      val (s, i) = vas.insert(TArray(signature), ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(vsm.sparkContext, splitMatrixType, vsm.value.localValue, ec)

    val localNSamples = vsm.nSamples
    val localRowType = vsm.rowType

    val localGlobalAnnotation = vsm.globalAnnotation
    val localVAnnotator = splitmulti.vAnnotator
    val localGAnnotator = splitmulti.gAnnotator
    val splitRowType = splitMatrixType.rowType

    val newMatrixType = vsm.matrixType.copy(vaType = newType)
    val newRowType = newMatrixType.rowType

    val newRDD2 = OrderedRVD(
      newMatrixType.orderedRVType,
      vsm.rdd2.partitioner,
      vsm.rdd2.mapPartitions { it =>
        val splitcontext = new SplitMultiPartitionContext(true, localNSamples, localGlobalAnnotation, localRowType,
          localVAnnotator, localGAnnotator, splitRowType)
        val rv2b = new RegionValueBuilder()
        val rv2 = RegionValue()
        it.map { rv =>
          val annotations = splitcontext.splitRow(rv,
            sortAlleles = false, removeLeftAligned = false, removeMoving = false, verifyLeftAligned = false)
            .map { splitrv =>
              val splitur = new UnsafeRow(splitRowType, splitrv)
              val v = splitur.getAs[Variant](1)
              val va = splitur.get(2)
              ec.setAll(localGlobalAnnotation, v, va)
              aggregateOption.foreach(f => f(splitrv))
              (f(), types).zipped.map { case (a, t) =>
                Annotation.copy(t, a)
              }
            }
            .toArray

          rv2b.set(rv.region)
          rv2b.start(newRowType)
          rv2b.startStruct()

          rv2b.addField(localRowType, rv, 0) // pk
          rv2b.addField(localRowType, rv, 1) // v

          val ur = new UnsafeRow(localRowType, rv.region, rv.offset)
          val va = ur.get(2)
          val newVA = inserters.zipWithIndex.foldLeft(va) { case (va, (inserter, i)) =>
            inserter(va, annotations.map(_ (i)): IndexedSeq[Any])
          }
          rv2b.addAnnotation(newType, newVA)

          rv2b.addField(localRowType, rv, 3) // gs
          rv2b.endStruct()

          rv2.set(rv.region, rv2b.end())
          rv2
        }
      })

    vsm.copy2(rdd2 = newRDD2, vaSignature = newType)
  }

  def concordance(other: MatrixTable): (IndexedSeq[IndexedSeq[Long]], Table, Table) = {
    require(vsm.wasSplit)
    require(other.wasSplit)

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

  def filterAllelesGeneric(filterExpr: String, variantExpr: String, genotypeExpr: String,
    keep: Boolean = true, leftAligned: Boolean = false, keepStar: Boolean = false): MatrixTable = {
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
    require(vsm.wasSplit)

    ImputeSexPlink(vsm,
      mafThreshold,
      includePAR,
      fMaleThreshold,
      fFemaleThreshold,
      popFreqExpr)
  }

  def ldMatrix(forceLocal: Boolean = false): LDMatrix = {
    require(vsm.wasSplit)
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

  /**
    * Remove multiallelic variants from this dataset.
    *
    * Useful for running methods that require biallelic variants without calling the more expensive split_multi step.
    */
  def filterMulti(): MatrixTable = {
    if (vsm.wasSplit) {
      warn("called redundant `filter_multi' on an already split or multiallelic-filtered VDS")
      vsm
    } else {
      vsm.filterVariants {
        case (v, va, gs) => v.asInstanceOf[Variant].isBiallelic
      }.copy2(wasSplit = true)
    }
  }

  def verifyBiallelic(): MatrixTable =
    verifyBiallelic("verifyBiallelic")

  def verifyBiallelic(method: String): MatrixTable = {
    if (vsm.wasSplit) {
      warn("called redundant `$method' on biallelic VDS")
      vsm
    } else {
      val localRowType = vsm.rowType
      vsm.copy2(
        rdd2 = vsm.rdd2.mapPreservesPartitioning(vsm.rdd2.typ) { rv =>
          val ur = new UnsafeRow(localRowType, rv.region, rv.offset)
          val v = ur.getAs[Variant](1)
          if (!v.isBiallelic)
            fatal("in $method: found non-biallelic variant: $v")
          rv
        },
        wasSplit = true)
    }
  }

  def splitMulti(keepStar: Boolean = false, leftAligned: Boolean = false): MatrixTable = {
    if (!vsm.genotypeSignature.isOfType(Genotype.htsGenotypeType))
      fatal(s"split_multi: genotype_schema must be the HTS genotype schema, found: ${ vsm.genotypeSignature }")

    vsm.splitMultiGeneric("va.aIndex = aIndex, va.wasSplit = wasSplit",
      s"""g =
    let
      newgt = downcode(g.GT, aIndex) and
      newad = if (isDefined(g.AD))
          let sum = g.AD.sum() and adi = g.AD[aIndex] in [sum - adi, adi]
        else
          NA: Array[Int] and
      newpl = if (isDefined(g.PL))
          range(3).map(i => range(g.PL.length).filter(j => downcode(Call(j), aIndex) == Call(i)).map(j => g.PL[j]).min())
        else
          NA: Array[Int] and
      newgq = gqFromPL(newpl)
    in { GT: newgt, AD: newad, DP: g.DP, GQ: newgq, PL: newpl }""",
      keepStar, leftAligned)
  }

  def splitMultiGeneric(variantExpr: String, genotypeExpr: String, keepStar: Boolean = false, leftAligned: Boolean = false): MatrixTable = {
    val splitmulti = new SplitMulti(vsm, variantExpr, genotypeExpr, keepStar, leftAligned)
    splitmulti.split()
  }
}
