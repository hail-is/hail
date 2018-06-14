package is.hail.testUtils

import is.hail.annotations._
import is.hail.expr.ir
import is.hail.expr.types._
import is.hail.expr.{EvalContext, Parser}
import is.hail.methods._
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant.{Locus, MatrixTable}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class RichMatrixTable(vsm: MatrixTable) {
  def expand(): RDD[(Annotation, Annotation, Annotation)] =
    mapWithKeys[(Annotation, Annotation, Annotation)]((v, s, g) => (v, s, g))

  def mapWithKeys[U](f: (Annotation, Annotation, Annotation) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = vsm.sparkContext.broadcast(vsm.stringSampleIds)

    rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith[Annotation, U](gs,
          (s, g) => f(v, s, g))
      }
  }

  def annotateSamplesF(signature: Type, path: List[String], annotation: (Annotation) => Annotation): MatrixTable = {
    val (t, i) = vsm.insertSA(signature, path)
    val sampleIds = vsm.stringSampleIds
    vsm.annotateCols(t, i) { case (_, i) => annotation(sampleIds(i)) }
  }

  def annotateColsExpr(exprs: (String, String)*): MatrixTable =
    vsm.selectCols(s"annotate(sa, {${ exprs.map { case (n, e) => s"`$n`: $e" }.mkString(",") }})", None)

  def annotateCols(annotations: Map[Annotation, Annotation], signature: Type, root: String): MatrixTable = {
    val (t, i) = vsm.insertSA(signature, List(root))
    vsm.annotateCols(t, i) { case (s, _) => annotations.getOrElse(s, null) }
  }

  def annotateRowsExpr(exprs: (String, String)*): MatrixTable =
    vsm.selectRows(s"annotate(va, {${ exprs.map { case (n, e) => s"`$n`: $e" }.mkString(",") }})", None)

  def annotateEntriesExpr(exprs: (String, String)*): MatrixTable =
    vsm.selectEntries(s"annotate(g, {${ exprs.map { case (n, e) => s"`$n`: $e" }.mkString(",") }})")

  def querySA(code: String): (Type, Querier) =
    vsm.query(code, Annotation.COL_HEAD, vsm.colType)

  def queryGA(code: String): (Type, Querier) =
    vsm.query(code, Annotation.ENTRY_HEAD, vsm.entryType)


  def queryVA(code: String): (Type, Querier) =
    query(code, Annotation.ROW_HEAD, vsm.rowType)

  def queryGlobal(path: String): (Type, Annotation) = {
    val (t, q) = query(path, Annotation.GLOBAL_HEAD, vsm.globalType)
    (t, q(vsm.globals.value))
  }

  def query(code: String, id: String, t: Type): (Type, Querier) = {
    val ec = EvalContext(Map(id -> (0, t)))
    val ast = Parser.parseToAST(code, ec)
    ast.toIROpt() match {
      case Some(x) =>
        val (resultType, f) = ir.Compile[Long, Long](id, t,
          ir.MakeTuple(FastSeq(x)))
        val f2 = (a: Annotation) => Region.scoped { region =>
          val argsOff =
            if (a != null) {
              val rvb = new RegionValueBuilder(region)
              rvb.start(t)
              rvb.addAnnotation(t, a)
              rvb.end()
            } else
              0
          val argsMissing = a == null
          val resultOff = f()(region, argsOff, argsMissing)
          SafeRow(resultType.asInstanceOf[TBaseStruct], region, resultOff).get(0)
        }
        (x.typ, f2)
    }
  }

  def stringSampleIdsAndAnnotations: IndexedSeq[(Annotation, Annotation)] = vsm.stringSampleIds.zip(vsm.colValues.value)

  def rdd: RDD[(Annotation, (Annotation, Iterable[Annotation]))] = {
    val fullRowType = vsm.rvRowType
    val localEntriesIndex = vsm.entriesIndex
    val localRowType = vsm.rowType
    val rowKeyF = vsm.rowKeysF
    vsm.rvd.map { rv =>
      val unsafeFullRow = new UnsafeRow(fullRowType, rv)
      val fullRow = SafeRow(fullRowType, rv.region, rv.offset)
      val row = fullRow.deleteField(localEntriesIndex)
      (rowKeyF(fullRow), (row, fullRow.getAs[IndexedSeq[Any]](localEntriesIndex)))
    }
  }

  def variantRDD: RDD[(Variant, (Annotation, Iterable[Annotation]))] =
    rdd.map { case (v, (va, gs)) =>
      Variant.fromLocusAlleles(v) -> (va, gs)
    }

  def typedRDD[RK](implicit rkct: ClassTag[RK]): RDD[(RK, (Annotation, Iterable[Annotation]))] =
    rdd.map { case (v, (va, gs)) =>
      (v.asInstanceOf[RK], (va, gs))
    }

  def variants: RDD[Variant] = variantRDD.keys

  def locusAlleles: RDD[(Locus, IndexedSeq[String])] =
    variants.map { v =>
      (v.locus, v.alleles)
    }

  def variantsAndAnnotations: RDD[(Variant, Annotation)] =
    variantRDD.map { case (v, (va, gs)) => (v, va) }

  def reorderCols(newIds: Array[Annotation]): MatrixTable = {
    require(newIds.length == vsm.numCols)
    require(newIds.areDistinct())

    val sampleSet = vsm.colKeys.toSet[Annotation]
    val newSampleSet = newIds.toSet

    val notInDataset = newSampleSet -- sampleSet
    if (notInDataset.nonEmpty)
      fatal(s"Found ${ notInDataset.size } ${ plural(notInDataset.size, "sample ID") } in new ordering that are not in dataset:\n  " +
        s"@1", notInDataset.truncatable("\n  "))

    val oldIndex = vsm.colKeys.zipWithIndex.toMap
    val newToOld = newIds.map(oldIndex)

    vsm.chooseCols(newToOld)
  }

  def linreg(yExpr: Array[String],
    xField: String,
    covExpr: Array[String] = Array.empty[String],
    root: String = "linreg",
    rowBlockSize: Int = 16): MatrixTable = {
    val vsmAnnot = vsm.annotateColsExpr(
      yExpr.zipWithIndex.map { case (e, i) => s"__y$i" -> e } ++
        covExpr.zipWithIndex.map { case (e, i) => s"__cov$i" -> e }: _*
    )
    LinearRegression(vsmAnnot,
      yExpr.indices.map(i => s"__y$i").toArray,
      xField,
      covExpr.indices.map(i => s"__cov$i").toArray,
      root,
      rowBlockSize)
  }

  def lmmreg(kinshipMatrix: KinshipMatrix,
    yExpr: String,
    xField: String,
    covExpr: Array[String] = Array.empty[String],
    useML: Boolean = false,
    rootGA: String = "lmmreg",
    rootVA: String = "lmmreg",
    runAssoc: Boolean = true,
    delta: Option[Double] = None,
    sparsityThreshold: Double = 1.0,
    nEigs: Option[Int] = None,
    optDroppedVarianceFraction: Option[Double] = None): MatrixTable = {
    LinearMixedRegression(vsm, kinshipMatrix, yExpr, xField, covExpr, useML, rootGA, rootVA,
      runAssoc, delta, sparsityThreshold, nEigs, optDroppedVarianceFraction)
  }

  def skat(keyExpr: String,
    weightExpr: String,
    yExpr: String,
    xField: String,
    covExpr: Array[String] = Array.empty[String],
    logistic: Boolean = false,
    maxSize: Int = 46340, // floor(sqrt(Int.MaxValue))
    accuracy: Double = 1e-6,
    iterations: Int = 10000): Table = {
    Skat(vsm, keyExpr, weightExpr, yExpr, xField, covExpr, logistic, maxSize, accuracy, iterations)
  }
}
