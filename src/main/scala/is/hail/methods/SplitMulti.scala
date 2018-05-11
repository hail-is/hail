package is.hail.methods

import is.hail.annotations._
import is.hail.asm4s.AsmFunction13
import is.hail.expr._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.rvd.{OrderedRVD, RVD, RVDContext}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row

class ExprAnnotator(val ec: EvalContext, t: TStruct, expr: String, head: Option[String]) extends Serializable {
  private val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, head)

  private val inserters = new Array[Inserter](types.length)
  val newT: TStruct = {
    var newT = t
    var i = 0
    while (i < types.length) {
      val (newSig, ins) = newT.structInsert(types(i), paths(i))
      inserters(i) = ins
      newT = newSig
      i += 1
    }
    newT
  }

  def insert(a: Annotation): Annotation = {
    var newA = a
    var i = 0
    val xs = f()
    while (i < xs.length) {
      newA = inserters(i)(newA, xs(i))
      i += 1
    }
    newA
  }
}

class SplitMultiRowIR(rowIRs: Array[(String, IR)], entryIRs: Array[(String, IR)], oldMatrixType: MatrixType) {
  val oldRowIR = Ref("va", oldMatrixType.rvRowType)
  val newEntries = ArrayMap(GetField(In(3, oldMatrixType.rvRowType), MatrixType.entriesIdentifier), "g", InsertFields(Ref("g", oldMatrixType.entryType), entryIRs))
  val changedFields: Seq[(String, IR)] =
    (rowIRs
      :+ ("locus", Ref("newLocus", oldMatrixType.rowType.fieldByName("locus").typ))
      :+ ("alleles", Ref("newAlleles", oldMatrixType.rowType.fieldByName("alleles").typ))
      :+ (MatrixType.entriesIdentifier, newEntries))

  val newRowIR: IR = InsertFields(oldRowIR, changedFields)

  val (t, splitRow): (Type, () => AsmFunction13[Region, Long, Boolean, Long, Boolean, Long, Boolean, Long, Boolean, Int, Boolean, Boolean, Boolean, Long]) =
    Compile[Long, Long, Long, Long, Int, Boolean, Long](
      "global", oldMatrixType.globalType,
      "newLocus", oldMatrixType.rowType.fieldByName("locus").typ,
      "newAlleles", oldMatrixType.rowType.fieldByName("alleles").typ,
      "va", oldMatrixType.rvRowType,
      "aIndex", TInt32(),
      "wasSplit", TBoolean(),
      newRowIR)

  val newMatrixType: MatrixType = oldMatrixType.copy(rvRowType = coerce[TStruct](t))
}

class SplitMultiPartitionContextIR(
  keepStar: Boolean,
  nSamples: Int, globalAnnotation: Annotation, matrixType: MatrixType,
  rowF: () => AsmFunction13[Region, Long, Boolean, Long, Boolean, Long, Boolean, Long, Boolean, Int, Boolean, Boolean, Boolean, Long],
  newRVRowType: TStruct,
  region: Region
) extends
  SplitMultiPartitionContext(keepStar, nSamples, globalAnnotation, matrixType, newRVRowType, region) {

  private val allelesType = matrixType.rowType.fieldByName("alleles").typ
  private val locusType = matrixType.rowType.fieldByName("locus").typ
  val f = rowF()

  def constructSplitRow(splitVariants: Iterator[(Locus, IndexedSeq[String], Int)], rv: RegionValue, wasSplit: Boolean): Iterator[RegionValue] = {
    splitVariants.map { case (newLocus, newAlleles, aIndex) =>
      rvb.set(splitRegion)
      rvb.start(matrixType.globalType)
      rvb.addAnnotation(matrixType.globalType, globalAnnotation)
      val globals = rvb.end()

      rvb.start(matrixType.rvRowType)
      rvb.addRegionValue(matrixType.rvRowType, rv)
      val oldRow = rvb.end()

      rvb.start(locusType)
      rvb.addAnnotation(locusType, newLocus)
      val locusOff = rvb.end()

      rvb.start(allelesType)
      rvb.addAnnotation(allelesType, newAlleles)
      val allelesOff = rvb.end()

      val off = f(splitRegion, globals, false, locusOff, false, allelesOff, false, oldRow, false, aIndex, false, wasSplit, false)
      splitrv.set(splitRegion, off)
      splitrv
    }
  }
}

class SplitMultiPartitionContextAST(
  keepStar: Boolean,
  nSamples: Int,
  globalAnnotation: Annotation,
  matrixType: MatrixType,
  vAnnotator: ExprAnnotator,
  gAnnotator: ExprAnnotator,
  newRVRowType: TStruct,
  region: Region
) extends
  SplitMultiPartitionContext(keepStar, nSamples, globalAnnotation, matrixType, newRVRowType, region) {

  val (t1, locusInserter) = vAnnotator.newT.insert(matrixType.rowType.fieldByName("locus").typ, "locus")
  assert(t1 == vAnnotator.newT)
  val (t2, allelesInserter) = vAnnotator.newT.insert(matrixType.rowType.fieldByName("alleles").typ, "alleles")
  assert(t2 == vAnnotator.newT)

  def constructSplitRow(splitVariants: Iterator[(Locus, IndexedSeq[String], Int)], rv: RegionValue, wasSplit: Boolean): Iterator[RegionValue] = {
    val gs = fullRow.getAs[IndexedSeq[Any]](matrixType.entriesIdx)
    splitVariants.map { case (newLocus, newAlleles, i) =>
      rvb.set(splitRegion)
      rvb.start(newRVRowType)
      rvb.startStruct()

      vAnnotator.ec.setAll(globalAnnotation, newLocus, newAlleles, fullRow, i, wasSplit)
      val newRow = allelesInserter(
        locusInserter(
          vAnnotator.insert(fullRow),
          newLocus),
        newAlleles
      ).asInstanceOf[Row]
      var fdIdx = 0
      while (fdIdx < newRow.length) {
        rvb.addAnnotation(newRVRowType.types(fdIdx), newRow(fdIdx))
        fdIdx += 1
      }

      rvb.startArray(nSamples) // gs
      gAnnotator.ec.setAll(globalAnnotation, newLocus, newAlleles, fullRow, i, wasSplit)
      var k = 0
      while (k < nSamples) {
        val g = gs(k)
        gAnnotator.ec.set(6, g)
        rvb.addAnnotation(gAnnotator.newT, gAnnotator.insert(g))
        k += 1
      }
      rvb.endArray() // gs
      rvb.endStruct()
      splitrv.set(splitRegion, rvb.end())
      splitrv
    }
  }
}


abstract class SplitMultiPartitionContext(
  keepStar: Boolean,
  nSamples: Int,
  globalAnnotation: Annotation,
  matrixType: MatrixType,
  newRVRowType: TStruct,
  val splitRegion: Region
) extends Serializable {

  var fullRow = new UnsafeRow(matrixType.rvRowType)
  var prevLocus: Locus = null
  val rvv = new RegionValueVariant(matrixType.rvRowType)
  val rvb = new RegionValueBuilder()
  val splitrv = RegionValue()
  val locusAllelesOrdering = matrixType.rowKeyStruct.ordering

  def constructSplitRow(splitVariants: Iterator[(Locus, IndexedSeq[String], Int)], rv: RegionValue, wasSplit: Boolean): Iterator[RegionValue]

  def splitRow(rv: RegionValue, sortAlleles: Boolean, removeLeftAligned: Boolean, removeMoving: Boolean, verifyLeftAligned: Boolean): Iterator[RegionValue] = {
    require(!(removeMoving && verifyLeftAligned))
    fullRow.set(rv)
    rvv.setRegion(rv)

    var isLeftAligned = true
    if (prevLocus != null && prevLocus == rvv.locus)
      isLeftAligned = false
    val ref = rvv.alleles()(0)
    val alts = rvv.alleles().tail
    var splitVariants = alts.iterator.zipWithIndex
      .filter(keepStar || _._1 != "*")
      .map { case (aa, aai) =>
        val splitLocus = rvv.locus()
        val splitAlleles: IndexedSeq[String] = Array(ref, aa)

        val (minLocus, minAlleles) = VariantMethods.minRep(splitLocus, splitAlleles)

        if (splitLocus != minLocus)
          isLeftAligned = false

        (minLocus, minAlleles, aai + 1)
      }.toArray

    if (splitVariants.isEmpty)
      return Iterator()

    val wasSplit = alts.length > 1

    if (isLeftAligned) {
      if (removeLeftAligned)
        return Iterator()
    } else {
      if (removeMoving)
        return Iterator()
      else if (verifyLeftAligned)
        fatal(s"found non-left aligned variant: ${ rvv.locus() }:$ref:${ alts.mkString(",") } ")
    }

    if (sortAlleles)
      splitVariants = splitVariants.sortBy { case (newLocus, newAlleles, i) => Annotation(newLocus, newAlleles) }(locusAllelesOrdering.toOrdering)

    val nAlleles = 1 + alts.length
    val nGenotypes = VariantMethods.nGenotypes(nAlleles)
    constructSplitRow(splitVariants.iterator, rv, wasSplit)
  }
}

object SplitMulti {

  def apply(vsm: MatrixTable, variantExpr: String, genotypeExpr: String, keepStar: Boolean = false, leftAligned: Boolean = false): MatrixTable = {
    val splitmulti = new SplitMulti(vsm, variantExpr, genotypeExpr, keepStar, leftAligned)
    splitmulti.split()
  }

  def unionMovedVariants(
    ordered: OrderedRVD,
    moved: RVD
  ): OrderedRVD = {
    val movedRVD = OrderedRVD.adjustBoundsAndShuffle(
      ordered.typ,
      ordered.partitioner,
      moved)

    ordered.copy(orderedPartitioner = movedRVD.partitioner).partitionSortedUnion(movedRVD)
  }
}

class SplitMulti(vsm: MatrixTable, variantExpr: String, genotypeExpr: String, keepStar: Boolean, leftAligned: Boolean) {
  val vEC = EvalContext(Map(
    "global" -> (0, vsm.globalType),
    "newLocus" -> (1, vsm.rowKeyStruct.types(0)),
    "newAlleles" -> (2, vsm.rowKeyStruct.types(1)),
    "va" -> (3, vsm.rvRowType),
    "aIndex" -> (4, TInt32()),
    "wasSplit" -> (5, TBoolean())))
  val vAnnotator = new ExprAnnotator(vEC, vsm.rowType, variantExpr, Some(Annotation.ROW_HEAD))

  val gEC = EvalContext(Map(
    "global" -> (0, vsm.globalType),
    "newLocus" -> (1, vsm.rowKeyStruct.types(0)),
    "newAlleles" -> (2, vsm.rowKeyStruct.types(1)),
    "va" -> (3, vsm.rvRowType),
    "aIndex" -> (4, TInt32()),
    "wasSplit" -> (5, TBoolean()),
    "g" -> (6, vsm.entryType)))
  val gAnnotator = new ExprAnnotator(gEC, vsm.entryType, genotypeExpr, Some(Annotation.ENTRY_HEAD))

  val rowASTs = Parser.parseAnnotationExprsToAST(variantExpr, vEC, Some("va"))
  val entryASTs = Parser.parseAnnotationExprsToAST(genotypeExpr, gEC, Some("g"))

  val rowIRs = rowASTs.flatMap { case (name, ast) =>
    for (ir <- ast.toIROpt()) yield {
      (name, ir)
    }
  }
  val entryIRs = entryASTs.flatMap { case (name, ast) =>
    for (ir <- ast.toIROpt()) yield {
      (name, ir)
    }
  }

  val (newMatrixType, useAST, rowF): (MatrixType, Boolean, () => AsmFunction13[Region, Long, Boolean, Long, Boolean, Long, Boolean, Long, Boolean, Int, Boolean, Boolean, Boolean, Long]) =
    if (rowASTs.length == rowIRs.length && entryASTs.length == entryIRs.length) {
      val ir = new SplitMultiRowIR(rowIRs, entryIRs, vsm.matrixType)
      (ir.newMatrixType, false, ir.splitRow)
    } else {
      val t = vsm.matrixType.copyParts(rowType = vAnnotator.newT, entryType = gAnnotator.newT)
      (t, true, null)
    }

  def split(sortAlleles: Boolean, removeLeftAligned: Boolean, removeMoving: Boolean, verifyLeftAligned: Boolean): RVD = {
    val localKeepStar = keepStar
    val globalsBc = vsm.globals.broadcast
    val localNSamples = vsm.numCols
    val localRowType = vsm.rvRowType
    val localMatrixType = vsm.matrixType
    val localVAnnotator = vAnnotator
    val localGAnnotator = gAnnotator
    val localSplitRow = rowF

    val newRowType = newMatrixType.rvRowType

    val locusIndex = localRowType.fieldIdx("locus")

    val makeContext = if (useAST) {
      (region: Region) => new SplitMultiPartitionContextAST(localKeepStar, localNSamples, globalsBc.value,
        localMatrixType, localVAnnotator, localGAnnotator, newRowType, region)
    } else {
      (region: Region) => new SplitMultiPartitionContextIR(localKeepStar, localNSamples, globalsBc.value,
        localMatrixType, localSplitRow, newRowType, region)
    }

    vsm.rvd.boundary.mapPartitions(newRowType, { (ctx, it) =>
      val splitMultiContext = makeContext(ctx.region)
      it.flatMap { rv =>
        val splitit = splitMultiContext.splitRow(rv, sortAlleles, removeLeftAligned, removeMoving, verifyLeftAligned)
        splitMultiContext.prevLocus = splitMultiContext.fullRow.getAs[Locus](locusIndex)
        splitit
      }
    })
  }

  def split(): MatrixTable = {
    val newRDD2: OrderedRVD =
      if (leftAligned)
        OrderedRVD(
          newMatrixType.orvdType,
          vsm.rvd.partitioner,
          split(sortAlleles = true, removeLeftAligned = false, removeMoving = false, verifyLeftAligned = true))
      else
        SplitMulti.unionMovedVariants(OrderedRVD(
          newMatrixType.orvdType,
          vsm.rvd.partitioner,
          split(sortAlleles = true, removeLeftAligned = false, removeMoving = true, verifyLeftAligned = false)),
          split(sortAlleles = false, removeLeftAligned = true, removeMoving = false, verifyLeftAligned = false))

    vsm.copyMT(rvd = newRDD2, matrixType = newMatrixType)
  }
}
