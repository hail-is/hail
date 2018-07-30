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
  nSamples: Int,
  globalAnnotation: Annotation,
  matrixType: MatrixType,
  rowF: () => AsmFunction13[Region, Long, Boolean, Long, Boolean, Long, Boolean, Long, Boolean, Int, Boolean, Boolean, Boolean, Long],
  oldRVRowType: TStruct,
  newRVRowType: TStruct,
  ctx: RVDContext,
  sortAlleles: Boolean,
  removeLeftAligned: Boolean,
  removeMoving: Boolean,
  verifyLeftAligned: Boolean
) {

  val splitRegion = ctx.region
  val locusIndex = oldRVRowType.fieldIdx("locus")
  var fullRow = new UnsafeRow(matrixType.rvRowType)
  var prevLocus: Locus = null
  val rvv = new RegionValueVariant(matrixType.rvRowType)
  val rvb = new RegionValueBuilder()
  val splitrv = RegionValue()
  val locusAllelesOrdering = matrixType.rowKeyStruct.ordering

  private val allelesType = matrixType.rowType.fieldByName("alleles").typ
  private val locusType = matrixType.rowType.fieldByName("locus").typ
  val f = rowF()
  private[this] val partitionRegion = ctx.freshRegion

  rvb.set(partitionRegion)
  rvb.start(matrixType.globalType)
  rvb.addAnnotation(matrixType.globalType, globalAnnotation)
  private[this] val partitionWideGlobals = rvb.end()

  def constructSplitRow(splitVariants: Iterator[(Locus, IndexedSeq[String], Int)], rv: RegionValue, wasSplit: Boolean): Iterator[RegionValue] = {
    splitVariants.map { case (newLocus, newAlleles, aIndex) =>
      rvb.set(splitRegion)
      rvb.start(matrixType.globalType)
      rvb.addRegionValue(matrixType.globalType, partitionRegion, partitionWideGlobals)
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

  def splitRow(rv: RegionValue): Iterator[RegionValue] = {
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

  def splitPartition(it: Iterator[RegionValue]): Iterator[RegionValue] =
    it.flatMap { rv =>
      val it = splitRow(rv)
      prevLocus = fullRow.getAs[Locus](locusIndex)
      it
    }
}

object SplitMulti {

  def apply(vsm: MatrixTable, variantExpr: String, genotypeExpr: String, keepStar: Boolean = false, leftAligned: Boolean = false): MatrixTable = {
    val refMap = Map(
      "global" -> vsm.globalType,
      "newLocus" -> vsm.rowKeyStruct.types(0),
      "newAlleles" -> vsm.rowKeyStruct.types(1),
      "va" -> vsm.rvRowType,
      "aIndex" -> TInt32(),
      "wasSplit" -> TBoolean(),
      "g" -> vsm.entryType)

    val variantIRs = Parser.parse_named_value_irs(variantExpr, refMap)
    val genotypeIRs = Parser.parse_named_value_irs(genotypeExpr, refMap)
    val splitmulti = new SplitMulti(vsm, variantIRs, genotypeIRs, keepStar, leftAligned)
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

    ordered.copy(partitioner = movedRVD.partitioner).partitionSortedUnion(movedRVD)
  }
}

class SplitMulti(vsm: MatrixTable, rowIRs: Array[(String, IR)], entryIRs: Array[(String, IR)], keepStar: Boolean, leftAligned: Boolean) {

  val (newMatrixType, rowF): (MatrixType, () => AsmFunction13[Region, Long, Boolean, Long, Boolean, Long, Boolean, Long, Boolean, Int, Boolean, Boolean, Boolean, Long]) = {
    val ir = new SplitMultiRowIR(rowIRs, entryIRs, vsm.matrixType)
    (ir.newMatrixType, ir.splitRow)
  }

  def split(sortAlleles: Boolean, removeLeftAligned: Boolean, removeMoving: Boolean, verifyLeftAligned: Boolean): RVD = {
    val localKeepStar = keepStar
    val globalsBc = vsm.globals.broadcast
    val localNSamples = vsm.numCols
    val oldRVRowType = vsm.rvRowType
    val localMatrixType = vsm.matrixType
    val localSplitRow = rowF

    val newRowType = newMatrixType.rvRowType

    val makeContext = {
      (ctx: RVDContext) => new SplitMultiPartitionContextIR(localKeepStar, localNSamples, globalsBc.value,
        localMatrixType, localSplitRow, oldRVRowType, newRowType, ctx, sortAlleles, removeLeftAligned, removeMoving, verifyLeftAligned)
    }

    vsm.rvd.boundary.mapPartitions(newRowType, { (ctx, it) =>
      makeContext(ctx).splitPartition(it)
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
