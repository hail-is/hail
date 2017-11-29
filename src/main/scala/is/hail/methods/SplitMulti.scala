package is.hail.methods

import is.hail.annotations._
import is.hail.expr._
import is.hail.sparkextras.OrderedRDD2
import is.hail.utils._
import is.hail.variant.{Locus, Variant, VariantSampleMatrix}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class ExprAnnotator(val ec: EvalContext, t: Type, expr: String, head: Option[String]) extends Serializable {
  private val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, head)

  private val inserters = new Array[Inserter](types.length)
  val newT: Type = {
    var newT = t
    var i = 0
    while (i < types.length) {
      val (newSig, ins) = newT.insert(types(i), paths(i))
      inserters(i) = ins
      newT = newSig
      i += 1
    }
    newT
  }

  def insert(a: Annotation): Annotation = {
    var newA = a
    var i = 0
    var xs = f()
    while (i < xs.length) {
      newA = inserters(i)(newA, xs(i))
      i += 1
    }
    newA
  }
}

class SplitMultiPartitionContext(
  keepStar: Boolean,
  nSamples: Int, globalAnnotation: Annotation, rowType: TStruct,
  vAnnotator: ExprAnnotator, gAnnotator: ExprAnnotator, newRowType: TStruct) {
  var prevLocus: Locus = null
  var ur = new UnsafeRow(rowType)
  val splitRegion = MemoryBuffer()
  val rvb = new RegionValueBuilder()
  val splitrv = RegionValue()

  def splitRow(rv: RegionValue, sortAlleles: Boolean, removeLeftAligned: Boolean, removeMoving: Boolean, verifyLeftAligned: Boolean): Iterator[RegionValue] = {
    require(!(removeMoving && verifyLeftAligned))

    ur.set(rv)
    val v = ur.getAs[Variant](1)

    var isLeftAligned = true
    if (prevLocus != null && prevLocus == v.locus)
      isLeftAligned = false
    var splitVariants = v.altAlleles.iterator.zipWithIndex
      .filter(keepStar || !_._1.isStar)
      .map { case (aa, aai) =>
        val splitv = Variant(v.contig, v.start, v.ref, Array(aa))
        val minsplitv = splitv.minRep

        if (splitv.locus != minsplitv.locus)
          isLeftAligned = false

        (minsplitv, aai + 1)
      }
      .toArray

    if (splitVariants.isEmpty)
      return Iterator()

    val wasSplit = !v.isBiallelic

    if (isLeftAligned) {
      if (removeLeftAligned)
        return Iterator()
    } else {
      if (removeMoving)
        return Iterator()
      else if (verifyLeftAligned)
        fatal("found non-left aligned variant: $v")
    }

    val va = ur.get(2)

    if (sortAlleles)
      splitVariants = splitVariants.sortBy { case (svj, i) => svj }

    val nAlleles = v.nAlleles
    val nGenotypes = v.nGenotypes

    val gs = ur.getAs[IndexedSeq[Any]](3)
    splitVariants.iterator
      .map { case (svj, i) =>
        splitRegion.clear()
        rvb.set(splitRegion)
        rvb.start(newRowType)
        rvb.startStruct()
        rvb.addAnnotation(newRowType.fieldType(0), svj.locus)
        rvb.addAnnotation(newRowType.fieldType(1), svj)

        vAnnotator.ec.setAll(globalAnnotation, v, svj, va, i, wasSplit)
        rvb.addAnnotation(vAnnotator.newT, vAnnotator.insert(va))

        rvb.startArray(nSamples) // gs
        gAnnotator.ec.setAll(globalAnnotation, v, svj, va, i, wasSplit)
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

object SplitMulti {
  def unionMovedVariants(ordered: OrderedRDD2,
    moved: RDD[RegionValue]): OrderedRDD2 = {
    ordered.partitionSortedUnion(OrderedRDD2.shuffle(ordered.typ,
      ordered.orderedPartitioner,
      moved))
  }
}

class SplitMulti(vsm: VariantSampleMatrix, variantExpr: String, genotypeExpr: String, keepStar: Boolean, leftAligned: Boolean) {
  val vEC = EvalContext(Map(
    "global" -> (0, vsm.globalSignature),
    "v" -> (1, vsm.vSignature),
    "newV" -> (2, vsm.vSignature),
    "va" -> (3, vsm.vaSignature),
    "aIndex" -> (4, TInt32()),
    "wasSplit" -> (5, TBoolean())))
  val vAnnotator = new ExprAnnotator(vEC, vsm.vaSignature, variantExpr, Some(Annotation.VARIANT_HEAD))

  val gEC = EvalContext(Map(
    "global" -> (0, vsm.globalSignature),
    "v" -> (1, vsm.vSignature),
    "newV" -> (2, vsm.vSignature),
    "va" -> (3, vsm.vaSignature),
    "aIndex" -> (4, TInt32()),
    "wasSplit" -> (5, TBoolean()),
    "g" -> (6, vsm.genotypeSignature)))
  val gAnnotator = new ExprAnnotator(gEC, vsm.genotypeSignature, genotypeExpr, Some(Annotation.GENOTYPE_HEAD))

  val newMatrixType = vsm.matrixType.copy(vaType = vAnnotator.newT, genotypeType = gAnnotator.newT)

  def split(sortAlleles: Boolean, removeLeftAligned: Boolean, removeMoving: Boolean, verifyLeftAligned: Boolean): RDD[RegionValue] = {
    val localKeepStar = keepStar
    val localGlobalAnnotation = vsm.globalAnnotation
    val localNSamples = vsm.nSamples
    val localRowType = vsm.rowType
    val localVAnnotator = vAnnotator
    val localGAnnotator = gAnnotator

    val newRowType = newMatrixType.rowType

    vsm.rdd2.mapPartitions { it =>
      val context = new SplitMultiPartitionContext(
        localKeepStar,
        localNSamples, localGlobalAnnotation, localRowType,
        localVAnnotator, localGAnnotator, newRowType)

      it.flatMap { rv =>
        val splitit = context.splitRow(rv, sortAlleles, removeLeftAligned, removeMoving, verifyLeftAligned)
        context.prevLocus = context.ur.getAs[Locus](0)
        splitit
      }
    }
  }

  def split(): VariantSampleMatrix = {
    val newRDD2: OrderedRDD2 =
      if (leftAligned)
        OrderedRDD2(
          newMatrixType.orderedRDD2Type,
          vsm.rdd2.orderedPartitioner,
          split(sortAlleles = true, removeLeftAligned = false, removeMoving = false, verifyLeftAligned = true))
      else
        SplitMulti.unionMovedVariants(OrderedRDD2(
          newMatrixType.orderedRDD2Type,
          vsm.rdd2.orderedPartitioner,
          split(sortAlleles = true, removeLeftAligned = false, removeMoving = true, verifyLeftAligned = false)),
          split(sortAlleles = false, removeLeftAligned = true, removeMoving = false, verifyLeftAligned = false))

    vsm.copy2(rdd2 = newRDD2, vaSignature = vAnnotator.newT, genotypeSignature = gAnnotator.newT, wasSplit = true)
  }
}
