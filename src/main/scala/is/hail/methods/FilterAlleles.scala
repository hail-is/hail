package is.hail.methods

import is.hail.annotations._
import is.hail.expr.{EvalContext, Parser, TArray, TInt32, TVariant}
import is.hail.sparkextras.OrderedRDD2
import is.hail.utils._
import is.hail.variant.{GenomeReference, Locus, Variant, VariantDataset, VariantSampleMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object FilterAlleles {
  def apply(vsm: VariantSampleMatrix, filterExpr: String,
    variantExpr: String = "",
    genotypeExpr: String = "",
    keep: Boolean = true, leftAligned: Boolean = false, keepStar: Boolean = false): VariantSampleMatrix = {
    if (vsm.wasSplit)
      warn("this VDS was already split; this module was designed to handle multi-allelics, perhaps you should use filter_variants instead.")

    val conditionEC = EvalContext(Map(
      "global" -> (0, vsm.globalSignature),
      "v" -> (1, vsm.vSignature),
      "va" -> (2, vsm.vaSignature),
      "aIndex" -> (3, TInt32())))
    val conditionE = Parser.parseTypedExpr[java.lang.Boolean](filterExpr, conditionEC)

    val vEC = EvalContext(Map(
      "global" -> (0, vsm.globalSignature),
      "v" -> (1, vsm.vSignature),
      "va" -> (2, vsm.vaSignature),
      "newV" -> (3, vsm.vSignature),
      "oldToNew" -> (4, TArray(TInt32())),
      "newToOld" -> (5, TArray(TInt32()))))

    val gEC = EvalContext(Map(
      "global" -> (0, vsm.globalSignature),
      "v" -> (1, vsm.vSignature),
      "va" -> (2, vsm.vaSignature),
      "newV" -> (3, vsm.vSignature),
      "oldToNew" -> (4, TArray(TInt32())),
      "newToOld" -> (5, TArray(TInt32())),
      "s" -> (6, vsm.sSignature),
      "sa" -> (7, vsm.saSignature),
      "g" -> (8, vsm.genotypeSignature)))

    val vAnnotator = new ExprAnnotator(vEC, vsm.vaSignature, variantExpr, Some(Annotation.VARIANT_HEAD))
    val gAnnotator = new ExprAnnotator(gEC, vsm.genotypeSignature, genotypeExpr, Some(Annotation.GENOTYPE_HEAD))

    val localGlobalAnnotation = vsm.globalAnnotation
    val localNSamples = vsm.nSamples

    val newMatrixType = vsm.matrixType.copy(vaType = vAnnotator.newT,
      genotypeType = gAnnotator.newT)

    def filter(rdd: RDD[RegionValue],
      removeLeftAligned: Boolean, removeMoving: Boolean, verifyLeftAligned: Boolean): RDD[RegionValue] = {

      def filterAllelesInVariant(prevlocus: Locus, v: Variant, va: Annotation): Option[(Variant, IndexedSeq[Int], IndexedSeq[Int])] = {
        var alive = 0
        val oldToNew = new Array[Int](v.nAlleles)
        for (aai <- v.altAlleles.indices) {
          val index = aai + 1
          conditionEC.setAll(localGlobalAnnotation, v, va, index)
          oldToNew(index) =
            if (Filter.boxedKeepThis(conditionE(), keep)) {
              alive += 1
              alive
            } else
              0
        }

        if (alive == 0)
          return None

        val newToOld = oldToNew.iterator
          .zipWithIndex
          .filter { case (newIdx, oldIdx) => oldIdx == 0 || newIdx != 0 }
          .map(_._2)
          .toArray

        val altAlleles = oldToNew.iterator
          .zipWithIndex
          .filter { case (newIdx, _) => newIdx != 0 }
          .map { case (_, idx) => v.altAlleles(idx - 1) }
          .toArray

        if (altAlleles.forall(_.isStar) && !keepStar)
          return None

        val filtv = v.copy(altAlleles = altAlleles).minRep
        val isLeftAligned = (prevlocus == null || prevlocus != filtv.locus) &&
          filtv.locus == v.locus

        if (isLeftAligned) {
          if (removeLeftAligned)
            return None
        } else {
          if (removeMoving)
            return None
          else if (verifyLeftAligned)
            fatal("found non-left aligned variant: $v")
        }

        Some((filtv, newToOld, oldToNew))
      }

      val localRowType = vsm.matrixType.rowType
      val newRowType = newMatrixType.rowType

      val localSampleIdsBc = vsm.sampleIdsBc
      val localSampleAnnotationsBc = vsm.sampleAnnotationsBc

      rdd.mapPartitions { it =>
        var prevLocus: Locus = null

        it.flatMap { rv =>
          val rvb = new RegionValueBuilder()
          val rv2 = RegionValue()

          val ur = new UnsafeRow(localRowType, rv.region, rv.offset)

          val v = ur.getAs[Variant](1)
          val va = ur.get(2)
          val gs = ur.getAs[IndexedSeq[Annotation]](3)

          filterAllelesInVariant(prevLocus, v, va)
            .map { case (newV, newToOld, oldToNew) =>
              rvb.set(rv.region)
              rvb.start(newRowType)
              rvb.startStruct()
              rvb.addAnnotation(newRowType.fieldType(0), newV.locus)
              rvb.addAnnotation(newRowType.fieldType(1), newV)

              vAnnotator.ec.setAll(localGlobalAnnotation, v, va, newV, oldToNew, newToOld)
              val newVA = vAnnotator.insert(va)
              rvb.addAnnotation(newRowType.fieldType(2), newVA)

              gAnnotator.ec.setAll(localGlobalAnnotation, v, va, newV, oldToNew, newToOld)

              rvb.startArray(localNSamples) // gs
              var k = 0
              while (k < localNSamples) {
                val g = gs(k)
                gAnnotator.ec.set(6, localSampleIdsBc.value(k))
                gAnnotator.ec.set(7, localSampleAnnotationsBc.value(k))
                gAnnotator.ec.set(8, g)
                rvb.addAnnotation(newRowType.fieldType(3).asInstanceOf[TArray].elementType, gAnnotator.insert(g))
                k += 1
              }
              rvb.endArray()
              rvb.endStruct()
              rv2.set(rv.region, rvb.end())

              prevLocus = newV.locus

              rv2
            }
        }
      }
    }

    val newRDD2: OrderedRDD2 =
      if (leftAligned) {
        OrderedRDD2(newMatrixType.orderedRDD2Type,
          vsm.rdd2.orderedPartitioner,
          filter(vsm.rdd2, removeLeftAligned = false, removeMoving = false, verifyLeftAligned = true))
      } else {
        val leftAlignedVariants = OrderedRDD2(newMatrixType.orderedRDD2Type,
          vsm.rdd2.orderedPartitioner,
          filter(vsm.rdd2, removeLeftAligned = false, removeMoving = true, verifyLeftAligned = false))

        val movingVariants = OrderedRDD2.shuffle(newMatrixType.orderedRDD2Type,
          vsm.rdd2.orderedPartitioner,
          filter(vsm.rdd2, removeLeftAligned = true, removeMoving = false, verifyLeftAligned = false))

        leftAlignedVariants.partitionSortedUnion(movingVariants)
      }

    vsm.copy2(rdd2 = newRDD2, vaSignature = vAnnotator.newT, genotypeSignature = gAnnotator.newT)
  }
}
