package is.hail.methods

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.expr.{EvalContext, Parser}
import is.hail.rvd.OrderedRVD
import is.hail.utils.ArrayBuilder
import is.hail.variant.{MatrixTable, Variant}

object AnnotateAllelesExpr {
  def apply(vsm: MatrixTable, splitVariantExpr: String, splitGenotypeExpr: String, variantExpr: String): MatrixTable = {
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
      val (s, i) = vas.structInsert(TArray(signature), ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(vsm.sparkContext, splitMatrixType, vsm.value.localValue, ec)

    val localNSamples = vsm.nSamples
    val localRowType = vsm.rvRowType

    val localGlobalAnnotation = vsm.globalAnnotation
    val localVAnnotator = splitmulti.vAnnotator
    val localGAnnotator = splitmulti.gAnnotator
    val splitRowType = splitMatrixType.rvRowType

    val newMatrixType = vsm.matrixType.copy(vaType = newType)
    val newRowType = newMatrixType.rvRowType

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
}
