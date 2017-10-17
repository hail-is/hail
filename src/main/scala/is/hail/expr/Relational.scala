package is.hail.expr

import is.hail.HailContext
import is.hail.annotations._
import is.hail.methods.Aggregators
import is.hail.sparkextras._
import is.hail.variant.{VSMFileMetadata, VSMLocalValue, VSMMetadata}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.json4s.jackson.JsonMethods

import scala.reflect.ClassTag

case class MatrixType(
  metadata: VSMMetadata) {
  def globalType: Type = metadata.globalSignature

  def sType: Type = metadata.sSignature

  def saType: Type = metadata.saSignature

  def locusType: Type = vType match {
    case t: TVariant =>
      TLocus(vType.asInstanceOf[TVariant].gr)
    case _ =>
      vType
  }

  def vType: Type = metadata.vSignature

  def vaType: Type = metadata.vaSignature

  def genotypeType: Type = metadata.genotypeSignature

  def rowType: TStruct =
    TStruct(
      "pk" -> locusType,
      "v" -> vType,
      "va" -> vaType,
      "gs" -> TArray(genotypeType))

  def orderedRDD2Type: OrderedRDD2Type = {
    new OrderedRDD2Type(Array("pk"),
      Array("pk", "v"),
      rowType)
  }

  def pkType: TStruct = orderedRDD2Type.pkType

  def kType: TStruct = orderedRDD2Type.kType

  def typ = TStruct(
    "v" -> vType,
    "va" -> vaType,
    "s" -> sType,
    "sa" -> saType,
    "g" -> genotypeType)

  def sampleEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "s" -> (1, sType),
      "sa" -> (2, saType),
      "g" -> (3, genotypeType),
      "v" -> (4, vType),
      "va" -> (5, vaType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "s" -> (1, sType),
      "sa" -> (2, saType),
      "gs" -> (3, TAggregable(genotypeType, aggregationST))))
  }

  def variantEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "v" -> (1, vType),
      "va" -> (2, vaType),
      "g" -> (3, genotypeType),
      "s" -> (4, sType),
      "sa" -> (5, saType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "v" -> (1, vType),
      "va" -> (2, vaType),
      "gs" -> (3, TAggregable(genotypeType, aggregationST))))
  }

  def copy(globalType: Type = globalType,
    sType: Type = sType, saType: Type = saType,
    vType: Type = vType, vaType: Type = vaType,
    genotypeType: Type = genotypeType): MatrixType =
    MatrixType(metadata = metadata.copy(
      globalSignature = globalType,
      sSignature = sType, saSignature = saType,
      vSignature = vType, vaSignature = vaType,
      genotypeSignature = genotypeType))
}

object NewAST {
  def genericRewriteTopDown(ast: NewAST, rule: PartialFunction[NewAST, NewAST]): NewAST = {
    def rewrite(ast: NewAST): NewAST = {
      rule.lift(ast) match {
        case Some(newAST) if newAST != ast =>
          rewrite(newAST)
        case None =>
          val newChildren = ast.children.map(rewrite)
          if ((ast.children, newChildren).zipped.forall(_ eq _))
            ast
          else
            ast.copy(newChildren)
      }
    }

    rewrite(ast)
  }

  def rewriteTopDown(ast: MatrixAST, rule: PartialFunction[NewAST, NewAST]): MatrixAST =
    genericRewriteTopDown(ast, rule).asInstanceOf[MatrixAST]

  def rewriteTopDown(ast: KeyTableAST, rule: PartialFunction[NewAST, NewAST]): KeyTableAST =
    genericRewriteTopDown(ast, rule).asInstanceOf[KeyTableAST]

  def genericRewriteBottomUp(ast: NewAST, rule: PartialFunction[NewAST, NewAST]): NewAST = {
    def rewrite(ast: NewAST): NewAST = {
      val newChildren = ast.children.map(rewrite)

      // only recons if necessary
      val rewritten =
        if ((ast.children, newChildren).zipped.forall(_ eq _))
          ast
        else
          ast.copy(newChildren)

      rule.lift(rewritten) match {
        case Some(newAST) if newAST != rewritten =>
          rewrite(newAST)
        case None =>
          rewritten
      }
    }

    rewrite(ast)
  }

  def rewriteBottomUp(ast: MatrixAST, rule: PartialFunction[NewAST, NewAST]): MatrixAST =
    genericRewriteBottomUp(ast, rule).asInstanceOf[MatrixAST]

  def rewriteBottomUp(ast: KeyTableAST, rule: PartialFunction[NewAST, NewAST]): KeyTableAST =
    genericRewriteBottomUp(ast, rule).asInstanceOf[KeyTableAST]
}

abstract class NewAST {
  def children: IndexedSeq[NewAST]

  def copy(newChildren: IndexedSeq[NewAST]): NewAST

  def mapChildren(f: (NewAST) => NewAST): NewAST = {
    copy(children.map(f))
  }
}

object MatrixValue {
  def apply[RPK, RK, T](
    typ: MatrixType,
    localValue: VSMLocalValue,
    rdd: OrderedRDD[RPK, RK, (Any, Iterable[T])]): MatrixValue = {
    implicit val kOk: OrderedKey[Annotation, Annotation] = typ.vType.orderedKey
    val sc = rdd.sparkContext
    val localRowType = typ.rowType
    val localGType = typ.genotypeType
    val localNSamples = localValue.nSamples
    val rangeBoundsType = TArray(typ.pkType)
    new MatrixValue(typ, localValue,
      OrderedRDD2(typ.orderedRDD2Type,
        new OrderedPartitioner2(rdd.orderedPartitioner.numPartitions,
          typ.orderedRDD2Type.partitionKey,
          typ.orderedRDD2Type.kType,
          UnsafeIndexedSeq(rangeBoundsType,
            rdd.orderedPartitioner.rangeBounds.map(b => Row(b)))),
        rdd.mapPartitions { it =>
          val region = MemoryBuffer()
          val rvb = new RegionValueBuilder(region)
          val rv = RegionValue(region)

          it.map { case (v, (va, gs)) =>
            region.clear()
            rvb.start(localRowType)
            rvb.startStruct()
            rvb.addAnnotation(localRowType.fieldType(0), kOk.project(v))
            rvb.addAnnotation(localRowType.fieldType(1), v)
            rvb.addAnnotation(localRowType.fieldType(2), va)
            rvb.startArray(localNSamples)
            var i = 0
            val git = gs.iterator
            while (git.hasNext) {
              rvb.addAnnotation(localGType, git.next())
              i += 1
            }
            rvb.endArray()
            rvb.endStruct()

            rv.setOffset(rvb.end())
            rv
          }
        }))
  }
}

case class MatrixValue(
  typ: MatrixType,
  localValue: VSMLocalValue,
  rdd2: OrderedRDD2) {

  def rdd[RPK, RK, T]: OrderedRDD[RPK, RK, (Any, Iterable[T])] = {
    warn("converting OrderedRDD2 => OrderedRDD")

    implicit val tct: ClassTag[T] = typ.genotypeType.scalaClassTag.asInstanceOf[ClassTag[T]]
    implicit val kOk: OrderedKey[RPK, RK] = typ.vType.typedOrderedKey[RPK, RK]

    import kOk._

    val localRowType = typ.rowType
    val localNSamples = localValue.nSamples
    OrderedRDD(
      rdd2.map { rv =>
        val ur = new UnsafeRow(localRowType, rv.region.copy(), rv.offset)

        val gs = ur.getAs[IndexedSeq[T]](3)
        assert(gs.length == localNSamples)

        (ur.getAs[RK](1),
          (ur.get(2),
            ur.getAs[IndexedSeq[T]](3): Iterable[T]))
      },
      OrderedPartitioner(
        rdd2.orderedPartitioner.rangeBounds.map { b =>
          b.asInstanceOf[Row].getAs[RPK](0)
        }.toArray(kOk.pkct),
        rdd2.orderedPartitioner.numPartitions))
  }

  def copyRDD[RPK2, RK2, T2](typ: MatrixType = typ,
    localValue: VSMLocalValue = localValue,
    rdd: OrderedRDD[RPK2, RK2, (Any, Iterable[T2])]): MatrixValue = {
    MatrixValue(typ, localValue, rdd)
  }

  def sparkContext: SparkContext = rdd2.sparkContext

  def nPartitions: Int = rdd2.partitions.length

  def nSamples: Int = localValue.nSamples

  def globalAnnotation: Annotation = localValue.globalAnnotation

  def sampleIds: IndexedSeq[Annotation] = localValue.sampleIds

  def sampleAnnotations: IndexedSeq[Annotation] = localValue.sampleAnnotations

  lazy val sampleIdsBc: Broadcast[IndexedSeq[Annotation]] = sparkContext.broadcast(sampleIds)

  lazy val sampleAnnotationsBc: Broadcast[IndexedSeq[Annotation]] = sparkContext.broadcast(sampleAnnotations)

  def sampleIdsAndAnnotations: IndexedSeq[(Annotation, Annotation)] = sampleIds.zip(sampleAnnotations)

  def filterSamples(p: (Annotation, Annotation) => Boolean): MatrixValue = {
    val keep = sampleIdsAndAnnotations.zipWithIndex
      .filter { case ((s, sa), i) => p(s, sa) }
      .map(_._2)
    val keepBc = sparkContext.broadcast(keep)
    val localRowType = typ.rowType
    val localNSamples = nSamples
    copy(localValue =
      localValue.copy(
        sampleIds = keep.map(sampleIds),
        sampleAnnotations = keep.map(sampleAnnotations)),
      rdd2 = rdd2.mapPartitionsPreservesPartitioning { it =>
        var rv2b = new RegionValueBuilder()
        var rv2 = RegionValue()

        it.map { rv =>
          rv2b.set(rv.region)
          rv2b.start(localRowType)
          rv2b.startStruct()
          rv2b.addField(localRowType, rv, 0)
          rv2b.addField(localRowType, rv, 1)
          rv2b.addField(localRowType, rv, 2)

          rv2b.startArray(keepBc.value.length)
          val gst = localRowType.fieldType(3).asInstanceOf[TArray]
          val gt = gst.elementType
          val aoff = localRowType.loadField(rv, 3)
          var i = 0
          val length = keepBc.value.length
          while (i < length) {
            val j = keepBc.value(i)
            rv2b.addElement(gst, rv.region, aoff, j)
            i += 1
          }
          rv2b.endArray()
          rv2b.endStruct()
          rv2.set(rv.region, rv2b.end())

          rv2
        }
      })
  }
}

object MatrixAST {
  def optimize(ast: MatrixAST): MatrixAST = {
    NewAST.rewriteTopDown(ast, {
      case FilterVariants(
      MatrixRead(hc, path, nPartitions, fileMetadata, dropSamples, _),
      Const(_, false, TBoolean)) =>
        MatrixRead(hc, path, nPartitions, fileMetadata, dropSamples, dropVariants = true)
      case FilterSamples(
      MatrixRead(hc, path, nPartitions, fileMetadata, _, dropVariants),
      Const(_, false, TBoolean)) =>
        MatrixRead(hc, path, nPartitions, fileMetadata, dropSamples = true, dropVariants)

      case FilterVariants(m, Const(_, true, TBoolean)) =>
        m
      case FilterSamples(m, Const(_, true, TBoolean)) =>
        m

      // minor, but push FilterVariants into FilterSamples
      case FilterVariants(FilterSamples(m, spred), vpred) =>
        FilterSamples(FilterVariants(m, vpred), spred)

      case FilterVariants(FilterVariants(m, pred1), pred2) =>
        FilterVariants(m, Apply(pred1.getPos, "&&", Array(pred1, pred2)))

      case FilterSamples(FilterSamples(m, pred1), pred2) =>
        FilterSamples(m, Apply(pred1.getPos, "&&", Array(pred1, pred2)))
    })
  }
}

abstract sealed class MatrixAST extends NewAST {
  def typ: MatrixType

  def execute(hc: HailContext): MatrixValue
}

case class MatrixLiteral(
  typ: MatrixType,
  value: MatrixValue) extends MatrixAST {

  def children: IndexedSeq[NewAST] = Array.empty[NewAST]

  def execute(hc: HailContext): MatrixValue = value

  def copy(newChildren: IndexedSeq[NewAST]): MatrixLiteral = {
    assert(newChildren.isEmpty)
    this
  }

  override def toString: String = "MatrixLiteral(...)"
}

case class MatrixRead(
  hc: HailContext,
  path: String,
  nPartitions: Int,
  fileMetadata: VSMFileMetadata,
  dropSamples: Boolean,
  dropVariants: Boolean) extends MatrixAST {

  def typ: MatrixType = MatrixType(fileMetadata.metadata)

  def children: IndexedSeq[NewAST] = Array.empty[NewAST]

  def copy(newChildren: IndexedSeq[NewAST]): MatrixRead = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): MatrixValue = {
    val metadata = fileMetadata.metadata
    val localValue =
      if (dropSamples)
        fileMetadata.localValue.dropSamples()
      else
        fileMetadata.localValue

    val rdd =
      if (dropVariants)
        OrderedRDD2.empty(hc.sc, typ.orderedRDD2Type)
      else {
        var rdd = OrderedRDD2(
          typ.orderedRDD2Type,
          OrderedPartitioner2(hc.sc,
            hc.hadoopConf.readFile(path + "/partitioner.json.gz")(JsonMethods.parse(_))),
          new ReadRowsRDD(hc.sc, path, typ.rowType, nPartitions))
        if (dropSamples) {
          val localRowType = typ.rowType
          rdd = rdd.mapPartitionsPreservesPartitioning { it =>
            var rv2b = new RegionValueBuilder(null)
            var rv2 = RegionValue()

            it.map { rv =>
              rv2b.set(rv.region)

              rv2b.start(localRowType)
              rv2b.startStruct()

              rv2b.addField(localRowType, rv, 0)
              rv2b.addField(localRowType, rv, 1)
              rv2b.addField(localRowType, rv, 2)

              rv2b.startArray(0) // gs
              rv2b.endArray()

              rv2b.endStruct()
              rv2.set(rv.region, rv2b.end())

              rv2
            }
          }
        }

        rdd
      }

    MatrixValue(
      typ,
      localValue,
      rdd)
  }

  override def toString: String = s"MatrixRead($path, dropSamples = $dropSamples, dropVariants = $dropVariants)"
}

case class FilterSamples(
  child: MatrixAST,
  pred: AST) extends MatrixAST {

  def children: IndexedSeq[NewAST] = Array(child)

  def copy(newChildren: IndexedSeq[NewAST]): FilterSamples = {
    assert(newChildren.length == 1)
    FilterSamples(newChildren(0).asInstanceOf[MatrixAST], pred)
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobalAnnotation = prev.localValue.globalAnnotation
    val sas = typ.metadata.saSignature
    val ec = typ.sampleEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val sampleAggregationOption = Aggregators.buildSampleAggregations[Annotation, Annotation, Annotation](hc, prev, ec)

    val p = (s: Annotation, sa: Annotation) => {
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.setAll(localGlobalAnnotation, s, sa)
      f() == true
    }
    prev.filterSamples(p)
  }
}

case class FilterVariants(
  child: MatrixAST,
  pred: AST) extends MatrixAST {

  def children: IndexedSeq[NewAST] = Array(child)

  def copy(newChildren: IndexedSeq[NewAST]): FilterVariants = {
    assert(newChildren.length == 1)
    FilterVariants(newChildren(0).asInstanceOf[MatrixAST], pred)
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobalAnnotation = prev.localValue.globalAnnotation
    val ec = child.typ.variantEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val aggregatorOption = Aggregators.buildVariantAggregations[Annotation, Annotation, Annotation](
      prev.rdd2.sparkContext, prev.localValue, ec)

    val localPrevRowType = prev.typ.rowType
    val p = (rv: RegionValue) => {
      val ur = new UnsafeRow(localPrevRowType, rv.region.copy(), rv.offset)

      val v = ur.get(1)
      val va = ur.get(2)
      val gs = ur.getAs[IndexedSeq[Annotation]](3)
      aggregatorOption.foreach(f => f(v, va, gs))

      ec.setAll(localGlobalAnnotation, v, va)

      // null => false
      f() == true
    }

    prev.copy(rdd2 = prev.rdd2.filter(p))
  }
}

abstract sealed class KeyTableAST extends NewAST {
  def execute(hc: HailContext): RDD[Row]
}

case class KeyTableLiteral(
  typ: TStruct,
  rdd: RDD[Row]) {

  def children: IndexedSeq[NewAST] = Array.empty[NewAST]

  def copy(newChildren: IndexedSeq[NewAST]): KeyTableLiteral = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): RDD[Row] = rdd
}
