package is.hail.expr

import is.hail.HailContext
import is.hail.annotations.{RegionValueBuilder, _}
import is.hail.asm4s.FunctionBuilder
import is.hail.expr.ir._
import is.hail.table.TableLocalValue
import is.hail.expr.types._
import is.hail.methods.Aggregators
import is.hail.sparkextras._
import is.hail.rvd._
import is.hail.variant.{VSMFileMetadata, VSMLocalValue, VSMMetadata}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.json4s.jackson.JsonMethods

import scala.collection.mutable

case class MatrixType(
  metadata: VSMMetadata) extends BaseType {
  def globalType: Type = metadata.globalSignature

  def sType: Type = metadata.sSignature

  def saType: Type = metadata.saSignature

  def locusType: Type = vType match {
    case t: TVariant => TLocus(t.gr)
    case _ => vType
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

  def orderedRVType: OrderedRVType = {
    new OrderedRVType(Array("pk"),
      Array("pk", "v"),
      rowType)
  }

  def pkType: TStruct = orderedRVType.pkType

  def kType: TStruct = orderedRVType.kType

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

  def genotypeEC: EvalContext = {
    EvalContext(Map(
      "global" -> (0, globalType),
      "v" -> (1, vType),
      "va" -> (2, vaType),
      "s" -> (3, sType),
      "sa" -> (4, saType),
      "g" -> (5, genotypeType)))
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

object BaseIR {
  def genericRewriteTopDown(ast: BaseIR, rule: PartialFunction[BaseIR, BaseIR]): BaseIR = {
    def rewrite(ast: BaseIR): BaseIR = {
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

  def rewriteTopDown(ast: MatrixIR, rule: PartialFunction[BaseIR, BaseIR]): MatrixIR =
    genericRewriteTopDown(ast, rule).asInstanceOf[MatrixIR]

  def rewriteTopDown(ast: TableIR, rule: PartialFunction[BaseIR, BaseIR]): TableIR =
    genericRewriteTopDown(ast, rule).asInstanceOf[TableIR]

  def genericRewriteBottomUp(ast: BaseIR, rule: PartialFunction[BaseIR, BaseIR]): BaseIR = {
    def rewrite(ast: BaseIR): BaseIR = {
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

  def rewriteBottomUp(ast: MatrixIR, rule: PartialFunction[BaseIR, BaseIR]): MatrixIR =
    genericRewriteBottomUp(ast, rule).asInstanceOf[MatrixIR]

  def rewriteBottomUp(ast: TableIR, rule: PartialFunction[BaseIR, BaseIR]): TableIR =
    genericRewriteBottomUp(ast, rule).asInstanceOf[TableIR]
}

abstract class BaseIR {
  def typ: BaseType

  def children: IndexedSeq[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR

  def mapChildren(f: (BaseIR) => BaseIR): BaseIR = {
    copy(children.map(f))
  }
}

object MatrixValue {
  def apply(
    typ: MatrixType,
    localValue: VSMLocalValue,
    rdd: OrderedRDD[Annotation, Annotation, (Any, Iterable[Annotation])]): MatrixValue = {
    implicit val kOk: OrderedKey[Annotation, Annotation] = typ.vType.orderedKey
    val sc = rdd.sparkContext
    val localRowType = typ.rowType
    val localGType = typ.genotypeType
    val localNSamples = localValue.nSamples
    val rangeBoundsType = TArray(typ.pkType)
    new MatrixValue(typ, localValue,
      OrderedRVD(typ.orderedRVType,
        new OrderedRVPartitioner(rdd.orderedPartitioner.numPartitions,
          typ.orderedRVType.partitionKey,
          typ.orderedRVType.kType,
          UnsafeIndexedSeq(rangeBoundsType,
            rdd.orderedPartitioner.rangeBounds.map(b => Row(b)))),
        rdd.mapPartitions { it =>
          val region = Region()
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
  rdd2: OrderedRVD) {

  def rdd: OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])] = {
    warn("converting OrderedRVD => OrderedRDD")

    implicit val kOk: OrderedKey[Annotation, Annotation] = typ.vType.orderedKey

    import kOk._

    val localRowType = typ.rowType
    val localNSamples = localValue.nSamples
    OrderedRDD(
      rdd2.map { rv =>
        val ur = new UnsafeRow(localRowType, rv.region.copy(), rv.offset)

        val gs = ur.getAs[IndexedSeq[Annotation]](3)
        assert(gs.length == localNSamples)

        (ur.get(1),
          (ur.get(2),
            ur.getAs[IndexedSeq[Annotation]](3): Iterable[Annotation]))
      },
      OrderedPartitioner(
        rdd2.partitioner.rangeBounds.map { b =>
          b.asInstanceOf[Row].get(0)
        }.toArray(kOk.pkct),
        rdd2.partitioner.numPartitions))
  }

  def copyRDD(typ: MatrixType = typ,
    localValue: VSMLocalValue = localValue,
    rdd: OrderedRDD[Annotation, Annotation, (Any, Iterable[Annotation])]): MatrixValue = {
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

  def filterSamplesKeep(keep: Array[Int]): MatrixValue = {
    val rowType = typ.rowType
    val keepType = TArray(!TInt32())
    val makeF = ir.Compile("row", ir.RegionValueRep[Long](rowType),
      "keep", ir.RegionValueRep[Long](keepType),
      ir.RegionValueRep[Long](rowType),
      body = ir.insertStruct(ir.Ref("row"), rowType, "gs",
        ir.ArrayMap(ir.Ref("keep"), "i",
          ir.ArrayRef(ir.GetField(ir.In(0, rowType), "gs"),
            ir.Ref("i")))))

    val keepBc = sparkContext.broadcast(keep)
    copy(localValue =
      localValue.copy(
        sampleIds = keep.map(sampleIds),
        sampleAnnotations = keep.map(sampleAnnotations)),
      rdd2 = rdd2.mapPartitionsPreservesPartitioning(typ.orderedRVType) { it =>
        val f = makeF()
        val keep = keepBc.value
        var rv2 = RegionValue()

        it.map { rv =>
          val region = rv.region
          val offset2 =
          rv2.set(region,
            f(region, rv.offset, false, region.appendArrayInt(keep), false))
          rv2
        }
      })
  }

  def filterSamples(p: (Annotation, Annotation) => Boolean): MatrixValue = {
    val keep = sampleIdsAndAnnotations.zipWithIndex
      .filter { case ((s, sa), i) => p(s, sa) }
      .map(_._2)
      .toArray
    filterSamplesKeep(keep)
  }
}

object MatrixIR {
  def optimize(ast: MatrixIR): MatrixIR = {
    BaseIR.rewriteTopDown(ast, {
      case FilterVariants(
      MatrixRead(path, nPartitions, fileMetadata, dropSamples, _),
      Const(_, false, TBoolean(_))) =>
        MatrixRead(path, nPartitions, fileMetadata, dropSamples, dropVariants = true)
      case FilterSamples(
      MatrixRead(path, nPartitions, fileMetadata, _, dropVariants),
      Const(_, false, TBoolean(_))) =>
        MatrixRead(path, nPartitions, fileMetadata, dropSamples = true, dropVariants)

      case FilterVariants(m, Const(_, true, TBoolean(_))) =>
        m
      case FilterSamples(m, Const(_, true, TBoolean(_))) =>
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

abstract sealed class MatrixIR extends BaseIR {
  def typ: MatrixType

  def partitionCounts: Option[Array[Long]] = None

  def execute(hc: HailContext): MatrixValue
}

case class MatrixLiteral(
  typ: MatrixType,
  value: MatrixValue) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def execute(hc: HailContext): MatrixValue = value

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixLiteral = {
    assert(newChildren.isEmpty)
    this
  }

  override def toString: String = "MatrixLiteral(...)"
}

case class MatrixRead(
  path: String,
  nPartitions: Int,
  fileMetadata: VSMFileMetadata,
  dropSamples: Boolean,
  dropVariants: Boolean) extends MatrixIR {
  def typ: MatrixType = MatrixType(fileMetadata.metadata)

  override def partitionCounts: Option[Array[Long]] = fileMetadata.partitionCounts

  def children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRead = {
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
        OrderedRVD.empty(hc.sc, typ.orderedRVType)
      else {
        var rdd = OrderedRVD(
          typ.orderedRVType,
          OrderedRVPartitioner(hc.sc,
            hc.hadoopConf.readFile(path + "/partitioner.json.gz")(JsonMethods.parse(_))),
          hc.readRows(path, typ.rowType, nPartitions))
        if (dropSamples) {
          val localRowType = typ.rowType
          rdd = rdd.mapPartitionsPreservesPartitioning(typ.orderedRVType) { it =>
            var rv2b = new RegionValueBuilder()
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
  child: MatrixIR,
  pred: AST) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): FilterSamples = {
    assert(newChildren.length == 1)
    FilterSamples(newChildren(0).asInstanceOf[MatrixIR], pred)
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobalAnnotation = prev.localValue.globalAnnotation
    val sas = typ.metadata.saSignature
    val ec = typ.sampleEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val sampleAggregationOption = Aggregators.buildSampleAggregations(hc, prev, ec)

    val p = (s: Annotation, sa: Annotation) => {
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.setAll(localGlobalAnnotation, s, sa)
      f() == true
    }
    prev.filterSamples(p)
  }
}

case class FilterVariants(
  child: MatrixIR,
  pred: AST) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): FilterVariants = {
    assert(newChildren.length == 1)
    FilterVariants(newChildren(0).asInstanceOf[MatrixIR], pred)
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobalAnnotation = prev.localValue.globalAnnotation
    val ec = child.typ.variantEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val aggregatorOption = Aggregators.buildVariantAggregations(
      prev.rdd2.sparkContext, prev.typ, prev.localValue, ec)

    val localPrevRowType = prev.typ.rowType
    val p = (rv: RegionValue) => {
      val ur = new UnsafeRow(localPrevRowType, rv.region.copy(), rv.offset)

      val v = ur.get(1)
      val va = ur.get(2)
      aggregatorOption.foreach(f => f(rv))

      ec.setAll(localGlobalAnnotation, v, va)

      // null => false
      f() == true
    }

    prev.copy(rdd2 = prev.rdd2.filter(p))
  }
}

case class TableValue(typ: TableType, localValue: TableLocalValue, rvd: RVD) {
  def rdd: RDD[Row] = {
    val localRowType = typ.rowType
    rvd.rdd.map { rv => new UnsafeRow(localRowType, rv.region.copy(), rv.offset) }
  }

  def filter(p: (RegionValue, RegionValue) => Boolean): TableValue = {
    val globalType = typ.globalType
    val globals = localValue.globals
    copy(rvd = rvd.mapPartitions(typ.rowType) { it =>
      val globalRV = RegionValue()
      val globalRVb = new RegionValueBuilder()
      it.filter { rv =>
        globalRVb.set(rv.region)
        globalRVb.start(globalType)
        globalRVb.addAnnotation(globalType, globals)
        globalRV.set(rv.region, globalRVb.end())
        p(rv, globalRV)
      }
    })
  }
}

case class TableType(rowType: TStruct, key: Array[String], globalType: TStruct) extends BaseType {
  def rowEC: EvalContext = EvalContext(rowType.fields.map { f => f.name -> f.typ } ++
      globalType.fields.map { f => f.name -> f.typ }: _*)
  def fields: Map[String, Type] = Map(rowType.fields.map { f => f.name -> f.typ } ++ globalType.fields.map { f => f.name -> f.typ }: _*)

  def remapIR(ir: IR): IR = ir match {
    case Ref(y, _) if rowType.selfField(y).isDefined => GetField(In(0, rowType), y, rowType.field(y).typ)
    case Ref(y, _) if globalType.selfField(y).isDefined => GetField(In(1, globalType), y, globalType.field(y).typ)
    case ir2 => Recur(remapIR)(ir2)
  }
}

object TableIR {
  def optimize(ir: TableIR): TableIR = {
    BaseIR.rewriteTopDown(ir, {
      case TableFilter(TableFilter(x, p1), p2) =>
        TableFilter(x, ApplyBinaryPrimOp(DoubleAmpersand(), p1, p2))
      case TableFilter(x, True()) => x
      case TableFilter(TableRead(path, ktType, localVal, _, nPart, pCounts), False() | NA(TBoolean(_))) =>
        TableRead(path, ktType, localVal, true, nPart, pCounts)
    })
  }
}

abstract sealed class TableIR extends BaseIR {
  def typ: TableType

  def partitionCounts: Option[Array[Long]] = None

  def execute(hc: HailContext): TableValue
}

case class TableLiteral(value: TableValue) extends TableIR {
  val typ: TableType = value.typ

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableLiteral = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): TableValue = value
}

case class TableRead(path: String,
  ktType: TableType,
  localValue: TableLocalValue,
  dropRows: Boolean,
  nPartitions: Int,
  override val partitionCounts: Option[Array[Long]]) extends TableIR {

  val typ: TableType = ktType

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRead = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): TableValue = {
    TableValue(typ,
      localValue,
      if (dropRows)
        RVD.empty(hc.sc, typ.rowType)
      else
        RVD(typ.rowType, hc.readRows(path, typ.rowType, nPartitions)))
  }
}

case class TableFilter(child: TableIR, pred: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, pred)

  val typ: TableType = child.typ

  def copy(newChildren: IndexedSeq[BaseIR]): TableFilter = {
    assert(newChildren.length == 2)
    TableFilter(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  def execute(hc: HailContext): TableValue = {
    val ktv = child.execute(hc)
    val mappedPred = typ.remapIR(pred)
    Infer(mappedPred)
    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Long, Boolean, Boolean]
    Emit(mappedPred, fb)
    val f = fb.result()
    ktv.filter((rv, globalRV) => f()(rv.region, rv.offset, false, globalRV.offset, false))
  }
}

case class TableAnnotate(child: TableIR, paths: IndexedSeq[List[String]], preds: IndexedSeq[IR]) extends TableIR {

  val children: IndexedSeq[BaseIR] = Array(child) ++ preds

  private def makeNestedStruct(path: List[String], ir: IR): IR = {
    if (path.isEmpty)
      ir
    else
      MakeStruct(Array((path.head, makeNestedStruct(path.tail, ir))))
  }

  private def insertIR(path: List[String], old: IR, ir: IR): IR = {
    val ir2 = if (path.tail.isEmpty)
      ir
    else {
      Infer(old)
      old.typ match {
        case t: TStruct if t.selfField(path.head).isDefined =>
          insertIR(path.tail, GetField(old, path.head), ir)
        case _ =>
          makeNestedStruct(path.tail, ir)
      }
    }
    InsertFields(old, Array((path.head, ir2)))
  }

  private var newIR: IR = In(0, child.typ.rowType)

  val typ: TableType = {
    var i = 0
    paths.zip(preds).foreach { case (path, pred) =>
      val mappedPred = child.typ.remapIR(pred)
      Infer(mappedPred)
      newIR = insertIR(path, newIR, mappedPred)
    }
    Infer(newIR)
    child.typ.copy(rowType = newIR.typ.asInstanceOf[TStruct])
  }

  def copy(newChildren: IndexedSeq[BaseIR]): TableAnnotate = {
    assert(newChildren.length == children.length)
    TableAnnotate(newChildren(0).asInstanceOf[TableIR], paths, newChildren.tail.asInstanceOf[IndexedSeq[IR]])
  }

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    Infer(newIR)
    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Long, Boolean, Long]
    Emit(newIR, fb)
    val f = fb.result()
    val globals = tv.localValue.globals
    val gType = typ.globalType
    TableValue(typ,
      tv.localValue,
      tv.rvd.mapPartitions(typ.rowType) { it =>
      val globalRV = RegionValue()
      val globalRVb = new RegionValueBuilder()
      val rv2 = RegionValue()
      val newRow = f()
      it.map { rv =>
        globalRVb.set(rv.region)
        globalRVb.start(gType)
        globalRVb.addAnnotation(gType, globals)
        globalRV.set(rv.region, globalRVb.end())
        rv2.set(rv.region, newRow(rv.region, rv.offset, false, globalRV.offset, false))
        rv2
      }
    })
  }
}

