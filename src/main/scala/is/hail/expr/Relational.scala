package is.hail.expr

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.ir._
import is.hail.table.TableLocalValue
import is.hail.expr.types._
import is.hail.methods.Aggregators
import is.hail.sparkextras._
import is.hail.rvd._
import is.hail.variant.{GenomeReference, Genotype, MatrixFileMetadata}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.json4s.jackson.JsonMethods

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

case class MatrixLocalValue(
  globalAnnotation: Annotation,
  sampleAnnotations: IndexedSeq[Annotation]) {

  def nSamples: Int = sampleAnnotations.length

  def dropSamples(): MatrixLocalValue = MatrixLocalValue(
    globalAnnotation,
    IndexedSeq.empty[Annotation])
}

object MatrixValue {

}

case class MatrixValue(
  typ: MatrixType,
  localValue: MatrixLocalValue,
  rdd2: OrderedRVD) {

  def sparkContext: SparkContext = rdd2.sparkContext

  def nPartitions: Int = rdd2.partitions.length

  def nSamples: Int = localValue.nSamples

  def globalAnnotation: Annotation = localValue.globalAnnotation

  def sampleIds: IndexedSeq[Row] = {
    val queriers = typ.colKey.map(field => typ.colType.query(field))
    localValue.sampleAnnotations.map(a => Row.fromSeq(queriers.map(_(a))))
  }

  def sampleAnnotations: IndexedSeq[Annotation] = localValue.sampleAnnotations

  lazy val sampleAnnotationsBc: Broadcast[IndexedSeq[Annotation]] = sparkContext.broadcast(sampleAnnotations)

  def filterSamplesKeep(keep: Array[Int]): MatrixValue = {
    val rowType = typ.rvRowType
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
      localValue.copy(sampleAnnotations = keep.map(sampleAnnotations)),
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

  def filterSamples(p: (Annotation, Int) => Boolean): MatrixValue = {
    val keep = (0 until nSamples)
      .view
      .filter { i => p(sampleAnnotations(i), i) }
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
  fileMetadata: MatrixFileMetadata,
  dropSamples: Boolean,
  dropVariants: Boolean) extends MatrixIR {
  def typ: MatrixType = fileMetadata.matrixType

  override def partitionCounts: Option[Array[Long]] = fileMetadata.partitionCounts

  def children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRead = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): MatrixValue = {
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
          hc.readRows(path, typ.rvRowType, nPartitions))
        if (dropSamples) {
          val localRowType = typ.rvRowType
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
    val sas = typ.saType
    val ec = typ.sampleEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val sampleAggregationOption = Aggregators.buildSampleAggregations(hc, prev, ec)

    val p = (sa: Annotation, i: Int) => {
      sampleAggregationOption.foreach(f => f.apply(i))
      ec.setAll(localGlobalAnnotation, sa)
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
    val ec = prev.typ.variantEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val aggregatorOption = Aggregators.buildVariantAggregations(
      prev.rdd2.sparkContext, prev.typ, prev.localValue, ec)

    val localPrevRowType = prev.typ.rvRowType
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

  def env: Env[IR] = {
    Env.empty[IR]
      .bind(typ.rowType.fieldNames.map {f => (f, GetField(In(0, typ.rowType), f)) }:_*)
      .bind(typ.globalType.fieldNames.map {f => (f, GetField(In(1, typ.globalType), f)) }:_*)
  }

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
    val f = ir.Compile(child.env, ir.RegionValueRep[Long](child.typ.rowType),
      ir.RegionValueRep[Long](child.typ.globalType),
      ir.RegionValueRep[Boolean](TBoolean()),
      pred)
    ktv.filter((rv, globalRV) => f()(rv.region, rv.offset, false, globalRV.offset, false))
  }
}

case class TableAnnotate(child: TableIR, paths: IndexedSeq[String], preds: IndexedSeq[IR]) extends TableIR {

  val children: IndexedSeq[BaseIR] = Array(child) ++ preds

  private val newIR: IR = InsertFields(In(0, child.typ.rowType), paths.zip(preds.map(child.typ.remapIR(_))).toArray)

  val typ: TableType = {
    Infer(newIR, None, child.typ.env)
    child.typ.copy(rowType = newIR.typ.asInstanceOf[TStruct])
  }

  def copy(newChildren: IndexedSeq[BaseIR]): TableAnnotate = {
    assert(newChildren.length == children.length)
    TableAnnotate(newChildren(0).asInstanceOf[TableIR], paths, newChildren.tail.asInstanceOf[IndexedSeq[IR]])
  }

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val f = ir.Compile(child.env, ir.RegionValueRep[Long](child.typ.rowType),
      ir.RegionValueRep[Long](child.typ.globalType),
      ir.RegionValueRep[Long](typ.rowType),
      newIR)
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

