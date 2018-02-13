package is.hail.expr

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.methods.Aggregators
import is.hail.rvd._
import is.hail.table.TableSpec
import is.hail.variant.MatrixTableSpec
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

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

case class MatrixValue(
  typ: MatrixType,
  globals: Annotation,
  colValues: IndexedSeq[Annotation],
  rvd: OrderedRVD) {

  def sparkContext: SparkContext = rvd.sparkContext

  def nPartitions: Int = rvd.partitions.length

  def nSamples: Int = colValues.length

  def sampleIds: IndexedSeq[Row] = {
    val queriers = typ.colKey.map(field => typ.colType.query(field))
    colValues.map(a => Row.fromSeq(queriers.map(_ (a))))
  }

  lazy val colValuesBc: Broadcast[IndexedSeq[Annotation]] = sparkContext.broadcast(colValues)

  def filterSamplesKeep(keep: Array[Int]): MatrixValue = {
    val rowType = typ.rvRowType
    val keepType = TArray(+TInt32())
    val makeF = ir.Compile("row", ir.RegionValueRep[Long](rowType),
      "keep", ir.RegionValueRep[Long](keepType),
      ir.RegionValueRep[Long](rowType),
      body = ir.insertStruct(ir.Ref("row"), rowType, MatrixType.entriesIdentifier,
        ir.ArrayMap(ir.Ref("keep"), "i",
          ir.ArrayRef(ir.GetField(ir.In(0, rowType), MatrixType.entriesIdentifier),
            ir.Ref("i")))))

    val keepBc = sparkContext.broadcast(keep)
    copy(colValues = keep.map(colValues),
      rvd = rvd.mapPartitionsPreservesPartitioning(typ.orvdType) { it =>
        val f = makeF()
        val keep = keepBc.value
        var rv2 = RegionValue()

        it.map { rv =>
          val region = rv.region
          rv2.set(region,
            f(region, rv.offset, false, region.appendArrayInt(keep), false))
          rv2
        }
      })
  }

  def filterSamples(p: (Annotation, Int) => Boolean): MatrixValue = {
    val keep = (0 until nSamples)
      .view
      .filter { i => p(colValues(i), i) }
      .toArray
    filterSamplesKeep(keep)
  }
}

object MatrixIR {
  def optimize(ast: MatrixIR): MatrixIR = {
    BaseIR.rewriteTopDown(ast, {
      case FilterVariants(
      MatrixRead(path, spec, dropSamples, _),
      Const(_, false, TBoolean(_))) =>
        MatrixRead(path, spec, dropSamples, dropVariants = true)
      case FilterSamples(
      MatrixRead(path, spec, _, dropVariants),
      Const(_, false, TBoolean(_))) =>
        MatrixRead(path, spec, dropSamples = true, dropVariants)

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
  spec: MatrixTableSpec,
  dropSamples: Boolean,
  dropVariants: Boolean) extends MatrixIR {
  def typ: MatrixType = spec.matrix_type

  override def partitionCounts: Option[Array[Long]] = Some(spec.partitionCounts)

  def children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRead = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): MatrixValue = {
    val hConf = hc.hadoopConf

    val globals = spec.globalsComponent.readLocal(hc, path)(0)

    val colAnnotations =
      if (dropSamples)
        IndexedSeq.empty[Annotation]
      else
        spec.colsComponent.readLocal(hc, path)

    val rvd =
      if (dropVariants)
        OrderedRVD.empty(hc.sc, typ.orvdType)
      else {
        val fullRowType = typ.rvRowType
        val localEntriesIndex = typ.entriesIdx

        val rowsRVD = spec.rowsComponent.read(hc, path).asInstanceOf[OrderedRVD]
        if (dropSamples) {
          rowsRVD.mapPartitionsPreservesPartitioning(typ.orvdType) { it =>
            var rv2b = new RegionValueBuilder()
            var rv2 = RegionValue()

            it.map { rv =>
              rv2b.set(rv.region)
              rv2b.start(fullRowType)
              rv2b.startStruct()
              var i = 0
              while (i < localEntriesIndex) {
                rv2b.addField(fullRowType, rv, i)
                i += 1
              }
              rv2b.startArray(0)
              rv2b.endArray()
              i += 1
              while (i < fullRowType.size) {
                rv2b.addField(fullRowType, rv, i - 1)
                i += 1
              }
              rv2b.endStruct()
              rv2.set(rv.region, rv2b.end())
              rv2
            }
          }
        } else {
          val entriesRVD = spec.entriesComponent.read(hc, path)
          val entriesRowType = entriesRVD.rowType
          OrderedRVD(typ.orvdType,
            rowsRVD.partitioner,
            rowsRVD.rdd.zipPartitions(entriesRVD.rdd) { case (it1, it2) =>
              val rvb = new RegionValueBuilder()

              new Iterator[RegionValue] {
                def hasNext: Boolean = {
                  val hn = it1.hasNext
                  assert(hn == it2.hasNext)
                  hn
                }

                def next(): RegionValue = {
                  val rv1 = it1.next()
                  val rv2 = it2.next()
                  val region = rv2.region
                  rvb.set(region)
                  rvb.start(fullRowType)
                  rvb.startStruct()
                  var i = 0
                  while (i < localEntriesIndex) {
                    rvb.addField(fullRowType, rv1, i)
                    i += 1
                  }
                  rvb.addField(entriesRowType, rv2, 0)
                  i += 1
                  while (i < fullRowType.size) {
                    rvb.addField(fullRowType, rv1, i - 1)
                    i += 1
                  }
                  rvb.endStruct()
                  rv2.set(region, rvb.end())
                  rv2
                }
              }
            })
        }
      }

    MatrixValue(
      typ,
      globals,
      colAnnotations,
      rvd)
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

    val localGlobals = prev.globals
    val sas = typ.colType
    val ec = typ.colEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val sampleAggregationOption = Aggregators.buildSampleAggregations(hc, prev, ec)

    val p = (sa: Annotation, i: Int) => {
      sampleAggregationOption.foreach(f => f.apply(i))
      ec.setAll(localGlobals, sa)
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

    val localGlobals = prev.globals
    val ec = prev.typ.rowEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val aggregatorOption = Aggregators.buildVariantAggregations(
      prev.rvd.sparkContext, prev.typ, prev.globals, prev.colValues, ec)

    val fullRowType = prev.typ.rvRowType
    val localRowType = prev.typ.rowType
    val localEntriesIndex = prev.typ.entriesIdx

    ec.set(0, prev.globals)

    val filteredRDD = prev.rvd.mapPartitionsPreservesPartitioning(prev.typ.orvdType) { it =>
      val fullRow = new UnsafeRow(fullRowType)
      val row = fullRow.deleteField(localEntriesIndex)
      it.filter { rv =>
        fullRow.set(rv)
        ec.set(1, row)
        aggregatorOption.foreach(_(rv))
        f() == true
      }
    }

    prev.copy(rvd = filteredRDD)
  }
}

case class TableValue(typ: TableType, globals: Row, rvd: RVD) {
  def rdd: RDD[Row] = {
    val localRowType = typ.rowType
    rvd.rdd.map { rv => new UnsafeRow(localRowType, rv.region.copy(), rv.offset) }
  }

  def filter(p: (RegionValue, RegionValue) => Boolean): TableValue = {
    val globalType = typ.globalType
    val localGlobals = globals
    copy(rvd = rvd.mapPartitions(typ.rowType) { it =>
      val globalRV = RegionValue()
      val globalRVb = new RegionValueBuilder()
      it.filter { rv =>
        globalRVb.set(rv.region)
        globalRVb.start(globalType)
        globalRVb.addAnnotation(globalType, localGlobals)
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
      case TableFilter(TableRead(path, spec, _), False() | NA(TBoolean(_))) =>
        TableRead(path, spec, true)
    })
  }
}

abstract sealed class TableIR extends BaseIR {
  def typ: TableType

  def partitionCounts: Option[Array[Long]] = None

  def env: Env[IR] = {
    Env.empty[IR]
      .bind(typ.rowType.fieldNames.map { f => (f, GetField(In(0, typ.rowType), f)) }: _*)
      .bind(typ.globalType.fieldNames.map { f => (f, GetField(In(1, typ.globalType), f)) }: _*)
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

case class TableRead(path: String, spec: TableSpec, dropRows: Boolean) extends TableIR {
  def typ: TableType = spec.table_type

  override def partitionCounts: Option[Array[Long]] = Some(spec.partitionCounts)

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRead = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): TableValue = {
    val globals = spec.globalsComponent.readLocal(hc, path)(0)
    TableValue(typ,
      globals,
      if (dropRows)
        UnpartitionedRVD.empty(hc.sc, typ.rowType)
      else
        spec.rowsComponent.read(hc, path))
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

  private val newIR: IR = InsertFields(In(0, child.typ.rowType), paths.zip(preds.map(child.typ.remapIR(_))))

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
    val localGlobals = tv.globals
    val gType = typ.globalType
    TableValue(typ,
      tv.globals,
      tv.rvd.mapPartitions(typ.rowType) { it =>
        val globalRV = RegionValue()
        val globalRVb = new RegionValueBuilder()
        val rv2 = RegionValue()
        val newRow = f()
        it.map { rv =>
          globalRVb.set(rv.region)
          globalRVb.start(gType)
          globalRVb.addAnnotation(gType, localGlobals)
          globalRV.set(rv.region, globalRVb.end())
          rv2.set(rv.region, newRow(rv.region, rv.offset, false, globalRV.offset, false))
          rv2
        }
      })
  }
}

