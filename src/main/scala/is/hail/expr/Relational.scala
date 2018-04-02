package is.hail.expr

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.methods.Aggregators
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
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
  globals: BroadcastValue,
  colValues: IndexedSeq[Annotation],
  rvd: OrderedRVD) {

  assert(rvd.typ == typ.orvdType)

  def sparkContext: SparkContext = rvd.sparkContext

  def nPartitions: Int = rvd.partitions.length

  def nCols: Int = colValues.length

  def sampleIds: IndexedSeq[Row] = {
    val queriers = typ.colKey.map(field => typ.colType.query(field))
    colValues.map(a => Row.fromSeq(queriers.map(_ (a))))
  }

  lazy val colValuesBc: Broadcast[IndexedSeq[Annotation]] = sparkContext.broadcast(colValues)
}

object MatrixIR {
  def optimize(ast: MatrixIR): MatrixIR = {
    BaseIR.rewriteTopDown(ast, {
      case FilterRows(
      MatrixRead(path, spec, dropSamples, _),
      Const(_, false, TBoolean(_))) =>
        MatrixRead(path, spec, dropSamples, dropRows = true)

      case FilterCols(
      MatrixRead(path, spec, _, dropVariants),
      Const(_, false, TBoolean(_))) =>
        MatrixRead(path, spec, dropCols = true, dropVariants)

      case FilterRows(m, Const(_, true, TBoolean(_))) =>
        m
      case FilterCols(m, Const(_, true, TBoolean(_))) =>
        m

      // minor, but push FilterVariants into FilterSamples
      case FilterRows(FilterCols(m, spred), vpred) =>
        FilterCols(FilterRows(m, vpred), spred)

      case FilterRows(FilterRows(m, pred1), pred2) =>
        FilterRows(m, Apply(pred1.getPos, "&&", Array(pred1, pred2)))

      case FilterCols(FilterCols(m, pred1), pred2) =>
        FilterCols(m, Apply(pred1.getPos, "&&", Array(pred1, pred2)))

      // Equivalent rewrites for the new Filter{Cols,Rows}IR
      case FilterRowsIR(MatrixRead(path, spec, dropSamples, _), False()) =>
        MatrixRead(path, spec, dropSamples, dropRows = true)

      case FilterColsIR(MatrixRead(path, spec, dropVariants, _), False()) =>
        MatrixRead(path, spec, dropCols = true, dropVariants)

      // Keep all rows/cols = do nothing
      case FilterRowsIR(m, True()) => m

      case FilterColsIR(m, True()) => m

      // Push FilterRowsIR into FilterColsIR
      case FilterRowsIR(FilterColsIR(m, colPred), rowPred) =>
        FilterColsIR(FilterRowsIR(m, rowPred), colPred)

      // Combine multiple filters into one
      /*
       * FIXME: optimizations disabled due to lack of DoubleAmpersand()
       *
      case FilterRowsIR(FilterRowsIR(m, pred1), pred2) =>
        FilterRowsIR(m,
          ApplyBinaryPrimOp(DoubleAmpersand(), pred1, pred2))

      case FilterColsIR(FilterColsIR(m, pred1), pred2) =>
        FilterColsIR(m,
          ApplyBinaryPrimOp(DoubleAmpersand(), pred1, pred2))
       */
    })
  }

  def chooseColsWithArray(typ: MatrixType): (MatrixType, (MatrixValue, Array[Int]) => MatrixValue) = {
    val rowType = typ.rvRowType
    val keepType = TArray(+TInt32())
    val (rTyp, makeF) = ir.Compile[Long, Long, Long]("row", rowType,
      "keep", keepType,
      body = InsertFields(ir.Ref("row"), Seq((MatrixType.entriesIdentifier,
        ir.ArrayMap(ir.Ref("keep"), "i",
          ir.ArrayRef(ir.GetField(ir.In(0, rowType), MatrixType.entriesIdentifier),
            ir.Ref("i")))))))
    assert(rTyp.isOfType(rowType))

    val newMatrixType = typ.copy(rvRowType = coerce[TStruct](rTyp))

    val keepF = { (mv: MatrixValue, keep: Array[Int]) =>
      val keepBc = mv.sparkContext.broadcast(keep)
      mv.copy(typ = newMatrixType,
        colValues = keep.map(mv.colValues),
        rvd = mv.rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType) { it =>
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
    (newMatrixType, keepF)
  }

  def filterCols(typ: MatrixType): (MatrixType, (MatrixValue, (Annotation, Int) => Boolean) => MatrixValue) = {
    val (t, keepF) = chooseColsWithArray(typ)
    (t, {(mv: MatrixValue, p :(Annotation, Int) => Boolean) =>
      val keep = (0 until mv.nCols)
        .view
        .filter { i => p(mv.colValues(i), i) }
        .toArray
      keepF(mv, keep)
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
  dropCols: Boolean,
  dropRows: Boolean) extends MatrixIR {
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
      if (dropCols)
        IndexedSeq.empty[Annotation]
      else
        spec.colsComponent.readLocal(hc, path)

    val rvd =
      if (dropRows)
        OrderedRVD.empty(hc.sc, typ.orvdType)
      else {
        val fullRowType = typ.rvRowType
        val localEntriesIndex = typ.entriesIdx

        val rowsRVD = spec.rowsComponent.read(hc, path).asInstanceOf[OrderedRVD]
        if (dropCols) {
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
          rowsRVD.zipPartitionsPreservesPartitioning(
            typ.orvdType,
            entriesRVD
          ) { case (it1, it2) =>
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
          }
        }
      }

    MatrixValue(
      typ,
      BroadcastValue(globals, typ.globalType, hc.sc),
      colAnnotations,
      rvd)
  }

  override def toString: String = s"MatrixRead($path, dropSamples = $dropCols, dropVariants = $dropRows)"
}

case class FilterCols(
  child: MatrixIR,
  pred: AST) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): FilterCols = {
    assert(newChildren.length == 1)
    FilterCols(newChildren(0).asInstanceOf[MatrixIR], pred)
  }

  val (typ, filterF) = MatrixIR.filterCols(child.typ)

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobals = prev.globals.value
    val sas = typ.colType
    val ec = typ.colEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val sampleAggregationOption = Aggregators.buildColAggregations(hc, prev, ec)

    val p = (sa: Annotation, i: Int) => {
      sampleAggregationOption.foreach(f => f.apply(i))
      ec.setAll(localGlobals, sa)
      f() == true
    }
    filterF(prev, p)
  }
}

case class FilterColsIR(
  child: MatrixIR,
  pred: IR) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): FilterColsIR = {
    assert(newChildren.length == 2)
    FilterColsIR(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val (typ, filterF) = MatrixIR.filterCols(child.typ)

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobals = prev.globals.broadcast
    val localColType = typ.colType

    //
    // Initialize a region containing the globals
    //
    val (rTyp, predCompiledFunc) = ir.Compile[Long, Long, Boolean](
      "global", typ.globalType,
      "sa", typ.colType,
      pred
    )
    // Note that we don't yet support IR aggregators
    val p = (sa: Annotation, i: Int) => {
      Region.scoped { colRegion =>
        // FIXME: it would be nice to only load the globals once per matrix
        val rvb = new RegionValueBuilder(colRegion)
        rvb.start(typ.globalType)
        rvb.addAnnotation(typ.globalType, localGlobals.value)
        val globalRVend = rvb.currentOffset()
        val globalRVoffset = rvb.end()

        colRegion.clear(globalRVend)
        val colRVb = new RegionValueBuilder(colRegion)
        colRVb.start(localColType)
        colRVb.addAnnotation(localColType, sa)
        val colRVoffset = colRVb.end()
        predCompiledFunc()(colRegion, globalRVoffset, false, colRVoffset, false)
      }
    }
    filterF(prev, p)
  }
}

case class FilterRows(
  child: MatrixIR,
  pred: AST) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): FilterRows = {
    assert(newChildren.length == 1)
    FilterRows(newChildren(0).asInstanceOf[MatrixIR], pred)
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val ec = prev.typ.rowEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val aggregatorOption = Aggregators.buildRowAggregations(
      prev.rvd.sparkContext, prev.typ, prev.globals, prev.colValues, ec)

    val fullRowType = prev.typ.rvRowType
    val localRowType = prev.typ.rowType
    val localEntriesIndex = prev.typ.entriesIdx

    val localGlobals = prev.globals.broadcast

    val filteredRDD = prev.rvd.mapPartitionsPreservesPartitioning(prev.typ.orvdType) { it =>
      val fullRow = new UnsafeRow(fullRowType)
      val row = fullRow.deleteField(localEntriesIndex)
      it.filter { rv =>
        fullRow.set(rv)
        ec.set(0, localGlobals.value)
        ec.set(1, row)
        aggregatorOption.foreach(_ (rv))
        f() == true
      }
    }

    prev.copy(rvd = filteredRDD)
  }
}

case class FilterRowsIR(
  child: MatrixIR,
  pred: IR) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): FilterRowsIR = {
    assert(newChildren.length == 2)
    FilterRowsIR(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val (rTyp, predCompiledFunc) = ir.Compile[Long, Long, Boolean](
      "va", typ.rvRowType,
      "global", typ.globalType,
      pred
    )

    // Note that we don't yet support IR aggregators
    //
    // Everything used inside the Spark iteration must be serializable,
    // so we pick out precisely what is needed.
    //
    val fullRowType = prev.typ.rvRowType
    val localRowType = prev.typ.rowType
    val localEntriesIndex = prev.typ.entriesIdx
    val localGlobalType = typ.globalType
    val localGlobals = prev.globals.broadcast

    val filteredRDD = prev.rvd.mapPartitionsPreservesPartitioning(prev.typ.orvdType) { it =>
      it.filter { rv =>
        // Append all the globals into this region
        val globalRVb = new RegionValueBuilder(rv.region)
        globalRVb.start(localGlobalType)
        globalRVb.addAnnotation(localGlobalType, localGlobals.value)
        val globalRVoffset = globalRVb.end()
        predCompiledFunc()(rv.region, rv.offset, false, globalRVoffset, false)
      }
    }

    prev.copy(rvd = filteredRDD)
  }
}

case class ChooseCols(child: MatrixIR, oldIndices: Array[Int]) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): ChooseCols = {
    assert(newChildren.length == 1)
    ChooseCols(newChildren(0).asInstanceOf[MatrixIR], oldIndices)
  }

  val (typ, colsF) = MatrixIR.chooseColsWithArray(child.typ)

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    colsF(prev, oldIndices)
  }
}

case class MapEntries(child: MatrixIR, newEntries: IR) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, newEntries)

  def copy(newChildren: IndexedSeq[BaseIR]): MapEntries = {
    assert(newChildren.length == 2)
    MapEntries(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val newRow = {
    val arrayLength = ArrayLen(GetField(Ref("va"), MatrixType.entriesIdentifier))
    val idxEnv = new Env[IR]()
      .bind("g", ArrayRef(GetField(Ref("va"), MatrixType.entriesIdentifier), Ref("i")))
      .bind("sa", ArrayRef(Ref("sa"), Ref("i")))
    val entries = ArrayMap(ArrayRange(I32(0), arrayLength, I32(1)), "i", Subst(newEntries, idxEnv))
    InsertFields(Ref("va"), Seq((MatrixType.entriesIdentifier, entries)))
  }

  val typ: MatrixType = {
    Infer(newRow, None, new Env[Type]()
      .bind("global", child.typ.globalType)
      .bind("va", child.typ.rvRowType)
      .bind("sa", TArray(child.typ.colType))
    )
    child.typ.copy(rvRowType = newRow.typ)
  }

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobalsType = typ.globalType
    val localColsType = TArray(typ.colType)
    val colValuesBc = prev.colValuesBc
    val globalsBc = prev.globals.broadcast

    val (rTyp, f) = ir.Compile[Long, Long, Long, Long](
      "global", localGlobalsType,
      "va", prev.typ.rvRowType,
      "sa", localColsType,
      newRow)
    assert(rTyp == typ.rvRowType)

    val newRVD = prev.rvd.mapPartitionsPreservesPartitioning(typ.orvdType) { it =>
      val rvb = new RegionValueBuilder()
      val newRV = RegionValue()
      val rowF = f()

      it.map { rv =>
        val region = rv.region
        val oldRow = rv.offset

        rvb.set(region)
        rvb.start(localGlobalsType)
        rvb.addAnnotation(localGlobalsType, globalsBc.value)
        val globals = rvb.end()

        rvb.start(localColsType)
        rvb.addAnnotation(localColsType, colValuesBc.value)
        val cols = rvb.end()

        val off = rowF(region, globals, false, oldRow, false, cols, false)

        newRV.set(region, off)
        newRV
      }
    }
    prev.copy(typ = typ, rvd = newRVD)
  }
}

case class TableValue(typ: TableType, globals: BroadcastValue, rvd: RVD) {
  def rdd: RDD[Row] =
    rvd.toRows

  def filter(p: (RegionValue, RegionValue) => Boolean): TableValue = {
    val globalType = typ.globalType
    val localGlobals = globals.broadcast
    copy(rvd = rvd.mapPartitions(typ.rowType) { it =>
      val globalRV = RegionValue()
      val globalRVb = new RegionValueBuilder()
      it.filter { rv =>
        globalRVb.set(rv.region)
        globalRVb.start(globalType)
        globalRVb.addAnnotation(globalType, localGlobals.value)
        globalRV.set(rv.region, globalRVb.end())
        p(rv, globalRV)
      }
    })
  }
}


object TableIR {
  def optimize(ir: TableIR): TableIR = {
    BaseIR.rewriteTopDown(ir, {
      case TableFilter(x, True()) => x
      case TableFilter(TableRead(path, spec, _), False() | NA(TBoolean(_))) =>
        TableRead(path, spec, true)
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
      BroadcastValue(globals, typ.globalType, hc.sc),
      if (dropRows)
        UnpartitionedRVD.empty(hc.sc, typ.rowType)
      else
        spec.rowsComponent.read(hc, path))
  }
}

case class TableParallelize(typ: TableType, rows: IndexedSeq[Row], nPartitions: Option[Int] = None) extends TableIR {
  assert(typ.globalType.size == 0)
  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableParallelize = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): TableValue = {
    val rowTyp = typ.rowType
    val rvd = ContextRDD.parallelize[RVDContext](hc.sc, rows, nPartitions)
      .cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, rowTyp))
    TableValue(typ, BroadcastValue(Annotation.empty, typ.globalType, hc.sc), new UnpartitionedRVD(rowTyp, rvd))
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
    val (rTyp, f) = ir.Compile[Long, Long, Boolean](
      "row", child.typ.rowType,
      "global", child.typ.globalType,
      pred)
    assert(rTyp == TBoolean())
    ktv.filter((rv, globalRV) => f()(rv.region, rv.offset, false, globalRV.offset, false))
  }
}

case class TableJoin(left: TableIR, right: TableIR, joinType: String) extends TableIR {
  require(left.typ.keyType isIsomorphicTo right.typ.keyType)

  val children: IndexedSeq[BaseIR] = Array(left, right)

  private val joinedFields = left.typ.keyType.fields ++
    left.typ.valueType.fields ++
    right.typ.valueType.fields
  private val preNames = joinedFields.map(_.name).toArray
  private val (finalColumnNames, remapped) = mangle(preNames)

  val newRowType = TStruct(joinedFields.zipWithIndex.map {
    case (fd, i) => (finalColumnNames(i), fd.typ)
  }: _*)

  val typ: TableType = left.typ.copy(rowType = newRowType)

  def copy(newChildren: IndexedSeq[BaseIR]): TableJoin = {
    assert(newChildren.length == 2)
    TableJoin(
      newChildren(0).asInstanceOf[TableIR],
      newChildren(1).asInstanceOf[TableIR],
      joinType)
  }

  def execute(hc: HailContext): TableValue = {
    val leftTV = left.execute(hc)
    val rightTV = right.execute(hc)
    val leftRowType = left.typ.rowType
    val rightRowType = right.typ.rowType
    val leftKeyFieldIdx = left.typ.keyFieldIdx
    val rightKeyFieldIdx = right.typ.keyFieldIdx
    val leftValueFieldIdx = left.typ.valueFieldIdx
    val rightValueFieldIdx = right.typ.valueFieldIdx
    val localNewRowType = newRowType
    val rvMerger: Iterator[JoinedRegionValue] => Iterator[RegionValue] = { it =>
      val rvb = new RegionValueBuilder()
      val rv = RegionValue()
      it.map { joined =>
        val lrv = joined._1
        val rrv = joined._2

        if (lrv != null)
          rvb.set(lrv.region)
        else {
          assert(rrv != null)
          rvb.set(rrv.region)
        }

        rvb.start(localNewRowType)
        rvb.startStruct()

        if (lrv != null)
          rvb.addFields(leftRowType, lrv, leftKeyFieldIdx)
        else {
          assert(rrv != null)
          rvb.addFields(rightRowType, rrv, rightKeyFieldIdx)
        }

        if (lrv != null)
          rvb.addFields(leftRowType, lrv, leftValueFieldIdx)
        else
          rvb.skipFields(leftValueFieldIdx.length)

        if (rrv != null)
          rvb.addFields(rightRowType, rrv, rightValueFieldIdx)
        else
          rvb.skipFields(rightValueFieldIdx.length)

        rvb.endStruct()
        rv.set(rvb.region, rvb.end())
        rv
      }
    }
    val leftORVD = leftTV.rvd match {
      case ordered: OrderedRVD => ordered
      case unordered =>
        OrderedRVD.coerce(
          new OrderedRVDType(left.typ.key.toArray, left.typ.key.toArray, leftRowType),
          unordered)
    }
    val rightORVD = rightTV.rvd match {
      case ordered: OrderedRVD => ordered
      case unordered =>
        val ordType =
          new OrderedRVDType(right.typ.key.toArray, right.typ.key.toArray, rightRowType)
        if (joinType == "left" || joinType == "inner")
          unordered.constrainToOrderedPartitioner(ordType, leftORVD.partitioner)
        else
          OrderedRVD.coerce(ordType, unordered, leftORVD.partitioner)
    }
    val joinedRVD = leftORVD.orderedJoin(
      rightORVD,
      joinType,
      rvMerger,
      new OrderedRVDType(leftORVD.typ.partitionKey, leftORVD.typ.key, newRowType))

    TableValue(typ, leftTV.globals, joinedRVD)
  }
}

case class TableMapRows(child: TableIR, newRow: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: TableType = {
    Infer(newRow, None, child.typ.env)
    val newRowType = newRow.typ.asInstanceOf[TStruct]
    val newKey = child.typ.key.filter(newRowType.fieldIdx.contains)
    child.typ.copy(rowType = newRowType, key = newKey)
  }

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapRows = {
    assert(newChildren.length == 2)
    TableMapRows(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val (rTyp, f) = ir.Compile[Long, Long, Long](
      "row", child.typ.rowType,
      "global", child.typ.globalType,
      newRow)
    assert(rTyp == typ.rowType)
    val globalsBc = tv.globals.broadcast
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
          globalRVb.addAnnotation(gType, globalsBc.value)
          globalRV.set(rv.region, globalRVb.end())
          rv2.set(rv.region, newRow(rv.region, rv.offset, false, globalRV.offset, false))
          rv2
        }
      })
  }
}

case class TableMapGlobals(child: TableIR, newRow: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: TableType = {
    Infer(newRow, None, child.typ.env)
    child.typ.copy(globalType = newRow.typ.asInstanceOf[TStruct])
  }

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapGlobals = {
    assert(newChildren.length == 2)
    TableMapGlobals(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val gType = typ.globalType

    val (rTyp, f) = ir.Compile[Long, Long](
      "global", child.typ.globalType,
      newRow)
    assert(rTyp == gType)

    val newGlobals = Region.scoped { region =>
      val rv = tv.globals.regionValue(region)
      val offset = f()(rv.region, rv.offset, false)

      tv.globals.copy(
        value = Annotation.safeFromRegionValue(rTyp, rv.region, offset),
        t = rTyp)
    }

    TableValue(typ, newGlobals, tv.rvd)
  }
}
