package is.hail.expr

import is.hail.HailContext
import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.io._
import is.hail.methods.Aggregators
import is.hail.rvd._
import is.hail.table.TableSpec
import is.hail.variant._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import is.hail.utils._
import org.apache.spark.SparkContext
import org.json4s._
import org.json4s.jackson.JsonMethods
import org.json4s.jackson.JsonMethods.parse

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
  globals: BroadcastRow,
  colValues: BroadcastIndexedSeq,
  rvd: OrderedRVD) {

  assert(rvd.typ == typ.orvdType)

  def sparkContext: SparkContext = rvd.sparkContext

  def nPartitions: Int = rvd.getNumPartitions

  def nCols: Int = colValues.value.length

  def sampleIds: IndexedSeq[Row] = {
    val queriers = typ.colKey.map(field => typ.colType.query(field))
    colValues.value.map(a => Row.fromSeq(queriers.map(_ (a))))
  }

  def colsTableValue: TableValue = TableValue(typ.colsTableType, globals, colsRVD())

  def rowsTableValue: TableValue = TableValue(typ.rowsTableType, globals, rowsRVD())

  def entriesTableValue: TableValue = TableValue(typ.entriesTableType, globals, entriesRVD())
  
  private def writeCols(path: String, codecSpec: CodecSpec) {
    val hc = HailContext.get
    val hadoopConf = hc.hadoopConf

    val partitionCounts = RVD.writeLocalUnpartitioned(hc, path + "/rows", typ.colType, codecSpec, colValues.value)

    val colsSpec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "../references",
      typ.colsTableType,
      Map("globals" -> RVDComponentSpec("../globals/rows"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    colsSpec.write(hc, path)

    hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }

  private def writeGlobals(path: String, codecSpec: CodecSpec) {
    val hc = HailContext.get
    val hadoopConf = hc.hadoopConf

    val partitionCounts = RVD.writeLocalUnpartitioned(hc, path + "/rows", typ.globalType, codecSpec, Array(globals.value))

    RVD.writeLocalUnpartitioned(hc, path + "/globals", TStruct.empty(), codecSpec, Array[Annotation](Row()))

    val globalsSpec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "../references",
      TableType(typ.globalType, Array.empty[String], TStruct.empty()),
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    globalsSpec.write(hc, path)

    hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }

  def write(path: String, overwrite: Boolean = false, codecSpecJSONStr: String = null): Unit = {
    val hc = HailContext.get
    val hadoopConf = hc.hadoopConf

    val codecSpec =
      if (codecSpecJSONStr != null) {
        implicit val formats = RVDSpec.formats
        val codecSpecJSON = parse(codecSpecJSONStr)
        codecSpecJSON.extract[CodecSpec]
      } else
        CodecSpec.default

    if (overwrite)
      hadoopConf.delete(path, recursive = true)
    else if (hadoopConf.exists(path))
      fatal(s"file already exists: $path")

    hc.hadoopConf.mkDir(path)

    val partitionCounts = rvd.writeRowsSplit(path, typ, codecSpec)

    val globalsPath = path + "/globals"
    hadoopConf.mkDir(globalsPath)
    writeGlobals(globalsPath, codecSpec)

    val rowsSpec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "../references",
      typ.rowsTableType,
      Map("globals" -> RVDComponentSpec("../globals/rows"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    rowsSpec.write(hc, path + "/rows")

    hadoopConf.writeTextFile(path + "/rows/_SUCCESS")(out => ())

    val entriesSpec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "../references",
      typ.rowsTableType,
      Map("globals" -> RVDComponentSpec("../globals/rows"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    entriesSpec.write(hc, path + "/entries")

    hadoopConf.writeTextFile(path + "/entries/_SUCCESS")(out => ())

    hadoopConf.mkDir(path + "/cols")
    writeCols(path + "/cols", codecSpec)

    val refPath = path + "/references"
    hc.hadoopConf.mkDir(refPath)
    Array(typ.colType, typ.rowType, typ.entryType, typ.globalType).foreach { t =>
      ReferenceGenome.exportReferences(hc, refPath, t)
    }

    val spec = MatrixTableSpec(
      FileFormat.version.rep,
      hc.version,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec("globals/rows"),
        "cols" -> RVDComponentSpec("cols/rows"),
        "rows" -> RVDComponentSpec("rows/rows"),
        "entries" -> RVDComponentSpec("entries/rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(hc, path)

    hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }

  def exportPlink(path: String): Unit =
    is.hail.io.plink.ExportPlink(this, path)

  def rowsRVD(): OrderedRVD = {
    val localRowType = typ.rowType
    val fullRowType = typ.rvRowType
    val localEntriesIndex = typ.entriesIdx
    rvd.mapPartitionsPreservesPartitioning(
      new OrderedRVDType(typ.rowPartitionKey.toArray, typ.rowKey.toArray, typ.rowType)
    ) { it =>
      val rv2b = new RegionValueBuilder()
      val rv2 = RegionValue()
      it.map { rv =>
        rv2b.set(rv.region)
        rv2b.start(localRowType)
        rv2b.startStruct()
        var i = 0
        while (i < fullRowType.size) {
          if (i != localEntriesIndex)
            rv2b.addField(fullRowType, rv, i)
          i += 1
        }
        rv2b.endStruct()
        rv2.set(rv.region, rv2b.end())
        rv2
      }
    }
  }

  def colsRVD(): RVD = {
    val hc = HailContext.get
    val signature = typ.colType

    new UnpartitionedRVD(signature, hc.sc.parallelize(colValues.value.toArray.map(_.asInstanceOf[Row]))
      .mapPartitions(_.toRegionValueIterator(signature)))
  }

  def entriesRVD(): RVD = {
    val resultStruct = typ.entriesTableType.rowType
    val fullRowType = typ.rvRowType
    val localEntriesIndex = typ.entriesIdx
    val localEntriesType = typ.entryArrayType
    val localColType = typ.colType
    val localEntryType = typ.entryType
    val localNCols = nCols

    val localColValues = colValues.broadcast.value

    rvd.mapPartitions(resultStruct) { it =>

      val rv2b = new RegionValueBuilder()
      val rv2 = RegionValue()

      it.flatMap { rv =>
        val rvEnd = rv.region.size
        rv2b.set(rv.region)
        val gsOffset = fullRowType.loadField(rv, localEntriesIndex)
        (0 until localNCols).iterator
          .filter { i =>
            localEntriesType.isElementDefined(rv.region, gsOffset, i)
          }
          .map { i =>
            rv.region.clear(rvEnd)
            rv2b.clear()
            rv2b.start(resultStruct)
            rv2b.startStruct()

            var j = 0
            while (j < fullRowType.size) {
              if (j != localEntriesIndex)
                rv2b.addField(fullRowType, rv, j)
              j += 1
            }

            rv2b.addInlineRow(localColType, localColValues(i).asInstanceOf[Row])
            rv2b.addAllFields(localEntryType, rv.region, localEntriesType.elementOffsetInRegion(rv.region, gsOffset, i))
            rv2b.endStruct()
            rv2.set(rv.region, rv2b.end())
            rv2
          }
      }
    }
  }
}

object MatrixIR {
  def chooseColsWithArray(typ: MatrixType): (MatrixType, (MatrixValue, Array[Int]) => MatrixValue) = {
    val rowType = typ.rvRowType
    val keepType = TArray(+TInt32())
    val (rTyp, makeF) = ir.Compile[Long, Long, Long]("row", rowType,
      "keep", keepType,
      body = InsertFields(ir.Ref("row", rowType), Seq((MatrixType.entriesIdentifier,
        ir.ArrayMap(ir.Ref("keep", keepType), "i",
          ir.ArrayRef(ir.GetField(ir.In(0, rowType), MatrixType.entriesIdentifier),
            ir.Ref("i", TInt32())))))))
    assert(rTyp.isOfType(rowType))

    val newMatrixType = typ.copy(rvRowType = coerce[TStruct](rTyp))

    val keepF = { (mv: MatrixValue, keep: Array[Int]) =>
      val keepBc = mv.sparkContext.broadcast(keep)
      mv.copy(typ = newMatrixType,
        colValues = mv.colValues.copy(value = keep.map(mv.colValues.value)),
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
    (t, { (mv: MatrixValue, p: (Annotation, Int) => Boolean) =>
      val keep = (0 until mv.nCols)
        .view
        .filter { i => p(mv.colValues.value(i), i) }
        .toArray
      keepF(mv, keep)
    })
  }

  def collectColsByKey(typ: MatrixType): (MatrixType, MatrixValue => MatrixValue) = {
    val oldRVRowType = typ.rvRowType
    val oldEntryArrayType = typ.entryArrayType
    val oldEntryType = typ.entryType

    val newColValueType = TStruct(typ.colValueStruct.fields.map(f => f.copy(typ = TArray(f.typ, required = true))))
    val newColType = typ.colKeyStruct ++ newColValueType
    val newEntryType = TStruct(typ.entryType.fields.map(f => f.copy(typ = TArray(f.typ, required = true))))
    val newMatrixType = typ.copyParts(colType = newColType, entryType = newEntryType)
    val newRVRowType = newMatrixType.rvRowType
    val localRowSize = newRVRowType.size

    (newMatrixType, { mv =>
      val colValMap: Map[Row, Array[Int]] = mv.colValues.value
        .map(_.asInstanceOf[Row])
        .zipWithIndex
        .groupBy[Row]{ case (r, i) => typ.extractColKey(r) }
        .mapValues{ _.map{ case (r, i) => i }.toArray }
      val idxMap = colValMap.values.toArray

      val newColValues: BroadcastIndexedSeq = mv.colValues.copy(
        value = colValMap.map { case (key, idx) =>
          Row.fromSeq(key.toSeq ++ newColValueType.fields.map { f =>
            idx.map { i =>
              mv.colValues.value(i).asInstanceOf[Row]
                .get(typ.colValueFieldIdx(f.index))
            }.toIndexedSeq
          })
        }.toIndexedSeq)

      val newRVD = mv.rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType){ it =>
        val rvb = new RegionValueBuilder()
        val rv2 = RegionValue()

        it.map { rv =>
          val entryArrayOffset = oldRVRowType.loadField(rv, oldRVRowType.fieldIdx(MatrixType.entriesIdentifier))

          rvb.set(rv.region)
          rvb.start(newRVRowType)
          rvb.startStruct()
          var i = 0
          while (i < localRowSize - 1) {
            rvb.addField(oldRVRowType, rv, i)
            i += 1
          }
          rvb.startArray(idxMap.length) // start entries array
          i = 0
          while (i < idxMap.length) {
            rvb.startStruct()
            var j = 0
            while (j < newEntryType.size) {
              rvb.startArray(idxMap(i).length)
              var k = 0
              while (k < idxMap(i).length) {
                rvb.addField(
                  oldEntryType,
                  rv.region,
                  oldEntryArrayType.loadElement(rv.region, entryArrayOffset, idxMap(i)(k)),
                  j
                )
                k += 1
              }
              rvb.endArray()
              j += 1
            }
            rvb.endStruct()
            i += 1
          }
          rvb.endArray()
          rvb.endStruct()
          rv2.set(rv.region, rvb.end())
          rv2
        }
      }

      mv.copy(
        typ = newMatrixType,
        colValues = newColValues,
        rvd = newRVD
      )
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

    val globals = Annotation.copy(typ.globalType, spec.globalsComponent.readLocal(hc, path)(0)).asInstanceOf[Row]

    val colAnnotations =
      if (dropCols)
        IndexedSeq.empty[Annotation]
      else
        Annotation.copy(TArray(typ.colType), spec.colsComponent.readLocal(hc, path)).asInstanceOf[IndexedSeq[Annotation]]

    val rvd =
      if (dropRows)
        OrderedRVD.empty(hc.sc, typ.orvdType)
      else {
        val fullRowType = typ.rvRowType
        val rowType = typ.rowType
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
                rv2b.addField(rowType, rv, i)
                i += 1
              }
              rv2b.startArray(0)
              rv2b.endArray()
              i += 1
              while (i < fullRowType.size) {
                rv2b.addField(rowType, rv, i - 1)
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
                  rvb.addField(rowType, rv1, i)
                  i += 1
                }
                rvb.addField(entriesRowType, rv2, 0)
                i += 1
                while (i < fullRowType.size) {
                  rvb.addField(rowType, rv1, i - 1)
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
      BroadcastRow(globals, typ.globalType, hc.sc),
      BroadcastIndexedSeq(colAnnotations, TArray(typ.colType), hc.sc),
      rvd)
  }

  override def toString: String = s"MatrixRead($path, dropSamples = $dropCols, dropVariants = $dropRows)"
}

case class MatrixRange(nRows: Int, nCols: Int, nPartitions: Int) extends MatrixIR {
  private val nPartitionsAdj = math.min(nRows, nPartitions)
  private val partCounts = partition(nRows, nPartitionsAdj)

  override val partitionCounts: Option[Array[Long]] = Some(partCounts.map(_.toLong))

  def children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRange = {
    assert(newChildren.isEmpty)
    this
  }

  val typ: MatrixType = MatrixType.fromParts(
    globalType = TStruct.empty(),
    colKey = Array("col_idx"),
    colType = TStruct("col_idx" -> TInt32()),
    rowPartitionKey = Array("row_idx"),
    rowKey = Array("row_idx"),
    rowType = TStruct("row_idx" -> TInt32()),
    entryType = TStruct.empty())

  def execute(hc: HailContext): MatrixValue = {
    val localRVType = typ.rvRowType
    val localPartCounts = partCounts
    val partStarts = partCounts.scanLeft(0)(_ + _)

    val rvd =
      OrderedRVD(typ.orvdType,
        new OrderedRVDPartitioner(typ.rowPartitionKey.toArray,
          typ.rowKeyStruct,
          Array.tabulate(nPartitionsAdj) { i =>
            val start = partStarts(i)
            Interval(Row(start), Row(start + localPartCounts(i)), includesStart = true, includesEnd = false)
          }),
        hc.sc.parallelize(Range(0, nPartitionsAdj), nPartitionsAdj)
          .mapPartitionsWithIndex { case (i, _) =>
            val region = Region()
            val rvb = new RegionValueBuilder(region)
            val rv = RegionValue(region)

            val start = partStarts(i)
            Iterator.range(start, start + localPartCounts(i))
              .map { j =>
                region.clear()
                rvb.start(localRVType)
                rvb.startStruct()

                // row idx field
                rvb.addInt(j)

                // entries field
                rvb.startArray(nCols)
                var i = 0
                while (i < nCols) {
                  rvb.startStruct()
                  rvb.endStruct()
                  i += 1
                }
                rvb.endArray()

                rvb.endStruct()
                rv.setOffset(rvb.end())
                rv
              }
          })

    MatrixValue(typ,
      BroadcastRow(Row(), typ.globalType, hc.sc),
      BroadcastIndexedSeq(
        Iterator.range(0, nCols)
          .map(Row(_))
          .toFastIndexedSeq,
        TArray(typ.colType),
        hc.sc),
      rvd)
  }
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
        val globalRVoffset = rvb.end()

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

    val localGlobals = prev.globals.broadcast

    val filteredRDD = prev.rvd.mapPartitionsPreservesPartitioning(prev.typ.orvdType) { it =>
      val fullRow = new UnsafeRow(fullRowType)
      it.filter { rv =>
        fullRow.set(rv)
        ec.set(0, localGlobals.value)
        ec.set(1, fullRow)
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
    assert(child.typ == prev.typ)

    val (rTyp, predCompiledFunc) = ir.Compile[Long, Long, Boolean](
      "va", typ.rvRowType,
      "global", typ.globalType,
      pred)

    // Note that we don't yet support IR aggregators
    //
    // Everything used inside the Spark iteration must be serializable,
    // so we pick out precisely what is needed.
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

  override def partitionCounts: Option[Array[Long]] = child.partitionCounts

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    colsF(prev, oldIndices)
  }
}

case class CollectColsByKey(child: MatrixIR) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): CollectColsByKey = {
    assert(newChildren.length == 1)
    CollectColsByKey(newChildren(0).asInstanceOf[MatrixIR])
  }

  val (typ, groupF) = MatrixIR.collectColsByKey(child.typ)

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    groupF(prev)
  }
}

case class MatrixMapEntries(child: MatrixIR, newEntries: IR) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, newEntries)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapEntries = {
    assert(newChildren.length == 2)
    MatrixMapEntries(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val newRow = {
    val vaType = child.typ.rvRowType
    val saType = TArray(child.typ.colType)

    val arrayLength = ArrayLen(GetField(Ref("va", vaType), MatrixType.entriesIdentifier))
    val idxEnv = new Env[IR]()
      .bind("g", ArrayRef(GetField(Ref("va", vaType), MatrixType.entriesIdentifier), Ref("i", TInt32())))
      .bind("sa", ArrayRef(Ref("sa", saType), Ref("i", TInt32())))
    val entries = ArrayMap(ArrayRange(I32(0), arrayLength, I32(1)), "i", Subst(newEntries, idxEnv))
    InsertFields(Ref("va", vaType), Seq((MatrixType.entriesIdentifier, entries)))
  }

  val typ: MatrixType =
    child.typ.copy(rvRowType = newRow.typ)

  override def partitionCounts: Option[Array[Long]] = child.partitionCounts

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobalsType = typ.globalType
    val localColsType = TArray(typ.colType)
    val colValuesBc = prev.colValues.broadcast
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

case class MatrixMapRows(child: MatrixIR, newRow: IR) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child, newRow)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapRows = {
    assert(newChildren.length == 2)
    MatrixMapRows(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val newRVRow = InsertFields(newRow, Seq(
    MatrixType.entriesIdentifier -> GetField(Ref("va", child.typ.rvRowType), MatrixType.entriesIdentifier)))

  val tAggElt: Type = child.typ.entryType
  val aggSymTab = Map(
    "global" -> (0, child.typ.globalType),
    "va" -> (1, child.typ.rvRowType),
    "g" -> (2, child.typ.entryType),
    "sa" -> (3, child.typ.colType))

  val tAgg = TAggregable(tAggElt, aggSymTab)

  val typ: MatrixType = {
    val newRVRowSet = newRVRow.typ.fieldNames.toSet
    val newRowKey = child.typ.rowKey.filter(newRVRowSet.contains)
    val newPartitionKey = child.typ.rowPartitionKey.filter(newRVRowSet.contains)
    child.typ.copy(rvRowType = newRVRow.typ, rowKey = newRowKey, rowPartitionKey = newPartitionKey)
  }

  override def partitionCounts: Option[Array[Long]] = child.partitionCounts

  private[this] def makeStructChangesKeys(fields: Seq[(String, IR)]): Boolean =
    fields.exists { case (name, ir) =>
      typ.rowKey.contains(name) && (
        ir match {
          case GetField(Ref("va", _), n) => n != name
          case _ => true
        })
    }

  val touchesKeys: Boolean = (typ.rowKey != child.typ.rowKey) ||
    (typ.rowPartitionKey != child.typ.rowPartitionKey) || (
    newRow match {
      case MakeStruct(fields) =>
        makeStructChangesKeys(fields)
      case InsertFields(MakeStruct(fields), toIns) =>
        makeStructChangesKeys(fields) || toIns.map(_._1).toSet.intersect(typ.rowKey.toSet).nonEmpty
      case _ => true
    })

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    assert(prev.typ == child.typ)

    val localRowType = prev.typ.rvRowType
    val localEntriesType = prev.typ.entryArrayType
    val entriesIdx = prev.typ.entriesIdx
    val localGlobalsType = prev.typ.globalType
    val localColsType = TArray(prev.typ.colType)
    val localNCols = prev.nCols
    val colValuesBc = prev.colValues.broadcast
    val globalsBc = prev.globals.broadcast

    val colValuesType = TArray(prev.typ.colType)
    val vaType = prev.typ.rvRowType
    val (rvAggs, seqOps, aggResultType, f, rTyp) = ir.CompileWithAggregators[Long, Long, Long, Long, Long, Long, Long](
      "global", prev.typ.globalType,
      "va", vaType,
      "AGG", tAgg,
      "global", prev.typ.globalType,
      "colValues", colValuesType,
      "va", vaType,
      newRVRow, { (nAggs: Int, aggIR: IR) =>
        def rewrite(x: IR): IR = {
          x match {
            case x@AggIn(_) =>
              val dummy = genUID()
              AggMap(x, dummy, ir.Ref("g", prev.typ.entryType))
            case _ =>
              ir.Recur(rewrite)(x)
          }
        }

        ir.ArrayFor(
          ir.ArrayRange(ir.I32(0), ir.I32(localNCols), ir.I32(1)),
          "i",
          ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", colValuesType), ir.Ref("i", TInt32())),
            ir.Let("g", ir.ArrayRef(
              ir.GetField(ir.Ref("va", vaType), MatrixType.entriesIdentifier),
              ir.Ref("i", TInt32())),
              rewrite(aggIR))))
      })
    assert(rTyp == typ.rvRowType, s"$rTyp, ${ typ.rvRowType }")

    val mapPartitionF = { it: Iterator[RegionValue] =>
      val rvb = new RegionValueBuilder()
      val newRV = RegionValue()
      val rowF = f()

      val partRegion = Region()

      rvb.set(partRegion)
      rvb.start(localGlobalsType)
      rvb.addAnnotation(localGlobalsType, globalsBc.value)
      val partGlobalsOff = rvb.end()

      rvb.start(localColsType)
      rvb.addAnnotation(localColsType, colValuesBc.value)
      val partColsOff = rvb.end()

      it.map { rv =>
        val region = rv.region
        val oldRow = rv.offset

        rvb.set(region)
        rvb.start(localGlobalsType)
        rvb.addRegionValue(localGlobalsType, partRegion, partGlobalsOff)
        val globals = rvb.end()

        rvb.set(region)
        rvb.start(localColsType)
        rvb.addRegionValue(localColsType, partRegion, partColsOff)
        val cols = rvb.end()

        val aggResultsOff = if (rvAggs.nonEmpty) {
          val newRVAggs = rvAggs.map(_.copy())

          seqOps()(region, newRVAggs, 0, true, globals, false, cols, false, oldRow, false)

          rvb.start(aggResultType)
          rvb.startStruct()

          var j = 0
          while (j < newRVAggs.length) {
            newRVAggs(j).result(rvb)
            j += 1
          }
          rvb.endStruct()
          val aggResultsOff = rvb.end()
          aggResultsOff
        } else
          0

        val off = rowF(region, aggResultsOff, false, globals, false, oldRow, false)

        newRV.set(region, off)
        newRV
      }
    }

    if (touchesKeys) {
      val newRDD = prev.rvd.mapPartitions(mapPartitionF)
      prev.copy(typ = typ,
        rvd = OrderedRVD.coerce(typ.orvdType, newRDD, None, None))
    } else {
      val newRVD = prev.rvd.mapPartitionsPreservesPartitioning(typ.orvdType)(mapPartitionF)
      prev.copy(typ = typ, rvd = newRVD)
    }
  }
}

case class MatrixMapCols(child: MatrixIR, newCol: IR) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child, newCol)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapCols = {
    assert(newChildren.length == 2)
    MatrixMapCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val tAggElt: Type = child.typ.entryType
  val aggSymTab = Map(
    "global" -> (0, child.typ.globalType),
    "va" -> (1, child.typ.rvRowType),
    "g" -> (2, child.typ.entryType),
    "sa" -> (3, child.typ.colType))

  val tAgg = TAggregable(tAggElt, aggSymTab)

  val typ: MatrixType = {
    val newColType = newCol.typ.asInstanceOf[TStruct]
    val newColNames = newColType.fieldNames.toSet
    val newColKey = child.typ.colKey.filter(newColNames.contains)
    child.typ.copy(colKey = newColKey, colType = newColType)
  }

  override def partitionCounts: Option[Array[Long]] = child.partitionCounts

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    assert(prev.typ == child.typ)

    val localRowType = prev.typ.rvRowType
    val localEntriesType = prev.typ.entryArrayType
    val entriesIdx = prev.typ.entriesIdx
    val localGlobalsType = prev.typ.globalType
    val localColsType = TArray(prev.typ.colType)
    val localNCols = prev.nCols
    val colValuesBc = prev.colValues.broadcast
    val globalsBc = prev.globals.broadcast

    val colValuesType = TArray(prev.typ.colType)
    val vaType = prev.typ.rvRowType
    val (rvAggs, seqOps, aggResultType, f, rTyp) = ir.CompileWithAggregators[Long, Long, Long, Long, Long, Long, Long](
      "global", localGlobalsType,
      "sa", prev.typ.colType,
      "AGG", tAgg,
      "global", localGlobalsType,
      "colValues", colValuesType,
      "va", vaType,
      newCol, { (nAggs, aggIR) =>
        val colIdx = ir.genUID()

        def rewrite(x: IR): IR = {
          x match {
            case SeqOp(a, i, agg) =>
              SeqOp(a,
                ir.ApplyBinaryPrimOp(ir.Add(),
                  ir.ApplyBinaryPrimOp(ir.Multiply(), ir.Ref(colIdx, TInt32()), ir.I32(nAggs)),
                i),
                agg)
            case x@AggIn(_) =>
              val dummy = genUID()
              AggMap(x, dummy, ir.Ref("g", prev.typ.entryType))
            case _ =>
              ir.Recur(rewrite)(x)
          }
        }

        ir.ArrayFor(
          ir.ArrayRange(ir.I32(0), ir.I32(localNCols), ir.I32(1)),
          colIdx,
          ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", colValuesType), ir.Ref(colIdx, TInt32())),
            ir.Let("g", ir.ArrayRef(
              ir.GetField(ir.Ref("va", vaType), MatrixType.entriesIdentifier),
              ir.Ref(colIdx, TInt32())),
              rewrite(aggIR))))
      })
    assert(rTyp == typ.colType, s"$rTyp, ${ typ.colType }")

    val depth = treeAggDepth(hc, prev.nPartitions)
    val nAggs = rvAggs.length

    val colRVAggs = new Array[RegionValueAggregator](nAggs * localNCols)
    var i = 0
    while (i < localNCols) {
      var j = 0
      while (j < nAggs) {
        colRVAggs(i * nAggs + j) = rvAggs(j).copy()
        j += 1
      }
      i += 1
    }

    val aggResults = if (nAggs > 0) {
      prev.rvd.treeAggregate[Array[RegionValueAggregator]](colRVAggs)({ (colRVAggs, rv) =>
        val rvb = new RegionValueBuilder()
        val region = rv.region
        val oldRow = rv.offset

        rvb.set(region)
        rvb.start(localGlobalsType)
        rvb.addAnnotation(localGlobalsType, globalsBc.value)
        val globals = rvb.end()

        rvb.start(localColsType)
        rvb.addAnnotation(localColsType, colValuesBc.value)
        val cols = rvb.end()

        seqOps()(region, colRVAggs, 0, true, globals, false, cols, false, oldRow, false)

        colRVAggs
      }, { (rvAggs1, rvAggs2) =>
        var i = 0
        while (i < rvAggs1.length) {
          rvAggs1(i).combOp(rvAggs2(i))
          i += 1
        }
        rvAggs1
      }, depth = depth)
    } else
      Array.empty[RegionValueAggregator]

    val prevColType = prev.typ.colType
    val rvb = new RegionValueBuilder()

    val mapF = (a: Annotation, i: Int) => {
      Region.scoped { region =>
        rvb.set(region)

        rvb.start(aggResultType)
        rvb.startStruct()
        var j = 0
        while (j < nAggs) {
          aggResults(i * nAggs + j).result(rvb)
          j += 1
        }
        rvb.endStruct()
        val aggResultsOffset = rvb.end()

        rvb.start(localGlobalsType)
        rvb.addAnnotation(localGlobalsType, globalsBc.value)
        val globalRVoffset = rvb.end()

        val colRVb = new RegionValueBuilder(region)
        colRVb.start(prevColType)
        colRVb.addAnnotation(prevColType, a)
        val colRVoffset = colRVb.end()

        val resultOffset = f()(region, aggResultsOffset, false, globalRVoffset, false, colRVoffset, false)

        SafeRow(coerce[TStruct](rTyp), region, resultOffset)
      }
    }

    val newColValues = BroadcastIndexedSeq(colValuesBc.value.zipWithIndex.map { case (a, i) => mapF(a, i) }, TArray(typ.colType), hc.sc)
    prev.copy(typ = typ, colValues = newColValues)
  }
}

case class MatrixMapGlobals(child: MatrixIR, newRow: IR, value: BroadcastRow) extends MatrixIR {
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: MatrixType =
    child.typ.copy(globalType = newRow.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapGlobals = {
    assert(newChildren.length == 2)
    MatrixMapGlobals(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR], value)
  }

  override def partitionCounts: Option[Array[Long]] = child.partitionCounts

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val (rTyp, f) = ir.Compile[Long, Long, Long](
      "global", child.typ.globalType,
      "value", value.t,
      newRow)
    assert(rTyp == typ.globalType)

    val globalRegion = Region()
    val globalOff = prev.globals.toRegion(globalRegion)
    val valueOff = value.toRegion(globalRegion)
    val newOff = f()(globalRegion, globalOff, false, valueOff, false)

    val newGlobals = prev.globals.copy(
      value = SafeRow(rTyp.asInstanceOf[TStruct], globalRegion, newOff),
      t = rTyp.asInstanceOf[TStruct])

    prev.copy(typ = typ, globals = newGlobals)
  }
}

case class TableValue(typ: TableType, globals: BroadcastRow, rvd: RVD) {
  require(typ.rowType == rvd.rowType)

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

  def write(path: String, overwrite: Boolean, codecSpecJSONStr: String) {
    val hc = HailContext.get

    val codecSpec =
      if (codecSpecJSONStr != null) {
        implicit val formats = RVDSpec.formats
        val codecSpecJSON = JsonMethods.parse(codecSpecJSONStr)
        codecSpecJSON.extract[CodecSpec]
      } else
        CodecSpec.default

    if (overwrite)
      hc.hadoopConf.delete(path, recursive = true)
    else if (hc.hadoopConf.exists(path))
      fatal(s"file already exists: $path")

    hc.hadoopConf.mkDir(path)

    val globalsPath = path + "/globals"
    hc.hadoopConf.mkDir(globalsPath)
    RVD.writeLocalUnpartitioned(hc, globalsPath, typ.globalType, codecSpec, Array(globals.value))

    val partitionCounts = rvd.write(path + "/rows", codecSpec)

    val referencesPath = path + "/references"
    hc.hadoopConf.mkDir(referencesPath)
    ReferenceGenome.exportReferences(hc, referencesPath, typ.rowType)
    ReferenceGenome.exportReferences(hc, referencesPath, typ.globalType)

    val spec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(hc, path)

    hc.hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }

  def export(path: String, typesFile: String = null, header: Boolean = true, exportType: Int = ExportType.CONCATENATED) {
    val hc = HailContext.get
    hc.hadoopConf.delete(path, recursive = true)

    val fields = typ.rowType.fields

    Option(typesFile).foreach { file =>
      exportTypes(file, hc.hadoopConf, fields.map(f => (f.name, f.typ)).toArray)
    }

    val localSignature = typ.rowType
    val localTypes = fields.map(_.typ)

    rvd.mapPartitions { it =>
      val sb = new StringBuilder()

      it.map { rv =>
        val ur = new UnsafeRow(localSignature, rv)
        sb.clear()
        localTypes.indices.foreachBetween { i =>
          sb.append(TableAnnotationImpex.exportAnnotation(ur.get(i), localTypes(i)))
        }(sb += '\t')

        sb.result()
      }
    }.writeTable(path, hc.tmpDir, Some(fields.map(_.name).mkString("\t")).filter(_ => header), exportType = exportType)
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
      BroadcastRow(globals, typ.globalType, hc.sc),
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
    val rvd = hc.sc.parallelize(rows, nPartitions.getOrElse(hc.sc.defaultParallelism))
      .mapPartitions(_.toRegionValueIterator(rowTyp))
    TableValue(typ, BroadcastRow(Row(), typ.globalType, hc.sc), new UnpartitionedRVD(rowTyp, rvd))
  }
}

case class TableImport(paths: Array[String], typ: TableType, readerOpts: TableReaderOptions) extends TableIR {
  assert(typ.globalType.size == 0)
  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableImport = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): TableValue = {
    val rowTyp = typ.rowType
    val nField = rowTyp.fields.length
    val rowFields = rowTyp.fields

    val rvd = hc.sc.textFilesLines(paths, readerOpts.nPartitions)
      .filter { line =>
        !readerOpts.isComment(line.value) &&
          (readerOpts.noHeader || readerOpts.header != line.value) &&
          !(readerOpts.skipBlankLines && line.value.isEmpty)
      }.mapPartitions { it =>
      val region = Region()
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)

      it.map {
        _.map { line =>
          region.clear()

          val split = TextTableReader.splitLine(line, readerOpts.separator, readerOpts.quote)
          if (split.length != nField)
            fatal(s"expected $nField fields, but found ${ split.length } fields")
          rvb.set(region)
          rvb.start(rowTyp)
          rvb.startStruct()

          var i = 0
          while (i < nField) {
            val Field(name, t, _) = rowFields(i)
            val field = split(i)
            try {
              if (field == readerOpts.missing)
                rvb.setMissing()
              else
                rvb.addAnnotation(t, TableAnnotationImpex.importAnnotation(field, t))
            } catch {
              case e: Exception =>
                fatal(s"""${ e.getClass.getName }: could not convert "$field" to $t in column "$name" """)
            }
            i += 1
          }

          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }.value
      }
    }

    TableValue(typ, BroadcastRow(Row.empty, typ.globalType, hc.sc), new UnpartitionedRVD(rowTyp, rvd))
  }
}

case class TableRange(n: Int, nPartitions: Int) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRange = {
    assert(newChildren.length == 0)
    this
  }

  private val partCounts = partition(n, nPartitions)

  override val partitionCounts = Some(partCounts.map(_.toLong))

  val typ: TableType = TableType(
    TStruct("idx" -> TInt32()),
    Array("idx"),
    TStruct.empty())

  def execute(hc: HailContext): TableValue = {
    val localRowType = typ.rowType
    val localPartCounts = partCounts
    val partStarts = partCounts.scanLeft(0)(_ + _)

    TableValue(typ,
      BroadcastRow(Row(), typ.globalType, hc.sc),
      new UnpartitionedRVD(typ.rowType,
        hc.sc.parallelize(Range(0, nPartitions), nPartitions)
          .mapPartitionsWithIndex { case (i, _) =>
            val region = Region()
            val rvb = new RegionValueBuilder(region)
            val rv = RegionValue(region)

            val start = partStarts(i)
            Iterator.range(start, start + localPartCounts(i))
              .map { j =>
                region.clear()
                rvb.start(localRowType)
                rvb.startStruct()
                rvb.addInt(j)
                rvb.endStruct()
                rv.setOffset(rvb.end())
                rv
              }
          }))
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
    val newRowType = newRow.typ.asInstanceOf[TStruct]
    val newKey = child.typ.key.filter(newRowType.fieldIdx.contains)
    child.typ.copy(rowType = newRowType, key = newKey)
  }

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapRows = {
    assert(newChildren.length == 2)
    TableMapRows(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def partitionCounts: Option[Array[Long]] = child.partitionCounts

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

case class TableMapGlobals(child: TableIR, newRow: IR, value: BroadcastRow) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: TableType =
    child.typ.copy(globalType = newRow.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapGlobals = {
    assert(newChildren.length == 2)
    TableMapGlobals(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR], value)
  }

  override def partitionCounts: Option[Array[Long]] = child.partitionCounts

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val gType = typ.globalType

    val (rTyp, f) = ir.Compile[Long, Long, Long](
      "global", child.typ.globalType,
      "value", value.t,
      newRow)
    assert(rTyp == gType)

    val newGlobals = Region.scoped { globalRegion =>
      val globalOff = tv.globals.toRegion(globalRegion)
      val valueOff = value.toRegion(globalRegion)
      val newOff = f()(globalRegion, globalOff, false, valueOff, false)

      tv.globals.copy(
        value = SafeRow(rTyp.asInstanceOf[TStruct], globalRegion, newOff),
        t = rTyp.asInstanceOf[TStruct])
    }

    TableValue(typ, newGlobals, tv.rvd)
  }
}


case class TableExplode(child: TableIR, column: String) extends TableIR {
  def children: IndexedSeq[BaseIR] = Array(child)

  private val columnIdx = child.typ.rowType.fieldIdx(column)
  private val columnType = child.typ.rowType.types(columnIdx).asInstanceOf[TContainer]

  val typ: TableType =
    child.typ.copy(
      rowType = child.typ.rowType.updateKey(column, columnIdx, columnType.elementType))

  def copy(newChildren: IndexedSeq[BaseIR]): TableExplode = {
    assert(newChildren.length == 1)
    TableExplode(newChildren(0).asInstanceOf[TableIR], column)
  }

  def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)

    val childRowType = child.typ.rowType

    val (_, isMissingF) = ir.Compile[Long, Boolean]("row", childRowType,
      ir.IsNA(ir.GetField(Ref("row", childRowType), column)))

    val (_, lengthF) = ir.Compile[Long, Int]("row", childRowType,
      ir.ArrayLen(ir.GetField(Ref("row", childRowType), column)))

    val (resultType, explodeF) = ir.Compile[Long, Int, Long]("row", childRowType,
      "i", TInt32(),
      ir.InsertFields(Ref("row", childRowType),
        Array(column -> ir.ArrayRef(
          ir.GetField(ir.Ref("row", childRowType), column),
          ir.Ref("i", TInt32())))))
    assert(resultType == typ.rowType)

    TableValue(typ, prev.globals,
      prev.rvd.mapPartitions(typ.rowType) { it =>
        val rv2 = RegionValue()

        it.flatMap { rv =>
          val isMissing = isMissingF()(rv.region, rv.offset, false)
          if (isMissing)
            Iterator.empty
          else {
            val end = rv.region.size
            val n = lengthF()(rv.region, rv.offset, false)
            Iterator.range(0, n)
              .map { i =>
                rv.region.clear(end)
                val off = explodeF()(rv.region, rv.offset, false, i, false)
                rv2.set(rv.region, off)
                rv2
              }
          }
        }
      })
  }
}

case class TableUnion(children: IndexedSeq[TableIR]) extends TableIR {
  assert(children.length > 0)
  assert(children.tail.forall(_.typ == children(0).typ))

  def copy(newChildren: IndexedSeq[BaseIR]): TableUnion = {
    TableUnion(newChildren.map(_.asInstanceOf[TableIR]))
  }

  val typ: TableType = children(0).typ

  def execute(hc: HailContext): TableValue = {
    val tvs = children.map(_.execute(hc))
    tvs(0).copy(
      rvd = RVD.union(tvs.map(_.rvd)))
  }
}

case class MatrixRowsTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRowsTable = {
    assert(newChildren.length == 1)
    MatrixRowsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.rowsTableType

  def execute(hc: HailContext): TableValue = {
    val mv = child.execute(hc)
    val rtv = mv.rowsTableValue
    assert(rtv.typ == typ)
    rtv
  }
}

case class MatrixColsTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixColsTable = {
    assert(newChildren.length == 1)
    MatrixColsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.colsTableType

  def execute(hc: HailContext): TableValue = {
    val mv = child.execute(hc)
    val ctv = mv.colsTableValue
    assert(ctv.typ == typ)
    ctv
  }
}

case class MatrixEntriesTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixEntriesTable = {
    assert(newChildren.length == 1)
    MatrixEntriesTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.entriesTableType

  def execute(hc: HailContext): TableValue = {
    val mv = child.execute(hc)
    val etv = mv.entriesTableValue
    assert(etv.typ == typ)
    etv
  }
}

case class MatrixFilterEntries(child: MatrixIR, pred: IR) extends MatrixIR {
  val children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterEntries = {
    assert(newChildren.length == 2)
    MatrixFilterEntries(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue = {
    val mv = child.execute(hc)

    val localGlobalType = child.typ.globalType
    val globalsBc = mv.globals.broadcast
    val localColValuesType = TArray(child.typ.colType)
    val colValuesBc = mv.colValues.broadcast

    val colValuesType = TArray(child.typ.colType)

    val x = ir.InsertFields(ir.Ref("va", child.typ.rvRowType),
      FastSeq(MatrixType.entriesIdentifier ->
        ir.ArrayMap(ir.ArrayRange(ir.I32(0), ir.I32(mv.nCols), ir.I32(1)),
          "i",
          ir.Let("g",
            ir.ArrayRef(
              ir.GetField(ir.Ref("va", child.typ.rvRowType), MatrixType.entriesIdentifier),
              ir.Ref("i", TInt32())),
            ir.If(
              ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", colValuesType), ir.Ref("i", TInt32())),
                pred),
              ir.Ref("g", child.typ.entryType),
              ir.NA(child.typ.entryType))))))

    val (t, f) = ir.Compile[Long, Long, Long, Long](
      "global", child.typ.globalType,
      "colValues", colValuesType,
      "va", child.typ.rvRowType,
      x)
    assert(t == typ.rvRowType)

    val mapPartitionF = { it: Iterator[RegionValue] =>
      val rvb = new RegionValueBuilder()
      val rowF = f()
      val newRV = RegionValue()

      it.map { rv =>
        val region = rv.region
        rvb.set(region)

        rvb.start(localGlobalType)
        rvb.addAnnotation(localGlobalType, globalsBc.value)
        val globalsOff = rvb.end()

        rvb.start(localColValuesType)
        rvb.addAnnotation(localColValuesType, colValuesBc.value)
        val colValuesOff = rvb.end()

        val off = rowF(region, globalsOff, false, colValuesOff, false, rv.offset, false)

        newRV.set(region, off)
        newRV
      }
    }

    val newRVD = mv.rvd.mapPartitionsPreservesPartitioning(typ.orvdType)(mapPartitionF)
    mv.copy(rvd = newRVD)
  }
}
