package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.types.{MatrixType, TArray, TStruct, TableType}
import is.hail.expr.types._
import is.hail.io.CodecSpec
import is.hail.rvd.{OrderedRVD, OrderedRVDType, RVDSpec, _}
import is.hail.sparkextras.ContextRDD
import is.hail.table.TableSpec
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods.parse

case class MatrixValue(
  typ: MatrixType,
  globals: BroadcastRow,
  colValues: BroadcastIndexedSeq,
  rvd: OrderedRVD) {

    assert(rvd.typ == typ.orvdType, s"\nrvdType: ${rvd.typ}\nmatType: ${typ.orvdType}")

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

      val partitionCounts = RVDSpec.writeLocal(hc, path + "/rows", typ.colType, codecSpec, colValues.value)

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

      val partitionCounts = RVDSpec.writeLocal(hc, path + "/rows", typ.globalType, codecSpec, Array(globals.value))

      RVDSpec.writeLocal(hc, path + "/globals", TStruct.empty(), codecSpec, Array[Annotation](Row()))

      val globalsSpec = TableSpec(
        FileFormat.version.rep,
        hc.version,
        "../references",
        TableType(typ.globalType, None, TStruct.empty()),
        Map("globals" -> RVDComponentSpec("globals"),
          "rows" -> RVDComponentSpec("rows"),
          "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
      globalsSpec.write(hc, path)

      hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
    }

    def write(path: String, overwrite: Boolean = false, stageLocally: Boolean = false, codecSpecJSONStr: String = null) = {
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

      val partitionCounts = rvd.writeRowsSplit(path, typ, codecSpec, stageLocally)

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
        TableType(typ.entriesRVType, None, typ.globalType),
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

    lazy val (sortedColValues, sortedColsToOldIdx): (BroadcastIndexedSeq, BroadcastIndexedSeq) = {
      val (_, keyF) = typ.colType.select(typ.colKey)
      if (typ.colKey.isEmpty)
        (colValues,
          BroadcastIndexedSeq(
            IndexedSeq.range(0, colValues.safeValue.length),
            TArray(TInt32()),
            colValues.sc))
      else {
        val sortedValsWithIdx = colValues.safeValue
          .zipWithIndex
          .map(colIdx => (keyF(colIdx._1.asInstanceOf[Row]), colIdx))
          .sortBy(_._1)(typ.colKeyStruct.ordering.toOrdering.asInstanceOf[Ordering[Row]])
          .map(_._2)

        (colValues.copy(value = sortedValsWithIdx.map(_._1)),
          BroadcastIndexedSeq(
            sortedValsWithIdx.map(_._2),
            TArray(TInt32()),
            colValues.sc))
      }
    }

    def rowsRVD(): OrderedRVD = {
      val localRowType = typ.rowType
      val fullRowType = typ.rvRowType
      val localEntriesIndex = typ.entriesIdx
      rvd.mapPartitions(
        OrderedRVDType(typ.rowType, typ.rowKey)
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

    def colsRVD(): OrderedRVD = {
      val hc = HailContext.get
      val signature = typ.colType

      OrderedRVD.coerce(
        typ.colsTableType.rvdType,
        ContextRDD.parallelize(hc.sc, sortedColValues.safeValue.asInstanceOf[IndexedSeq[Row]])
          .cmapPartitions { (ctx, it) => it.toRegionValueIterator(ctx.region, signature) }
      )
    }

    def entriesRVD(): OrderedRVD = {
      val resultStruct = typ.entriesTableType.rowType
      val fullRowType = typ.rvRowType
      val localEntriesIndex = typ.entriesIdx
      val localEntriesType = typ.entryArrayType
      val localColType = typ.colType
      val localEntryType = typ.entryType
      val localOrvdType = typ.orvdType
      val localNCols = nCols

      val localSortedColValues = sortedColValues.broadcast
      val localSortedColsToOldIdx = sortedColsToOldIdx.broadcast

      rvd.repartition(rvd.partitioner.strictify).boundary
        .mapPartitions(typ.entriesTableType.rvdType.copy(key = typ.rowKey),
          { (ctx, it) =>
            val rv2b = ctx.rvb
            val rv2 = RegionValue(ctx.region)

            val colRegion = ctx.freshRegion
            val colRVB = new RegionValueBuilder(colRegion)
            val colsType = TArray(localColType, required = true)
            colRVB.start(colsType)
            colRVB.addAnnotation(colsType, localSortedColValues.value)
            val colsOffset = colRVB.end()

            val rowBuffer = new RegionValueArrayBuffer(fullRowType, ctx.freshRegion)

            val colsNewToOldIdx = localSortedColsToOldIdx.value.asInstanceOf[IndexedSeq[Int]]

            OrderedRVIterator(localOrvdType, it, ctx).staircase.flatMap { step =>
              rowBuffer.clear()
              rowBuffer ++= step
              (0 until localNCols).iterator.flatMap { i =>
                rowBuffer.iterator
                  .filter { row =>
                    localEntriesType.isElementDefined(row.region, fullRowType.loadField(row, localEntriesIndex), colsNewToOldIdx(i))
                  }.map { row =>
                    rv2b.clear()
                    rv2b.start(resultStruct)
                    rv2b.startStruct()

                    var j = 0
                    while (j < fullRowType.size) {
                      if (j != localEntriesIndex)
                        rv2b.addField(fullRowType, row, j)
                      j += 1
                    }

                    rv2b.addAllFields(localColType, colRegion, colsType.loadElement(colRegion, colsOffset, i))
                    rv2b.addAllFields(
                      localEntryType,
                      row.region,
                      localEntriesType.loadElement(row.region, fullRowType.loadField(row, localEntriesIndex), colsNewToOldIdx(i)))
                    rv2b.endStruct()
                    rv2.setOffset(rv2b.end())
                    rv2
                  }
              }
            }
          }).extendKeyPreservesPartitioning(typ.entriesTableType.rvdType.key)
    }

    def insertEntries[PC](makePartitionContext: () => PC, newColType: TStruct = typ.colType,
      newColKey: IndexedSeq[String] = typ.colKey,
      newColValues: BroadcastIndexedSeq = colValues,
      newGlobalType: TStruct = typ.globalType,
      newGlobals: BroadcastRow = globals)(newEntryType: TStruct,
      inserter: (PC, RegionValue, RegionValueBuilder) => Unit): MatrixValue = {
      insertIntoRow(makePartitionContext, newColType, newColKey, newColValues, newGlobalType, newGlobals)(
        TArray(newEntryType), MatrixType.entriesIdentifier, inserter)
    }

    def insertIntoRow[PC](makePartitionContext: () => PC, newColType: TStruct = typ.colType,
      newColKey: IndexedSeq[String] = typ.colKey,
      newColValues: BroadcastIndexedSeq = colValues,
      newGlobalType: TStruct = typ.globalType,
      newGlobals: BroadcastRow = globals)(typeToInsert: Type, path: String,
      inserter: (PC, RegionValue, RegionValueBuilder) => Unit): MatrixValue = {
      assert(!typ.rowKey.contains(path))

      val fullRowType = typ.rvRowType
      val localEntriesIndex = typ.entriesIdx

      val (newRVPType, ins) = fullRowType.physicalType.unsafeStructInsert(typeToInsert.physicalType, List(path))
      val newRVType = newRVPType.virtualType


      val newMatrixType = typ.copy(rvRowType = newRVType, colType = newColType,
        colKey = newColKey, globalType = newGlobalType)

      MatrixValue(
        newMatrixType,
        newGlobals,
        newColValues,
        rvd.mapPartitions(newMatrixType.orvdType) { it =>

          val pc = makePartitionContext()

          val rv2 = RegionValue()
          val rvb = new RegionValueBuilder()
          it.map { rv =>
            rvb.set(rv.region)
            rvb.start(newRVType)

            ins(rv.region, rv.offset, rvb,
              () => inserter(pc, rv, rvb)
            )

            rv2.set(rv.region, rvb.end())
            rv2
          }
        })
    }
  }
