package is.hail.expr.ir

import is.hail.HailContext
import is.hail.rvd._
import is.hail.annotations._
import is.hail.expr.types.{MatrixType, TStruct, TableType}
import is.hail.io.CodecSpec
import is.hail.rvd.{OrderedRVD, OrderedRVDType, RVD, RVDSpec, UnpartitionedRVD}
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
        TableType(typ.globalType, None, TStruct.empty()),
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

      new UnpartitionedRVD(
        signature,
        ContextRDD.parallelize(hc.sc, colValues.value.toArray.map(_.asInstanceOf[Row]))
          .cmapPartitions { (ctx, it) => it.toRegionValueIterator(ctx.region, signature) })
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

      rvd.boundary.mapPartitions(resultStruct, { (ctx, it) =>
        val rv2b = ctx.rvb
        val rv2 = RegionValue(ctx.region)

        it.flatMap { rv =>
          val gsOffset = fullRowType.loadField(rv, localEntriesIndex)
          (0 until localNCols).iterator
            .filter { i =>
              localEntriesType.isElementDefined(rv.region, gsOffset, i)
            }
            .map { i =>
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
              rv2.setOffset(rv2b.end())
              rv2
            }
        }
      })
    }
  }
