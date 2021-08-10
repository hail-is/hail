package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.backend.spark.SparkBackend
import is.hail.expr.JSONAnnotationImpex
import is.hail.types.physical.{PArray, PCanonicalStruct, PStruct, PType}
import is.hail.types.virtual._
import is.hail.types.{MatrixType, TableType}
import is.hail.io.{BufferSpec, FileWriteMetadata, MatrixWriteCheckpoint}
import is.hail.io.fs.FS
import is.hail.linalg.RowMatrix
import is.hail.rvd.{AbstractRVDSpec, RVD, _}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.commons.lang3.StringUtils
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods
import org.json4s.jackson.JsonMethods.parse

case class MatrixValue(
  typ: MatrixType,
  tv: TableValue) {
  val colFieldType = tv.globals.t.fieldType(LowerMatrixIR.colsFieldName).asInstanceOf[PArray]
  assert(colFieldType.required)
  assert(colFieldType.elementType.required)

  lazy val globals: BroadcastRow = {
    val prevGlobals = tv.globals
    val newT = prevGlobals.t.deleteField(LowerMatrixIR.colsFieldName)
    val rvb = new RegionValueBuilder(prevGlobals.value.region)
    rvb.start(newT)
    rvb.startStruct()
    rvb.addFields(prevGlobals.t, prevGlobals.value,
      prevGlobals.t.fields.filter(_.name != LowerMatrixIR.colsFieldName).map(_.index).toArray)
    rvb.endStruct()
    BroadcastRow(tv.ctx, RegionValue(prevGlobals.value.region, rvb.end()), newT)
  }

  lazy val colValues: BroadcastIndexedSeq = {
    val prevGlobals = tv.globals
    val field = prevGlobals.t.field(LowerMatrixIR.colsFieldName)
    val t = field.typ.asInstanceOf[PArray]
    BroadcastIndexedSeq(tv.ctx,
      RegionValue(prevGlobals.value.region, prevGlobals.t.loadField(prevGlobals.value.offset, field.index)),
      t)
  }

  val rvd: RVD = tv.rvd
  lazy val rvRowPType: PStruct = rvd.typ.rowType
  lazy val rvRowType: TStruct = rvRowPType.virtualType
  lazy val entriesIdx: Int = rvRowPType.fieldIdx(MatrixType.entriesIdentifier)
  lazy val entryArrayPType: PArray = rvRowPType.types(entriesIdx).asInstanceOf[PArray]
  lazy val entryArrayType: TArray = rvRowType.types(entriesIdx).asInstanceOf[TArray]
  lazy val entryPType: PStruct = entryArrayPType.elementType.asInstanceOf[PStruct]
  lazy val entryType: TStruct = entryArrayType.elementType.asInstanceOf[TStruct]

  lazy val entriesRVType: TStruct = TStruct(
    MatrixType.entriesIdentifier -> TArray(entryType))

  require(rvd.typ.key.startsWith(typ.rowKey), s"\nmat row key: ${ typ.rowKey }\nrvd key: ${ rvd.typ.key }")

  def sparkContext: SparkContext = rvd.sparkContext

  def nPartitions: Int = rvd.getNumPartitions

  lazy val nCols: Int = colValues.t.loadLength(colValues.value.offset)

  def stringSampleIds: IndexedSeq[String] = {
    val colKeyTypes = typ.colKeyStruct.types
    assert(colKeyTypes.length == 1 && colKeyTypes(0) == TString, colKeyTypes.toSeq)
    val querier = typ.colType.query(typ.colKey(0))
    colValues.javaValue.map(querier(_).asInstanceOf[String])
  }

  def requireUniqueSamples(method: String) {
    val dups = stringSampleIds.counter().filter(_._2 > 1).toArray
    if (dups.nonEmpty)
      fatal(s"Method '$method' does not support duplicate column keys. Duplicates:" +
        s"\n  @1", dups.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
  }

  def referenceGenome: ReferenceGenome = typ.referenceGenome

  def colsTableValue(ctx: ExecuteContext): TableValue =
    TableValue(ctx, typ.colsTableType, globals, colsRVD(ctx))

  private def writeCols(ctx: ExecuteContext, path: String, bufferSpec: BufferSpec): Long = {
    val fs = ctx.fs
    val fileData = AbstractRVDSpec.writeSingle(ctx, path + "/rows", colValues.t.elementType.asInstanceOf[PStruct], bufferSpec, colValues.javaValue)
    val partitionCounts = fileData.map(_.rowsWritten)

    val colsSpec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "../references",
      typ.colsTableType.copy(key = FastIndexedSeq[String]()),
      Map("globals" -> RVDComponentSpec("../globals/rows"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    colsSpec.write(fs, path)

    using(fs.create(path + "/_SUCCESS"))(out => ())

    fileData.map(_.bytesWritten).sum
  }

  private def writeGlobals(ctx: ExecuteContext, path: String, bufferSpec: BufferSpec): Long = {
    val fs = ctx.fs
    val fileData = AbstractRVDSpec.writeSingle(ctx, path + "/rows", globals.t, bufferSpec, Array(globals.javaValue))
    val partitionCounts = fileData.map(_.rowsWritten)

    AbstractRVDSpec.writeSingle(ctx, path + "/globals", PCanonicalStruct.empty(required = true), bufferSpec, Array[Annotation](Row()))

    val globalsSpec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "../references",
      TableType(typ.globalType, FastIndexedSeq(), TStruct.empty),
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    globalsSpec.write(fs, path)

    using(fs.create(path + "/_SUCCESS"))(out => ())
    fileData.map(_.bytesWritten).sum
  }

  private def finalizeWrite(
    ctx: ExecuteContext,
    path: String,
    bufferSpec: BufferSpec,
    fileData: Array[FileWriteMetadata],
    consoleInfo: Boolean
  ): Unit = {
    val fs = ctx.fs
    val globalsPath = path + "/globals"
    fs.mkDir(globalsPath)
    val globalBytesWritten = writeGlobals(ctx, globalsPath, bufferSpec)

    val partitionCounts = fileData.map(_.rowsWritten)

    val rowsSpec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "../references",
      typ.rowsTableType,
      Map("globals" -> RVDComponentSpec("../globals/rows"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    rowsSpec.write(fs, path + "/rows")

    using(fs.create(path + "/rows/_SUCCESS"))(out => ())

    val entriesSpec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "../references",
      TableType(entriesRVType, FastIndexedSeq(), typ.globalType),
      Map("globals" -> RVDComponentSpec("../globals/rows"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    entriesSpec.write(fs, path + "/entries")

    using(fs.create(path + "/entries/_SUCCESS"))(out => ())

    fs.mkDir(path + "/cols")
    val colBytesWritten = writeCols(ctx, path + "/cols", bufferSpec)

    val refPath = path + "/references"
    fs.mkDir(refPath)
    Array(typ.colType, typ.rowType, entryType, typ.globalType).foreach { t =>
      ReferenceGenome.exportReferences(fs, refPath, t)
    }

    val spec = MatrixTableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec("globals/rows"),
        "cols" -> RVDComponentSpec("cols/rows"),
        "rows" -> RVDComponentSpec("rows/rows"),
        "entries" -> RVDComponentSpec("entries/rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(fs, path)

    writeNativeFileReadMe(fs, path)

    using(fs.create(path + "/_SUCCESS"))(_ => ())

    val nRows = partitionCounts.sum
    val printer: String => Unit = if (consoleInfo) info else log.info

    val partitionBytesWritten = fileData.map(_.bytesWritten)
    val totalRowsEntriesBytes = partitionBytesWritten.sum
    val totalBytesWritten: Long = totalRowsEntriesBytes + colBytesWritten + globalBytesWritten
    val smallestPartition = fileData.minBy(_.bytesWritten)
    val largestPartition = fileData.maxBy(_.bytesWritten)
    val smallestStr = s"${ smallestPartition.rowsWritten } rows (${ formatSpace(smallestPartition.bytesWritten) })"
    val largestStr = s"${ largestPartition.rowsWritten } rows (${ formatSpace(largestPartition.bytesWritten) })"

    printer(s"wrote matrix table with $nRows ${ plural(nRows, "row") } " +
      s"and $nCols ${ plural(nCols, "column") } " +
      s"in ${ partitionCounts.length } ${ plural(partitionCounts.length, "partition") } " +
      s"to $path" +
      s"\n    Total size: ${ formatSpace(totalBytesWritten) }" +
      s"\n    * Rows/entries: ${ formatSpace(totalRowsEntriesBytes) }" +
      s"\n    * Columns: ${ formatSpace(colBytesWritten) }" +
      s"\n    * Globals: ${ formatSpace(globalBytesWritten) }" +
      s"\n    * Smallest partition: $smallestStr" +
      s"\n    * Largest partition:  $largestStr")
  }

  def write(ctx: ExecuteContext,
    path: String,
    overwrite: Boolean,
    stageLocally: Boolean,
    codecSpecJSON: String,
    partitions: String,
    partitionsTypeStr: String,
    checkpointFile: String): Unit = {
    assert(typ.isCanonical)
    val fs = ctx.fs

    val bufferSpec = BufferSpec.parseOrDefault(codecSpecJSON)

    if (overwrite) {
      if (checkpointFile != null)
        fatal(s"cannot currently use a checkpoint file with overwrite=True")
      fs.delete(path, recursive = true)
    } else if (fs.exists(path))
      if (checkpointFile == null || fs.exists(path + "/_SUCCESS"))
        fatal(s"file already exists: $path")

    fs.mkDir(path)

    val targetPartitioner =
      if (partitions != null) {
        if (checkpointFile != null)
          fatal(s"cannot currently use a checkpoint file with `partitions` argument")
        val partitionsType = IRParser.parseType(partitionsTypeStr)
        val jv = JsonMethods.parse(partitions)
        val rangeBounds = JSONAnnotationImpex.importAnnotation(jv, partitionsType)
          .asInstanceOf[IndexedSeq[Interval]]
        new RVDPartitioner(typ.rowKey.toArray, typ.rowKeyStruct, rangeBounds)
      } else
        null

    val checkpoint = Option(checkpointFile).map(path => MatrixWriteCheckpoint.read(fs, path, path, rvd.getNumPartitions))

    val fileData = rvd.writeRowsSplit(ctx, path, bufferSpec, stageLocally, targetPartitioner, checkpoint)

    finalizeWrite(ctx, path, bufferSpec, fileData, consoleInfo = true)
  }

  def colsRVD(ctx: ExecuteContext): RVD = {
    // only used in exportPlink
    assert(typ.colKey.isEmpty)
    val colPType = PType.canonical(typ.colType).setRequired(true).asInstanceOf[PStruct]

    RVD.coerce(ctx,
      typ.colsTableType.canonicalRVDType,
      ContextRDD.parallelize(colValues.safeJavaValue)
        .cmapPartitions { (ctx, it) => it.copyToRegion(ctx.region, colPType) })
  }

  def toRowMatrix(entryField: String): RowMatrix = {
    val partCounts: Array[Long] = rvd.countPerPartition()
    val partStarts = partCounts.scanLeft(0L)(_ + _)
    assert(partStarts.length == rvd.getNumPartitions + 1)
    val partStartsBc = HailContext.backend.broadcast(partStarts)

    val localRvRowPType = rvRowPType
    val localEntryArrayPType = entryArrayPType
    val localEntryPType = entryPType
    val fieldType = entryPType.field(entryField).typ

    assert(fieldType.virtualType == TFloat64)

    val localEntryArrayIdx = entriesIdx
    val fieldIdx = entryType.fieldIdx(entryField)
    val numColsLocal = nCols

    val rows = rvd.mapPartitionsWithIndex { (pi, _, it) =>
      var i = partStartsBc.value(pi)
      it.map { ptr =>
        val data = new Array[Double](numColsLocal)
        val entryArrayOffset = localRvRowPType.loadField(ptr, localEntryArrayIdx)
        var j = 0
        while (j < numColsLocal) {
          if (localEntryArrayPType.isElementDefined(entryArrayOffset, j)) {
            val entryOffset = localEntryArrayPType.loadElement(entryArrayOffset, j)
            if (localEntryPType.isFieldDefined(entryOffset, fieldIdx)) {
              val fieldOffset = localEntryPType.loadField(entryOffset, fieldIdx)
              data(j) = Region.loadDouble(fieldOffset)
            } else
              fatal(s"Cannot create RowMatrix: missing value at row $i and col $j")
          } else
            fatal(s"Cannot create RowMatrix: filtered entry at row $i and col $j")
          j += 1
        }
        val row = (i, data)
        i += 1
        row
      }
    }

    new RowMatrix(rows, nCols, Some(partStarts.last), Some(partCounts))
  }

  def typeCheck(): Unit = {
    assert(typ.globalType.typeCheck(globals.value))
    assert(TArray(typ.colType).typeCheck(colValues.value))
    val localRVRowType = rvRowType
    assert(rvd.toRows.forall(r => localRVRowType.typeCheck(r)))
  }

  def toTableValue: TableValue = tv
}

object MatrixValue {
  def writeMultiple(
    ctx: ExecuteContext,
    mvs: IndexedSeq[MatrixValue],
    prefix: String,
    overwrite: Boolean,
    stageLocally: Boolean
  ): Unit = {
    val first = mvs.head
    require(mvs.forall(_.typ == first.typ))
    val fs = ctx.fs
    val bufferSpec = BufferSpec.default

    val d = digitsNeeded(mvs.length)
    val paths = (0 until mvs.length).map { i => prefix + StringUtils.leftPad(i.toString, d, '0') + ".mt" }
    paths.foreach { path =>
      if (overwrite)
        fs.delete(path, recursive = true)
      else if (fs.exists(path))
        fatal(s"file already exists: $path")
      fs.mkDir(path)
    }

    val fileData = RVD.writeRowsSplitFiles(ctx, mvs.map(_.rvd), prefix, bufferSpec, stageLocally)
    for ((mv, path, fd) <- (mvs, paths, fileData).zipped) {
      mv.finalizeWrite(ctx, path, bufferSpec, fd, consoleInfo = false)
    }
  }

  def apply(
    ctx: ExecuteContext,
    typ: MatrixType,
    globals: Row,
    colValues: IndexedSeq[Row],
    rvd: RVD): MatrixValue = {
    val globalsType = typ.globalType.appendKey(LowerMatrixIR.colsFieldName, TArray(typ.colType))
    val globalsPType = PType.canonical(globalsType).asInstanceOf[PStruct]
    val rvb = new RegionValueBuilder(ctx.r)
    rvb.start(globalsPType)
    rvb.startStruct()
    typ.globalType.fields.foreach { f =>
      rvb.addAnnotation(f.typ, globals.get(f.index))
    }
    rvb.addAnnotation(TArray(typ.colType), colValues)

    MatrixValue(typ,
      TableValue(ctx, TableType(
        rowType = rvd.rowType,
        key = typ.rowKey,
        globalType = globalsType),
        BroadcastRow(ctx, RegionValue(ctx.r, rvb.end()), globalsPType),
        rvd))
  }

}
