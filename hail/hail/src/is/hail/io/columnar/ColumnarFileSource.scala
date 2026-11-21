package is.hail.io.columnar

import is.hail.io.fs.{FS, SeekableDataInputStream}
import is.hail.utils.{fatal, Interval}

import scala.jdk.CollectionConverters._

import org.apache.arrow.vector.types.pojo.Schema
import org.apache.parquet.ParquetReadOptions
import org.apache.parquet.column.{ColumnDescriptor, ColumnReader}
import org.apache.parquet.column.impl.ColumnReadStoreImpl
import org.apache.parquet.column.page.PageReadStore
import org.apache.parquet.conf.PlainParquetConfiguration
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.internal.filter2.columnindex.RowRanges
import org.apache.parquet.io.{DelegatingSeekableInputStream, InputFile, SeekableInputStream}
import org.apache.parquet.io.api.{Converter, GroupConverter, PrimitiveConverter}
import org.apache.parquet.schema.{GroupType, MessageType}
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName
import org.apache.spark.sql.Row

/** What a task needs to open a [[ColumnarFileSource]]: the Arrow schema the file must have, as a
  * base64 encapsulated IPC message, the Parquet paths of the leaf columns to read, and for a
  * key-range read, the key.
  *
  * Arrow schemas aren't serializable, so this carries the encoded form to workers.
  */
case class ColumnarSourceSpec(
  arrowSchema: String,
  leaves: IndexedSeq[IndexedSeq[String]],
  keyRange: Option[ColumnarKeyRange],
) {
  def open(fs: FS, path: String): ColumnarFileSource = new ColumnarFileSource(fs, path, this, None)

  /** Opens the file at `path` for a key-range read. `context` is the partition reader's context, a
    * row of the file's path and [[ColumnarKeyRange.contextFields]].
    */
  def openKeyRange(fs: FS, context: AnyRef): ColumnarFileSource = {
    val Row(path: String, rowFile: String, intervals: IndexedSeq[Interval @unchecked]) = context
    new ColumnarFileSource(fs, path, this, Some((rowFile, intervals)))
  }
}

object ColumnarFileSource {
  private class HailInputFile(fs: FS, path: String) extends InputFile {
    override def getLength: Long = fs.getFileSize(path)

    override def newStream(): SeekableInputStream = {
      val is: SeekableDataInputStream = fs.openNoCompression(path)
      new DelegatingSeekableInputStream(is) {
        override def getPos: Long = is.getPosition

        override def seek(newPos: Long): Unit = is.seek(newPos)
      }
    }
  }

  private object NoConverter extends PrimitiveConverter

  /** Converts the record of `t` to nothing: column readers need a converter for each leaf. */
  private class NoGroupConverter(t: GroupType) extends GroupConverter {
    private[this] val children: Array[Converter] = t.getFields.asScala.map { f =>
      if (f.isPrimitive) NoConverter else new NoGroupConverter(f.asGroupType): Converter
    }.toArray

    override def getConverter(i: Int): Converter = children(i)

    override def start(): Unit = {}

    override def end(): Unit = {}
  }

  /** `interval` in the representation `partitionerContains` takes, as an interval of key prefixes.
    */
  private def fromIRRepresentation(interval: Interval): Interval = {
    def prefix(p: Any): Row = {
      val Row(key: Row, length: Int) = p
      Row.fromSeq(key.toSeq.take(length))
    }
    Interval(
      prefix(interval.start),
      prefix(interval.end),
      interval.includesStart,
      interval.includesEnd,
    )
  }

  /** Opens the Parquet file at `path`, reading only its footer. */
  def openFile(fs: FS, path: String): ParquetFileReader =
    ParquetFileReader.open(
      new HailInputFile(fs, path),
      ParquetReadOptions.builder(new PlainParquetConfiguration()).build(),
    )

  /** The Arrow schema in the footer of an open file (spec section 4.6). */
  def arrowSchema(reader: ParquetFileReader, path: String): Schema =
    Option(reader.getFileMetaData.getKeyValueMetaData.get(ColumnarFileSink.ArrowSchemaKey)) match {
      case Some(s) => ColumnarFileSink.decodeSchema(s)
      case None => fatal(s"columnar reader: '$path' has no ${ColumnarFileSink.ArrowSchemaKey}")
    }

  /** The Arrow schema in the footer of the Parquet file at `path`. */
  def readArrowSchema(fs: FS, path: String): Schema = {
    val reader = openFile(fs, path)
    try arrowSchema(reader, path)
    finally reader.close()
  }
}

/** Reads the leaf columns of one Parquet data file a record at a time, for emitted code to assemble
  * into Hail values.
  *
  * Only the column chunks of the requested leaves are read. `nextRecord` buffers each leaf's
  * repetition levels, definition levels and values for the next record in a [[ColumnarLeaf]], so
  * emitted code can look ahead within a record, for example to find a list's length.
  *
  * In a key-range read, `keyRange` is the row partition's row file and its key intervals, in the
  * representation `partitionerContains` takes. Only the rows [[ColumnarKeyRange.rowRanges]] picks
  * are read, and only the pages that hold them.
  */
final class ColumnarFileSource(
  fs: FS,
  path: String,
  spec: ColumnarSourceSpec,
  keyRange: Option[(String, IndexedSeq[Interval])],
) {
  import ColumnarFileSource._

  private[this] val reader: ParquetFileReader = openFile(fs, path)

  /** The rows of each row group of the row file open in `rowFileReader` that may hold keys in the
    * key intervals of `keyRange`.
    */
  private[this] def pickRows(rowFileReader: ParquetFileReader, intervals: IndexedSeq[Interval])
    : IndexedSeq[RowRanges] =
    spec.keyRange.get.rowRanges(rowFileReader, intervals.map(fromIRRepresentation))

  // The rows to read of each row group.
  private[this] val rowRanges: IndexedSeq[RowRanges] =
    try {
      if (arrowSchema(reader, path) != ColumnarFileSink.decodeSchema(spec.arrowSchema))
        fatal(s"columnar reader: the Arrow schema of '$path' differs from its manifest's")
      keyRange match {
        case None => reader.getRowGroups.asScala.toIndexedSeq.map(ColumnarKeyRange.allRows)
        case Some((rowFile, intervals)) if rowFile == path => pickRows(reader, intervals)
        case Some((rowFile, intervals)) =>
          val rowFileReader = openFile(fs, rowFile)
          try {
            def rowCounts(r: ParquetFileReader) = r.getRowGroups.asScala.map(_.getRowCount)
            if (rowCounts(rowFileReader) != rowCounts(reader))
              fatal(s"columnar reader: the row groups of '$path' differ from those of its row " +
                s"file '$rowFile'")
            pickRows(rowFileReader, intervals)
          } finally rowFileReader.close()
      }
    } catch {
      case e: Throwable =>
        reader.close()
        throw e
    }

  private[this] val descriptors: IndexedSeq[ColumnDescriptor] =
    try {
      val schema = reader.getFileMetaData.getSchema
      val ds = spec.leaves.map { p =>
        if (!schema.containsPath(p.toArray))
          fatal(s"columnar reader: '$path' has no Parquet column ${p.mkString(".")}")
        schema.getColumnDescription(p.toArray)
      }
      reader.setRequestedSchema(ds.asJava)
      ds
    } catch {
      case e: Throwable =>
        reader.close()
        throw e
    }

  private[this] val schema: MessageType = reader.getFileMetaData.getSchema

  private[this] val converter: GroupConverter = new NoGroupConverter(schema)

  private[this] val leaves: Array[ColumnarLeaf] =
    descriptors.map(d =>
      new ColumnarLeaf(d.getMaxDefinitionLevel, d.getPrimitiveType.getPrimitiveTypeName)
    ).toArray

  private[this] var nextRowGroup: Int = 0
  private[this] var recordsLeft: Long = 0

  def leaf(i: Int): ColumnarLeaf = leaves(i)

  /** Buffers the next record of every leaf. False at the end of the file. */
  def nextRecord(): Boolean = {
    while (recordsLeft == 0) {
      if (nextRowGroup == rowRanges.length) return false
      val i = nextRowGroup
      nextRowGroup += 1
      val ranges = rowRanges(i)
      recordsLeft = ranges.rowCount
      // With no leaves requested, records are only counted, and no column chunk is read.
      if (recordsLeft > 0 && leaves.nonEmpty) {
        val pages: PageReadStore =
          if (recordsLeft == reader.getRowGroups.get(i).getRowCount) reader.readRowGroup(i)
          else reader.readFilteredRowGroup(i, ranges)
        val columns = new ColumnReadStoreImpl(
          pages,
          converter,
          schema,
          reader.getFileMetaData.getCreatedBy,
        )
        descriptors.indices.foreach(j => leaves(j).reset(columns.getColumnReader(descriptors(j))))
      }
    }
    leaves.foreach(_.fill())
    recordsLeft -= 1
    true
  }

  def close(): Unit = reader.close()
}

/** One record of a leaf column: its (repetition level, definition level, value) triples, and a
  * cursor over them. A value is only defined where the definition level is the maximum.
  *
  * Primitive values are held as longs: integers and booleans by value, floats and doubles by their
  * bits.
  */
final class ColumnarLeaf(maxDefinitionLevel: Int, typeName: PrimitiveTypeName) {
  private[this] var column: ColumnReader = _

  private[this] var reps: Array[Int] = new Array[Int](8)
  private[this] var defs: Array[Int] = new Array[Int](8)

  private[this] var values: Array[Long] =
    if (typeName == PrimitiveTypeName.BINARY) null else new Array[Long](8)

  private[this] var binaries: Array[Array[Byte]] =
    if (typeName == PrimitiveTypeName.BINARY) new Array[Array[Byte]](8) else null

  private[this] var size: Int = 0
  private[this] var pos: Int = 0

  /** Starts reading `column`, a row group's column chunk. */
  private[columnar] def reset(column: ColumnReader): Unit =
    this.column = column

  private[this] def grow(): Unit = {
    val n = reps.length * 2
    reps = java.util.Arrays.copyOf(reps, n)
    defs = java.util.Arrays.copyOf(defs, n)
    if (values != null) values = java.util.Arrays.copyOf(values, n)
    if (binaries != null) binaries = java.util.Arrays.copyOf(binaries, n)
  }

  private[columnar] def fill(): Unit = {
    // Drop the previous record's values, so that slots this record leaves unset hold no stale ones.
    if (binaries != null) {
      var i = 0
      while (i < size) { binaries(i) = null; i += 1 }
    }
    size = 0
    pos = 0
    do {
      if (size == reps.length) grow()
      val d = column.getCurrentDefinitionLevel
      reps(size) = column.getCurrentRepetitionLevel
      defs(size) = d
      if (d == maxDefinitionLevel) typeName match {
        case PrimitiveTypeName.BOOLEAN => values(size) = if (column.getBoolean) 1L else 0L
        case PrimitiveTypeName.INT32 => values(size) = column.getInteger.toLong
        case PrimitiveTypeName.INT64 => values(size) = column.getLong
        case PrimitiveTypeName.FLOAT =>
          values(size) = java.lang.Float.floatToRawIntBits(column.getFloat).toLong
        case PrimitiveTypeName.DOUBLE =>
          values(size) = java.lang.Double.doubleToRawLongBits(column.getDouble)
        case PrimitiveTypeName.BINARY => binaries(size) = column.getBinary.getBytes
        case t => fatal(s"columnar reader: unsupported Parquet type $t")
      }
      size += 1
      column.consume()
      // At the end of the column chunk, the repetition level is 0.
    } while (column.getCurrentRepetitionLevel != 0)
  }

  /** The definition level at the cursor. */
  def definitionLevel(): Int = defs(pos)

  def getBoolean(): Boolean = values(pos) != 0

  def getInt(): Int = values(pos).toInt

  def getLong(): Long = values(pos)

  def getFloat(): Float = java.lang.Float.intBitsToFloat(values(pos).toInt)

  def getDouble(): Double = java.lang.Double.longBitsToDouble(values(pos))

  def getBinary(): Array[Byte] = binaries(pos)

  /** Moves the cursor past one value of a node at repetition level `repetitionLevel`: the triple at
    * the cursor and every following one that repeats a node nested inside it.
    */
  def skip(repetitionLevel: Int): Unit = {
    pos += 1
    while (pos < size && reps(pos) > repetitionLevel) pos += 1
  }

  /** The number of elements of the non-empty list starting at the cursor, whose elements have
    * repetition level `repetitionLevel`.
    */
  def countElements(repetitionLevel: Int): Int = {
    var n = 1
    var i = pos + 1
    while (i < size && reps(i) >= repetitionLevel) {
      if (reps(i) == repetitionLevel) n += 1
      i += 1
    }
    n
  }
}
