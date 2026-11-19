package is.hail.io.columnar

import is.hail.io.fs.{FS, SeekableDataInputStream}
import is.hail.types.arrow.ExtensionTypes
import is.hail.utils.fatal

import scala.jdk.CollectionConverters._

import org.apache.arrow.vector.types.pojo.Schema
import org.apache.parquet.{ParquetReadOptions, VersionParser}
import org.apache.parquet.column.{ColumnDescriptor, ColumnReader}
import org.apache.parquet.column.impl.ColumnReaderImpl
import org.apache.parquet.column.page.PageReadStore
import org.apache.parquet.conf.PlainParquetConfiguration
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.io.{DelegatingSeekableInputStream, InputFile, SeekableInputStream}
import org.apache.parquet.io.api.PrimitiveConverter
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName

/** What a task needs to open a [[ColumnarFileSource]]: the Arrow schema the file must have, as a
  * base64 encapsulated IPC message, and the Parquet paths of the leaf columns to read.
  *
  * Arrow schemas aren't serializable, so this carries the encoded form to workers.
  */
case class ColumnarSourceSpec(arrowSchema: String, leaves: IndexedSeq[IndexedSeq[String]]) {
  def open(fs: FS, path: String): ColumnarFileSource = new ColumnarFileSource(fs, path, this)
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

  /** Opens the Parquet file at `path`, reading only its footer. */
  def openFile(fs: FS, path: String): ParquetFileReader =
    ParquetFileReader.open(
      new HailInputFile(fs, path),
      ParquetReadOptions.builder(new PlainParquetConfiguration()).build(),
    )

  /** The Arrow schema in the footer of an open file (spec section 4.6). */
  def arrowSchema(reader: ParquetFileReader, path: String): Schema = {
    ExtensionTypes.register()
    Option(reader.getFileMetaData.getKeyValueMetaData.get(ColumnarFileSink.ArrowSchemaKey)) match {
      case Some(s) => ColumnarFileSink.decodeSchema(s)
      case None => fatal(s"columnar reader: '$path' has no ${ColumnarFileSink.ArrowSchemaKey}")
    }
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
  */
final class ColumnarFileSource(fs: FS, path: String, spec: ColumnarSourceSpec) {
  import ColumnarFileSource._

  private[this] val reader: ParquetFileReader = openFile(fs, path)

  try
    if (arrowSchema(reader, path) != ColumnarFileSink.decodeSchema(spec.arrowSchema))
      fatal(s"columnar reader: the Arrow schema of '$path' differs from its manifest's")
  catch {
    case e: Throwable =>
      reader.close()
      throw e
  }

  private[this] val descriptors: IndexedSeq[ColumnDescriptor] = {
    val schema = reader.getFileMetaData.getSchema
    spec.leaves.map(p => schema.getColumnDescription(p.toArray))
  }

  reader.setRequestedSchema(descriptors.asJava)

  private[this] val writerVersion: VersionParser.ParsedVersion =
    try VersionParser.parse(reader.getFileMetaData.getCreatedBy)
    catch { case _: Exception => null }

  private[this] val leaves: Array[ColumnarLeaf] =
    descriptors.map(d =>
      new ColumnarLeaf(d.getMaxDefinitionLevel, d.getPrimitiveType.getPrimitiveTypeName)
    ).toArray

  // parquet-java's readNextRowGroup skips row groups with no rows, so they are left out here too.
  private[this] val rowGroupSizes: Array[Long] =
    reader.getRowGroups.asScala.map(_.getRowCount).filter(_ > 0).toArray

  private[this] var nextRowGroup: Int = 0
  private[this] var recordsLeft: Long = 0

  def leaf(i: Int): ColumnarLeaf = leaves(i)

  /** Buffers the next record of every leaf. False at the end of the file. */
  def nextRecord(): Boolean = {
    while (recordsLeft == 0) {
      if (nextRowGroup == rowGroupSizes.length) return false
      recordsLeft = rowGroupSizes(nextRowGroup)
      nextRowGroup += 1
      // With no leaves requested, records are only counted, and no column chunk is read.
      if (leaves.nonEmpty) {
        val pages: PageReadStore = reader.readNextRowGroup()
        descriptors.indices.foreach { i =>
          val pageReader = pages.getPageReader(descriptors(i))
          leaves(i).reset(
            new ColumnReaderImpl(descriptors(i), pageReader, NoConverter, writerVersion),
            pageReader.getTotalValueCount,
          )
        }
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
  private[this] var columnValuesLeft: Long = 0

  private[this] var reps: Array[Int] = new Array[Int](8)
  private[this] var defs: Array[Int] = new Array[Int](8)

  private[this] var values: Array[Long] =
    if (typeName == PrimitiveTypeName.BINARY) null else new Array[Long](8)

  private[this] var binaries: Array[Array[Byte]] =
    if (typeName == PrimitiveTypeName.BINARY) new Array[Array[Byte]](8) else null

  private[this] var size: Int = 0
  private[this] var pos: Int = 0

  /** Starts reading `column`, a row group's column chunk holding `values` triples. */
  private[columnar] def reset(column: ColumnReader, values: Long): Unit = {
    this.column = column
    columnValuesLeft = values
  }

  private[this] def grow(): Unit = {
    val n = reps.length * 2
    reps = java.util.Arrays.copyOf(reps, n)
    defs = java.util.Arrays.copyOf(defs, n)
    if (values != null) values = java.util.Arrays.copyOf(values, n)
    if (binaries != null) binaries = java.util.Arrays.copyOf(binaries, n)
  }

  private[columnar] def fill(): Unit = {
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
      columnValuesLeft -= 1
    } while (columnValuesLeft > 0 && column.getCurrentRepetitionLevel != 0)
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
