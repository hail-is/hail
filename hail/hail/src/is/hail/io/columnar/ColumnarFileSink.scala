package is.hail.io.columnar

import is.hail.io.fs.FS

import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

import java.io.{BufferedOutputStream, ByteArrayInputStream, OutputStream}
import java.nio.channels.Channels
import java.util.Base64

import org.apache.arrow.vector.ipc.ReadChannel
import org.apache.arrow.vector.ipc.message.MessageSerializer
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.parquet.column.{ColumnWriteStore, ParquetProperties}
import org.apache.parquet.compression.CompressionCodecFactory
import org.apache.parquet.conf.PlainParquetConfiguration
import org.apache.parquet.hadoop.{
  CodecFactory, ColumnChunkPageWriteStore, ParquetFileWriter, ParquetWriter,
}
import org.apache.parquet.hadoop.metadata.CompressionCodecName
import org.apache.parquet.io.{ColumnIOFactory, MessageColumnIO, OutputFile, PositionOutputStream}
import org.apache.parquet.io.api.{Binary, RecordConsumer}
import org.apache.parquet.schema.MessageType

/** What a task needs to open a [[ColumnarFileSink]]: the file's Arrow schema, as the base64
  * encapsulated IPC message that also goes in the footer's `ARROW:schema`.
  *
  * Arrow schemas aren't serializable, so this carries the encoded form to workers.
  */
case class ColumnarFileSpec(arrowSchema: String) {
  def open(fs: FS, path: String): ColumnarFileSink = new ColumnarFileSink(fs, path, arrowSchema)
}

object ColumnarFileSink {
  val ArrowSchemaKey: String = "ARROW:schema"

  def decodeSchema(arrowSchema: String): Schema =
    MessageSerializer.deserializeSchema(
      new ReadChannel(Channels.newChannel(
        new ByteArrayInputStream(Base64.getDecoder.decode(arrowSchema))
      ))
    )

  private class HailOutputFile(fs: FS, path: String) extends OutputFile {
    override def create(blockSizeHint: Long): PositionOutputStream =
      createOrOverwrite(blockSizeHint)

    override def createOrOverwrite(blockSizeHint: Long): PositionOutputStream =
      new PositionOutputStream {
        private[this] val out: OutputStream =
          new BufferedOutputStream(fs.createNoCompression(path), 1 << 20)

        private[this] var pos: Long = 0

        override def getPos: Long = pos

        override def write(b: Int): Unit = { out.write(b); pos += 1 }

        override def write(b: Array[Byte], off: Int, len: Int): Unit = {
          out.write(b, off, len); pos += len
        }

        override def flush(): Unit = out.flush()

        override def close(): Unit = out.close()
      }

    override def supportsBlockSize(): Boolean = false

    override def defaultBlockSize(): Long = 0

    override def getPath: String = path
  }
}

/** Writes one Parquet data file of the columnar format, one record at a time.
  *
  * Emitted code drives this directly: `startRecord`, then a sequence of field, group and value
  * calls following parquet-java's [[RecordConsumer]] protocol for the file's Parquet schema, then
  * `endRecord`. A row group ends only at [[endRowGroup]] or [[close]], so that the caller can give
  * several files identical row-group boundaries (spec section 4.6.1). The footer carries the Arrow
  * schema under `ARROW:schema`. Statistics and the page index are written for every leaf, and pages
  * are zstd-compressed.
  *
  * parquet-java's `ParquetWriter` ends row groups itself, and the class that does so is private, so
  * this does the same work: one `ColumnWriteStore` per row group, flushed to the file when the row
  * group ends.
  */
final class ColumnarFileSink(fs: FS, path: String, arrowSchema: String) {
  import ColumnarFileSink._

  private[this] val schema: MessageType = ColumnarSchema.toParquet(decodeSchema(arrowSchema))

  private[this] val props: ParquetProperties =
    ParquetProperties.builder().withStatisticsEnabled(true).build()

  private[this] val codecs: CodecFactory =
    new CodecFactory(new PlainParquetConfiguration(), props.getPageSizeThreshold)

  private[this] val compressor: CompressionCodecFactory.BytesInputCompressor =
    codecs.getCompressor(CompressionCodecName.ZSTD)

  private[this] val file: ParquetFileWriter = new ParquetFileWriter(
    new HailOutputFile(fs, path),
    schema,
    ParquetFileWriter.Mode.OVERWRITE,
    ParquetWriter.DEFAULT_BLOCK_SIZE.toLong,
    0, // HailOutputFile doesn't support block alignment, so padding is never used.
    null, // no encryption
    props,
  )

  file.start()

  private[this] val columnIO: MessageColumnIO = new ColumnIOFactory().getColumnIO(schema)

  // The current row group's buffers, replaced when it ends.
  private[this] var pages: ColumnChunkPageWriteStore = _
  private[this] var columns: ColumnWriteStore = _
  private[this] var consumer: RecordConsumer = _

  private[this] var rowGroupRecords: Long = 0

  // The number of row groups written so far, which is the current row group's ordinal.
  private[this] var rowGroupsWritten: Int = 0

  startRowGroup()

  private[this] def startRowGroup(): Unit = {
    pages = new ColumnChunkPageWriteStore(
      compressor,
      schema,
      props.getAllocator,
      props.getColumnIndexTruncateLength,
      props.getPageWriteChecksumEnabled,
      null, // no encryption
      rowGroupsWritten,
    )
    columns = props.newColumnWriteStore(schema, pages, pages)
    consumer = columnIO.getRecordWriter(columns)
  }

  /** Writes the current row group to the file, if it has any records, and frees its buffers. */
  private[this] def finishRowGroup(): Unit = {
    if (rowGroupRecords > 0) {
      consumer.flush()
      file.startBlock(rowGroupRecords)
      columns.flush()
      pages.flushToFileWriter(file)
      file.endBlock()
      rowGroupRecords = 0
      rowGroupsWritten += 1
    }
    columns.close()
    pages.close()
  }

  def startRecord(): Unit = consumer.startMessage()

  def endRecord(): Unit = {
    consumer.endMessage()
    rowGroupRecords += 1
  }

  def startField(name: String, index: Int): Unit = consumer.startField(name, index)

  def endField(name: String, index: Int): Unit = consumer.endField(name, index)

  def startGroup(): Unit = consumer.startGroup()

  def endGroup(): Unit = consumer.endGroup()

  def addBoolean(v: Boolean): Unit = consumer.addBoolean(v)

  def addInt(v: Int): Unit = consumer.addInteger(v)

  def addLong(v: Long): Unit = consumer.addLong(v)

  def addFloat(v: Float): Unit = consumer.addFloat(v)

  def addDouble(v: Double): Unit = consumer.addDouble(v)

  def addBinary(v: Array[Byte]): Unit = consumer.addBinary(Binary.fromConstantByteArray(v))

  /** The bytes the current row group holds in memory. */
  def bufferedSize: Long = columns.getBufferedSize

  /** Ends the current row group. Does nothing if it has no records. */
  def endRowGroup(): Unit =
    if (rowGroupRecords > 0) {
      finishRowGroup()
      startRowGroup()
    }

  /** Ends the last row group, if it has any records, and writes the footer. */
  def close(): Unit = {
    finishRowGroup()
    file.end(Map(ArrowSchemaKey -> arrowSchema).asJava)
    codecs.release()
  }
}

object ColumnarRowGroups {

  /** The size, in buffered bytes, at which a row group ends: parquet-java's default. */
  val TargetBytes: Long = ParquetWriter.DEFAULT_BLOCK_SIZE.toLong

  /** At most this many records are written between checks of the buffered size. */
  private val MaxRecordsBetweenChecks: Long = 10000
}

/** The data files of one row partition, whose row groups end together (spec section 4.6.1).
  *
  * The caller ends a row group when [[checkFull]] says it is full, or for reasons of its own, such
  * as the leaf cap. Each record of the partition is written to every file, followed by one call to
  * [[endRecord]].
  */
final class ColumnarRowGroups(targetBytes: Long) {
  import ColumnarRowGroups._

  private[this] val sinks = ArrayBuffer[ColumnarFileSink]()

  private[this] var records: Long = 0

  // The buffered size is summed over every leaf of every file, so it is checked only once enough
  // records may have been written to reach the target, estimated from the average record size, as
  // parquet-java's `ParquetWriter` does. The estimate is only as good as the records it averages,
  // so the next check is never more records away than have been written: a few small records
  // followed by large ones can't defer it for long.
  private[this] var nextSizeCheck: Long = 1

  def add(sink: ColumnarFileSink): Unit = sinks += sink

  /** The number of records in the current row group. */
  def recordCount: Long = records

  def endRecord(): Unit = records += 1

  /** Whether the current row group has reached the target size. If it hasn't, also estimates when
    * to check its size next.
    */
  def checkFull(): Boolean =
    records >= nextSizeCheck && {
      val size = sinks.iterator.map(_.bufferedSize).max
      size >= targetBytes || {
        val recordSize = math.max(size / records, 1)
        nextSizeCheck =
          records + math.min(
            math.max((targetBytes - size) / recordSize / 2, 1),
            math.min(records, MaxRecordsBetweenChecks),
          )
        false
      }
    }

  def endRowGroup(): Unit = {
    sinks.foreach(_.endRowGroup())
    records = 0
    nextSizeCheck = 1
  }

  def close(): Unit = sinks.foreach(_.close())
}
