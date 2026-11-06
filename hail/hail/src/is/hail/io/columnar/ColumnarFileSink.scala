package is.hail.io.columnar

import is.hail.io.fs.FS

import scala.jdk.CollectionConverters._

import java.io.{BufferedOutputStream, ByteArrayInputStream, OutputStream}
import java.nio.channels.Channels
import java.util.Base64

import org.apache.arrow.vector.ipc.ReadChannel
import org.apache.arrow.vector.ipc.message.MessageSerializer
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.conf.{ParquetConfiguration, PlainParquetConfiguration}
import org.apache.parquet.hadoop.{ParquetFileWriter, ParquetWriter}
import org.apache.parquet.hadoop.api.WriteSupport
import org.apache.parquet.hadoop.metadata.CompressionCodecName
import org.apache.parquet.io.{OutputFile, PositionOutputStream}
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

  /** Hands parquet-java's record consumer to [[ColumnarFileSink]], which emits each record itself.
    * parquet-java replaces the consumer whenever it starts a new row group.
    */
  private class ConsumerWriteSupport(schema: MessageType, arrowSchema: String)
      extends WriteSupport[AnyRef] {
    var consumer: RecordConsumer = _

    override def init(conf: Configuration): WriteSupport.WriteContext =
      new WriteSupport.WriteContext(schema, Map(ArrowSchemaKey -> arrowSchema).asJava)

    override def init(conf: ParquetConfiguration): WriteSupport.WriteContext =
      new WriteSupport.WriteContext(schema, Map(ArrowSchemaKey -> arrowSchema).asJava)

    override def prepareForWrite(recordConsumer: RecordConsumer): Unit =
      consumer = recordConsumer

    // The record was already emitted through `consumer`.
    override def write(record: AnyRef): Unit = ()
  }

  private class Builder(file: OutputFile, support: ConsumerWriteSupport)
      extends ParquetWriter.Builder[AnyRef, Builder](file) {
    override def self(): Builder = this

    override def getWriteSupport(conf: Configuration): WriteSupport[AnyRef] = support

    override def getWriteSupport(conf: ParquetConfiguration): WriteSupport[AnyRef] = support
  }
}

/** Writes one Parquet data file of the columnar format, one record at a time.
  *
  * Emitted code drives this directly: `startRecord`, then a sequence of field, group and value
  * calls following parquet-java's [[RecordConsumer]] protocol for the file's Parquet schema, then
  * `endRecord`. The footer carries the Arrow schema under `ARROW:schema`. Statistics and the page
  * index are written for every leaf, and pages are zstd-compressed.
  */
final class ColumnarFileSink(fs: FS, path: String, arrowSchema: String) {
  import ColumnarFileSink._

  private[this] val support = new ConsumerWriteSupport(
    ColumnarSchema.toParquet(decodeSchema(arrowSchema)),
    arrowSchema,
  )

  private[this] val writer: ParquetWriter[AnyRef] =
    new Builder(new HailOutputFile(fs, path), support)
      .withConf(new PlainParquetConfiguration())
      .withWriteMode(ParquetFileWriter.Mode.OVERWRITE)
      .withCompressionCodec(CompressionCodecName.ZSTD)
      .withStatisticsEnabled(true)
      .build()

  private[this] def consumer: RecordConsumer = support.consumer

  def startRecord(): Unit = consumer.startMessage()

  def endRecord(): Unit = {
    consumer.endMessage()
    // Counts the record emitted above, and flushes the row group when it is full.
    writer.write(null)
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

  def close(): Unit = writer.close()
}
