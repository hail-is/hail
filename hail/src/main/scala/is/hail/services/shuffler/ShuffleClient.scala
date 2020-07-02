package is.hail.services.shuffler

import java.net.Socket

import is.hail._
import is.hail.annotations._
import is.hail.services.tls._
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.io._
import is.hail.utils._
import javax.net.ssl._
import org.apache.log4j.Logger

object ShuffleClient {
  private[this] val log = Logger.getLogger(getClass.getName())

  lazy val sslContext = is.hail.services.shuffler.sslContext(
    getClass.getResourceAsStream("/non-secret-key-and-trust-stores/client-keystore.p12"),
    "hailhail",
    "PKCS12",
    getClass.getResourceAsStream("/non-secret-key-and-trust-stores/client-truststore.p12"),
    "hailhail",
    "JKS"
  )

  def socket(): Socket = {
    val host = "localhost"
    val port = 8080
    socket(host, port)
  }

  def socket(host: String, port: Int): Socket = {
    val s = sslContext.getSocketFactory().createSocket(host, port)
    log.info(s"connected to ${host}:${port} (socket())")
    s
  }

  def codeSocket(): Code[Socket] =
    Code.invokeScalaObject0[Socket](ShuffleClient.getClass, "socket")
}

object CodeShuffleClient {
  def create(ecb: EmitClassBuilder[_], shuffleType: TShuffle): ValueShuffleClient =
    new ValueShuffleClient(
      ecb.genLazyFieldThisRef[ShuffleClient](
        Code.newInstance[ShuffleClient, TShuffle](ecb.getType(shuffleType)),
        "shuffleClient"))

  def create(ecb: EmitClassBuilder[_], shuffleType: TShuffle, uuid: Code[Array[Byte]]): ValueShuffleClient =
    new ValueShuffleClient(
      ecb.genLazyFieldThisRef[ShuffleClient](
        Code.newInstance[ShuffleClient, TShuffle, Array[Byte]](ecb.getType(shuffleType), uuid),
        "shuffleClient"))
}

class ValueShuffleClient(
  val code: Value[ShuffleClient]
) extends AnyVal {
  def start(): Code[Unit] =
    code.invoke[Unit]("start")

  def startPut(): Code[Unit] =
    code.invoke[Unit]("startPut")

  def putValue(value: Code[Long]): Code[Unit] =
    code.invoke[Long, Unit]("putValue", value)

  def putValueDone(): Code[Unit] =
    code.invoke[Unit]("putValueDone")

  def endPut(): Code[Unit] =
    code.invoke[Unit]("endPut")

  def uuid: Code[Array[Byte]] = code.getField[Array[Byte]]("uuid")

  def startGet(
    start: Code[Long],
    startInclusive: Code[Boolean],
    end: Code[Long],
    endInclusive: Code[Boolean]
  ): Code[Unit] =
    code.invoke[Long, Boolean, Long, Boolean, Unit](
      "startGet", start, startInclusive, end, endInclusive)

  def getValue(region: Code[Region]): Code[Long] =
    code.invoke[Region, Long]("getValue", region)

  def getValueFinished(): Code[Boolean] =
    code.invoke[Boolean]("getValueFinished")

  def getDone(): Code[Unit] =
    code.invoke[Unit]("getDone")

  def startPartitionBounds(nPartitions: Code[Int]): Code[Unit] =
    code.invoke[Int, Unit]("startPartitionBounds", nPartitions)

  def partitionBoundsValue(region: Code[Region]): Code[Long] =
    code.invoke[Region, Long]("partitionBoundsValue", region)

  def partitionBoundsValueFinished(): Code[Boolean] =
    code.invoke[Boolean]("partitionBoundsValueFinished")

  def endPartitionBounds(): Code[Unit] =
    code.invoke[Unit]("endPartitionBounds")

  def stop(): Code[Unit] =
    code.invoke[Unit]("stop")

  def close(): Code[Unit] =
    code.invoke[Unit]("close")
}

class ShuffleClient (
  shuffleType: TShuffle,
  var uuid: Array[Byte]
) extends AutoCloseable {
  private[this] val log = Logger.getLogger(getClass.getName())
  private[this] val ctx = new ExecuteContext("/tmp", "file:///tmp", null, null, Region(), new ExecutionTimer())

  def this(shuffleType: TShuffle) = this(shuffleType, null)

  val codecs = new ShuffleCodecSpec(ctx, shuffleType)

  private[this] val s = ShuffleClient.socket()
  private[this] val in = shuffleBufferSpec.buildInputBuffer(s.getInputStream())
  private[this] val out = shuffleBufferSpec.buildOutputBuffer(s.getOutputStream())

  private[this] def startOperation(op: Byte): Unit = {
    out.writeByte(op)
    if (op != Wire.START) {
      assert(uuid != null)
      log.info(s"operation $op uuid ${uuidToString(uuid)}")
      Wire.writeByteArray(out, uuid)
    }
  }

  def start(): Unit = {
    log.info(s"start")
    startOperation(Wire.START)
    Wire.writeTStruct(out, shuffleType.rowType)
    Wire.writeEBaseStruct(out, shuffleType.rowEType)
    Wire.writeSortFieldArray(out, shuffleType.keyFields)
    Wire.writeEBaseStruct(out, shuffleType.keyEType)
    out.flush()
    uuid = Wire.readByteArray(in)
    assert(uuid.length == Wire.ID_SIZE, s"${uuid.length} ${Wire.ID_SIZE}")
    log.info(s"start done")
  }

  private[this] var encoder: Encoder = null

  def startPut(): Unit = {
    log.info(s"put")
    startOperation(Wire.PUT)
    out.flush()
    encoder = codecs.makeRowEncoder(out)
  }

  def put(values: Array[Long]): Unit = {
    startPut()
    writeRegionValueArray(encoder, values)
    endPut()
  }

  def putValue(value: Long): Unit = {
    encoder.writeByte(1)
    encoder.writeRegionValue(value)
  }

  def putValueDone(): Unit = {
    encoder.writeByte(0)
  }

  def endPut(): Unit = {
    // fixme: server needs to send uuid for the successful partition
    out.flush()
    assert(in.readByte() == 0.toByte)
    log.info(s"put done")
  }

  private[this] var decoder: Decoder = null

  def startGet(
    start: Long,
    startInclusive: Boolean,
    end: Long,
    endInclusive: Boolean
  ): Unit = {
    log.info(s"get ${Region.pretty(codecs.keyDecodedPType, start)} ${startInclusive} " +
      s"${Region.pretty(codecs.keyDecodedPType, end)} ${endInclusive}")
    val keyEncoder = codecs.makeKeyEncoder(out)
    decoder = codecs.makeRowDecoder(in)
    startOperation(Wire.GET)
    out.flush()
    keyEncoder.writeRegionValue(start)
    keyEncoder.writeByte(if (startInclusive) 1.toByte else 0.toByte)
    keyEncoder.writeRegionValue(end)
    keyEncoder.writeByte(if (endInclusive) 1.toByte else 0.toByte)
    keyEncoder.flush()
    log.info(s"get receiving values")
  }

  def get(
    region: Region,
    start: Long,
    startInclusive: Boolean,
    end: Long,
    endInclusive: Boolean
  ): Array[Long] = {
    startGet(start, startInclusive, end, endInclusive)
    val values = readRegionValueArray(region, decoder)
    getDone()
    values
  }

  def getValue(region: Region): Long = {
    decoder.readRegionValue(region)
  }

  def getValueFinished(): Boolean = {
    decoder.readByte() == 0.toByte
  }

  def getDone(): Unit = {
    log.info(s"get done")
  }

  private[this] var keyDecoder: Decoder = null

  def startPartitionBounds(
    nPartitions: Int
  ): Unit = {
    log.info(s"partitionBounds")
    startOperation(Wire.PARTITION_BOUNDS)
    out.writeInt(nPartitions)
    out.flush()
    log.info(s"partitionBounds receiving values")
    keyDecoder = codecs.makeKeyDecoder(in)
  }

  def partitionBounds(region: Region, nPartitions: Int): Array[Long] = {
    startPartitionBounds(nPartitions)
    val keys = readRegionValueArray(region, keyDecoder, nPartitions + 1)
    endPartitionBounds()
    keys
  }

  def partitionBoundsValue(region: Region): Long = {
    keyDecoder.readRegionValue(region)
  }

  def partitionBoundsValueFinished(): Boolean = {
    keyDecoder.readByte() == 0.toByte
  }

  def endPartitionBounds(): Unit = {
    log.info(s"partitionBounds done")
  }

  def stop(): Unit = {
    log.info(s"stop")
    out.writeByte(Wire.STOP)
    Wire.writeByteArray(out, uuid)
    out.flush()
    assert(in.readByte() == 0.toByte)
    log.info(s"stop done")
  }

  def close(): Unit = {
    try {
      try {
        out.writeByte(Wire.EOS)
        out.flush()
        assert(in.readByte() == Wire.EOS)
      } finally {
        s.close()
      }
    } finally {
      ctx.close()
    }
  }
}
