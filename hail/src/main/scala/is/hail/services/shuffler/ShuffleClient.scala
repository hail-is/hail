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

  def openConnection(
    code: ArrayBuilder[Code[Unit]],
    mb: EmitMethodBuilder[_],
    typ: TShuffle
  ): (Value[Socket], Value[InputBuffer], Value[OutputBuffer], Value[Logger]) = {
    val socket = mb.newLocal[Socket]("shuffleClientSocket")
    code += (socket := codeSocket())

    val in = mb.newLocal[InputBuffer]("shuffleClientInputBuffer")
    code += (in := typ.bufferSpec.buildCodeInputBuffer(socket.getInputStream()))

    val out = mb.newLocal[OutputBuffer]("shuffleClientOutputBuffer")
    code += (out := typ.bufferSpec.buildCodeOutputBuffer(socket.getOutputStream()))

    val log = mb.newLocal[Logger]("shuffleClientLogger")
    code += (log := CodeLogger.getLogger[ShuffleClient]())

    (socket, in, out, log)
  }
}

object CodeShuffleClient {
  def create(ecb: EmitClassBuilder, shuffleType: TShuffle): CodeShuffleClient =
    new CodeShuffleClient(
      Code.newInstance[ShuffleClient, TShuffle](ecb.getPType(shuffleType)))

  def create(ecb: EmitClassBuilder, shuffleType: TShuffle, uuid: Array[Byte]): CodeShuffleClient =
    new CodeShuffleClient(
      Code.newInstance[ShuffleClient, TShuffle, Array[Byte]](ecb.getPType(shuffleType), uuid))
}

class ValueShuffleClient(
  val code: Value[ShuffleClient]
) extends AnyVal {
  def start(): Code[Unit] =
    code.invoke("start")

  def startPut(): Code[Unit] =
    code.invoke("startPut", values)

  def put(values: Code[Array[Long]]): Unit =
    code.invoke[Array[Long], Unit]("put", values)

  def endPut(): Code[Unit] =
    code.invoke("endPut", values)

  def uuid: Code[Array[Byte]] = code.getField("uuid")

  def get(
    region: Code[Region],
    start: Code[Long],
    startInclusive: Code[Boolean],
    end: Code[Long],
    endInclusive: Code[Boolean]
  ): Code[Array[Long]] =
    code.invoke[Region, Long, Boolean, Long, Boolean, Array[Long]](
      "get", region, start, startInclusive, end, endInclusive)

  def partitionBounds(region: Code[Region], nPartitions: Code[Int]): Code[Array[Long]] =
    code.invoke[Region, Int, Array[Long]]("partitionBounds", region, nPartitions)

  def stop(): Code[Unit] =
    code.invoke("stop")

  def close(): Code[Unit] =
    code.invoke("close")
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
    encoder.writeRegionValue(values(i))
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

  def get(
    region: Region,
    start: Long,
    startInclusive: Boolean,
    end: Long,
    endInclusive: Boolean
  ): Array[Long] = {
    log.info(s"get ${Region.pretty(codecs.keyDecodedPType, start)} ${startInclusive} " +
      s"${Region.pretty(codecs.keyDecodedPType, end)} ${endInclusive}")
    val keyEncoder = codecs.makeKeyEncoder(out)
    val decoder = codecs.makeRowDecoder(in)
    startOperation(Wire.GET)
    out.flush()
    keyEncoder.writeRegionValue(start)
    keyEncoder.writeByte(if (startInclusive) 1.toByte else 0.toByte)
    keyEncoder.writeRegionValue(end)
    keyEncoder.writeByte(if (endInclusive) 1.toByte else 0.toByte)
    keyEncoder.flush()
    log.info(s"get receiving values")
    val values = readRegionValueArray(region, decoder)
    log.info(s"get done")
    values
  }

  def partitionBounds(region: Region, nPartitions: Int): Array[Long] = {
    log.info(s"partitionBounds")
    val keyDecoder = codecs.makeKeyDecoder(in)
    startOperation(Wire.PARTITION_BOUNDS)
    out.writeInt(nPartitions)
    out.flush()
    log.info(s"partitionBounds receiving values")
    val keys = readRegionValueArray(region, keyDecoder, nPartitions + 1)
    log.info(s"partitionBounds done")
    keys
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
