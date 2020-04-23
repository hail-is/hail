package is.hail.shuffler

import java.net.Socket

import is.hail._
import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.io._
import is.hail.utils._
import javax.net.ssl._
import org.apache.log4j.Logger

object ShuffleClient {
  private[this] val log = Logger.getLogger(getClass.getName())

  lazy val sslContext = {
    val keyStore = System.getenv("SHUFFLER_SSL_CLIENT_KEY_STORE")
    val trustStore = System.getenv("SHUFFLER_SSL_CLIENT_TRUST_STORE")
    if (keyStore == null && trustStore != null ||
      trustStore == null && keyStore != null) {
      fatal("you must specify both or neither of the hail context flags: " +
        "shuffler_ssl_client_key_store and shuffler_ssl_client_trust_store")
    }
    if (keyStore == null) {
      is.hail.shuffler.sslContext(
        getClass.getResourceAsStream("/non-secret-key-and-trust-stores/client-keystore.p12"),
        "hailhail",
        "PKCS12",
        getClass.getResourceAsStream("/non-secret-key-and-trust-stores/client-truststore.p12"),
        "hailhail",
        "JKS"
      )
    } else {
      is.hail.shuffler.sslContext(keyStore, "", "PKCS12", trustStore, "", "PKCS12")
    }
  }

  def socket(): Socket = {
    var host = System.getenv("SHUFFLER_SSL_CLIENT_HOST")
    if (host == null) {
      host = "localhost"
    }
    var portStr = System.getenv("SHUFFLER_SSL_CLIENT_PORT")
    if (portStr == null) {
      portStr = "8080"
    }
    val port = java.lang.Integer.valueOf(portStr)
    socket(host, port)
  }

  def socket(host: String, port: Int): Socket = {
    val s = sslContext.getSocketFactory().createSocket(host, port)
    log.info(s"connected to ${host}:${port} (socket())")
    s
  }

  def codeSocket(): Code[Socket] =
    Code.invokeScalaObject0[Socket](ShuffleClient.getClass, "socket")

  def openConnection(cb: EmitCodeBuilder, typ: TShuffle): (SocketCode, InputBufferValue, OutputBufferValue, LoggerCode) = {
    val socketLocal = cb.emb.newLocal[Socket]("shuffleClientSocket")
    cb.append(socketLocal := codeSocket())
    val socket = new SocketCode(socketLocal)

    val inputBufferLocal = cb.emb.newLocal[InputBuffer]("shuffleClientInputBuffer")
    cb.append(inputBufferLocal := typ.bufferSpec.buildCodeInputBuffer(socket.getInputStream()))
    val in = new InputBufferValue(inputBufferLocal)

    val outputBufferLocal = cb.emb.newLocal[OutputBuffer]("shuffleClientOutputBuffer")
    cb.append(outputBufferLocal := typ.bufferSpec.buildCodeOutputBuffer(socket.getOutputStream()))
    val out = new OutputBufferValue(outputBufferLocal)

    val loggerLocal = cb.emb.newLocal[Logger]("shuffleClientLogger")
    cb.append(loggerLocal := LoggerCode.getLogger[ShuffleClient]())
    val log = new LoggerCode(loggerLocal)

    (socket, in, out, log)
  }

  def openConnection(code: ArrayBuilder[Code[Unit]], mb: EmitMethodBuilder[_], typ: TShuffle): (SocketCode, InputBufferValue, OutputBufferValue, LoggerCode) = {
    val socketLocal = mb.newLocal[Socket]("shuffleClientSocket")
    code += (socketLocal := codeSocket())
    val socket = new SocketCode(socketLocal)

    val inputBufferLocal = mb.newLocal[InputBuffer]("shuffleClientInputBuffer")
    code += (inputBufferLocal := typ.bufferSpec.buildCodeInputBuffer(socket.getInputStream()))
    val in = new InputBufferValue(inputBufferLocal)

    val outputBufferLocal = mb.newLocal[OutputBuffer]("shuffleClientOutputBuffer")
    code += (outputBufferLocal := typ.bufferSpec.buildCodeOutputBuffer(socket.getOutputStream()))
    val out = new OutputBufferValue(outputBufferLocal)

    val loggerLocal = mb.newLocal[Logger]("shuffleClientLogger")
    code += (loggerLocal := LoggerCode.getLogger[ShuffleClient]())
    val log = new LoggerCode(loggerLocal)

    (socket, in, out, log)
  }
}

class ShuffleClient (
  shuffleType: TShuffle,
  ssl: SSLContext,
  host: String,
  port: Int
) extends AutoCloseable {
  private[this] val log = Logger.getLogger(getClass.getName())
  private[this] var uuid: Array[Byte] = null
  private[this] val ctx = new ExecuteContext("/tmp", "file:///tmp", null, null, Region(), new ExecutionTimer())

  val codecs = new ShuffleCodecSpec(ctx, shuffleType)

  private[this] val s = ShuffleClient.socket(host, port)
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

  def put(values: Array[Long]): Unit = {
    log.info(s"put")
    startOperation(Wire.PUT)
    out.flush()
    val encoder = codecs.makeRowEncoder(out)
    writeRegionValueArray(encoder, values)
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
