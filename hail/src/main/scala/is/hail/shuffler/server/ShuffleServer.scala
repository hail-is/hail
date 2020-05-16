package is.hail.shuffler.server

import java.net._
import java.security.SecureRandom
import java.util.concurrent.{ConcurrentSkipListMap, Executors, _}

import is.hail.annotations.Region
import is.hail.expr.ir._
import is.hail.expr.types.encoded._
import is.hail.expr.types.virtual._
import is.hail.io._
import is.hail.shuffler._
import is.hail.utils._
import javax.net.ssl._
import org.apache.log4j.Logger

import scala.annotation.switch

class Handler (
  private[this] val server: ShuffleServer,
  private[this] val socket: Socket
) extends Runnable {
  private[this] val log = Logger.getLogger(getClass.getName())
  private[this] val in = shuffleBufferSpec.buildInputBuffer(socket.getInputStream)
  private[this] val out = shuffleBufferSpec.buildOutputBuffer(socket.getOutputStream)
  private[this] val random = new SecureRandom();

  def run(): Unit = {
    try {
      log.info(s"SERV handle")
      try {
        var continue = true
        while (continue) {
          val op = in.readByte()
          log.info(s"SERV operation ${op}")
            (op: @switch) match {
            case Wire.START => start()
            case Wire.PUT => put()
            case Wire.GET => get()
            case Wire.STOP => stop()
            case Wire.PARTITION_BOUNDS => partitionBounds()
            case Wire.EOS =>
              log.info(s"client ended session, replying, then exiting cleanly")
              eos()
              continue = false
            case op => fatal(s"bad operation number $op")
          }
        }
      } finally {
        socket.close()
      }
    } catch {
      case e: Exception =>
        log.warn(s"exception while serving", e)
    }
  }

  def readShuffleUUID(): Shuffle = {
    val uuid = Wire.readByteArray(in)
    assert(uuid.length == Wire.ID_SIZE, s"${uuid.length} ${Wire.ID_SIZE}")
    log.info(s"SERV uuid ${uuidToString(uuid)}")
    val shuffle = server.shuffles.get(uuid)
    if (shuffle == null) {
      throw new RuntimeException(s"shuffle does not exist $uuid")
    }
    shuffle
  }

  def start(): Unit = {
    log.info(s"SERV start")
    val rowType = Wire.readTStruct(in)
    log.info(s"SERV start got row type ${rowType}")
    val rowEType = Wire.readEBaseStruct(in)
    log.info(s"SERV start got row encoded type ${rowEType}")
    val keyFields = Wire.readSortFieldArray(in)
    log.info(s"SERV start got key fields ${keyFields.mkString("[", ",", "]")}")
    val keyEType = Wire.readEBaseStruct(in)
    log.info(s"SERV start got key encoded type ${keyEType}")
    val uuid = new Array[Byte](Wire.ID_SIZE)
    random.nextBytes(uuid)
    server.shuffles.put(uuid, new Shuffle(uuid, keyFields, rowType, rowEType, keyEType))
    Wire.writeByteArray(out, uuid)
    log.info(s"SERV start wrote uuid")
    out.flush()
    log.info(s"SERV start flush")
    log.info(s"SERV start done")
  }

  def put(): Unit = {
    log.info(s"SERV put")
    val shuffle = readShuffleUUID()
    shuffle.put(in, out)
    log.info(s"SERV put done")
  }

  def get(): Unit = {
    log.info(s"SERV get")
    val shuffle = readShuffleUUID()
    shuffle.get(in, out)
    log.info(s"SERV get done")
  }

  def stop(): Unit = {
    log.info(s"SERV stop")
    val uuid = Wire.readByteArray(in)
    assert(uuid.length == Wire.ID_SIZE, s"${uuid.length} ${Wire.ID_SIZE}")
    val shuffle = server.shuffles.remove(uuid)
    if (shuffle != null) {
      shuffle.close()
    }
    out.writeByte(0.toByte)
    out.flush()
    log.info(s"SERV stop done")
  }

  def partitionBounds(): Unit = {
    log.info(s"SERV partitionBounds")
    val shuffle = readShuffleUUID()
    shuffle.partitionBounds(in, out)
    log.info(s"SERV partitionBounds done")
  }

  def eos(): Unit = {
    out.writeByte(Wire.EOS)
    out.flush()
  }
}

class Shuffle (
  uuid: Array[Byte],
  keyFields: Array[SortField],
  rowType: TStruct,
  rowEType: EBaseStruct,
  keyEType: EBaseStruct
) extends AutoCloseable {
  private[this] val log = Logger.getLogger(getClass.getName)
  private[this] val rootRegion = Region()
  private[this] val ctx = new ExecuteContext("/tmp", "file:///tmp", null, null, rootRegion, new ExecutionTimer())
  private[this] val codecs = new ShuffleCodecSpec(ctx, keyFields, rowType, rowEType, keyEType)
  private[this] val store = new LSM(s"/tmp/${uuidToString(uuid)}", codecs)

  private[this] def makeRegion(): Region = {
    val region = Region()
    rootRegion.addReferenceTo(region)
    region
  }

  def close(): Unit = {
    rootRegion.close()
    ctx.close()
  }

  def put(in: InputBuffer, out: OutputBuffer) {
    val decoder = codecs.makeRowDecoder(in)
    val region = makeRegion()
    var hasNext = in.readByte()
    assert(hasNext != -1)
    while (hasNext == 1) {
      val off = decoder.readRegionValue(region)
      val koff = codecs.keyDecodedPType.copyFromAddress(region, codecs.rowDecodedPType, off, false)
      store.put(koff, off)
      hasNext = in.readByte()
    }
    // fixme: server needs to send uuid for the successful partition
    out.writeByte(0)
    out.flush()
  }

  def get(in: InputBuffer, out: OutputBuffer) {
    val region = makeRegion()
    val keyDecoder = codecs.makeKeyDecoder(in)
    val encoder = codecs.makeRowEncoder(out)
    val start = keyDecoder.readRegionValue(region)
    val startInclusive = keyDecoder.readByte() == 1.toByte
    val end = keyDecoder.readRegionValue(region)
    val endInclusive = keyDecoder.readByte() == 1.toByte

    log.info(s"SERV get start ${rvstr(codecs.keyDecodedPType, start)} ${startInclusive} end ${rvstr(codecs.keyDecodedPType, end)} ${endInclusive}")
    val it = store.iterator(start, startInclusive)
    var continue = it.hasNext
    val inRange =
      if (endInclusive) (key: Long) => store.keyOrd.lteq(key, end)
      else              (key: Long) => store.keyOrd.lt(key, end)
    while (continue) {
      val kv = it.next
      val k = kv.getKey
      val v = kv.getValue
      continue = inRange(k)
      if (continue) {
        encoder.writeByte(1)
        encoder.writeRegionValue(v)
        continue = it.hasNext
      }
    }
    encoder.writeByte(0)
    encoder.flush()
  }

  def partitionBounds(in: InputBuffer, out: OutputBuffer) {
    val nPartitionsLong = in.readLong()
    assert(nPartitionsLong < Int.MaxValue)
    val nPartitions = nPartitionsLong.toInt

    val region = makeRegion()
    val keyEncoder = codecs.makeKeyEncoder(out)

    log.info(s"SERV partitionBounds ${nPartitions}")
    val keys = store.partitionKeys(nPartitions)
    assert((nPartitions == 0 && keys.length == 0) ||
      keys.length == nPartitions + 1)
    writeRegionValueArray(keyEncoder, keys)
    keyEncoder.flush()
  }
}

class ShuffleServer (
  ssl: SSLContext,
  port: Int
) {
  val log = Logger.getLogger(this.getClass.getName());

  val shuffles = new ConcurrentSkipListMap[Array[Byte], Shuffle](new SameLengthByteArrayComparator())

  val ssf = ssl.getServerSocketFactory()
  val ss = ssf.createServerSocket(port)
  ss.asInstanceOf[SSLServerSocket].setNeedClientAuth(true)

  val executor = Executors.newCachedThreadPool()
  var stopped = false

  def serveInBackground(): Future[_] =
    executor.submit(new Runnable() { def run(): Unit = serve() })

  def serve(): Unit = {
    try {
      log.info(s"SERV serving on ${port}")
      while (true) {
        val sock = ss.accept()
        log.info(s"SERV accepted")
        executor.execute(new Handler(this, sock))
      }
    } catch {
      case se: SocketException =>
        if (stopped) {
          log.info(s"SERV exiting")
          return
        } else {
          fatal("unexpected closed server socket", se)
        }
    }
  }

  def stop(): Unit = {
    log.info(s"SERV stopping")
    stopped = true
    ss.close()
    executor.shutdownNow()
  }
}
