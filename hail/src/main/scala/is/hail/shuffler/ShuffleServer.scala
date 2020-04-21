package is.hail.shuffler

import java.io._
import java.net._
import java.security.SecureRandom
import java.util.Base64
import java.util.concurrent.{ConcurrentSkipListMap, Executors, _}

import is.hail.annotations.Region
import is.hail.expr.ir.ExecuteContext
import is.hail.expr.types.virtual.TStruct
import is.hail.io.TypedCodecSpec
import is.hail.utils._
import javax.net.ssl._
import org.apache.log4j.Logger

import scala.annotation.switch

class Handler (
  private[this] val server: ShuffleServer,
  private[this] val socket: Socket
) extends Runnable {
  private[this] val log = Logger.getLogger(getClass.getName())
  private[this] val in = new DataInputStream(socket.getInputStream)
  private[this] val out = new DataOutputStream(socket.getOutputStream)
  private[this] val random = new SecureRandom();

  def run(): Unit = {
    try {
      log.info(s"SERV handle")
      try {
        var continue = true
        while (continue) {
          val op = in.read()
          log.info(s"SERV operation ${op}")
            (op: @switch) match {
            case Wire.START => start()
            case Wire.PUT => put()
            case Wire.GET => get()
            case Wire.STOP => stop()
            case Wire.EOS =>
              log.info(s"client closed socket, exiting cleanly")
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
    val shuffle = server.shuffles.get(uuid)
    if (shuffle == null) {
      throw new RuntimeException(s"shuffle does not exist $uuid")
    }
    shuffle
  }

  def start(): Unit = {
    log.info(s"SERV start")
    val t = Wire.readTStruct(in)
    val codecSpec = Wire.readTypedCodecSpec(in)
    val key = Wire.readStringArray(in)
    val uuid = new Array[Byte](Wire.ID_SIZE)
    random.nextBytes(uuid)
    server.shuffles.put(uuid, new Shuffle(uuid, t, codecSpec, key))
    Wire.writeByteArray(out, uuid)
    out.flush()
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
    val shuffle = server.shuffles.remove(uuid)
    if (shuffle != null) {
      shuffle.close()
    }
    log.info(s"SERV stop done")
  }
}

class Shuffle (
  uuid: Array[Byte],
  t: TStruct,
  codecSpec: TypedCodecSpec,
  key: Array[String]
) extends AutoCloseable {
  private[this] val log = Logger.getLogger(getClass.getName)
  private[this] val ctx = new ExecuteContext("/tmp", "file:///tmp", null, null, Region(), new ExecutionTimer())
  private[this] val b64uuid = Base64.getEncoder.encode(uuid)
  private[this] val codecs = new KeyedCodecSpec(ctx, t, codecSpec, key)
  private[this] val rootRegion = ctx.r
  private[this] val store = new LSM(s"/tmp/${b64uuid}", codecs, rootRegion)

  private[this] def makeRegion(): Region = {
    val region = Region()
    rootRegion.addReferenceTo(region)
    region
  }

  def close(): Unit = {
    rootRegion.close()
    ctx.close()
  }

  def put(in: DataInputStream, out: DataOutputStream) {
    val decoder = codecs.makeDec(in)
    val region = makeRegion()
    var hasNext = in.read()
    assert(hasNext != -1)
    while (hasNext == 1) {
      val off = decoder.readRegionValue(region)
      val koff = codecs.keyPType.copyFromAddress(region, codecs.pType, off, false)
      store.store.put(koff, off)
      hasNext = in.read()
    }
    out.write(0)
    out.flush()
  }

  def get(in: DataInputStream, out: DataOutputStream) {
    val region = makeRegion()
    val keyDecoder = codecs.makeKeyDec(in)
    val encoder = codecs.makeEnc(out)
    val l = keyDecoder.readRegionValue(region)
    val r = keyDecoder.readRegionValue(region)

    log.info(s"SERV get l ${rvstr(codecs.keyPType, l)} r ${rvstr(codecs.keyPType, r)}")
    val it = store.store.iterator(l, true)
    var continue = it.hasNext
    while (continue) {
      val kv = it.next
      val k = kv.getKey
      val v = kv.getValue
      if (store.ord.lt(k, r)) {
        encoder.writeByte(1)
        encoder.writeRegionValue(v)
        encoder.flush()
        continue = it.hasNext
      } else {
        continue = false
      }
    }
    encoder.writeByte(0)
    out.flush()
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
