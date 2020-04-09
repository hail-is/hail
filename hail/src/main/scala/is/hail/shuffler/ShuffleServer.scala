package is.hail.shuffler

import is.hail.shuffler.ShuffleUtils._
import org.apache.log4j.Logger;
import is.hail.HailLSM
import is.hail.annotations.{Region, UnsafeRow}
import is.hail.expr.ir.IRParser
import is.hail.expr.types.encoded.EType
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.expr.types.virtual.{TStruct, Type}
import is.hail.io.TypedCodecSpec
import is.hail.rvd.AbstractRVDSpec
import java.io._
import java.net._
import java.security.KeyStore;
import java.util.UUID
import java.util.concurrent.{ConcurrentSkipListMap, Executors}
import javax.net._
import javax.net.ssl._
import javax.security.cert.X509Certificate;
import java.util.concurrent._
import org.json4s.jackson.{JsonMethods, Serialization}
import java.nio.charset.StandardCharsets

import scala.annotation.switch

import is.hail.utils._

class Handler (
  val server: ShuffleServer,
  val socket: Socket
) extends Runnable {
  val log = Logger.getLogger(this.getClass.getName());
  val in = new DataInputStream(socket.getInputStream)
  val out = new DataOutputStream(socket.getOutputStream)

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
            case -1 =>
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
        println(s"exception while serving ${e.getMessage}")
        log.info(s"exception while serving", e)
    }
  }

  def readShuffleUUID(): Shuffle = {
    val uuid = in.readUTF()
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
    val key = Wire.readListOfStrings(in)
    val uuid = UUID.randomUUID().toString()
    server.shuffles.put(uuid, new Shuffle(uuid, t, codecSpec, key))
    out.writeUTF(uuid)
    out.flush()
    log.info(s"SERV start done")
  }

  def put(): Unit = {
    log.info(s"SERV put")
    val shuffle = readShuffleUUID
    val decoder = shuffle.keyedCodecSpec.makeDec(in)
    val region = shuffle.region()
    var hasNext = in.read()
    assert(hasNext != -1)
    while (hasNext == 1) {
      val off = decoder.readRegionValue(region)
      val koff = shuffle.keyedCodecSpec.keyPType.copyFromAddress(region, shuffle.keyedCodecSpec.pType, off, false)
      shuffle.store.store.put(koff, off)
      hasNext = in.read()
    }
    out.write(0)
    out.flush()
    log.info(s"SERV put done")
  }

  def get(): Unit = {
    log.info(s"SERV get")
    val shuffle = readShuffleUUID
    val region = shuffle.region()
    val keyDecoder = shuffle.keyedCodecSpec.makeKeyDec(in)
    val encoder = shuffle.keyedCodecSpec.makeEnc(out)
    val l = keyDecoder.readRegionValue(region)
    val r = keyDecoder.readRegionValue(region)

    log.info(s"SERV get l ${rvstr(shuffle.keyedCodecSpec.keyPType, l)} r ${rvstr(shuffle.keyedCodecSpec.keyPType, r)}")
    val it = shuffle.store.store.iterator(l, true)
    var continue = it.hasNext
    while (continue) {
      val kv = it.next
      val k = kv.getKey
      val v = kv.getValue
      if (shuffle.store.ord.lt(k, r)) {
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
    log.info(s"SERV get done")
  }
}

class Shuffle (
  uuid: String,
  t: TStruct,
  codecSpec: TypedCodecSpec,
  key: Array[String]
) extends AutoCloseable {
  val keyedCodecSpec = new KeyedCodecSpec(t, codecSpec, key)
  import keyedCodecSpec._

  val rootRegion = Region()

  val store = {
    def keyDec(in: InputStream) = {
      val dec = makeKeyDec(in)
      () => dec.readRegionValue(rootRegion)
    }
    def keyEnc(out: OutputStream) = {
      val enc = makeKeyEnc(out)

      { (x: Long) =>
        enc.writeRegionValue(x)
        enc.flush()
      }
    }
    def dec(in: InputStream) = {
      val dec = makeDec(in)
      () => dec.readRegionValue(rootRegion)
    }
    def enc(out: OutputStream) = {
      val enc = makeEnc(out)

      { (x: Long) =>
        enc.writeRegionValue(x)
        enc.flush()
      }
    }

    new HailLSM(
      s"/tmp/${uuid}",
      decodedKeyPType.unsafeOrdering,
      keyDec,
      keyEnc,
      dec,
      enc
    )
  }

  def region(): Region = {
    val region = Region()
    rootRegion.addReferenceTo(region)
    region
  }

  def close(): Unit = rootRegion.close()
}

class ShuffleServer (
  ssl: SSLContext,
  port: Int
) {
  val log = Logger.getLogger(this.getClass.getName());

  val shuffles = new ConcurrentSkipListMap[String, Shuffle]()

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
