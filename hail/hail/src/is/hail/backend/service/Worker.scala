package is.hail.backend.service

import is.hail.{HailFeatureFlags, Revision}
import is.hail.asm4s._
import is.hail.backend.Backend.PartitionFn
import is.hail.backend.HailTaskContext
import is.hail.io.fs._
import is.hail.services._
import is.hail.utils._

import scala.collection.mutable
import scala.concurrent.{Await, ExecutionContext, ExecutionException, Future}
import scala.concurrent.duration.Duration

import java.io._
import java.nio.charset._
import java.nio.file.Path
import java.util
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

class ServiceTaskContext(val partitionId: Int) extends HailTaskContext {
  override def stageId(): Int = 0

  override def attemptNumber(): Int = 0
}

class WorkerTimer extends Logging {

  var startTimes: mutable.Map[String, Long] = mutable.Map()

  def start(label: String): Unit =
    startTimes.update(label, System.nanoTime())

  def end(label: String): Unit = {
    val endTime = System.nanoTime()
    val startTime = startTimes.get(label)
    startTime.foreach { s =>
      val durationMS = "%.6f".format((endTime - s).toDouble / 1000000.0)
      logger.info(s"$label took $durationMS ms.")
    }
  }

  def enter[A](label: String)(f: => A): A = {
    start(label)
    try f
    finally end(label)
  }
}

// Java's ObjectInputStream does not properly use the context classloader that
// we set on the thread and knows about the hail jar. We need to explicitly force
// the ObjectInputStream to use the correct classloader, but it is not configurable
// so we override the behavior ourselves.
// For more context, see: https://github.com/scala/bug/issues/9237#issuecomment-292436652
object ExplicitClassLoaderInputStream {
  val primClasses: util.HashMap[String, Class[_]] = {
    val m = new util.HashMap[String, Class[_]](8, 1.0f)
    m.put("boolean", Boolean.getClass)
    m.put("byte", Byte.getClass)
    m.put("char", Char.getClass)
    m.put("short", Short.getClass)
    m.put("int", Int.getClass)
    m.put("long", Long.getClass)
    m.put("float", Float.getClass)
    m.put("double", Double.getClass)
    // FIXME: I (ps) don't understand what this code is doing. In Scala 2.13 the Unit object isn't
    // allowed to be named anymore. Why are we loading scala companion objects here?
    m.put("void", ().getClass)
    m
  }
}

class ExplicitClassLoaderInputStream(is: InputStream, cl: ClassLoader)
    extends ObjectInputStream(is) {

  override def resolveClass(desc: ObjectStreamClass): Class[_] = {
    val name = desc.getName
    try Class.forName(name, false, cl)
    catch {
      case ex: ClassNotFoundException =>
        val cl = ExplicitClassLoaderInputStream.primClasses.get(name)
        if (cl != null) cl
        else throw ex
    }
  }
}

object WireProtocol {

  implicit class Write(private val os: DataOutputStream) extends AnyVal {

    def writeBytes(bs: Array[Byte]): Unit = {
      os.writeInt(bs.length)
      os.write(bs)
    }

    def writeString(s: String): Unit =
      os.writeBytes(s.getBytes(StandardCharsets.UTF_8))

    def writeSuccess(partId: Int, bs: Array[Byte]): Unit = {
      os.writeInt(partId)
      os.writeBytes(bs)
    }

    def writeFailure(partId: Int, e: Throwable): Unit = {
      val (shortMessage, expandedMessage, errorId) = handleForPython(e)
      os.writeInt(partId)
      os.writeString(shortMessage)
      os.writeString(expandedMessage)
      os.writeInt(errorId)
    }
  }

  implicit class Read(private val is: DataInputStream) extends AnyVal {

    def readBytes(): Array[Byte] = {
      val length = is.readInt()
      val bs = new Array[Byte](length)
      is.readFully(bs)
      bs
    }

    def readString(): String =
      new String(is.readBytes(), StandardCharsets.UTF_8)

    def readSuccess(): (Array[Byte], Int) = {
      val partId = is.readInt()
      val bytes = is.readBytes()
      bytes -> partId
    }

    def readFailure(): HailWorkerException =
      HailWorkerException(
        partitionId = is.readInt(),
        shortMessage = is.readString(),
        expandedMessage = is.readString(),
        errorId = is.readInt(),
      )
  }

  def write(os: DataOutputStream, partId: Int, result: Either[Throwable, Array[Byte]]): Unit =
    result match {
      case Left(throwable) =>
        os.writeByte(0)
        os.writeFailure(partId, throwable)

      case Right(bytes) =>
        os.writeByte(1)
        os.writeSuccess(partId, bytes)
    }

  def read(is: DataInputStream): Either[HailWorkerException, (Array[Byte], Int)] =
    is.readByte() match {
      case 0 => Left(is.readFailure())
      case 1 => Right(is.readSuccess())
    }
}

object Worker extends Logging {

  def main(argv: Array[String]): Unit = {
    if (argv.length != 7) {
      throw new IllegalArgumentException(s"expected seven arguments, not: ${argv.length}")
    }

    val scratchDir = argv(0)
    // val logFile = argv(1)
    // var jarLocation = argv(2)
    val kind = argv(3)
    assert(kind == Main.WORKER)
    val root = argv(4)
    val partition = argv(5).toInt
    val index = argv(6).toInt
    val timer = new WorkerTimer()

    sys.env.get("HAIL_SSL_CONFIG_DIR").foreach(tls.setSSLConfigFromDir)

    logger.info(s"${getClass.getName} $Revision")
    logger.info(s"running partition $partition root '$root' with scratch directory '$scratchDir'")

    timer.start(s"partition $partition")

    val hcl = new HailClassLoader(getClass.getClassLoader)

    val fs = RouterFS.buildRoutes(
      CloudStorageFSConfig.fromFlagsAndEnv(
        Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
        HailFeatureFlags.fromEnv(),
      )
    )

    implicit val ec: ExecutionContext =
      ExecutionContext.fromExecutor(
        Executors.newCachedThreadPool {
          val threadFactory = Executors.defaultThreadFactory()
          val counter = new AtomicInteger(0)
          task =>
            val thread = threadFactory.newThread(task)
            thread.setName(f"hail-worker-thread-${counter.getAndIncrement()}")
            thread.setDaemon(true)
            thread
        }
      )

    def open(x: String): SeekableDataInputStream =
      fs.openNoCompression(x)

    val inputs: Either[Throwable, (Array[Byte], Array[Byte], PartitionFn)] =
      timer.enter("read inputs") {
        val globals = Future {
          retryTransientErrors {
            using(open(s"$root/globals"))(_.readAllBytes())
          }
        }

        val context = Future {
          retryTransientErrors {
            using(open(s"$root/contexts")) { is =>
              is.seek(index.toLong * 12)
              val offset = is.readLong()
              val length = is.readInt()
              is.seek(offset)
              val context = new Array[Byte](length)
              is.readFully(context)
              context
            }
          }
        }

        val partitionFn = Future {
          retryTransientErrors {
            using(new ExplicitClassLoaderInputStream(open(s"$root/f"), hcl)) {
              _.readObject().asInstanceOf[PartitionFn]
            }
          }
        }

        try Await.result(
            globals.zip(context).zipWith(partitionFn)((gc, f) => Right((gc._1, gc._2, f))),
            Duration.Inf,
          )
        catch {
          case t: ExecutionException => Left(t.getCause)
          case t: Throwable => Left(t)
        }
      }

    val result: Either[Throwable, Array[Byte]] =
      inputs.flatMap { case (globals, context, f) =>
        timer.enter("execute") {
          try
            using(new ServiceTaskContext(partition)) { htc =>
              retryTransientErrors {
                Right(f(globals, context, htc, hcl, fs))
              }
            }
          catch {
            case t: Throwable => Left(t)
          }
        }
      }

    timer.enter("write outputs") {
      retryTransientErrors {
        fs.writePDOS(s"$root/result.$index")(dos => WireProtocol.write(dos, partition, result))
      }
    }

    timer.end(s"partition $partition")
    logger.info(s"finished job $index at root $root")

    result.left.foreach { throwableWhileExecutingUserCode =>
      logger.info("throwing the exception so that this Worker job is marked as failed.")
      throw throwableWhileExecutingUserCode
    }
  }
}
