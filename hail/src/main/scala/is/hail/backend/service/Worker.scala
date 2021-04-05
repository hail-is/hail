package is.hail.backend.service

import java.io._
import java.net._
import java.nio.charset._
import java.util.concurrent._

import is.hail.HAIL_REVISION
import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.{Backend, BackendContext, BroadcastValue, HailTaskContext}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering.{DArrayLowering, LoweringPipeline, TableStage, TableStageDependency}
import is.hail.expr.ir.{Compile, ExecuteContext, IR, IRParser, Literal, MakeArray, MakeTuple, ShuffleRead, ShuffleWrite, SortField, ToStream}
import is.hail.io.fs.GoogleStorageFS
import is.hail.linalg.BlockMatrix
import is.hail.rvd.RVDPartitioner
import is.hail.services._
import is.hail.services.batch_client.BatchClient
import is.hail.services.shuffler.ShuffleClient
import is.hail.types._
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.commons.io.IOUtils
import org.apache.log4j.Logger
import org.json4s.JsonAST._
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats}
import org.newsclub.net.unix.{AFUNIXServerSocket, AFUNIXSocketAddress}

import scala.annotation.switch
import scala.collection.mutable
import scala.reflect.ClassTag
import java.lang.reflect.InvocationTargetException
import is.hail.io.fs.FS

class ServiceTaskContext(val partitionId: Int) extends HailTaskContext {
  override type BackendType = ServiceBackend

  override def stageId(): Int = 0

  override def attemptNumber(): Int = 0
}

object WorkerTimer {
  private val log = Logger.getLogger(getClass.getName())
}

class WorkerTimer() {
  import WorkerTimer._

  var startTimes: mutable.Map[String, Long] = mutable.Map()
  def start(label: String): Unit = {
    startTimes.put(label, System.nanoTime())
  }

  def end(label: String): Unit = {
    val endTime = System.nanoTime()
    val startTime = startTimes.get(label)
    startTime.foreach { s =>
      val durationMS = "%.6f".format((endTime - s).toDouble / 1000000.0)
      log.info(s"$label took $durationMS ms.")
    }
  }
}

object Worker {
  val myRevision = HAIL_REVISION

  private[this] val log = Logger.getLogger(getClass.getName())
  private[this] val revisions = mutable.Map[String, Array[String] => Unit]()
  private[this] val scratchDir = {
    val x = System.getenv("HAIL_WORKER_SCRATCH_DIR")
    if (x == null) "" else x
  }
  private[this] val workerGCSFS = retryTransientErrors {
    using(new FileInputStream(s"/worker-key.json")) { is =>
      new GoogleStorageFS(IOUtils.toString(is, Charset.defaultCharset().toString()))
    }
  }

  private[this] def withThreadContextClassLoader[T](custom: ClassLoader)(f: => T): T = {
    val original = Thread.currentThread().getContextClassLoader()
    try {
      Thread.currentThread().setContextClassLoader(custom)
      f
    } finally {
      Thread.currentThread().setContextClassLoader(original)
    }
  }

  private[this] val STRING_ARRAY_CLASS = Class.forName("[Ljava.lang.String;")
  private[this] def loadMainFromLocalJar(revision: String, localJarURL: URL): Array[String] => Unit = {
    val classLoader = new LoadSelfFirstURLClassLoader(Array(localJarURL))
    val reflectedMain = withThreadContextClassLoader(classLoader) {
      val workerClass = classLoader.loadClass("is.hail.backend.service.Worker")

      val actualRevision = workerClass.getMethod("myRevision").invoke(null).asInstanceOf[String]
      assert(revision == actualRevision, s"$revision != $actualRevision")

      workerClass.getMethod("main", STRING_ARRAY_CLASS)
    }

    { (args: Array[String]) =>
      withThreadContextClassLoader(classLoader) {
        try {
          reflectedMain.invoke(null, args): Unit
        } catch {
          case e: InvocationTargetException =>
            throw e.getCause()
        }
      }
    }
  }

  private[this] def downloadFile(remoteFS: FS, remote: String, local: String): Unit =
    retryTransientErrors {
      using(remoteFS.openNoCompression(remote)) { is =>
        using(new FileOutputStream(local)) { os =>
          IOUtils.copy(is, os)
        }
      }
    }

  private[this] def mainForRevision(revision: String, remoteJarLocation: String): Array[String] => Unit = {
    revisions.get(revision) match {
      case Some(mainMethod) =>
        log.info(s"$revision found in the memory cache")
        mainMethod
      case None =>
        log.info(s"$revision not in the memory cache")

        val localJarLocation = s"/hail-jars/${revision}.jar"
        val localJarFile = new File(localJarLocation)
        if (!localJarFile.exists()) {
          log.info(s"$revision not in the disk cache")
          downloadFile(workerGCSFS, remoteJarLocation, localJarLocation)
          log.info(s"$revision added to the disk cache at $localJarLocation")
        }

        val localJarURL = localJarFile.toURI().toURL()
        val mainMethod = loadMainFromLocalJar(revision, localJarURL)
        log.info(s"$revision loaded")
        revisions.put(revision, mainMethod)
        mainMethod
    }
  }

  def main(args: Array[String]): Unit = {
    val timer = new WorkerTimer()
    timer.start("main")

    log.info(s"is.hail.backend.service.Worker $myRevision")
    if (args.length < 2) {
      throw new IllegalArgumentException(s"expected at least two arguments, not: ${ args.length }")
    }

    val revision = args(0)
    val remoteJarLocation = args(1)
    if (revision != myRevision) {
      timer.start("classLoading")
      log.info(s"received job for different revision: $revision; I am $myRevision")
      val mainMethod = mainForRevision(revision, remoteJarLocation)
      timer.end("classLoading")
      return mainMethod(args)
    }

    if (args.length != 4) {
      throw new IllegalArgumentException(s"expected at four arguments, not: ${ args.length }")
    }
    val root = args(2)
    val i = args(3).toInt

    log.info(s"running job $i at root $root with scratch directory '$scratchDir'")

    timer.start(s"Job $i")
    timer.start("readInputs")

    val fs = retryTransientErrors {
      using(new FileInputStream(s"$scratchDir/gsa-key/key.json")) { is =>
        new GoogleStorageFS(IOUtils.toString(is, Charset.defaultCharset().toString()))
      }
    }

    val f = retryTransientErrors {
      using(new ObjectInputStream(fs.openNoCompression(s"$root/f"))) { is =>
        is.readObject().asInstanceOf[(Array[Byte], HailTaskContext) => Array[Byte]]
      }
    }

    var offset = 0L
    var length = 0

    retryTransientErrors {
      using(fs.openNoCompression(s"$root/context.offsets")) { is =>
        is.seek(i * 12)
        offset = is.readLong()
        length = is.readInt()
      }
    }

    val context = retryTransientErrors {
      using(fs.openNoCompression(s"$root/contexts")) { is =>
        is.seek(offset)
        val context = new Array[Byte](length)
        is.readFully(context)
        context
      }
    }
    timer.end("readInputs")
    timer.start("executeFunction")

    val hailContext = HailContext(
      new WorkerBackend(), skipLoggingConfiguration = true, quiet = true)
    val htc = new ServiceTaskContext(i)
    HailTaskContext.setTaskContext(htc)
    val result = f(context, htc)
    HailTaskContext.finish()

    timer.end("executeFunction")
    timer.start("writeOutputs")

    using(fs.createNoCompression(s"$root/result.$i")) { os =>
      os.write(result)
    }
    timer.end("writeOutputs")
    timer.end(s"Job $i")
    timer.end(s"main")
    log.info(s"finished job $i at root $root")
  }
}

class WorkerBackend extends Backend {
  def defaultParallelism: Int = 1

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] = ???

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String): Unit = ???

  def unpersist(backendContext: BackendContext, id: String): Unit = ???

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix = ???

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType = ???

  def parallelizeAndComputeWithIndex(backendContext: BackendContext, collection: Array[Array[Byte]], dependency: Option[TableStageDependency] = None)(f: (Array[Byte], HailTaskContext) => Array[Byte]): Array[Array[Byte]] = ???

  def stop(): Unit = {
  }

  def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    relationalLetsAbove: Map[String, IR],
    rowTypeRequiredness: RStruct
  ): TableStage = ???
}
