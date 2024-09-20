package is.hail.backend.api

import is.hail.{HailContext, HailFeatureFlags}
import is.hail.annotations.Memory
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{Backend, ExecuteContext, HttpLikeBackendRpc}
import is.hail.backend.caching.NoCaching
import is.hail.backend.service._
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.io.fs.{CloudStorageFSConfig, FS, RouterFS}
import is.hail.io.reference.{IndexedFastaSequenceFile, LiftOver}
import is.hail.services._
import is.hail.types.virtual.Kinds
import is.hail.utils.{ErrorHandling, ExecutionTimer, HailWorkerException, Logging, toRichIterable, using}
import is.hail.utils.ExecutionTimer.Timings
import is.hail.variant.ReferenceGenome
import org.json4s.{DefaultFormats, Formats}

import scala.annotation.switch
import scala.collection.mutable
import java.io.OutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import org.json4s.JsonAST.JValue
import org.json4s.jackson.JsonMethods

object ServiceBackendApi extends HttpLikeBackendRpc[Request] with Logging {

  implicit object Handler
      extends Routing with Write[Request] with Context[Request] with ErrorHandling {
    import Routes._

    override def route(a: Request): Route =
      (a.action: @switch) match {
        case 1 => TypeOf(Kinds.Value)
        case 2 => TypeOf(Kinds.Table)
        case 3 => TypeOf(Kinds.Matrix)
        case 4 => TypeOf(Kinds.BlockMatrix)
        case 5 => Execute
        case 6 => ParseVcfMetadata
        case 7 => ImportFam
        case 8 => LoadReferencesFromDataset
        case 9 => LoadReferencesFromFASTA
      }

    override def payload(a: Request): JValue = a.payload

    // service backend doesn't support sending timings back to the python client
    override def timings(env: Request)(t: Timings): Unit =
      ()

    override def result(env: Request)(result: Array[Byte]): Unit =
      retryTransientErrors {
        using(env.fs.createNoCompression(env.outputUrl)) { outputStream =>
          val output = new HailSocketAPIOutputStream(outputStream)
          output.writeBool(true)
          output.writeBytes(result)
        }
      }

    override def error(env: Request)(t: Throwable): Unit =
      retryTransientErrors {
        val (shortMessage, expandedMessage, errorId) =
          t match {
            case t: HailWorkerException =>
              log.error(
                "A worker failed. The exception was written for Python but we will also throw an exception to fail this driver job.",
                t,
              )
              (t.shortMessage, t.expandedMessage, t.errorId)
            case _ =>
              log.error(
                "An exception occurred in the driver. The exception was written for Python but we will re-throw to fail this driver job.",
                t,
              )
              handleForPython(t)
          }

        using(env.fs.createNoCompression(env.outputUrl)) { outputStream =>
          val output = new HailSocketAPIOutputStream(outputStream)
          output.writeBool(false)
          output.writeString(shortMessage)
          output.writeString(expandedMessage)
          output.writeInt(errorId)
        }

        throw t
      }

    override def scoped[A](env: Request)(f: ExecuteContext => A): (A, Timings) =
      ExecutionTimer.time { timer =>
        ExecuteContext.scoped(
          env.rpcConfig.tmp_dir,
          env.rpcConfig.remote_tmpdir,
          env.backend,
          env.fs,
          timer,
          null,
          env.hcl,
          env.flags,
          new IrMetadata(),
          mutable.Map(env.references.toSeq: _*),
          NoCaching,
          NoCaching,
          NoCaching,
          NoCaching,
        )(f)
      }
  }

  def main(argv: Array[String]): Unit = {
    assert(argv.length == 7, argv.toFastSeq)

    val scratchDir = argv(0)
    // val logFile = argv(1)
    val jarLocation = argv(2)
    val kind = argv(3)
    assert(kind == Main.DRIVER)
    val name = argv(4)
    val inputURL = argv(5)
    val outputURL = argv(6)

    val deployConfig = DeployConfig.fromConfigFile("/deploy-config/deploy-config.json")
    DeployConfig.set(deployConfig)
    sys.env.get("HAIL_SSL_CONFIG_DIR").foreach(tls.setSSLConfigFromDir)

    var fs = RouterFS.buildRoutes(
      CloudStorageFSConfig.fromFlagsAndEnv(
        Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
        HailFeatureFlags.fromEnv(),
      )
    )

    val (rpcConfig, jobConfig, action, payload) =
      using(fs.openNoCompression(inputURL)) { is =>
        val input = JsonMethods.parse(is)
        (
          (input \ "rpc_config").extract[ServiceBackendRPCPayload],
          (input \ "job_config").extract[BatchJobConfig],
          (input \ "action").extract[Int],
          input \ "payload",
        )
      }

    // requester pays config is conveyed in feature flags currently
    val featureFlags = HailFeatureFlags.fromEnv(rpcConfig.flags)
    fs = RouterFS.buildRoutes(
      CloudStorageFSConfig.fromFlagsAndEnv(
        Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
        featureFlags,
      )
    )

    val references: Map[String, ReferenceGenome] =
      ReferenceGenome.builtinReferences() ++
        rpcConfig.custom_references.map(ReferenceGenome.fromJSON).map(rg => rg.name -> rg)

    rpcConfig.liftovers.foreach { case (sourceGenome, liftoversForSource) =>
      liftoversForSource.foreach { case (destGenome, chainFile) =>
        references(sourceGenome).addLiftover(references(destGenome), LiftOver(fs, chainFile))
      }
    }

    rpcConfig.sequences.foreach { case (rg, seq) =>
      references(rg).addSequence(IndexedFastaSequenceFile(fs, seq.fasta, seq.index))
    }

    val backend = new ServiceBackend(
      name,
      BatchClient(deployConfig, Path.of(scratchDir, "secrets/gsa-key/key.json")),
      JarUrl(jarLocation),
      BatchConfig.fromConfigFile(Path.of(scratchDir, "batch-config/batch-config.json")),
      jobConfig,
    )

    log.info("ServiceBackend allocated.")
    if (HailContext.isInitialized) {
      HailContext.get.backend = backend
      log.info("Default references added to already initialized HailContexet.")
    } else {
      HailContext(backend, 50, 3)
      log.info("HailContexet initialized.")
    }

    // FIXME: when can the classloader be shared? (optimizer benefits!)
    runRpc(
      Request(
        backend,
        featureFlags,
        new HailClassLoader(getClass.getClassLoader),
        rpcConfig,
        fs,
        references,
        outputURL,
        action,
        payload,
      )
    )
  }
}

case class Request(
  backend: Backend,
  flags: HailFeatureFlags,
  hcl: HailClassLoader,
  rpcConfig: ServiceBackendRPCPayload,
  fs: FS,
  references: Map[String, ReferenceGenome],
  outputUrl: String,
  action: Int,
  payload: JValue,
)

private class HailSocketAPIOutputStream(
  private[this] val out: OutputStream
) extends AutoCloseable {
  private[this] var closed: Boolean = false
  private[this] val dummy = new Array[Byte](8)

  def writeBool(b: Boolean): Unit =
    out.write(if (b) 1 else 0)

  def writeInt(v: Int): Unit = {
    Memory.storeInt(dummy, 0, v)
    out.write(dummy, 0, 4)
  }

  def writeLong(v: Long): Unit = {
    Memory.storeLong(dummy, 0, v)
    out.write(dummy)
  }

  def writeBytes(bytes: Array[Byte]): Unit = {
    writeInt(bytes.length)
    out.write(bytes)
  }

  def writeString(s: String): Unit = writeBytes(s.getBytes(StandardCharsets.UTF_8))

  def close(): Unit =
    if (!closed) {
      out.close()
      closed = true
    }
}

case class SequenceConfig(fasta: String, index: String)

case class ServiceBackendRPCPayload(
  tmp_dir: String,
  remote_tmpdir: String,
  flags: Map[String, String],
  custom_references: Array[String],
  liftovers: Map[String, Map[String, String]],
  sequences: Map[String, SequenceConfig],
)

case class BatchJobConfig(
  token: String,
  billing_project: String,
  worker_cores: String,
  worker_memory: String,
  storage: String,
  cloudfuse_configs: Array[CloudfuseConfig],
  regions: Array[String],
)
