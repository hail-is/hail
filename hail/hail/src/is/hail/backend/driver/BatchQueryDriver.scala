package is.hail.backend.driver

import is.hail.{HailFeatureFlags, PrettyVersion}
import is.hail.annotations.Memory
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{Backend, ExecuteContext, OwningTempFileManager}
import is.hail.backend.service._
import is.hail.collection.ImmutableMap
import is.hail.collection.implicits.toRichIterable
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.io.fs.{CloudStorageFSConfig, FS, RouterFS}
import is.hail.io.reference.{IndexedFastaSequenceFile, LiftOver}
import is.hail.services._
import is.hail.services.oauth2.CloudCredentials
import is.hail.types.virtual.Kinds._
import is.hail.utils._
import is.hail.utils.ExecutionTimer.Timings
import is.hail.variant.ReferenceGenome

import scala.annotation.switch

import java.io.OutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Path

import org.json4s.JsonAST.JValue
import org.json4s.jackson.JsonMethods

object BatchQueryDriver extends HttpLikeRpc with Logging {

  case class Env(
    backend: Backend,
    flags: HailFeatureFlags,
    hcl: HailClassLoader,
    fs: FS,
    tmpdir: String,
    references: Map[String, ReferenceGenome],
    outputUrl: String,
    action: Int,
    payload: JValue,
  )

  implicit object Request extends HttpLikeRequest {

    override def route(env: Env): Route =
      (env.action: @switch) match {
        case 1 => Routes.TypeOf(Value)
        case 2 => Routes.TypeOf(Table)
        case 3 => Routes.TypeOf(Matrix)
        case 4 => Routes.TypeOf(BlockMatrix)
        case 5 => Routes.Execute
        case 6 => Routes.ParseVcfMetadata
        case 7 => Routes.ImportFam
        case 8 => Routes.LoadReferencesFromDataset
        case 9 => Routes.LoadReferencesFromFASTA
      }

    override def payload(env: Env): JValue = env.payload
    // service backend doesn't support sending timings back to the python client
    override def timings(env: Env, t: Timings): Unit = ()

    override def result(env: Env, result: Array[Byte]): Unit =
      retryTransientErrors {
        using(env.fs.createNoCompression(env.outputUrl)) { outputStream =>
          val output = new HailSocketAPIOutputStream(outputStream)
          output.writeBool(true)
          output.writeBytes(result)
        }
      }

    override def failure(env: Env, t: Throwable): Unit =
      retryTransientErrors {
        val (shortMessage, expandedMessage, errorId) =
          t match {
            case t: HailWorkerException =>
              logger.error(
                "A worker failed. The exception was written for Python but we will also throw an exception to fail this driver job.",
                t,
              )
              (t.shortMessage, t.expandedMessage, t.errorId)
            case _ =>
              logger.error(
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
  }

  implicit object Context extends Context {
    override def scoped[A](env: Env)(f: ExecuteContext => A): (A, Timings) =
      ExecutionTimer.time { timer =>
        ExecuteContext.scoped(
          tmpdir = env.tmpdir,
          localTmpdir = "file:///tmp",
          backend = env.backend,
          references = env.references,
          fs = env.fs,
          timer = timer,
          tempFileManager = new OwningTempFileManager(env.fs),
          theHailClassLoader = env.hcl,
          flags = env.flags,
          irMetadata = new IrMetadata(),
          blockMatrixCache = ImmutableMap.empty,
          codeCache = ImmutableMap.empty,
          irCache = ImmutableMap.empty,
          coercerCache = ImmutableMap.empty,
        )(f)
      }

    override def putReferences(env: Env)(refs: Iterable[ReferenceGenome]): Unit =
      // evaluate for effects
      ReferenceGenome.addFatalOnCollision(env.references, refs): Unit
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

    logger.info(f"${getClass.getName} $PrettyVersion")

    sys.env.get("HAIL_SSL_CONFIG_DIR").foreach(tls.setSSLConfigFromDir)

    val (rpcConfig, jobConfig, action, payload) = {
      val bootstrapFs =
        RouterFS.buildRoutes(
          CloudStorageFSConfig.fromFlagsAndEnv(
            Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
            HailFeatureFlags.fromEnv(),
          )
        )

      using(bootstrapFs.openNoCompression(inputURL)) { is =>
        val input = JsonMethods.parse(is)
        (
          (input \ "rpc_config").extract[ServiceBackendRPCPayload],
          (input \ "job_config").extract[BatchJobConfig],
          (input \ "action").extract[Int],
          input \ "payload",
        )
      }
    }

    // requester pays config is conveyed in feature flags currently
    val featureFlags =
      HailFeatureFlags.fromEnv(rpcConfig.flags)

    val fs =
      RouterFS.buildRoutes(
        CloudStorageFSConfig.fromFlagsAndEnv(
          Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
          featureFlags,
        )
      )

    val references =
      ReferenceGenome.addFatalOnCollision(
        ReferenceGenome.builtinReferences(),
        rpcConfig.custom_references.map(ReferenceGenome.fromJSON),
      )

    for ((sourceGenome, liftOversForSource) <- rpcConfig.liftovers)
      for ((destGenome, chainFile) <- liftOversForSource)
        references(sourceGenome).addLiftover(references(destGenome), LiftOver(fs, chainFile))

    for ((rg, seq) <- rpcConfig.sequences)
      references(rg).addSequence(IndexedFastaSequenceFile(fs, seq.fasta, seq.index))

    val backend =
      new ServiceBackend(
        name,
        BatchClient(
          DeployConfig.fromConfigFile("/deploy-config/deploy-config.json"),
          CloudCredentials(Some(Path.of(scratchDir, "secrets/gsa-key/key.json"))),
        ),
        JarUrl(jarLocation),
        BatchConfig.fromConfigFile(Path.of(scratchDir, "batch-config/batch-config.json")),
        jobConfig,
      )

    // FIXME: when can the classloader be shared? (optimizer benefits!)
    try runRpc(
        Env(
          backend,
          featureFlags,
          new HailClassLoader(getClass.getClassLoader),
          fs,
          rpcConfig.tmp_dir,
          references,
          outputURL,
          action,
          payload,
        )
      )
    finally backend.close()
  }
}

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

  override def close(): Unit =
    if (!closed) {
      out.close()
      closed = true
    }
}

case class SequenceConfig(fasta: String, index: String)

case class ServiceBackendRPCPayload(
  tmp_dir: String,
  flags: Map[String, String],
  custom_references: Array[String],
  liftovers: Map[String, Map[String, String]],
  sequences: Map[String, SequenceConfig],
)
