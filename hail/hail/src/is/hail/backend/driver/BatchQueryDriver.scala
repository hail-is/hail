package is.hail.backend.driver

import is.hail.{HailFeatureFlags, PrettyVersion}
import is.hail.annotations.Memory
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{Backend, ExecuteContext, OwningTempFileManager}
import is.hail.backend.service._
import is.hail.backend.service.ServiceBackend.DefaultMaxReadParallelism
import is.hail.collection.ImmutableMap
import is.hail.collection.implicits.toRichIterable
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.io.fs.{CloudStorageConfig, FS, RequesterPaysConfig, RouterFS}
import is.hail.io.reference.{Liftover, ReferenceSequence, RgAttachments}
import is.hail.services._
import is.hail.services.oauth2.CloudCredentials
import is.hail.types.virtual.Kinds._
import is.hail.utils._
import is.hail.variant.{ReferenceGenome, ReferenceGenomeEnv}

import scala.annotation.switch

import java.io.OutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Path

import org.json4s.{CustomSerializer, DefaultFormats, Formats}
import org.json4s.JsonAST.{JArray, JValue}
import org.json4s.jackson.JsonMethods

object BatchQueryDriver extends HttpLikeRpc with Logging {

  case class Env(
    backend: Backend,
    flags: HailFeatureFlags,
    hcl: HailClassLoader,
    fs: FS,
    tmpdir: String,
    references: ReferenceGenomeEnv,
    attachments: RgAttachments,
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
    override def scoped[A](env: Env)(f: ExecuteContext => A): A =
      TimedBlock.enter {
        ExecuteContext.scoped(
          tmpdir = env.tmpdir,
          localTmpdir = "file:///tmp",
          backend = env.backend,
          references = env.references,
          attachments = env.attachments,
          fs = env.fs,
          tempFileManager = new OwningTempFileManager(env.fs),
          theHailClassLoader = env.hcl,
          flags = env.flags,
          irMetadata = new IrMetadata(),
          blockMatrixCache = ImmutableMap.empty,
          compileCache = ImmutableMap.empty,
          irCache = ImmutableMap.empty,
          coercerCache = ImmutableMap.empty,
        )(f)
      }

    // Only checks that `refs` could be added: Python sends them with the next request.
    override def putReferences(env: Env)(refs: Iterable[ReferenceGenome]): Unit =
      env.references.withReferences(refs): Unit
  }

  /** The reference genome attachments Python sends: liftover chain files by source then destination
    * name, and sequences by reference genome name. A liftover or sequence attached to a reference
    * genome not in `references` is skipped with a warning, so that only a use of it fails.
    */
  private[driver] def loadAttachments(
    fs: FS,
    references: ReferenceGenomeEnv,
    liftovers: Map[String, Map[String, String]],
    sequences: Map[String, SequenceConfig],
  ): RgAttachments = {
    def inScope(description: String, names: String*): Boolean =
      names.filter(references.get(_).isEmpty) match {
        case Seq() => true
        case missing =>
          logger.warn(
            s"Skipping $description: reference genome ${missing.map(n => s"'$n'").mkString(", ")} " +
              "is not in scope."
          )
          false
      }

    val withLiftovers =
      liftovers.foldLeft(RgAttachments.empty) { case (acc, (source, chainFiles)) =>
        chainFiles.foldLeft(acc) { case (acc, (dest, chainFile)) =>
          if (inScope(s"liftover from '$source' to '$dest'", source, dest))
            acc.withLiftover(Liftover(fs, references(source), references(dest), chainFile))
          else acc
        }
      }

    sequences.foldLeft(withLiftovers) { case (acc, (rg, seq)) =>
      if (inScope(s"reference sequence of '$rg'", rg))
        acc.withSequence(ReferenceSequence(fs, references(rg), seq.fasta, seq.index))
      else acc
    }
  }

  implicit override val fmts: Formats =
    DefaultFormats + RequesterPaysConfigFormats

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

    logger.info(f"Hail $PrettyVersion")

    sys.env.get("HAIL_SSL_CONFIG_DIR").foreach(tls.setSSLConfigFromDir)

    val fsConfig =
      CloudStorageConfig.readEnv(
        Some(Path.of(scratchDir, "secrets/gsa-key/key.json"))
      )

    val (rpcConfig, jobConfig, action, payload) = {
      val bootstrapFs = RouterFS.buildRoutes(fsConfig)

      using(bootstrapFs.openNoCompression(inputURL)) { is =>
        val input = JsonMethods.parse(is)
        (
          (input \ "rpc_config").extract[ServiceBackendRPCConfig],
          (input \ "job_config").extract[BatchJobConfig],
          (input \ "action").extract[Int],
          input \ "payload",
        )
      }
    }

    val fs =
      RouterFS.buildRoutes(
        fsConfig.copy(
          google = fsConfig.google.map(_.copy(
            requester_pays_config = rpcConfig.requester_pays_config
          ))
        )
      )

    val references =
      ReferenceGenomeEnv(rpcConfig.custom_references.map(ReferenceGenome.fromJSON))

    val attachments =
      loadAttachments(fs, references, rpcConfig.liftovers, rpcConfig.sequences)

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
        rpcConfig.max_read_parallelism.getOrElse(DefaultMaxReadParallelism),
      )

    // FIXME: when can the classloader be shared? (optimizer benefits!)
    try runRpc(
        Env(
          backend,
          HailFeatureFlags.fromEnv(rpcConfig.flags),
          new HailClassLoader(getClass.getClassLoader),
          fs,
          rpcConfig.tmp_dir,
          references,
          attachments,
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

case class ServiceBackendRPCConfig(
  tmp_dir: String,
  flags: Map[String, String],
  requester_pays_config: Option[RequesterPaysConfig],
  custom_references: Array[String],
  liftovers: Map[String, Map[String, String]],
  sequences: Map[String, SequenceConfig],
  max_read_parallelism: Option[Int],
)

object RequesterPaysConfigFormats
    extends CustomSerializer[RequesterPaysConfig](implicit fmts =>
      (
        {
          case JArray(List(project, buckets)) =>
            RequesterPaysConfig(
              project = project.extract[String],
              buckets = buckets.extract[Option[Set[String]]],
            )
        },
        PartialFunction.empty,
      )
    )
