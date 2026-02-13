package is.hail.backend.driver

import is.hail.backend.{Backend, ExecuteContext}
import is.hail.collection.FastSeq
import is.hail.expr.ir.IRParser
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.functions.IRFunctionRegistry.UserDefinedFnKey
import is.hail.io.BufferSpec
import is.hail.io.plink.LoadPlink
import is.hail.io.vcf.LoadVCF
import is.hail.types.virtual.{Kind, TFloat64}
import is.hail.types.virtual.Kinds._
import is.hail.utils.{jsonToBytes, using, ExecutionTimer}
import is.hail.utils.ExecutionTimer.Timings
import is.hail.variant.ReferenceGenome

import scala.collection.mutable
import scala.util.control.NonFatal

import java.io.ByteArrayOutputStream

import org.json4s.{DefaultFormats, Extraction, Formats, JArray, JValue}

case class SerializedIRFunction(
  name: String,
  type_parameters: Array[String],
  value_parameter_names: Array[String],
  value_parameter_types: Array[String],
  return_type: String,
  rendered_body: String,
)

trait BackendRpc {

  sealed trait Command extends Product with Serializable

  object Commands {
    case class TypeOf(k: Kind, ir: String) extends Command
    case class Execute(ir: String, fs: Array[SerializedIRFunction], codec: String) extends Command
    case class ParseVcfMetadata(path: String) extends Command

    case class ImportFam(path: String, isQuantPheno: Boolean, delimiter: String, missing: String)
        extends Command

    case class LoadReferencesFromDataset(path: String) extends Command

    case class LoadReferencesFromFASTA(
      name: String,
      fasta_file: String,
      index_file: String,
      x_contigs: Array[String],
      y_contigs: Array[String],
      mt_contigs: Array[String],
      par: Array[String],
    ) extends Command
  }

  type Env

  trait Request {
    def command(env: Env): Command
    def result(env: Env, r: Array[Byte]): Unit
    def timings(env: Env, t: ExecutionTimer.Timings): Unit
    def failure(env: Env, t: Throwable): Unit
  }

  trait Context {
    def scoped[A](env: Env)(f: ExecuteContext => A): (A, Timings)
    def putReferences(env: Env)(refs: Iterable[ReferenceGenome]): Unit
  }

  implicit val fmts: Formats = DefaultFormats

  import Commands._

  final def runRpc(env: Env)(implicit R: Request, C: Context): Unit =
    try {
      val command = R.command(env)
      val (result, timings) =
        C.scoped(env) { ctx =>
          command match {
            case TypeOf(kind, s) =>
              jsonToBytes {
                (kind match {
                  case Value => IRParser.parse_value_ir(ctx, s)
                  case Table => IRParser.parse_table_ir(ctx, s)
                  case Matrix => IRParser.parse_matrix_ir(ctx, s)
                  case BlockMatrix => IRParser.parse_blockmatrix_ir(ctx, s)
                }).typ.toJSON
              }

            case Execute(s, fns, codec) =>
              val bufferSpec = BufferSpec.parseOrDefault(codec)
              withRegisterSerializedFns(ctx, fns) {
                val ir = IRParser.parse_value_ir(ctx, s)
                val res = ctx.backend.execute(ctx, ir)
                res match {
                  case Left(_) => Array.empty[Byte]
                  case Right((pt, off)) =>
                    using(new ByteArrayOutputStream()) { os =>
                      Backend.encodeToOutputStream(ctx, pt, off, bufferSpec, os)
                      os.toByteArray
                    }
                }
              }

            case ParseVcfMetadata(path) =>
              jsonToBytes {
                val metadata = LoadVCF.parseHeaderMetadata(ctx.fs, Set.empty, TFloat64, path)
                Extraction.decompose(metadata)
              }

            case ImportFam(path, isQuantPheno, delimiter, missing) =>
              jsonToBytes {
                LoadPlink.importFamJSON(ctx.fs, path, isQuantPheno, delimiter, missing)
              }

            case LoadReferencesFromDataset(path) =>
              jsonToBytes {
                val rgs = ReferenceGenome.fromHailDataset(ctx.fs, path)
                C.putReferences(env)(rgs)
                JArray(rgs.map(_.toJSON).toList)
              }

            case LoadReferencesFromFASTA(name, fasta, index, xContigs, yContigs, mtContigs, par) =>
              jsonToBytes {
                val rg = ReferenceGenome.fromFASTAFile(
                  ctx,
                  name,
                  fasta,
                  index,
                  xContigs,
                  yContigs,
                  mtContigs,
                  par,
                )
                C.putReferences(env)(FastSeq(rg))
                rg.toJSON
              }
          }
        }

      R.result(env, result)
      R.timings(env, timings)
    } catch {
      case NonFatal(error) => R.failure(env, error)
    }

  private[this] def withRegisterSerializedFns[A](
    ctx: ExecuteContext,
    serializedFns: Array[SerializedIRFunction],
  )(
    body: => A
  ): A = {
    val fns = mutable.ArrayBuffer.empty[UserDefinedFnKey]
    fns.sizeHint(serializedFns.length)
    try {
      for (func <- serializedFns) {
        fns += IRFunctionRegistry.registerIR(
          ctx,
          func.name,
          func.type_parameters,
          func.value_parameter_names,
          func.value_parameter_types,
          func.return_type,
          func.rendered_body,
        )
      }

      body
    } finally
      fns.view.reverse.foreach(IRFunctionRegistry.unregisterIr)
  }
}

object HttpLikeRpc {
  object Payloads {
    case class IRTypePayload(ir: String)

    case class LoadReferencesFromDatasetPayload(path: String)

    case class FromFASTAFilePayload(
      name: String,
      fasta_file: String,
      index_file: String,
      x_contigs: Array[String],
      y_contigs: Array[String],
      mt_contigs: Array[String],
      par: Array[String],
    )

    case class ParseVCFMetadataPayload(path: String)

    case class ImportFamPayload(
      path: String,
      quant_pheno: Boolean,
      delimiter: String,
      missing: String,
    )

    case class ExecutePayload(
      ir: String,
      fns: Array[SerializedIRFunction],
      stream_codec: String,
    )
  }
}

trait HttpLikeRpc extends BackendRpc {

  trait HttpLikeRequest extends Request {

    sealed trait Route extends Product with Serializable

    object Routes {
      case class TypeOf(kind: Kind) extends Route
      case object Execute extends Route
      case object ParseVcfMetadata extends Route
      case object ImportFam extends Route
      case object LoadReferencesFromDataset extends Route
      case object LoadReferencesFromFASTA extends Route
    }

    def route(a: Env): Route
    def payload(a: Env): JValue

    import HttpLikeRpc.Payloads._

    final override def command(a: Env): Command =
      route(a) match {
        case Routes.TypeOf(k) =>
          Commands.TypeOf(k, payload(a).extract[IRTypePayload].ir)
        case Routes.Execute =>
          val ExecutePayload(ir, fns, codec) = payload(a).extract[ExecutePayload]
          Commands.Execute(ir, fns, codec)
        case Routes.ParseVcfMetadata =>
          Commands.ParseVcfMetadata(payload(a).extract[ParseVCFMetadataPayload].path)
        case Routes.ImportFam =>
          val config = payload(a).extract[ImportFamPayload]
          Commands.ImportFam(config.path, config.quant_pheno, config.delimiter, config.missing)
        case Routes.LoadReferencesFromDataset =>
          val path = payload(a).extract[LoadReferencesFromDatasetPayload].path
          Commands.LoadReferencesFromDataset(path)
        case Routes.LoadReferencesFromFASTA =>
          val config = payload(a).extract[FromFASTAFilePayload]
          Commands.LoadReferencesFromFASTA(
            config.name,
            config.fasta_file,
            config.index_file,
            config.x_contigs,
            config.y_contigs,
            config.mt_contigs,
            config.par,
          )
      }
  }
}
