package is.hail.backend

import is.hail.expr.ir.IRParser
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.io.BufferSpec
import is.hail.io.plink.LoadPlink
import is.hail.io.vcf.LoadVCF
import is.hail.linalg.RowMatrix
import is.hail.services.retryTransientErrors
import is.hail.types.virtual.{Kind, TFloat64, VType}
import is.hail.types.virtual.Kinds._
import is.hail.utils.{toRichIterable, using, ExecutionTimer}
import is.hail.utils.ExecutionTimer.Timings
import is.hail.variant.ReferenceGenome

import scala.util.control.NonFatal

import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets

import org.json4s.{DefaultFormats, Extraction, Formats, JValue}
import org.json4s.jackson.{JsonMethods, Serialization}

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
case class ImportFamPayload(path: String, quant_pheno: Boolean, delimiter: String, missing: String)

case class ExecutePayload(
  ir: String,
  fs: Array[SerializedIRFunction],
  stream_codec: String,
)

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
    case class TypeOf(k: Kind[_ <: VType], ir: String) extends Command
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

    case class ExportBlockMatrix(
      pathIn: String,
      pathOut: String,
      delimiter: String,
      header: String,
      addIndex: Boolean,
      exportType: String,
      partitionSize: Int,
      entries: String,
    ) extends Command
  }

  trait Ask[Env] {
    def command(env: Env): Command
  }

  trait Context[Env] {
    def scoped[A](env: Env)(f: ExecuteContext => A): (A, Timings)
  }

  trait Write[Env] {
    def timings(env: Env)(t: ExecutionTimer.Timings): Unit
    def result(env: Env)(r: Array[Byte]): Unit
    def error(env: Env)(t: Throwable): Unit
  }

  implicit val fmts: Formats = DefaultFormats

  import Commands._

  final def runRpc[A](env: A)(implicit Ask: Ask[A], Context: Context[A], Write: Write[A]): Unit =
    try {
      val command = Ask.command(env)
      val (result, timings) = retryTransientErrors {
        Context.scoped(env) { ctx =>
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
              withIRFunctionsReadFromInput(ctx, fns) {
                val ir = IRParser.parse_value_ir(ctx, s)
                val res = ctx.backend.execute(ctx, ir)
                res match {
                  case Left(_) => Array()
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
              LoadPlink
                .importFamJSON(ctx.fs, path, isQuantPheno, delimiter, missing)
                .getBytes(StandardCharsets.UTF_8)

            case LoadReferencesFromDataset(path) =>
              val rgs = ReferenceGenome.fromHailDataset(ctx.fs, path)
              ctx.References ++= rgs.map(rg => rg.name -> rg)
              Serialization.write(rgs.map(_.toJSON).toFastSeq).getBytes(StandardCharsets.UTF_8)

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
                ctx.References += rg.name -> rg
                rg.toJSON
              }

            case ExportBlockMatrix(pathIn, pathOut, delimiter, header, addIndex, exportType,
                  partitionSize, entries) =>
              val rm = RowMatrix.readBlockMatrix(ctx.fs, pathIn, partitionSize)
              entries match {
                case "full" =>
                  rm.export(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
                case "lower" =>
                  rm.exportLowerTriangle(
                    ctx,
                    pathOut,
                    delimiter,
                    Option(header),
                    addIndex,
                    exportType,
                  )
                case "strict_lower" =>
                  rm.exportStrictLowerTriangle(
                    ctx,
                    pathOut,
                    delimiter,
                    Option(header),
                    addIndex,
                    exportType,
                  )
                case "upper" =>
                  rm.exportUpperTriangle(
                    ctx,
                    pathOut,
                    delimiter,
                    Option(header),
                    addIndex,
                    exportType,
                  )
                case "strict_upper" =>
                  rm.exportStrictUpperTriangle(
                    ctx,
                    pathOut,
                    delimiter,
                    Option(header),
                    addIndex,
                    exportType,
                  )
              }
              Array()
          }
        }
      }

      Write.result(env)(result.asInstanceOf[Array[Byte]])
      Write.timings(env)(timings)
    } catch {
      case NonFatal(error) => Write.error(env)(error)
    }

  def jsonToBytes(v: JValue): Array[Byte] =
    JsonMethods.compact(v).getBytes(StandardCharsets.UTF_8)

  private[this] def withIRFunctionsReadFromInput[A](
    ctx: ExecuteContext,
    serializedFunctions: Array[SerializedIRFunction],
  )(
    body: => A
  ): A = {
    try {
      serializedFunctions.foreach { func =>
        IRFunctionRegistry.registerIR(
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
      IRFunctionRegistry.clearUserFunctions()
  }
}

trait HttpLikeBackendRpc[A] extends BackendRpc {

  import Commands._

  trait Routing extends Ask[A] {

    sealed trait Route extends Product with Serializable

    object Routes {
      case class TypeOf(kind: Kind[_ <: VType]) extends Route
      case object Execute extends Route
      case object ParseVcfMetadata extends Route
      case object ImportFam extends Route
      case object LoadReferencesFromDataset extends Route
      case object LoadReferencesFromFASTA extends Route
    }

    def route(a: A): Route
    def payload(a: A): JValue

    final override def command(a: A): Command =
      route(a) match {
        case Routes.TypeOf(k) =>
          TypeOf(k, payload(a).extract[IRTypePayload].ir)
        case Routes.Execute =>
          val ExecutePayload(ir, fs, codec) = payload(a).extract[ExecutePayload]
          Execute(ir, fs, codec)
        case Routes.ParseVcfMetadata =>
          ParseVcfMetadata(payload(a).extract[ParseVCFMetadataPayload].path)
        case Routes.ImportFam =>
          val config = payload(a).extract[ImportFamPayload]
          ImportFam(config.path, config.quant_pheno, config.delimiter, config.missing)
        case Routes.LoadReferencesFromDataset =>
          val path = payload(a).extract[LoadReferencesFromDatasetPayload].path
          LoadReferencesFromDataset(path)
        case Routes.LoadReferencesFromFASTA =>
          val config = payload(a).extract[FromFASTAFilePayload]
          LoadReferencesFromFASTA(
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

  implicit protected def Ask: Routing
  implicit protected def Write: Write[A]
  implicit protected def Context: Context[A]
}
