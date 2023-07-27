package is.hail.expr.ir.analyses

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.{TableCalculateNewPartitions, WrappedMatrixToValueFunction}
import is.hail.expr.ir.{MatrixRangeReader, _}
import is.hail.io.fs.FS
import is.hail.io.vcf.MatrixVCFReader
import is.hail.methods._
import is.hail.types.virtual._
import is.hail.utils.{Logging, TreeTraversal}
import org.apache.commons.codec.digest.MurmurHash3
import org.apache.spark.unsafe.types.ByteArray
import sun.jvm.hotspot.runtime.Bytes

import java.nio.ByteBuffer
import scala.collection.mutable.ArrayBuffer
import scala.compat.Platform

case object SemanticHash extends Logging {
  type Type = Int
  type NullableType = Integer

  // Picked from https://softwareengineering.stackexchange.com/a/145633
  def apply(ctx: ExecuteContext)(root: BaseIR): Option[Type] =
    ctx.timer.time("SemanticHash") {
      val normalized = ctx.timer.time("NormalizeNames") {
        new NormalizeNames(_.toString, allowFreeVariables = true)(root)
      }

      val semhash = ctx.timer.time("Hash") {
        encode(ctx.fs, normalized).map { bytestream =>

          val buffSize = 256 * 4
          val buff = Array.ofDim[Byte](buffSize)

          val it = bytestream.iterator
          var hash = MurmurHash3.DEFAULT_SEED
          while (it.hasNext) {
            val k = 0
            while (it.hasNext && k < buffSize) {
             buff(k) = it.next()
            }
            hash = MurmurHash3.hash32x86(buff, 0, k, hash)
          }
          hash
        }
      }

      log.info(s"IR Semantic Hash: $semhash")
      semhash
    }

  private def encode(fs: FS, root: BaseIR): Option[Stream[Byte]] =
    Some(
      TreeTraversal
        .levelOrder((_: BaseIR, children: Iterable[BaseIR]) => children.map(c => (c, c.children) )(root)
        .foldLeft(Stream.empty[Byte]) { case (bstream, ir) =>
        (bstream += ir.getClass.getName.getBytes) += (ir match {
        case a: AggExplode =>
           a.getClass.getName.getBytes +

        case a: AggFilter =>
          Type(a.isScan)

        case a: AggFold =>
          Type(a.isScan)

        case a: AggGroupBy =>
          Type(a.isScan)

        case a: AggLet =>
          Type(a.isScan)

        case a: AggArrayPerElement =>
          Type(a.isScan)

        case Apply(fname, _, _, _, _) =>
          Type(fname)

        case ApplyAggOp(_, _, AggSignature(op, _, _)) =>
          Type(op.getClass)

        case ApplyBinaryPrimOp(op, _, _) =>
          Type(op.getClass)

        case ApplyComparisonOp(op, _, _) =>
          Type(op.getClass)

        case ApplyIR(fname, _, _, _) =>
          Type(fname)

        case ApplySeeded(fname, _, _, _, _) =>
          Type(fname)

        case ApplySpecial(fname, _, _, _, _) =>
          Type(fname)

        case ApplyUnaryPrimOp(op, _) =>
          Type(op.getClass)

        case BlockMatrixToTableApply(_, _, function) =>
          function match {
            case PCRelate(maf, blockSize, kinship, stats) =>
              Type(classOf[PCRelate]) <> Type(maf) <> Type(blockSize) <> Type(kinship) <> Type(stats)
          }

        case BlockMatrixRead(reader) =>
          Type(reader.getClass) <> (reader match {
            case _: BlockMatrixNativeReader =>
              reader.pathsUsed
                .flatMap(p => fs.glob(p + "/**").filter(_.isFile))
                .map(g => getFileHash(fs)(g.getPath))
                .reduce(_ <> _)

            case BlockMatrixBinaryReader(path, _, _) =>
              getFileHash(fs)(path)

            case _ =>
              log.warn(s"SemanticHash unknown: ${reader.getClass.getName}")
              return None
          })


        case Cast(_, typ) =>
          Type(SemanticTypeName(typ))

        case EncodedLiteral(_, bytes) =>
          bytes.ba.foldLeft(Type.init)(_ <> Type(_))

        case GetField(struct, name) =>
          Type(struct.typ.asInstanceOf[TStruct].fieldIdx(name))

        case GetTupleElement(_, idx) =>
          Type(idx)

        case Literal(typ, value) =>
          Type(typ.toJSON(value))

        case MakeTuple(fields) =>
          fields.foldLeft(Type.init)({ case (hash, (index, _)) => hash <> Type(index) })

        case MatrixRead(_, _, _, reader) =>
          Type(reader.getClass) <> (reader match {
            case MatrixRangeReader(params, nPartitionsAdj) =>
              Type(params.nRows) <> Type(params.nCols) <> params.nPartitions.foldLeft(Type(nPartitionsAdj))(_ <> Type(_))

            case _: MatrixNativeReader =>
              reader
                .pathsUsed
                .flatMap(p => fs.glob(p + "/**").filter(_.isFile))
                .map(g => getFileHash(fs)(g.getPath))
                .reduce(_ <> _)

            case _: MatrixVCFReader =>
              reader.pathsUsed.map(getFileHash(fs)).reduce(_ <> _)

            case _ =>
              log.warn(s"SemanticHash unknown: ${reader.getClass.getName}")
              return None
          })

        case MatrixWrite(_, writer) =>
          Type(writer.path)

        case NDArrayReindex(_, indices) =>
          indices.foldLeft(Type(classOf[NDArrayReindex]))(_ <> Type(_))

        case ReadPartition(_, _, reader) =>
          Type(reader.toJValue)

        case Ref(name, _) =>
          Type(name)

        case RelationalRef(name, _) =>
          Type(name)

        case SelectFields(struct, names) =>
          val getFieldIndex = struct.typ.asInstanceOf[TStruct].fieldIdx
          names.map(getFieldIndex).foldLeft(Type.init)(_ <> Type(_))

        case StreamZip(_, _, _, behaviour, _) =>
          Type(behaviour)

        case TableKeyBy(table, keys, _) =>
          val getFieldIndex = table.typ.rowType.fieldIdx
          keys.map(getFieldIndex).foldLeft(Type.init)(_ <> Type(_))

        case TableJoin(_, _, joinop, key) =>
          Type(joinop) <> Type(key)

        case TableParallelize(_, nPartitions) =>
          nPartitions.foldLeft(Type.init)(_ <> Type(_))

        case TableRange(count, numPartitions) =>
          Type(count) <> Type(numPartitions)

        case TableRead(_, dropRows, reader) =>
          Type(dropRows) <> Type(reader.getClass) <> (reader match {
            case StringTableReader(_, fileStatuses) =>
              fileStatuses.foldLeft(Type(classOf[StringTableReader])) { (h, s) =>
                h <> getFileHash(fs)(s.getPath)
              }

            case reader@(_: TableNativeReader | _: TableNativeZippedReader) =>
              reader.pathsUsed
                .flatMap(p => fs.glob(p + "/**").filter(_.isFile))
                .foldLeft(Type(reader.getClass)) { (h, g) =>
                  h <> getFileHash(fs)(g.getPath)
                }

            case _ =>
              log.warn(s"SemanticHash unknown: ${reader.getClass.getName}")
              return None
          })

        case TableToValueApply(_, op) =>
          op match {
            case TableCalculateNewPartitions(nPartitions) =>
              Type(classOf[TableCalculateNewPartitions]) <> Type(nPartitions)

            case WrappedMatrixToValueFunction(op, _, _, _) =>
              Type(classOf[WrappedMatrixToValueFunction]) <> (op match {
                case _: ForceCountMatrixTable | _: NPartitionsMatrixTable =>
                  Type(op.getClass)

                case _: MatrixExportEntriesByCol =>
                  log.warn("SemanticHash unknown: MatrixExportEntriesByCol")
                  return None
              })

            case _: ForceCountTable | _: NPartitionsTable =>
              Type(op.getClass)
          }

        case TableWrite(_, writer) =>
          Type(writer.path)

        case WritePartition(_, _, writer) =>
          Type(writer.toJValue)

        case WriteMetadata(_, writer) =>
          Type(writer.toJValue)

        case WriteValue(_, _, writer, _) =>
          Type(writer.toJValue)

        // The following are parameterized entirely by the operation's input and the operation itself
        case _: ArrayLen |
             _: ArrayRef |
             _: ArraySlice |
             _: ArraySort |
             _: ArrayZeros |
             _: Begin |
             _: BlockMatrixCollect |
             _: CastToArray |
             _: Coalesce |
             _: CollectDistributedArray |
             _: ConsoleLog |
             _: Consume |
             _: Die |
             _: GroupByKey |
             _: If |
             _: InsertFields |
             _: IsNA |
             _: Let |
             _: LiftMeOut |
             _: MakeArray |
             _: MakeNDArray |
             _: MakeStream |
             _: MakeStruct |
             _: MatrixAggregate |
             _: MatrixColsTable |
             _: MatrixCount |
             _: MatrixMapGlobals |
             _: MatrixLiteral |
             _: MatrixMapRows |
             _: MatrixFilterRows |
             _: MatrixMapCols |
             _: MatrixFilterCols |
             _: MatrixMapEntries |
             _: MatrixFilterEntries |
             _: MatrixDistinctByRow |
             _: NDArrayShape |
             _: NDArraySlice |
             _: NDArrayReshape |
             _: NDArrayWrite |
             _: RelationalLet |
             _: RNGSplit |
             _: RNGStateLiteral |
             _: StreamAgg |
             _: StreamDrop |
             _: StreamDropWhile |
             _: StreamFilter |
             _: StreamFlatMap |
             _: StreamFold |
             _: StreamFold2 |
             _: StreamFor |
             _: StreamIota |
             _: StreamLen |
             _: StreamMap |
             _: StreamRange |
             _: StreamTake |
             _: StreamTakeWhile |
             _: TableGetGlobals |
             _: TableAggregate |
             _: TableCollect |
             _: TableCount |
             _: TableDistinct |
             _: TableFilter |
             _: TableMapGlobals |
             _: TableMapRows |
             _: TableRename |
             _: ToArray |
             _: ToDict |
             _: ToSet |
             _: ToStream |
             _: Trap =>
          Type.init

        // Discrete values
        case _: Void | _: True | _: False =>
          Type.init

        // integral constants
        case I32(x) => Type(x)
        case I64(x) => Type(x)
        case F32(x) => Type(x)
        case F64(x) => Type(x)
        case NA(typ) => Type(SemanticTypeName(typ))
        case Str(x) => Type(x)

        // In these cases, just return None meaning that two
        // invocations will never return the same thing.
        case _ =>
          log.warn(s"SemanticHash unknown: ${ir.getClass.getName}")
          return None
      }
    })

  def getFileHash(fs: FS)(path: String): Type =
    fs.eTag(path) match {
      case Some(etag) => Type(etag)
      case None =>
        Type(path) <> Type(fs.fileStatus(path).getModificationTime)
    }
}

case object SemanticTypeName {
  def apply(t: Type): String = {
    val sb = StringBuilder.newBuilder

    def go(typ: Type): Unit = {
      sb.append(typ.getClass.getSimpleName)
      val children = typ.children
      if (children.nonEmpty) {
        sb.append('[')
        go(children.head)

        children.tail.foreach { t =>
          sb.append(',')
          go(t)
        }

        sb.append(']')
      }
    }

    go(t)

    sb.toString()
  }
}