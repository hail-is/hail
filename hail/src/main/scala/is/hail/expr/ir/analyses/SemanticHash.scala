package is.hail.expr.ir.analyses

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.{TableCalculateNewPartitions, WrappedMatrixToValueFunction}
import is.hail.expr.ir.{MatrixRangeReader, _}
import is.hail.io.fs.FS
import is.hail.io.vcf.MatrixVCFReader
import is.hail.methods._
import is.hail.types.virtual._
import is.hail.utils.Logging
import org.apache.commons.codec.digest.MurmurHash3

case object SemanticHash extends Logging {

  type Type = Int
  type NullableType = Integer

  // Picked from https://softwareengineering.stackexchange.com/a/145633
  object Hash {
    def apply(c: Class[_]): Type =
      apply(c.getName)

    def apply(a: Any): Type =
      apply(a.toString)

    def apply(s: String): Type =
      apply(s.getBytes)

    def apply(bytes: Array[Byte]): Type =
      MurmurHash3.hash32x86(bytes)

    val init: Type =
      0

    @inline def combine(a: SemanticHash.Type, b: SemanticHash.Type): SemanticHash.Type =
      MurmurHash3.hash32(a, b, MurmurHash3.DEFAULT_SEED)
  }

  implicit class MagmaInstanceForSemanticHash(val a: SemanticHash.Type) extends AnyVal {
    @inline def <>(b: Type): Type = Hash.combine(a, b)
  }

  def apply(ctx: ExecuteContext)(root: BaseIR): Option[Type] =
    ctx.timer.time("SemanticHash") {
      val normalized = ctx.timer.time("NormalizeNames") {
        new NormalizeNames(_.toString, allowFreeVariables = true)(root)
      }

      val semhash = ctx.timer.time("Hash") {
        go(ctx.fs, normalized)
      }

      log.info(s"IR Semantic Hash: $semhash")
      semhash
    }

  private def go(fs: FS, root: BaseIR): Option[Type] =
    Some(IRTraversal.levelOrder(root).foldLeft(Hash.init) { (semhash, ir) =>
      semhash <> Hash(ir.getClass) <> (ir match {
        case a: AggExplode =>
          Hash(a.isScan)

        case a: AggFilter =>
          Hash(a.isScan)

        case a: AggFold =>
          Hash(a.isScan)

        case a: AggGroupBy =>
          Hash(a.isScan)

        case a: AggLet =>
          Hash(a.isScan)

        case a: AggArrayPerElement =>
          Hash(a.isScan)

        case Apply(fname, _, _, _, _) =>
          Hash(fname)

        case ApplyAggOp(_, _, AggSignature(op, _, _)) =>
          Hash(op.getClass)

        case ApplyBinaryPrimOp(op, _, _) =>
          Hash(op.getClass)

        case ApplyComparisonOp(op, _, _) =>
          Hash(op.getClass)

        case ApplyIR(fname, _, _, _) =>
          Hash(fname)

        case ApplySeeded(fname, _, _, _, _) =>
          Hash(fname)

        case ApplySpecial(fname, _, _, _, _) =>
          Hash(fname)

        case ApplyUnaryPrimOp(op, _) =>
          Hash(op.getClass)

        case BlockMatrixToTableApply(_, _, function) =>
          function match {
            case PCRelate(maf, blockSize, kinship, stats) =>
              Hash(classOf[PCRelate]) <> Hash(maf) <> Hash(blockSize) <> Hash(kinship) <> Hash(stats)
          }

        case BlockMatrixRead(reader) =>
          Hash(reader.getClass) <> (reader match {
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
          Hash(SemanticTypeName(typ))

        case EncodedLiteral(_, bytes) =>
          bytes.ba.foldLeft(Hash.init)(_ <> Hash(_))

        case GetField(struct, name) =>
          Hash(struct.typ.asInstanceOf[TStruct].fieldIdx(name))

        case GetTupleElement(_, idx) =>
          Hash(idx)

        case Literal(typ, value) =>
          Hash(typ.toJSON(value))

        case MakeTuple(fields) =>
          fields.foldLeft(Hash.init)({ case (hash, (index, _)) => hash <> Hash(index) })

        case MatrixRead(_, _, _, reader) =>
          Hash(reader.getClass) <> (reader match {
            case MatrixRangeReader(params, nPartitionsAdj) =>
              Hash(params.nRows) <> Hash(params.nCols) <> params.nPartitions.foldLeft(Hash(nPartitionsAdj))(_ <> Hash(_))

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
          Hash(writer.path)

        case ReadPartition(_, _, reader) =>
          Hash(reader.toJValue)

        case Ref(name, _) =>
          Hash(name)

        case RelationalRef(name, _) =>
          Hash(name)

        case SelectFields(struct, names) =>
          val getFieldIndex = struct.typ.asInstanceOf[TStruct].fieldIdx
          names.map(getFieldIndex).foldLeft(Hash.init)(_ <> Hash(_))

        case StreamZip(_, _, _, behaviour, _) =>
          Hash(behaviour)

        case TableKeyBy(table, keys, _) =>
          val getFieldIndex = table.typ.rowType.fieldIdx
          keys.map(getFieldIndex).foldLeft(Hash.init)(_ <> Hash(_))

        case TableJoin(_, _, joinop, key) =>
          Hash(joinop) <> Hash(key)

        case TableParallelize(_, nPartitions) =>
          nPartitions.foldLeft(Hash.init)(_ <> Hash(_))

        case TableRange(count, numPartitions) =>
          Hash(count) <> Hash(numPartitions)

        case TableRead(_, dropRows, reader) =>
          Hash(dropRows) <> Hash(reader.getClass) <> (reader match {
            case StringTableReader(_, fileStatuses) =>
              fileStatuses.foldLeft(Hash(classOf[StringTableReader])) { (h, s) =>
                h <> getFileHash(fs)(s.getPath)
              }

            case reader@(_: TableNativeReader | _: TableNativeZippedReader) =>
              reader.pathsUsed
                .flatMap(p => fs.glob(p + "/**").filter(_.isFile))
                .foldLeft(Hash(reader.getClass)) { (h, g) =>
                  h <> getFileHash(fs)(g.getPath)
                }

            case _ =>
              log.warn(s"SemanticHash unknown: ${reader.getClass.getName}")
              return None
          })

        case TableToValueApply(_, op) =>
          op match {
            case TableCalculateNewPartitions(nPartitions) =>
              Hash(classOf[TableCalculateNewPartitions]) <> Hash(nPartitions)

            case WrappedMatrixToValueFunction(op, _, _, _) =>
              Hash(classOf[WrappedMatrixToValueFunction]) <> (op match {
                case _: ForceCountMatrixTable | _: NPartitionsMatrixTable =>
                  Hash(op.getClass)

                case _: MatrixExportEntriesByCol =>
                  log.warn("SemanticHash unknown: MatrixExportEntriesByCol")
                  return None
              })

            case _: ForceCountTable | _: NPartitionsTable =>
              Hash(op.getClass)
          }

        case TableWrite(_, writer) =>
          Hash(writer.path)

        case WritePartition(_, _, writer) =>
          Hash(writer.toJValue)

        case WriteMetadata(_, writer) =>
          Hash(writer.toJValue)

        case WriteValue(_, _, writer, _) =>
          Hash(writer.toJValue)

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
          Hash.init

        // Discrete values
        case _: Void | _: True | _: False =>
          Hash.init

        // integral constants
        case I32(x) => Hash(x)
        case I64(x) => Hash(x)
        case F32(x) => Hash(x)
        case F64(x) => Hash(x)
        case NA(typ) => Hash(SemanticTypeName(typ))
        case Str(x) => Hash(x)

        // In these cases, just return None meaning that two
        // invocations will never return the same thing.
        case _ =>
          log.warn(s"SemanticHash unknown: ${ir.getClass.getName}")
          return None
      })
    })

  def getFileHash(fs: FS)(path: String): Type =
    Hash(fs.fileChecksum(path))

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