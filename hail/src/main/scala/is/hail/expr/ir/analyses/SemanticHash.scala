package is.hail.expr.ir.analyses

import is.hail.expr.ir._
import is.hail.expr.ir.functions.{TableCalculateNewPartitions, WrappedMatrixToValueFunction}
import is.hail.expr.ir.lowering.RVDTableReader
import is.hail.io.fs.FS
import is.hail.methods.{ForceCountTable, NPartitionsTable}
import is.hail.types.virtual._
import is.hail.utils.Logging
import org.apache.commons.codec.digest.MurmurHash3

import java.util.UUID
import scala.language.implicitConversions

case object SemanticHash extends Logging {

  type Type = Int
  type NextHash = () => Type

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
  }

  object Implicits {
    implicit class MagmaHashType(a: Type) {
      def <>(b: Type): Type =
        MurmurHash3.hash32(a, b, MurmurHash3.DEFAULT_SEED)
    }
  }

  import Implicits._

  def getFileHash(fs: FS)(path: String): Type =
    Hash(fs.fileChecksum(path))

  def unique: Type =
    Hash(UUID.randomUUID.toString)

  def apply(fs: FS)(root: BaseIR): NextHash = {
    val memo = Memo.empty[SemanticHash.Type]
    val normalized = new NormalizeNames(_.toString, allowFreeVariables = true)(root)
    for (ir <- IRTraversal.postOrder(normalized)) {
      memo.bind(ir, ir match {
        case AggOrScan(a, isScan) =>
          a.children.foldLeft(Hash(a.getClass) <> Hash(isScan))(_ <> memo(_))

        case Apply(fname, _, args, _, _) =>
          args.foldLeft(Hash(classOf[Apply]) <> Hash(fname))(_ <> memo(_))

        case ApplyAggOp(init, seq, AggSignature(op, _, _)) =>
          seq.foldLeft(init.foldLeft(Hash(classOf[ApplyAggOp]) <> Hash(op.getClass))(_ <> memo(_)))(_ <> memo(_))

        case ApplyBinaryPrimOp(op, x, y) =>
          Hash(classOf[ApplyBinaryPrimOp]) <> Hash(op.getClass) <> memo(x) <> memo(y)

        case ApplyComparisonOp(op, x, y) =>
          Hash(classOf[ApplyComparisonOp]) <> Hash(op.getClass) <> memo(x) <> memo(y)

        case ApplyIR(fname, _, args, _) =>
          args.foldLeft(Hash(classOf[ApplyIR]) <> Hash(fname))(_ <> memo(_))

        case ApplySeeded(fname, args, rngState: IR, _, _) =>
          args.foldLeft(Hash(classOf[ApplySeeded]) <> Hash(fname))(_ <> memo(_)) <> memo(rngState)

        case ApplySpecial(fname, _, args, _, _) =>
          args.foldLeft(Hash(classOf[ApplySpecial]) <> Hash(fname))(_ <> memo(_))

        case ApplyUnaryPrimOp(op, x) =>
          Hash(classOf[ApplyUnaryPrimOp]) <> Hash(op.getClass) <> memo(x)

        case Cast(ir, typ) =>
          Hash(classOf[Cast]) <> memo(ir) <> Hash(SemanticTypeName(typ))

        case EncodedLiteral(_, bytes) =>
          bytes.ba.foldLeft(Hash(classOf[EncodedLiteral]))(_ <> Hash(_))

        case GetField(ir, name) =>
          Hash(classOf[GetField]) <> memo(ir) <> Hash(ir.typ.asInstanceOf[TStruct].fieldIdx(name))

        case GetTupleElement(ir, idx) =>
          Hash(classOf[GetTupleElement]) <> memo(ir) <> Hash(idx)

        case Literal(typ, value) =>
          Hash(classOf[Literal]) <> Hash(typ.toJSON(value))

        case MakeStruct(fields) =>
          fields.zipWithIndex.foldLeft(Hash(classOf[MakeStruct])) { case (result, ((_, ir), index)) =>
            result <> Hash(index) <> memo(ir)
          }

        case MakeTuple(fields) =>
          fields.foldLeft(Hash(classOf[MakeTuple])) { case (result, (index, ir)) =>
            result <> Hash(index) <> memo(ir)
          }

        case MatrixRead(_, _, _, reader) =>
          reader.pathsUsed.map(getFileHash(fs)).foldLeft(Hash(classOf[MatrixRead]))(_ <> _)

        case MatrixWrite(child, writer) =>
          Hash(classOf[MatrixWrite]) <> memo(child) <> Hash(writer.path)

        case ReadPartition(context, _, reader) =>
          Hash(classOf[ReadPartition]) <> memo(context) <> Hash(reader.toJValue)

        // Notes:
        // - If the name in the (Relational)Ref is free then it's impossible to know the semantic hash
        // - The semantic hash of a (Relational)Ref cannot be the same as the value it binds to as
        //   that would imply that the evaluation of the value to is the same as evaluating the ref itself.
        case Ref(name, _) =>
          Hash(classOf[Ref]) <> Hash(name)

        case RelationalRef(name, _) =>
          Hash(classOf[RelationalRef]) <> Hash(name)

        case SelectFields(struct, names) =>
          Hash(classOf[SelectFields]) <> names.foldLeft(memo(struct))(_ <> Hash(_))

        case StreamZip(streams, _, body, behaviour, _) =>
          streams.foldLeft(Hash(classOf[StreamZip]))(_ <> memo(_)) <> memo(body) <> Hash(behaviour)

        case TableKeyBy(child, keys, _) =>
          keys.foldLeft(Hash(classOf[TableKeyBy]) <> memo(child))(_ <> Hash(_))

        case TableJoin(left, right, joinop, key) =>
          Hash(classOf[TableJoin]) <> memo(left) <> memo(right) <> Hash(joinop) <> Hash(key)

        case TableParallelize(rowsAndGlobal, nPartitions) =>
          nPartitions.foldLeft(Hash(classOf[TableParallelize]) <> memo(rowsAndGlobal))(_ <> Hash(_))

        case TableRange(count, numPartitions) =>
          Hash(classOf[TableRange]) <> Hash(count) <> Hash(numPartitions)

        case TableRead(_, dropRows, reader)  =>
          Hash(classOf[TableRead]) <> Hash(dropRows) <> (reader match {
            case RVDTableReader(_, _, _, semhash) =>
              semhash

            case _ =>
              Hash(reader.getClass) <> reader
                .pathsUsed
                .flatMap(fs.globWithPrefix(_, "**"))
                .filter(_.isFile)
                .map(s => getFileHash(fs)(s.getPath))
                .reduce(_ <> _)
          })

        case TableToValueApply(table, op) =>
          Hash(classOf[TableToValueApply]) <> memo(table) <> (op match {
            case TableCalculateNewPartitions(nPartitions) =>
              Hash(classOf[TableCalculateNewPartitions]) <> Hash(nPartitions)

            case _: WrappedMatrixToValueFunction =>
              unique

            case _: ForceCountTable | _: NPartitionsTable =>
              Hash(op.getClass)
          })

        case TableWrite(child, writer) =>
          Hash(classOf[TableWrite]) <> memo(child) <> Hash(writer.path)

        case WritePartition(partition, context, writer) =>
          Hash(classOf[WritePartition]) <> memo(partition) <> memo(context) <> Hash(writer.toJValue)

        case WriteMetadata(writeAnnotations, writer) =>
          Hash(classOf[WriteMetadata]) <> memo(writeAnnotations) <> Hash(writer.toJValue)

        case WriteValue(value, path, _, stagingFile) =>
          stagingFile.foldLeft(Hash(classOf[WriteValue]) <> memo(value) <> memo(path))(_ <> memo(_))

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
             _: MatrixAggregate |
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
          ir.children.map(memo(_)).foldLeft(Hash(ir.getClass))(_ <> _)

        // Discrete values
        case c@(_: Void | _: True | _: False) =>
          Hash(c.getClass)

        // integral constants
        case I32(x) => Hash(classOf[I32]) <> Hash(x)
        case I64(x) => Hash(classOf[I64]) <> Hash(x)
        case F32(x) => Hash(classOf[F32]) <> Hash(x)
        case F64(x) => Hash(classOf[F64]) <> Hash(x)
        case NA(typ) => Hash(classOf[NA]) <> Hash(SemanticTypeName(typ))
        case Str(x) => Hash(classOf[Str]) <> Hash(x)

        // In these cases, just return a random SemanticHash meaning that two
        // invocations will never return the same thing.
        case _ =>
          log.info(s"SemanticHash unknown: ${ir.getClass.getName}")
          unique
      })

      log.info(s"[${memo(ir)}]: $ir")
    }

    val semhash = memo(normalized)
    log.info(s"Query semantic hash = $semhash")
    var count = 0
    () => { val h = count; count += 1; semhash <> Hash(h)}
  }
}

private object AggOrScan {
  def unapply(ir: BaseIR): Option[(IR, Boolean)] = ir match {
    case a: AggExplode => Some((a, a.isScan))
    case a: AggFilter => Some((a, a.isScan))
    case a: AggFold => Some((a, a.isScan))
    case a: AggGroupBy => Some((a, a.isScan))
    case a: AggLet => Some((a, a.isScan))
    case a: AggArrayPerElement => Some((a, a.isScan))
    case _ => None
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
