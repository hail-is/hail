package is.hail.expr.ir.analyses

import is.hail.expr.ir._
import is.hail.io.fs.FS
import is.hail.utils.{FastIndexedSeq, Logging, TreeTraversal}

import java.util.UUID
import scala.collection.mutable

case object SemanticHash extends Logging {

  object Hash {
    type Type = Int

    def apply(o: Any): Type =
      o.hashCode
  }

  implicit class MagmaHash(a: Hash.Type) {
    def <>(b: Hash.Type): Hash.Type =
      Hash(FastIndexedSeq(a, b))
  }

  def getFileHash(fs: FS)(path: String): Hash.Type = {
    val status = fs.fileStatus(path)
    Hash(status.getPath) <> Hash(status.getLen) <> Hash(status.getModificationTime)
  }

  def apply(fs: FS)(root: BaseIR): (Hash.Type, Memo[Hash.Type]) = {
    val ueIRs = mutable.HashMap.empty[String, BaseIR]
    val memo = Memo.empty[Hash.Type]
    for (ir <- TreeTraversal.postOrder(bindUEIRs(ueIRs))(root)) {
      memo.bind(ir, ir match {
        case Apply(fname, _, args, _, _) =>
          args.foldLeft(Hash(classOf[Apply]) <> Hash(fname))(_ <> memo(_))

        case ApplyComparisonOp(op, x, y) =>
          Hash(classOf[ApplyComparisonOp]) <> Hash(op.getClass) <> memo(x) <> memo(y)

        case ApplySeeded(fname, args, rngState: IR, _, _) =>
          args.foldLeft(Hash(classOf[ApplySeeded]) <> Hash(fname))(_ <> memo(_)) <> memo(rngState)

        case ApplySpecial(fname, _, args, _, _) =>
          args.foldLeft(Hash(classOf[ApplySpecial]) <> Hash(fname))(_ <> memo(_))

        case GetField(ir, name) =>
          Hash(classOf[GetField]) <> memo(ir) <> Hash(name)

        case GetTupleElement(ir, idx) =>
          Hash(classOf[GetTupleElement]) <> memo(ir) <> Hash(idx)

        case Literal(typ, value) =>
          Hash(classOf[Literal]) <> Hash(typ.toJSON(value))

        case MakeStruct(fields) =>
          fields.foldLeft(Hash(classOf[MakeStruct])) { case (result, (name, ir)) =>
            result <> Hash(name) <> memo(ir)
          }

        case MakeTuple(fields) =>
          fields.foldLeft(Hash(classOf[MakeTuple])) { case (result, (index, ir)) =>
            result <> Hash(index) <> memo(ir)
          }

        case MatrixRead(_, _, _, reader) =>
          reader.pathsUsed.map(getFileHash(fs)).foldLeft(Hash(classOf[MatrixRead]))(_ <> _)

        case MatrixWrite(child, writer) =>
          Hash(classOf[MatrixWrite]) <> memo(child) <> Hash(writer.path)

        // Notes:
        // - If the name in the (Relational)Ref is free then it's impossible to know the semantic hash
        // - The semantic hash of a (Relational)Ref cannot be the same as the value it binds to as
        //   that would imply that the evaluation of the value to is the same as evaluating the ref itself.
        case Ref(name, _) =>
          Hash(classOf[Ref]) <> memo(ueIRs(name))

        case RelationalRef(name, _) =>
          Hash(classOf[RelationalRef]) <> memo(ueIRs(name))

        case SelectFields(struct, names) =>
          Hash(classOf[SelectFields]) <> names.foldLeft(memo(struct))(_ <> Hash(_))

        case StreamZip(streams, _, body, behaviour, _) =>
          streams.foldLeft(Hash(classOf[StreamZip]))(_ <> memo(_)) <> memo(body) <> Hash(behaviour)

        case TableRead(_, _, reader) =>
          reader.pathsUsed.map(getFileHash(fs)).foldLeft(Hash(classOf[TableRead]))(_ <> _)

        case TableWrite(child, writer) =>
          Hash(classOf[TableWrite]) <> memo(child) <> Hash(writer.path)

        case TableKeyBy(child, keys, _) =>
          keys.foldLeft(Hash(classOf[TableKeyBy]) <> memo(child))(_ <> Hash(_))

        case WritePartition(partition, context, writer) =>
          Hash(classOf[WritePartition]) <> memo(partition) <> memo(context) <> Hash(writer.toJValue)

        case WriteMetadata(writeAnnotations, writer) =>
          Hash(classOf[WriteMetadata]) <> memo(writeAnnotations) <> Hash(writer.toJValue)

        // The following are parameterized entirely by the operation's input and the operation itself
        case _: ArrayLen |
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
             _: StreamFilter |
             _: StreamMap |
             _: StreamRange |
             _: TableGetGlobals |
             _: TableCollect |
             _: TableAggregate |
             _: TableCount |
             _: TableMapRows |
             _: TableMapGlobals |
             _: TableFilter |
             _: TableDistinct |
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
        case NA(typ) => Hash(classOf[NA]) <> Hash(typ)
        case Str(x) => Hash(classOf[Str]) <> Hash(x)
        case UUID4(x) => Hash(classOf[UUID4]) <> Hash(x)

        // In these cases, just return a random SemanticHash meaning that two
        // invocations will never return the same thing.
        case _ =>
          log.info(s"SemanticHash unknown: ${ir.getClass.getName}")
          Hash(UUID.randomUUID)
      })
    }
    (memo(root), memo)
  }

  // Assume all upwardly-exposed IR bindings are in SSA form
  def bindUEIRs(ueIRs: mutable.HashMap[String, BaseIR]): BaseIR => Iterator[BaseIR] = {
    case CollectDistributedArray(contexts, globals, cname, gname, body, dynamicID, _, _) =>
      assert(ueIRs.put(cname, contexts).isEmpty)
      assert(ueIRs.put(gname, globals).isEmpty)
      FastIndexedSeq(contexts, globals, body, dynamicID).iterator

    case Let(name, value, body) =>
      assert(ueIRs.put(name, value).isEmpty)
      FastIndexedSeq(value, body).iterator

    case RelationalLet(name, value, body) =>
      assert(ueIRs.put(name, value).isEmpty)
      FastIndexedSeq(value, body).iterator

    case StreamFilter(stream, name, pred) =>
      assert(ueIRs.put(name, stream).isEmpty)
      FastIndexedSeq(stream, pred).iterator

    case StreamFold(stream, zero, accumulator, value, body) =>
      assert(ueIRs.put(accumulator, zero).isEmpty)
      assert(ueIRs.put(value, stream).isEmpty)
      FastIndexedSeq(stream, zero, body).iterator

    case StreamMap(stream, name, body) =>
      assert(ueIRs.put(name, stream).isEmpty)
      FastIndexedSeq(stream, body).iterator

    case StreamZip(streams, names, body, _, _) =>
      assert(names.zip(streams).forall { case (name, stream) => ueIRs.put(name, stream).isEmpty })
      streams.iterator ++ Iterator.single(body)

    case ir =>
      ir.children.iterator
  }

}
