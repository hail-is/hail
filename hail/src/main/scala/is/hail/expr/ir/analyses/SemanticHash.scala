package is.hail.expr.ir.analyses

import is.hail.expr.ir._
import is.hail.io.fs.FS
import is.hail.utils.{FastIndexedSeq, TreeTraversal}

import java.util.UUID
import scala.collection.mutable

case object SemanticHash {

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
    val lets = mutable.HashMap.empty[String, BaseIR]
    val memo = Memo.empty[Hash.Type]
    for (ir <- TreeTraversal.postOrder(computeLets(lets))(root)) {
      memo.bind(ir, ir match {
        case TableRead(_, _, reader) =>
          reader.pathsUsed.map(getFileHash(fs)).foldLeft(Hash(classOf[TableRead]))(_ <> _)

        case TableWrite(child, writer) =>
          Hash(classOf[TableWrite]) <> memo(child) <> Hash(writer.path)

        case TableKeyBy(child, keys, _) =>
          Hash(classOf[TableKeyBy]) <> memo(child) <> keys.map(Hash(_)).reduce(_ <> _)

        case MatrixRead(_, _, _, reader) =>
          reader.pathsUsed.map(getFileHash(fs)).foldLeft(Hash(classOf[MatrixRead]))(_ <> _)

        case MatrixWrite(child, writer) =>
          Hash(classOf[MatrixWrite]) <> memo(child) <> Hash(writer.path)

        case MakeStruct(fields) =>
          fields.foldLeft(Hash(classOf[MakeStruct])) { case (result, (name, ir)) =>
            result <> Hash(name) <> memo(ir)
          }

        case MakeTuple(fields) =>
          fields.foldLeft(Hash(classOf[MakeTuple])) { case (result, (index, ir)) =>
            result <> Hash(index) <> memo(ir)
          }

        // Notes:
        // - Assume copy propagation has run at this point (no refs to refs)
        // - If the name in the (Relational)Ref is free then it's impossible to know the semantic hash
        // - The semantic hash of a (Relational)Ref cannot be the same as the value it binds to as
        //   that would imply that the evaluation of the value to is the same as evaluating the ref itself.
        case Ref(name, _) =>
          Hash(classOf[Ref]) <> lets.get(name).map(memo(_)).getOrElse(Hash(UUID.randomUUID))

        case RelationalRef(name, _) =>
          Hash(classOf[RelationalRef]) <> lets.get(name).map(memo(_)).getOrElse(Hash(UUID.randomUUID))

        // The following are parameterized entirely by the operation's input and the operation itself
        case _: ArrayLen |
             _: ArrayZeros |
             _: Begin |
             _: BlockMatrixCollect |
             _: Coalesce |
             _: ConsoleLog |
             _: Consume |
             _: If |
             _: IsNA |
             _: Let |
             _: LiftMeOut |
             _: MakeNDArray |
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
             _: TableGetGlobals |
             _: TableCollect |
             _: TableAggregate |
             _: TableCount |
             _: TableMapRows |
             _: TableMapGlobals |
             _: TableFilter |
             _: TableDistinct =>
          ir.children.map(memo(_)).foldLeft(Hash(ir.getClass))(_ <> _)

        // Discrete values
        case c@(_: Void | _: True | _: False) =>
          Hash(c.getClass)

        // integral constants
        case I32(x) => Hash(classOf[I32]) <> Hash(x)
        case I64(x) => Hash(classOf[I64]) <> Hash(x)
        case F32(x) => Hash(classOf[F32]) <> Hash(x)
        case F64(x) => Hash(classOf[F64]) <> Hash(x)
        case Str(x) => Hash(classOf[Str]) <> Hash(x)
        case UUID4(x) => Hash(classOf[UUID4]) <> Hash(x)

        // In these cases, just return a random SemanticHash meaning that two
        // invocations will never return the same thing.
        case _ =>
          Hash(UUID.randomUUID)
      })
    }
    (memo(root), memo)
  }

  // Assume all let-bindings are in SSA form
  def computeLets(lets: mutable.HashMap[String, BaseIR]): BaseIR => Iterator[BaseIR] = {
    case Let(name, value, body) =>
      assert(lets.put(name, value).isEmpty)
      FastIndexedSeq(value, body).iterator

    case RelationalLet(name, value, body) =>
      assert(lets.put(name, value).isEmpty)
      FastIndexedSeq(value, body).iterator

    case ir =>
      ir.children.iterator
  }

}
