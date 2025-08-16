package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.defs._
import is.hail.expr.ir.streams.StreamUtils
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.StackSafe._

import scala.reflect.ClassTag

object TypeCheck {
  def apply(ctx: ExecuteContext, ir: BaseIR): Unit =
    apply(ctx, ir, BindingEnv.empty)

  def apply(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): Unit =
    ctx.time {
      try
        check(ctx, ir, env).run()
      catch {
        case e: Throwable =>
          fatal(
            s"Error while typechecking IR:\n${Pretty(ctx, ir, preserveNames = true, allowUnboundRefs = true)}",
            e,
          )
      }
    }

  def check(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): StackFrame[Unit] = {
    for {
      _ <- ir.forEachChildWithEnvStackSafe(env) { (child, i, childEnv) =>
        for {
          _ <- call(check(ctx, child, childEnv))
        } yield
          if (child.typ == TVoid) {
            checkVoidTypedChild(ctx, ir, i, env)
          } else ()
      }
    } yield checkSingleNode(ctx, ir, env)
  }

  private def checkVoidTypedChild(ctx: ExecuteContext, ir: BaseIR, i: Int, env: BindingEnv[Type])
    : Unit = ir match {
    case l: Block if i == l.bindings.length || l.body.typ == TVoid =>
    case _: StreamFor if i == 1 =>
    case _: RunAggScan if (i == 1 || i == 2) =>
    case _: StreamBufferedAggregate if (i == 1 || i == 3) =>
    case _: RunAgg if i == 0 =>
    case _: SeqOp => // let seqop checking below catch bad void arguments
    case _: InitOp => // let initop checking below catch bad void arguments
    case _: If if i != 0 =>
    case _: RelationalLet if i == 1 =>
    case _: WriteMetadata =>
    case _ =>
      throw new RuntimeException(
        s"unexpected void-typed IR at child $i of ${ir.getClass.getSimpleName}" +
          s"\n  IR: ${Pretty(ctx, ir)}"
      )
  }

  private def checkSingleNode(ctx: ExecuteContext, ir: BaseIR, env: BindingEnv[Type]): Unit = {
    ir match {
      case Ref(name, t) =>
        env.eval.lookupOption(name) match {
          case Some(expected) =>
            assert(
              t == expected,
              s"type mismatch:\n  name: $name\n  actual: ${t.parsableString()}\n  expect: ${expected.parsableString()}",
            )
          case None =>
            throw new NoSuchElementException(
              s"Ref with name $name could not be resolved in env $env"
            )
        }

      case RelationalRef(name, t) =>
        env.relational.lookupOption(name) match {
          case Some(t2) =>
            if (t != t2)
              throw new RuntimeException(s"RelationalRef type mismatch:\n  node=$t\n   env=$t2")
          case None =>
            throw new RuntimeException(s"RelationalRef not found in env: $name")
        }
      case TailLoop(name, _, _, body) =>
        def recurInTail(node: IR, tailPosition: Boolean): Boolean = node match {
          case x: Recur =>
            x.name != name || tailPosition
          case _ =>
            node.children.zipWithIndex
              .forall {
                case (c: IR, i) => recurInTail(c, tailPosition && InTailPosition(node, i))
                case _ => true
              }
        }
        assert(recurInTail(body, tailPosition = true))
      case Recur(name, args, typ) =>
        val TTuple(IndexedSeq(TupleField(_, argTypes), TupleField(_, rt))) = env.eval.lookup(name)
        assert(argTypes.asInstanceOf[TTuple].types.zip(args).forall { case (t, ir) => t == ir.typ })
        assert(typ == rt)

      case TableAggregateByKey(child, _) =>
        assert(child.typ.key.nonEmpty)
      case TableExplode(child, path) =>
        assert(!child.typ.key.contains(path.head))
      case TableGen(contexts, globals, _, _, body, partitioner, _) =>
        coerce[TStream]("contexts", contexts.typ)
        coerce[TStruct]("globals", globals.typ)
        val bodyType = coerce[TStream]("body", body.typ)
        val rowType = coerce[TStruct]("body.elementType", bodyType.elementType)

        if (!partitioner.kType.isSubsetOf(rowType))
          throw new IllegalArgumentException(
            s"""'partitioner': key type contains fields absent from row type
               |  Key type: ${partitioner.kType}
               |  Row type: $rowType""".stripMargin
          )
      case TableJoin(left, right, _, joinKey) =>
        assert(left.typ.key.length >= joinKey)
        assert(right.typ.key.length >= joinKey)
        assert(
          left.typ.keyType.truncate(joinKey) isJoinableWith right.typ.keyType.truncate(joinKey)
        )
        assert(
          left.typ.globalType.fieldNames.toSet
            .intersect(right.typ.globalType.fieldNames.toSet)
            .isEmpty
        )
      case TableKeyBy(child, keys, _) =>
        val fields = child.typ.rowType.fieldNames.toSet
        assert(
          keys.forall(fields.contains),
          s"${keys.filter(k => !fields.contains(k)).mkString(", ")}",
        )
      case TableKeyByAndAggregate(_, expr, newKey, _, _) =>
        assert(expr.typ.isInstanceOf[TStruct])
        assert(newKey.typ.isInstanceOf[TStruct])
      case TableLeftJoinRightDistinct(left, right, _) =>
        assert(
          right.typ.keyType isPrefixOf left.typ.keyType,
          s"\n  L: ${left.typ}\n  R: ${right.typ}",
        )
      case TableMapPartitions(child, _, partitionStreamName, body, requestedKey, allowedOverlap) =>
        assert(allowedOverlap >= -1)
        assert(allowedOverlap <= child.typ.key.size)
        assert(requestedKey >= 0)
        assert(requestedKey <= child.typ.key.size)
        assert(
          StreamUtils.isIterationLinear(body, partitionStreamName),
          "must iterate over the partition exactly once",
        )
        val newRowType = tcoerce[TStream](body.typ).elementType.asInstanceOf[TStruct]
        child.typ.key.foreach { k =>
          if (!newRowType.hasField(k))
            throw new RuntimeException(s"prev key: ${child.typ.key}, new row: $newRowType")
        }
      case TableMapRows(child, newRow) =>
        val newFieldSet = newRow.typ.asInstanceOf[TStruct].fieldNames.toSet
        assert(child.typ.key.forall(newFieldSet.contains))
      case TableMultiWayZipJoin(childrenSeq, _, _) =>
        val first = childrenSeq.head
        val rest = childrenSeq.tail
        assert(
          rest.forall(e => e.typ.rowType == first.typ.rowType),
          "all rows must have the same type",
        )
        assert(rest.forall(e => e.typ.key == first.typ.key), "all keys must be the same")
        assert(
          rest.forall(e => e.typ.globalType == first.typ.globalType),
          "all globals must have the same type",
        )
      case TableParallelize(rowsAndGlobal, nPartitions) =>
        assert(tcoerce[TStruct](rowsAndGlobal.typ).fieldNames.sameElements(Array(
          "rows",
          "global",
        )))
        assert(nPartitions.forall(_ > 0))
      case TableRename(child, rowMap, globalMap) =>
        assert(rowMap.keys.forall(child.typ.rowType.hasField))
        assert(globalMap.keys.forall(child.typ.globalType.hasField))
      case TableUnion(childrenSeq) =>
        assert(childrenSeq.tail.forall(_.typ.rowType == childrenSeq(0).typ.rowType))
        assert(childrenSeq.tail.forall(_.typ.key == childrenSeq(0).typ.key))
      case CastTableToMatrix(child, entriesFieldName, _, _) =>
        child.typ.rowType.fieldType(entriesFieldName) match {
          case TArray(TStruct(_)) =>
          case t => fatal(s"expected entry field to be an array of structs, found $t")
        }
      case MatrixAggregateColsByKey(child, _, _) =>
        assert(child.typ.colKey.nonEmpty)
      case MatrixAggregateRowsByKey(child, _, _) =>
        assert(child.typ.rowKey.nonEmpty)
      case MatrixAnnotateColsTable(child, _, root) =>
        assert(child.typ.colType.selfField(root).isEmpty)
      case MatrixAnnotateRowsTable(child, table, _, product) =>
        assert(
          (!product && table.typ.keyType.isPrefixOf(child.typ.rowKeyStruct)) ||
            (table.typ.keyType.size == 1 && table.typ.keyType.types(0) == TInterval(
              child.typ.rowKeyStruct.types(0)
            )),
          s"\n  L: ${child.typ}\n  R: ${table.typ}",
        )
      case MatrixKeyRowsBy(child, keys, _) =>
        val fields = child.typ.rowType.fieldNames.toSet
        assert(
          keys.forall(fields.contains),
          s"${keys.filter(k => !fields.contains(k)).mkString(", ")}",
        )
      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        assert(globalMap.keys.forall(child.typ.globalType.hasField))
        assert(colMap.keys.forall(child.typ.colType.hasField))
        assert(rowMap.keys.forall(child.typ.rowType.hasField))
        assert(entryMap.keys.forall(child.typ.entryType.hasField))
      case MatrixUnionCols(left, right, _) =>
        assert(
          left.typ.rowKeyStruct == right.typ.rowKeyStruct,
          s"${left.typ.rowKeyStruct} != ${right.typ.rowKeyStruct}",
        )
        assert(
          left.typ.colType == right.typ.colType,
          s"${left.typ.colType} != ${right.typ.colType}",
        )
        assert(
          left.typ.entryType == right.typ.entryType,
          s"${left.typ.entryType} != ${right.typ.entryType}",
        )
      case MatrixUnionRows(children) =>
        def compatible(t1: MatrixType, t2: MatrixType): Boolean =
          t1.colKeyStruct == t2.colKeyStruct &&
            t1.rowType == t2.rowType &&
            t1.rowKey == t2.rowKey &&
            t1.entryType == t2.entryType
        assert(
          children.tail.forall(c => compatible(c.typ, children.head.typ)),
          children.map(_.typ),
        )
      case BlockMatrixBroadcast(child, inIndexExpr, shape, _) =>
        val (nRows, nCols) = BlockMatrixIR.tensorShapeToMatrixShape(child)
        val childMatrixShape = IndexedSeq(nRows, nCols)

        assert(inIndexExpr.zipWithIndex.forall { case (out: Int, in: Int) =>
          !child.typ.shape.contains(in) || childMatrixShape(in) == shape(out)
        })
      case BlockMatrixMap(child, _, _, needsDense) =>
        assert(!(needsDense && child.typ.isSparse))
      case BlockMatrixMap2(left, right, _, _, _, _) =>
        assert(left.typ.nRows == right.typ.nRows)
        assert(left.typ.nCols == right.typ.nCols)
        assert(left.typ.blockSize == right.typ.blockSize)
      case ValueToBlockMatrix(child, _, _) =>
        assert(
          child.typ.isInstanceOf[TArray] || child.typ.isInstanceOf[TNDArray] || child.typ == TFloat64
        )
      case _ =>
    }
  }

  def coerce[A <: Type](argname: String, typ: Type)(implicit tag: ClassTag[A]): A =
    if (tag.runtimeClass.isInstance(typ)) typ.asInstanceOf[A]
    else throw new IllegalArgumentException(
      s"""'$argname': Type mismatch.
         |  Expected: ${tag.runtimeClass.getName}
         |    Actual: ${typ.getClass.getName}""".stripMargin
    )

}
