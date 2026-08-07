package is.hail.expr.ir

import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.defs._
import is.hail.types.tcoerce
import is.hail.types.virtual._

import scala.reflect.ClassTag

sealed abstract class AggEnv {
  def empty: AggEnv = this match {
    case AggEnv.Create(_) => AggEnv.Create(Seq.empty)
    case AggEnv.Bind(_) => AggEnv.NoOp
    case AggEnv.NoOp => AggEnv.NoOp
    case AggEnv.Drop => AggEnv.Drop
    case AggEnv.Promote => AggEnv.Promote
  }

  def isEmpty: Boolean = this match {
    case a: AggEnv.Modify => a.bindings.isEmpty
    case _ => true
  }
}

object AggEnv {
  case object NoOp extends AggEnv
  case object Drop extends AggEnv
  case object Promote extends AggEnv

  abstract class Modify extends AggEnv { def bindings: Seq[Int] }
  final case class Create(override val bindings: Seq[Int]) extends Modify
  final case class Bind(override val bindings: Seq[Int]) extends Modify

  def bindOrNoOp(bindings: Seq[Int]): AggEnv =
    if (bindings.nonEmpty) Bind(bindings) else NoOp
}

object Binds {
  def apply(x: IR, v: Name, i: Int): Boolean = {
    val bindings = Bindings.get[Unit](x, i)
    bindings.all.zipWithIndex.exists { case ((name, _), i) =>
      name == v && bindings.eval.contains(i)
    }
  }
}

final case class Bindings[+T](
  all: IndexedSeq[(Name, T)],
  eval: IndexedSeq[Int],
  agg: AggEnv,
  scan: AggEnv,
  relational: IndexedSeq[Int],
  dropEval: Boolean,
) {
  def map[U](f: (Name, T) => U): Bindings[U] =
    copy(all = all.map { case (n, t) => (n, f(n, t)) })

  def allEmpty: Boolean =
    eval.isEmpty && agg.isEmpty && scan.isEmpty && relational.isEmpty

  def dropBindings[U]: Bindings[U] =
    Bindings(FastSeq.empty, FastSeq.empty, agg.empty, scan.empty, FastSeq.empty, dropEval)
}

// An algebra of type-like values, used by `Bindings.get` to compute the value
// bound to each name in a child environment. A domain supplies values of type
// T for expressions and a family of relational values (`Algebra.Table` etc.)
// for the other IR kinds, reached only through `denote`. Instantiated at
// `Type` (see `Algebra.Virtual`) it yields the virtual-type environments
// consumed by TypeCheck and friends; dataflow analyses supply instances that
// denote nodes by references into their own analysis state (see Requiredness).
trait Algebra[T] {
  // the relational members of this domain's family of values
  type TableValue <: Algebra.Table[T]
  type MatrixValue <: Algebra.Matrix[T]
  type BlockMatrixValue <: Algebra.BlockMatrix[T]

  // what each kind of syntax denotes in this domain; a denotation need not be
  // derived from a node's virtual type - Requiredness denotes nodes by
  // references into its own analysis state
  def denote(ir: IR): T
  def denote(table: TableIR): TableValue
  def denote(matrix: MatrixIR): MatrixValue
  def denote(bm: BlockMatrixIR): BlockMatrixValue

  // constructors and projections on abstract values
  def elementOf(t: T): T
  def selectFields(t: T, fields: IndexedSeq[String]): T
  def firstField(t: T): T
  def array(element: T): T
  def stream(element: T): T
  def tuple(ts: IndexedSeq[T]): T

  // a fresh value of a virtual type, for synthetic bindings that are never
  // missing (e.g. element indices)
  def lift(t: Type): T

  // the meet of several defining expressions, for bindings whose value may
  // come from more than one definition (e.g. fold accumulators)
  def meet(ts: IndexedSeq[T]): T
  // weakens a binding that may bind a missing value even when its definitions
  // are present (e.g. the sides of outer joins)
  def weakened(t: T): T
  // strengthens a binding that never binds a missing value even when its
  // definitions may be missing (e.g. sort comparator arguments)
  def strengthened(t: T): T
}

object Algebra {
  trait Table[+T] {
    def global: T
    def row: T
  }

  trait Matrix[+T] {
    def global: T
    def col: T
    def row: T
    def entry: T
  }

  trait BlockMatrix[+T] {
    def element: T
  }

  implicit val Types: Algebra[Type] = TypeAlgebra
  implicit val Structural: Algebra[Unit] = StructuralDomain
}

private object TypeAlgebra extends Algebra[Type] {
  type TableValue = Algebra.Table[Type]
  type MatrixValue = Algebra.Matrix[Type]
  type BlockMatrixValue = Algebra.BlockMatrix[Type]

  override def denote(ir: IR): Type = ir.typ

  override def denote(table: TableIR): TableValue =
    new Algebra.Table[Type] {
      override def global: Type = table.typ.globalType
      override def row: Type = table.typ.rowType
    }

  override def denote(matrix: MatrixIR): MatrixValue =
    new Algebra.Matrix[Type] {
      override def global: Type = matrix.typ.globalType
      override def col: Type = matrix.typ.colType
      override def row: Type = matrix.typ.rowType
      override def entry: Type = matrix.typ.entryType
    }

  override def denote(bm: BlockMatrixIR): BlockMatrixValue =
    new Algebra.BlockMatrix[Type] {
      override def element: Type = bm.typ.elementType
    }

  override def elementOf(t: Type): Type =
    t match {
      case t: TIterable => t.elementType
      case t: TNDArray => t.elementType
    }

  override def selectFields(t: Type, fields: IndexedSeq[String]): Type =
    tcoerce[TStruct](t).typeAfterSelectNames(fields)

  override def firstField(t: Type): Type = tcoerce[TBaseStruct](t).types.head
  override def array(element: Type): Type = TArray(element)
  override def stream(element: Type): Type = TStream(element)
  override def tuple(ts: IndexedSeq[Type]): Type = TTuple(ts: _*)
  override def lift(t: Type): Type = t
  override def meet(ts: IndexedSeq[Type]): Type = ts.head
  override def weakened(t: Type): Type = t
  override def strengthened(t: Type): Type = t
}

private object StructuralDomain extends Algebra[Unit] {
  type TableValue = Algebra.Table[Unit]
  type MatrixValue = Algebra.Matrix[Unit]
  type BlockMatrixValue = Algebra.BlockMatrix[Unit]

  private val table =
    new Algebra.Table[Unit] {
      override def global: Unit = ()
      override def row: Unit = ()
    }

  private val matrix =
    new Algebra.Matrix[Unit] {
      override def global: Unit = ()
      override def col: Unit = ()
      override def row: Unit = ()
      override def entry: Unit = ()
    }

  private val blockMatrix =
    new Algebra.BlockMatrix[Unit] {
      override def element: Unit = ()
    }

  override def denote(ir: IR): Unit = ()
  override def denote(t: TableIR): TableValue = table
  override def denote(m: MatrixIR): MatrixValue = matrix
  override def denote(bm: BlockMatrixIR): BlockMatrixValue = blockMatrix

  override def elementOf(t: Unit): Unit = ()
  override def selectFields(t: Unit, fields: IndexedSeq[String]): Unit = ()
  override def firstField(t: Unit): Unit = ()
  override def array(element: Unit): Unit = ()
  override def stream(element: Unit): Unit = ()
  override def tuple(ts: IndexedSeq[Unit]): Unit = ()
  override def lift(t: Type): Unit = ()
  override def meet(ts: IndexedSeq[Unit]): Unit = ()
  override def weakened(t: Unit): Unit = ()
  override def strengthened(t: Unit): Unit = ()
}

object Bindings {
  def apply[T](
    bindings: IndexedSeq[(Name, T)] = FastSeq.empty,
    eval: IndexedSeq[Int] = FastSeq.empty,
    agg: AggEnv = AggEnv.NoOp,
    scan: AggEnv = AggEnv.NoOp,
    relational: IndexedSeq[Int] = FastSeq.empty,
    dropEval: Boolean = false,
  ): Bindings[T] =
    if (eval.isEmpty && agg.isEmpty && scan.isEmpty && relational.isEmpty)
      new Bindings(bindings, bindings.indices, agg, scan, relational, dropEval)
    else
      new Bindings(bindings, eval, agg, scan, relational, dropEval)

  val empty: Bindings[Nothing] =
    Bindings(FastSeq.empty, FastSeq.empty, AggEnv.NoOp, AggEnv.NoOp, FastSeq.empty, false)

  /** Returns the environment of the `i`th child or `ir` in an arbitrary domain, given the
    * environment of the parent node, `ir`.
    *
    * Binding values may only be projected from children preceding the bound child: later children
    * may not be typed yet when their environments are computed (see IRParser.annotateTypes).
    * Recursive definitions (loop parameters, fold accumulators) are therefore contributed by
    * analyses directly.
    */
  def get[T: ClassTag: Algebra](ir: BaseIR, i: Int): Bindings[T] =
    ir match {
      case ir: BlockMatrixIR => BlockMatrix.childEnv(ir, i)
      case ir: MatrixIR => Matrix.childEnv(ir, i)
      case ir: TableIR => Table.childEnv(ir, i)
      case ir: IR => Value.childEnv(ir, i)
    }

  // Create a `Bindings` which cannot see anything bound in the enclosing context.
  private def inFreshScope[T](
    bindings: IndexedSeq[(Name, T)] = FastSeq.empty,
    eval: IndexedSeq[Int] = FastSeq.empty,
    agg: Option[IndexedSeq[Int]] = None,
    scan: Option[IndexedSeq[Int]] = None,
    relational: IndexedSeq[Int] = FastSeq.empty,
  ): Bindings[T] = Bindings(
    bindings,
    eval,
    agg.map(AggEnv.Create(_)).getOrElse(AggEnv.Drop),
    scan.map(AggEnv.Create(_)).getOrElse(AggEnv.Drop),
    relational,
    dropEval = true,
  )

  object BlockMatrix {
    def childEnv[T](ir: BlockMatrixIR, i: Int)(implicit A: Algebra[T]): Bindings[T] =
      ir match {
        case BlockMatrixMap(child, eltName, _, _) if i == 1 =>
          Bindings.inFreshScope(FastSeq(eltName -> A.denote(child).element))

        case BlockMatrixMap2(leftChild, rightChild, lName, rName, _, _) if i == 2 =>
          Bindings.inFreshScope(FastSeq(
            lName -> A.denote(leftChild).element,
            rName -> A.denote(rightChild).element,
          ))

        case _ =>
          Bindings.inFreshScope()
      }
  }

  object Matrix {
    val globalBindings: IndexedSeq[Int] = FastSeq(0)
    val rowInRowBindings: IndexedSeq[Int] = FastSeq(0, 1)
    val colInColBindings: IndexedSeq[Int] = FastSeq(0, 1)
    val rowInEntryBindings: IndexedSeq[Int] = FastSeq(0, 2)
    val colInEntryBindings: IndexedSeq[Int] = FastSeq(0, 1)
    val entryBindings: IndexedSeq[Int] = FastSeq(0, 1, 2, 3)

    def globalBindings[T](m: Algebra.Matrix[T]): IndexedSeq[(Name, T)] =
      FastSeq(MatrixIR.globalName -> m.global)

    def colBindings[T](m: Algebra.Matrix[T]): IndexedSeq[(Name, T)] =
      FastSeq(
        MatrixIR.globalName -> m.global,
        MatrixIR.colName -> m.col,
      )

    def rowBindings[T](m: Algebra.Matrix[T]): IndexedSeq[(Name, T)] =
      FastSeq(
        MatrixIR.globalName -> m.global,
        MatrixIR.rowName -> m.row,
      )

    def entryBindings[T](m: Algebra.Matrix[T]): IndexedSeq[(Name, T)] =
      FastSeq(
        MatrixIR.globalName -> m.global,
        MatrixIR.colName -> m.col,
        MatrixIR.rowName -> m.row,
        MatrixIR.entryName -> m.entry,
      )

    def globalEnv[T](m: Algebra.Matrix[T]): Env[T] = Env.fromSeq(globalBindings(m))
    def colEnv[T](m: Algebra.Matrix[T]): Env[T] = Env.fromSeq(colBindings(m))
    def rowEnv[T](m: Algebra.Matrix[T]): Env[T] = Env.fromSeq(rowBindings(m))
    def entryEnv[T](m: Algebra.Matrix[T]): Env[T] = Env.fromSeq(entryBindings(m))

    def childEnv[T](ir: MatrixIR, i: Int)(implicit A: Algebra[T]): Bindings[T] =
      ir match {
        case MatrixMapRows(child, _) if i == 1 =>
          Bindings.inFreshScope(
            entryBindings(A.denote(child)) :+ Name("n_cols") -> A.lift(TInt32),
            eval = rowInEntryBindings :+ 4,
            agg = Some(entryBindings),
            scan = Some(rowInEntryBindings),
          )
        case MatrixFilterRows(child, _) if i == 1 =>
          Bindings.inFreshScope(rowBindings(A.denote(child)))
        case MatrixMapCols(child, _, _) if i == 1 =>
          Bindings.inFreshScope(
            entryBindings(A.denote(child)) :+ Name("n_rows") -> A.lift(TInt64),
            eval = colInEntryBindings :+ 4,
            agg = Some(entryBindings),
            scan = Some(colInEntryBindings),
          )
        case MatrixFilterCols(child, _) if i == 1 =>
          Bindings.inFreshScope(colBindings(A.denote(child)))
        case MatrixMapEntries(child, _) if i == 1 =>
          Bindings.inFreshScope(entryBindings(A.denote(child)))
        case MatrixFilterEntries(child, _) if i == 1 =>
          Bindings.inFreshScope(entryBindings(A.denote(child)))
        case MatrixMapGlobals(child, _) if i == 1 =>
          Bindings.inFreshScope(globalBindings(A.denote(child)))
        case MatrixAggregateColsByKey(child, _, _) =>
          if (i == 1)
            Bindings.inFreshScope(
              entryBindings(A.denote(child)),
              eval = rowInEntryBindings,
              agg = Some(entryBindings),
            )
          else if (i == 2)
            Bindings.inFreshScope(
              colBindings(A.denote(child)),
              eval = globalBindings,
              agg = Some(colInColBindings),
            )
          else
            Bindings.inFreshScope()
        case MatrixAggregateRowsByKey(child, _, _) =>
          if (i == 1)
            Bindings.inFreshScope(
              entryBindings(A.denote(child)),
              eval = colInEntryBindings,
              agg = Some(entryBindings),
            )
          else if (i == 2)
            Bindings.inFreshScope(
              rowBindings(A.denote(child)),
              eval = globalBindings,
              agg = Some(rowInRowBindings),
            )
          else
            Bindings.inFreshScope()
        case RelationalLetMatrixTable(name, value, _) if i == 1 =>
          Bindings.inFreshScope(FastSeq(name -> A.denote(value)), relational = FastSeq(0))
        case _ =>
          Bindings.inFreshScope()
      }
  }

  object Table {
    val globalBindings: IndexedSeq[Int] = FastSeq(0)
    val rowBindings: IndexedSeq[Int] = FastSeq(0, 1)

    def globalBindings[T](t: Algebra.Table[T]): IndexedSeq[(Name, T)] =
      FastSeq(TableIR.globalName -> t.global)

    def rowBindings[T](t: Algebra.Table[T]): IndexedSeq[(Name, T)] =
      FastSeq(
        TableIR.globalName -> t.global,
        TableIR.rowName -> t.row,
      )

    def globalEnv[T](t: Algebra.Table[T]): Env[T] = Env.fromSeq(globalBindings(t))
    def rowEnv[T](t: Algebra.Table[T]): Env[T] = Env.fromSeq(rowBindings(t))

    def childEnv[T](ir: TableIR, i: Int)(implicit A: Algebra[T]): Bindings[T] =
      ir match {
        case TableFilter(child, _) if i == 1 =>
          Bindings.inFreshScope(rowBindings(A.denote(child)))
        case TableGen(contexts, globals, cname, gname, _, _, _) if i == 2 =>
          Bindings.inFreshScope(FastSeq(
            cname -> A.elementOf(A.denote(contexts)),
            gname -> A.denote(globals),
          ))
        case TableMapGlobals(child, _) if i == 1 =>
          Bindings.inFreshScope(globalBindings(A.denote(child)))
        case TableMapRows(child, _) if i == 1 =>
          Bindings.inFreshScope(
            rowBindings(A.denote(child)),
            eval = rowBindings,
            scan = Some(rowBindings),
          )
        case TableAggregateByKey(child, _) if i == 1 =>
          Bindings.inFreshScope(
            rowBindings(A.denote(child)),
            eval = globalBindings,
            agg = Some(rowBindings),
          )
        case TableKeyByAndAggregate(child, _, _, _, _) =>
          if (i == 1)
            Bindings.inFreshScope(
              rowBindings(A.denote(child)),
              eval = globalBindings,
              agg = Some(rowBindings),
            )
          else if (i == 2)
            Bindings.inFreshScope(rowBindings(A.denote(child)))
          else Bindings.inFreshScope()
        case TableMapPartitions(child, g, p, _, _, _) if i == 1 =>
          val t = A.denote(child)
          Bindings.inFreshScope(FastSeq(
            g -> t.global,
            p -> A.stream(t.row),
          ))
        case RelationalLetTable(name, value, _) if i == 1 =>
          Bindings.inFreshScope(FastSeq(name -> A.denote(value)), relational = FastSeq(0))
        case _ =>
          Bindings.inFreshScope()
      }
  }

  object Value {
    def childEnv[T: ClassTag](ir: IR, i: Int)(implicit A: Algebra[T]): Bindings[T] =
      ir match {
        case Block(bindings, _) =>
          val types = ArraySeq.newBuilder[(Name, T)]
          types.sizeHint(i)

          val eval = ArraySeq.newBuilder[Int]
          eval.sizeHint(i) // most likely binding in eval

          val agg = ArraySeq.newBuilder[Int]
          val scan = ArraySeq.newBuilder[Int]

          for (k <- 0 until i) {
            val Binding(name, value, scope) = bindings(k)
            types += name -> A.denote(value)
            scope match {
              case Scope.EVAL =>
                eval += k
              case Scope.AGG =>
                agg += k
              case Scope.SCAN =>
                scan += k
            }
          }

          if (i == bindings.length || bindings(i).scope == Scope.EVAL)
            Bindings(
              types.result(),
              eval.result(),
              AggEnv.bindOrNoOp(agg.result()),
              AggEnv.bindOrNoOp(scan.result()),
            )
          else if (bindings(i).scope == Scope.AGG)
            Bindings(types.result(), agg.result(), AggEnv.Promote, AggEnv.bindOrNoOp(scan.result()))
          else // SCAN
            Bindings(types.result(), scan.result(), AggEnv.bindOrNoOp(agg.result()), AggEnv.Promote)

        case TailLoop(name, args, resultType, _) if i == args.length =>
          Bindings(
            args.map { case (name, ir) => name -> A.denote(ir) } :+
              name -> A.tuple(FastSeq(
                A.tuple(args.map(a => A.denote(a._2))),
                A.lift(resultType),
              ))
          )
        case StreamMap(a, name, _) if i == 1 =>
          Bindings(FastSeq(name -> A.elementOf(A.denote(a))))
        case StreamZip(as, names, _, behavior, _) if i == as.length =>
          val elts = as.map { a =>
            val elt = A.elementOf(A.denote(a))
            if (behavior == ArrayZipBehavior.ExtendNA) A.weakened(elt) else elt
          }
          Bindings(names.zip(elts))
        case StreamZipJoin(as, key, curKey, curVals, _) if i == as.length =>
          val elts = as.map(a => A.elementOf(A.denote(a)))
          Bindings(FastSeq(
            curKey -> A.meet(elts.map(A.selectFields(_, key))),
            curVals -> A.meet(elts.map(e => A.array(A.weakened(e)))),
          ))
        case StreamZipJoinProducers(contexts, ctxName, makeProducer, key, curKey, curVals, _) =>
          if (i == 1) {
            val contextType = A.elementOf(A.denote(contexts))
            Bindings(FastSeq(ctxName -> contextType))
          } else if (i == 2) {
            val eltType = A.elementOf(A.denote(makeProducer))
            Bindings(FastSeq(
              curKey -> A.selectFields(eltType, key),
              curVals -> A.array(A.weakened(eltType)),
            ))
          } else Bindings.empty
        case StreamLeftIntervalJoin(left, right, _, _, lEltName, rEltName, _) if i == 2 =>
          Bindings(FastSeq(
            lEltName -> A.elementOf(A.denote(left)),
            rEltName -> A.array(A.elementOf(A.denote(right))),
          ))
        case StreamFor(a, name, _) if i == 1 =>
          Bindings(FastSeq(name -> A.elementOf(A.denote(a))))
        case StreamFlatMap(a, name, _) if i == 1 =>
          Bindings(FastSeq(name -> A.elementOf(A.denote(a))))
        case StreamFilter(a, name, _) if i == 1 =>
          Bindings(FastSeq(name -> A.elementOf(A.denote(a))))
        case StreamTakeWhile(a, name, _) if i == 1 =>
          Bindings(FastSeq(name -> A.elementOf(A.denote(a))))
        case StreamDropWhile(a, name, _) if i == 1 =>
          Bindings(FastSeq(name -> A.elementOf(A.denote(a))))
        case StreamFold(a, zero, accumName, valueName, _) if i == 2 =>
          Bindings(FastSeq(
            accumName -> A.denote(zero),
            valueName -> A.elementOf(A.denote(a)),
          ))
        case StreamFold2(a, accum, valueName, _, _) =>
          if (i < accum.length + 1)
            Bindings.empty
          else if (i < 2 * accum.length + 1)
            Bindings(
              (valueName -> A.elementOf(A.denote(a))) +:
                accum.map { case (name, value) => (name, A.denote(value)) }
            )
          else
            Bindings(accum.map { case (name, value) => (name, A.denote(value)) })
        case StreamBufferedAggregate(stream, _, _, _, name, _, _) if i > 1 =>
          Bindings(FastSeq(name -> A.elementOf(A.denote(stream))))
        case RunAggScan(a, name, _, _, _, _) if i == 2 || i == 3 =>
          Bindings(FastSeq(name -> A.elementOf(A.denote(a))))
        case StreamAgg(a, name, _) if i == 1 =>
          Bindings(
            FastSeq(name -> A.elementOf(A.denote(a))),
            agg = AggEnv.Create(FastSeq(0)),
            scan = AggEnv.Drop,
          )
        case StreamScan(a, zero, accumName, valueName, _) if i == 2 =>
          Bindings(FastSeq(
            accumName -> A.denote(zero),
            valueName -> A.elementOf(A.denote(a)),
          ))
        case StreamAggScan(a, name, _) if i == 1 =>
          Bindings(
            FastSeq(name -> A.elementOf(A.denote(a))),
            eval = FastSeq(0),
            agg = AggEnv.Drop,
            scan = AggEnv.Create(FastSeq(0)),
          )
        case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, joinType) if i == 2 =>
          val lElt = A.elementOf(A.denote(ll))
          val rElt = A.elementOf(A.denote(rr))
          Bindings(FastSeq(
            l -> (if (joinType == "outer" || joinType == "right") A.weakened(lElt) else lElt),
            r -> (if (joinType == "outer" || joinType == "left") A.weakened(rElt) else rElt),
          ))
        case ArraySort(a, left, right, _) if i == 1 =>
          val elt = A.strengthened(A.elementOf(A.denote(a)))
          Bindings(FastSeq(left -> elt, right -> elt))
        case ArrayMaximalIndependentSet(a, Some((left, right, _))) if i == 1 =>
          val tupleType =
            A.strengthened(A.tuple(FastSeq(A.firstField(A.elementOf(A.denote(a))))))
          Bindings(FastSeq(left -> tupleType, right -> tupleType), dropEval = true)
        case AggArrayPerElement(a, elementName, indexName, _, _, isScan) =>
          if (i == 0)
            Bindings(
              agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
              scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
            )
          else if (i == 1) {
            Bindings(
              FastSeq(
                elementName -> A.elementOf(A.denote(a)),
                indexName -> A.lift(TInt32),
              ),
              eval = FastSeq(1),
              agg = if (isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0, 1)),
              scan = if (!isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0, 1)),
            )
          } else Bindings.empty
        case AggFold(zero, _, _, accumName, otherAccumName, isScan) =>
          lazy val accum = A.denote(zero)
          if (i == 0)
            Bindings(
              agg = if (isScan) AggEnv.NoOp else AggEnv.Drop,
              scan = if (!isScan) AggEnv.NoOp else AggEnv.Drop,
            )
          else if (i == 1)
            Bindings(
              FastSeq(accumName -> accum),
              agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
              scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
            )
          else
            Bindings(
              FastSeq(accumName -> accum, otherAccumName -> accum),
              agg = if (isScan) AggEnv.NoOp else AggEnv.Drop,
              scan = if (!isScan) AggEnv.NoOp else AggEnv.Drop,
              dropEval = true,
            )
        case NDArrayMap(nd, name, _) if i == 1 =>
          Bindings(FastSeq(name -> A.elementOf(A.denote(nd))))
        case NDArrayMap2(l, r, lName, rName, _, _) if i == 2 =>
          Bindings(FastSeq(
            lName -> A.elementOf(A.denote(l)),
            rName -> A.elementOf(A.denote(r)),
          ))
        case CollectDistributedArray(contexts, globals, cname, gname, _, _, _, _) if i == 2 =>
          Bindings.inFreshScope(
            FastSeq(
              cname -> A.elementOf(A.denote(contexts)),
              gname -> A.denote(globals),
            )
          )
        case TableAggregate(child, _) =>
          if (i == 1)
            Bindings.inFreshScope(
              Table.rowBindings(A.denote(child)),
              eval = Table.globalBindings,
              agg = Some(Table.rowBindings),
            )
          else Bindings(agg = AggEnv.Drop, scan = AggEnv.Drop, dropEval = true)
        case MatrixAggregate(child, _) =>
          if (i == 1)
            Bindings.inFreshScope(
              Matrix.entryBindings(A.denote(child)),
              eval = Matrix.globalBindings,
              agg = Some(Matrix.entryBindings),
            )
          else Bindings(agg = AggEnv.Drop, scan = AggEnv.Drop, dropEval = true)
        case ApplyAggOp(init, _, _) =>
          if (i < init.length) Bindings(agg = AggEnv.Drop)
          else Bindings(agg = AggEnv.Promote)
        case ApplyScanOp(init, _, _) =>
          if (i < init.length) Bindings(scan = AggEnv.Drop)
          else Bindings(scan = AggEnv.Promote)
        case AggFilter(_, _, isScan) if i == 0 =>
          Bindings(
            agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
          )
        case AggGroupBy(_, _, isScan) if i == 0 =>
          Bindings(
            agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
          )
        case AggExplode(a, name, _, isScan) =>
          if (i == 0)
            Bindings(
              agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
              scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
            )
          else
            Bindings(
              FastSeq(name -> A.elementOf(A.denote(a))),
              agg = if (isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0)),
              scan = if (!isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0)),
            )
        case RelationalLet(name, value, _) =>
          if (i == 1)
            Bindings(
              FastSeq(name -> A.denote(value)),
              agg = AggEnv.Drop,
              scan = AggEnv.Drop,
              relational = FastSeq(0),
            )
          else
            Bindings(
              agg = AggEnv.Drop,
              scan = AggEnv.Drop,
              dropEval = true,
            )
        case _ =>
          if (UsesAggEnv(ir, i))
            Bindings(agg = AggEnv.Promote)
          else if (UsesScanEnv(ir, i))
            Bindings(scan = AggEnv.Promote)
          else Bindings.empty
      }
  }
}
