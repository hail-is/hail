package is.hail.scalacheck

import is.hail.annotations.RowSeq
import is.hail.expr.ir._
import is.hail.expr.ir.Scopes.ScopesSyntax
import is.hail.expr.ir.defs._
import is.hail.types.virtual._

import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag

import org.scalacheck.Gen

object ArbitraryIR {

  private val snd: (Name, Type) => Type = (_, t) => t

  private val numeric = ArraySeq(TInt32, TInt64, TFloat64)
  private val primitive = numeric ++ ArraySeq(TBoolean, TString)

  def genType(depth: Int = 2): Gen[Type] =
    if (depth == 0) Gen.oneOf(primitive)
    else
      Gen.frequency(
        4 -> Gen.oneOf(primitive),
        1 -> genType(depth - 1).map(TArray(_)),
        1 -> genTypes(depth - 1).map(ts =>
          TStruct(ts.zipWithIndex.map { case (t, i) => s"f$i" -> t }: _*)
        ),
        1 -> genTypes(depth - 1).map(TTuple(_: _*)),
      )

  private def genTypes(depth: Int): Gen[IndexedSeq[Type]] =
    Gen.choose(1, 3).flatMap(n => sequence(ArraySeq.fill(n)(genType(depth))))

  def genValue(t: Type): Gen[Any] =
    t match {
      case TInt32 => Gen.choose(-100, 100)
      case TInt64 => Gen.choose(-100L, 100L)
      case TFloat64 => Gen.choose(-100.0, 100.0)
      case TBoolean => Gen.oneOf(true, false)
      case TString => Gen.alphaNumStr.map(_.take(8))
      case TArray(e) => Gen.choose(0, 4).flatMap(n => sequence(ArraySeq.fill(n)(genValue(e))))
      case t: TBaseStruct => sequence(t.types.map(genValue)).map(RowSeq.fromSeq)
    }

  def genIR(t: Type, env: BindingEnv[Type], size: Int): Gen[IR] =
    pick(if (size <= 1) leaves(t, env) else nodes(t, env, size - 1))

  private type Rule = (Int, () => Gen[IR])

  // Builds only the chosen rule's Gen; building every rule's would recurse into all of them.
  private def pick[A](rules: Seq[(Int, () => Gen[A])]): Gen[A] =
    Gen.frequency(rules.map { case (w, r) => w -> Gen.const(r) }: _*).flatMap(_())

  private def leaves(t: Type, env: BindingEnv[Type]): Seq[Rule] = {
    val refs = env.eval.m.collect { case (n, u) if u == t => n }.toIndexedSeq
    val fieldRefs =
      env.eval.m.to(ArraySeq).flatMap {
        case (n, u: TStruct) => u.fields.filter(_.typ == t).map(f => GetField(Ref(n, u), f.name))
        case _ => ArraySeq()
      }
    val const =
      t match {
        case TInt32 => Gen.choose(-100, 100).map(I32)
        case TInt64 => Gen.choose(-100L, 100L).map(I64)
        case TFloat64 => Gen.choose(-100.0, 100.0).map(F64)
        case TBoolean => Gen.oneOf(True(), False())
        case TString => Gen.alphaNumStr.map(s => Str(s.take(8)))
        case TStream(TInt32) => Gen.choose(0, 10).map(n => StreamRange(I32(0), I32(n), I32(1)))
        case TStream(e) => genValue(TArray(e)).map(v => ToStream(Literal(TArray(e), v)))
        case _ => genValue(t).map(Literal(t, _))
      }

    ArraySeq.untagged.concat(
      Some(4 -> (() => const)),
      Option.when(refs.nonEmpty)(4 -> (() => Gen.oneOf(refs).map(Ref(_, t)))),
      Option.when(fieldRefs.nonEmpty)(4 -> (() => Gen.oneOf(fieldRefs))),
      Option.when(!t.isInstanceOf[TStream])(1 -> (() => Gen.const(NA(t)))),
    )
  }

  private def nodes(t: Type, env: BindingEnv[Type], budget: Int): Seq[Rule] =
    t match {
      case TStream(e) => streams(e, env, budget)
      case _ => any(t, env, budget) ++ byType(t, env, budget) ++ relational(t, budget)
    }

  private def any(t: Type, env: BindingEnv[Type], budget: Int): Seq[Rule] =
    ArraySeq(
      2 -> (() =>
        for {
          cond <- genIR(TBoolean, env, budget / 3)
          cnsq <- genIR(t, env, budget / 3)
          altr <- genIR(t, env, budget / 3)
        } yield If(cond, cnsq, altr)
      ),
      3 -> (() =>
        for {
          k <- Gen.choose(1, 3)
          bindings <- genBindings(env, k, budget / (k + 1), ArraySeq())
          body <- genIR(t, ChildEnv.block(env, k, bindings)(snd), budget / (k + 1))
        } yield Block(bindings, body)
      ),
      1 -> (() =>
        for {
          others <- genTypes(1)
          i <- Gen.choose(0, others.length)
          ts = others.patch(i, ArraySeq(t), 0)
          o <- genIR(TStruct(ts.zipWithIndex.map { case (t, j) => s"f$j" -> t }: _*), env, budget)
        } yield GetField(o, s"f$i")
      ),
      1 -> (() =>
        for {
          others <- genTypes(1)
          i <- Gen.choose(0, others.length)
          o <- genIR(TTuple(others.patch(i, ArraySeq(t), 0): _*), env, budget)
        } yield GetTupleElement(o, i)
      ),
      1 -> (() =>
        for {
          a <- genIR(TArray(t), env, budget * 2 / 3)
          i <- genIR(TInt32, env, budget / 3)
        } yield ArrayRef(a, i)
      ),
      1 -> (() =>
        for {
          e <- genType(1)
          accumName <- genName
          valueName <- genName
          a <- genIR(TStream(e), env, budget / 3)
          zero <- genIR(t, env, budget / 3)
          bodyEnv = ChildEnv.streamFold(env, 2, accumName, valueName, ArraySeq(a, zero))(snd)
          body <- genIR(t, bodyEnv, budget / 3)
        } yield StreamFold(a, zero, accumName, valueName, body)
      ),
      1 -> (() =>
        for {
          e <- genType(1)
          name <- genName
          a <- genIR(TStream(e), env, budget / 2)
          queryEnv = ChildEnv.streamAgg(env, 1, name, ArraySeq(a))(snd)
          query <- genIR(t, queryEnv, budget / 2)
        } yield StreamAgg(a, name, query)
      ),
    )

  private def relational(t: Type, budget: Int): Seq[Rule] =
    ArraySeq.untagged.concat(
      Some(1 -> (() =>
        for {
          child <- genTable(budget / 2)
          queryEnv = ChildEnv.tableAggregate(BindingEnv.empty[Type], 1, ArraySeq(child))(snd)
          query <- genIR(t, queryEnv, budget / 2)
        } yield TableAggregate(child, query)
      )),
      Some(1 -> (() =>
        for {
          child <- genMatrix(budget / 2)
          queryEnv = ChildEnv.matrixAggregate(BindingEnv.empty[Type], 1, ArraySeq(child))(snd)
          query <- genIR(t, queryEnv, budget / 2)
        } yield MatrixAggregate(child, query)
      )),
      Option.when(t == TInt64)(1 -> (() => genTable(budget).map(TableCount))),
    )

  // New rows and globals insert fields into the old ones, so key fields are kept.
  def genTable(size: Int): Gen[TableIR] = {
    val budget = size - 1
    def mapEnv(child: TableIR): BindingEnv[Type] =
      ChildEnv.tableMapRows(BindingEnv.empty[Type], 1, ArraySeq(child))(snd)
    if (size <= 1)
      for {
        n <- Gen.choose(0, 100)
        nPartitions <- Gen.choose(1, 8)
      } yield TableRange(n, nPartitions)
    else
      pick(ArraySeq(
        4 -> (() =>
          for {
            child <- genTable(budget / 2)
            newRow <- insertFields(TableIR.rowName, mapEnv(child), budget / 2)
          } yield TableMapRows(child, newRow)
        ),
        2 -> (() =>
          for {
            child <- genTable(budget / 2)
            env = ChildEnv.tableMapGlobals(BindingEnv.empty[Type], 1, ArraySeq(child))(snd)
            newGlobals <- insertFields(TableIR.globalName, env, budget / 2)
          } yield TableMapGlobals(child, newGlobals)
        ),
        2 -> (() =>
          for {
            child <- genTable(budget / 2)
            env = ChildEnv.tableFilter(BindingEnv.empty[Type], 1, ArraySeq(child))(snd)
            pred <- genIR(TBoolean, env, budget / 2)
          } yield TableFilter(child, pred)
        ),
        1 -> (() =>
          for {
            child <- genTable(budget)
            key <- Gen.oneOf(child.typ.rowType.fields.filter(f => primitive.contains(f.typ)))
          } yield TableKeyBy(child, ArraySeq(key.name))
        ),
        1 -> (() =>
          Gen.zip(genTable(budget), Gen.choose(0L, 50L)).map { case (c, n) => TableHead(c, n) }
        ),
        1 -> (() => genMatrix(budget).map(MatrixRowsTable)),
        1 -> (() => genMatrix(budget).map(MatrixColsTable)),
      ))
  }

  def genMatrix(size: Int): Gen[MatrixIR] = {
    val budget = size - 1
    if (size <= 1)
      for {
        n <- Gen.choose(0, 100)
        nPartitions <- Gen.choose(1, 8)
        nCols <- Gen.choose(0, 5)
      } yield {
        val cols = Literal(TArray(TStruct("s" -> TInt32)), (0 until nCols).map(RowSeq(_)))
        val entries =
          Literal(TArray(TStruct("x" -> TFloat64)), (0 until nCols).map(i => RowSeq(i + 0.5)))
        val range = TableRange(n, nPartitions)
        val withCols = TableMapGlobals(range, MakeStruct(ArraySeq("cols" -> cols)))
        val row = Ref(TableIR.rowName, range.typ.rowType)
        val withEntries = TableMapRows(withCols, InsertFields(row, ArraySeq("entries" -> entries)))
        CastTableToMatrix(withEntries, "entries", "cols", ArraySeq("s"))
      }
    else {
      def map(
        name: Name,
        childEnv: (BindingEnv[Type], Int, IndexedSeq[BaseIR]) => BindingEnv[Type],
      )(
        node: (MatrixIR, IR) => MatrixIR
      ): () => Gen[MatrixIR] = () =>
        for {
          child <- genMatrix(budget / 2)
          ir <- insertFields(name, childEnv(BindingEnv.empty[Type], 1, ArraySeq(child)), budget / 2)
        } yield node(child, ir)
      def filter(
        childEnv: (BindingEnv[Type], Int, IndexedSeq[BaseIR]) => BindingEnv[Type]
      )(
        node: (MatrixIR, IR) => MatrixIR
      ): () => Gen[MatrixIR] = () =>
        for {
          child <- genMatrix(budget / 2)
          pred <- genIR(TBoolean, childEnv(BindingEnv.empty[Type], 1, ArraySeq(child)), budget / 2)
        } yield node(child, pred)
      pick(ArraySeq(
        3 -> map(MatrixIR.rowName, ChildEnv.matrixMapRows(_, _, _)(snd))(MatrixMapRows),
        2 -> map(MatrixIR.colName, ChildEnv.matrixMapCols(_, _, _)(snd))(MatrixMapCols(_, _, None)),
        3 -> map(MatrixIR.entryName, ChildEnv.matrixMapEntries(_, _, _)(snd))(MatrixMapEntries),
        1 -> map(MatrixIR.globalName, ChildEnv.matrixMapGlobals(_, _, _)(snd))(MatrixMapGlobals),
        2 -> filter(ChildEnv.matrixFilterRows(_, _, _)(snd))(MatrixFilterRows),
        1 -> filter(ChildEnv.matrixFilterCols(_, _, _)(snd))(MatrixFilterCols),
        1 -> filter(ChildEnv.matrixFilterEntries(_, _, _)(snd))(MatrixFilterEntries),
      ))
    }
  }

  private def insertFields(name: Name, env: BindingEnv[Type], budget: Int): Gen[IR] = {
    val old = Ref(name, env.eval(name))
    val oldFields = old.typ.asInstanceOf[TStruct].fieldNames.toSet
    val fresh = Iterator.from(0).map(i => s"f$i").filterNot(oldFields)
    for {
      n <- Gen.choose(1, 2)
      types <- sequence(ArraySeq.fill(n)(genType(1)))
      fields <- genFields(fresh.take(n).zip(types).to(ArraySeq), env, budget)
    } yield InsertFields(old, fields)
  }

  private def genBindings(env: BindingEnv[Type], k: Int, budget: Int, acc: IndexedSeq[Binding])
    : Gen[IndexedSeq[Binding]] =
    if (acc.length == k) Gen.const(acc)
    else {
      val scopes =
        ArraySeq.untagged.concat(
          Some(4 -> Scope.EVAL),
          Option.when(env.agg.isDefined)(1 -> Scope.AGG),
          Option.when(env.scan.isDefined)(1 -> Scope.SCAN),
        )

      for {
        scope <- Gen.frequency(scopes.map { case (w, sc) => w -> Gen.const(sc) }: _*)
        typ <- genType(1)
        name <- genName
        value <- genIR(typ, Binds.aggSeq(env, scope), budget)
        newEnv = env.bind(scope, ArraySeq(name -> typ))
        bindings <- genBindings(newEnv, k, budget, acc :+ Binding(name, value, scope))
      } yield bindings
    }

  private def byType(t: Type, env: BindingEnv[Type], budget: Int): Seq[Rule] = {
    def aggOp(op: AggOp, args: Type*): Rule = {
      val seqArgEnv = ChildEnv.applyAggOp(env, 0, 0)
      3 -> (() =>
        sequence(args.map(genIR(_, seqArgEnv, budget / args.length.max(1))))
          .map(ApplyAggOp(ArraySeq(), _, op))
      )
    }
    val aggs =
      if (env.agg.isEmpty) ArraySeq()
      else t match {
        case TInt64 => ArraySeq(aggOp(Count()), aggOp(Sum(), TInt64))
        case TFloat64 => ArraySeq(aggOp(Sum(), TFloat64))
        case TArray(e) => ArraySeq(aggOp(Collect(), e))
        case _ => ArraySeq()
      }
    val rules =
      t match {
        case TInt32 | TInt64 | TFloat64 =>
          ArraySeq.untagged.concat(
            Some(3 -> (() =>
              for {
                op <- Gen.oneOf(Add(), Subtract(), Multiply())
                l <- genIR(t, env, budget / 2)
                r <- genIR(t, env, budget / 2)
              } yield ApplyBinaryPrimOp(op, l, r)
            )),
            Some(1 -> (() => genIR(t, env, budget).map(ApplyUnaryPrimOp(Negate, _)))),
            Some(1 -> (() => Gen.oneOf(numeric).flatMap(genIR(_, env, budget)).map(Cast(_, t)))),
            Option.when(t == TInt32)(1 -> (() =>
              genType(1).flatMap(e => genIR(TArray(e), env, budget)).map(ArrayLen)
            )),
          )
        case TBoolean =>
          ArraySeq(
            3 -> (() =>
              for {
                op <- Gen.oneOf(LT, LTEQ, GT, GTEQ, EQ, NEQ)
                u <- Gen.oneOf(numeric)
                l <- genIR(u, env, budget / 2)
                r <- genIR(u, env, budget / 2)
              } yield ApplyComparisonOp(op, l, r)
            ),
            1 -> (() => genIR(t, env, budget).map(ApplyUnaryPrimOp(Bang, _))),
            1 -> (() => genType(1).flatMap(genIR(_, env, budget)).map(IsNA)),
          )
        case TArray(e) =>
          ArraySeq(
            2 -> (() =>
              Gen.choose(1, 4).flatMap(n => sequence(ArraySeq.fill(n)(genIR(e, env, budget / n))))
                .map(MakeArray(_, TArray(e)))
            ),
            3 -> (() => genIR(TStream(e), env, budget).map(ToArray)),
          )
        case t: TStruct =>
          val fields = t.fields.map(f => f.name -> f.typ)
          ArraySeq.untagged.concat(
            Some(2 -> (() => genFields(fields, env, budget).map(MakeStruct(_)))),
            Option.when(fields.length >= 2)(1 -> (() =>
              for {
                k <- Gen.choose(1, fields.length - 1)
                (oldFields, newFields) = fields.splitAt(k)
                old <- genIR(TStruct(oldFields: _*), env, budget / 2)
                inserted <- genFields(newFields, env, budget / 2)
              } yield InsertFields(old, inserted)
            )),
          )
        case t: TTuple =>
          ArraySeq(
            2 -> (() =>
              sequence(t.types.map(genIR(_, env, budget / t.size))).map(fs => MakeTuple.ordered(fs))
            )
          )
        case _ => ArraySeq()
      }
    ArraySeq.untagged.concat(aggs, rules)
  }

  private def genFields(fields: IndexedSeq[(String, Type)], env: BindingEnv[Type], budget: Int)
    : Gen[IndexedSeq[(String, IR)]] =
    sequence(fields.map { case (n, t) => genIR(t, env, budget / fields.length).map(n -> _) })

  private def streams(e: Type, env: BindingEnv[Type], budget: Int): Seq[Rule] =
    ArraySeq.untagged.concat(
      Some(2 -> (() => genIR(TArray(e), env, budget).map(ToStream(_)))),
      Some(2 -> (() =>
        for {
          u <- genType(1)
          name <- genName
          a <- genIR(TStream(u), env, budget / 2)
          body <- genIR(e, ChildEnv.streamMap(env, 1, name, ArraySeq(a))(snd), budget / 2)
        } yield StreamMap(a, name, body)
      )),
      Some(1 -> (() =>
        for {
          name <- genName
          a <- genIR(TStream(e), env, budget / 2)
          cond <- genIR(TBoolean, ChildEnv.streamFilter(env, 1, name, ArraySeq(a))(snd), budget / 2)
        } yield StreamFilter(a, name, cond)
      )),
      Option.when(e == TInt32)(1 -> (() =>
        for {
          start <- genIR(TInt32, env, budget / 2)
          stop <- genIR(TInt32, env, budget / 2)
        } yield StreamRange(start, stop, I32(1))
      )),
    )

  private val genName =
    Gen.choose(0, 999999).map(i => Name(s"x$i"))

  private def sequence[A: ClassTag](gs: Seq[Gen[A]]): Gen[ArraySeq[A]] =
    Gen.sequence[ArraySeq[A], A](gs)
}
