package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.expr.Nat
import is.hail.expr.ir.agg.CallStatsState
import is.hail.expr.ir.defs._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.stats.fetStruct
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.interfaces.SStream
import is.hail.types.physical.stypes.primitives.SInt32
import is.hail.types.virtual._
import is.hail.utils.{BoxedArrayBuilder, FastSeq}

import org.apache.spark.sql.Row
import org.scalatest
import org.scalatest.Inspectors.forAll
import org.scalatest.enablers.InspectorAsserting.assertingNatureOfAssertion
import org.testng.annotations.{DataProvider, Test}

class RequirednessSuite extends HailSuite {
  val required: Boolean = true
  val optional: Boolean = false

  val tarray = TArray(TInt32)
  val tstream = TStream(TInt32)
  val tstruct = TStruct("x" -> TInt32, "y" -> TArray(TInt32))
  val ttuple = TTuple(TInt32, TArray(TInt32))
  val tnd = TNDArray(TInt32, Nat(2))
  val tnestednd = TNDArray(tarray, Nat(2))
  val tnestedarray = TArray(tarray)
  val tnestedstream = TStream(tarray)

  def int(r: Boolean): IR = if (r) I32(5) else NA(TInt32)

  def call(r: Boolean): IR = if (r) invoke("Call", TCall, Str("0/0")) else NA(TCall)

  def stream(r: Boolean, elt: Boolean): IR =
    if (r)
      MakeStream(FastSeq(int(elt), int(required)), tstream)
    else
      mapIR(NA(tstream))(x => x + int(elt))

  def array(r: Boolean, elt: Boolean): IR = ToArray(stream(r, elt))

  def struct(r: Boolean, i: Boolean, a: Boolean, elt: Boolean): IR = {
    val fields = FastSeq("x" -> int(i), "y" -> array(a, elt))
    if (r) MakeStruct(fields) else InsertFields(NA(tstruct), fields)
  }

  def tuple(r: Boolean, i: Boolean, a: Boolean, elt: Boolean): IR = {
    val t = MakeTuple.ordered(FastSeq(int(i), array(a, elt)))
    if (r) t else If(True(), NA(ttuple), t)
  }

  def nd(r: Boolean): IR =
    if (r) MakeNDArray.fill(int(optional), FastSeq(1L, 2L), True()) else NA(tnd)

// FIXME: Currently ndarrays don't support elements that have pointers.
//  def nestednd(r: Boolean, aelt: Boolean): IR = {
//    if (r)
//      MakeNDArray.fill(array(optional, aelt), FasSeq(1, 2), True())
//    else
//      NDArrayMap(NA(tnestednd), genUID(), array(optional, aelt))
//  }

  def nestedstream(r: Boolean, a: Boolean, aelt: Boolean): IR =
    if (r)
      MakeStream(FastSeq(array(a, required), array(required, aelt)), tnestedstream)
    else
      mapIR(NA(tnestedstream))(x => array(a, aelt))

  def nestedarray(r: Boolean, a: Boolean, aelt: Boolean): IR = ToArray(nestedstream(r, a, aelt))

  def pint(r: Boolean): PInt32 = PInt32(r)

  def pstream(r: Boolean, elt: Boolean): EmitType = EmitType(SStream(EmitType(SInt32, elt)), r)

  def parray(r: Boolean, elt: Boolean): PArray = PCanonicalArray(pint(elt), r)

  def pstruct(r: Boolean, i: Boolean, a: Boolean, elt: Boolean): PStruct =
    PCanonicalStruct(r, "x" -> pint(i), "y" -> parray(a, elt))

  def ptuple(r: Boolean, i: Boolean, a: Boolean, elt: Boolean): PTuple =
    PCanonicalTuple(r, pint(i), parray(a, elt))

  def pnd(r: Boolean): PNDArray = PCanonicalNDArray(pint(required), 2, r)

  def pnestednd(r: Boolean, aelt: Boolean): PNDArray =
    PCanonicalNDArray(parray(required, aelt), 2, r)

  def pnestedarray(r: Boolean, a: Boolean, aelt: Boolean): PArray =
    PCanonicalArray(parray(a, aelt), r)

  def interval(point: IR, r: Boolean): IR = invoke(
    "Interval",
    TInterval(point.typ),
    point,
    point.deepCopy(),
    True(),
    if (r) True() else NA(TBoolean),
  )

  def pinterval(point: PType, r: Boolean): PInterval = PCanonicalInterval(point, r)

  @DataProvider(name = "valueIR")
  def valueIR(): Array[Array[Any]] = {
    val nodes = new BoxedArrayBuilder[Array[Any]](50)

    val allRequired = Array(
      I32(5),
      I64(5),
      F32(3.14f),
      F64(3.14),
      Str("foo"),
      True(),
      False(),
      IsNA(I32(5)),
      Cast(I32(5), TFloat64),
      Die("mumblefoo", TFloat64),
      Literal(TStruct("x" -> TInt32), Row(1)),
      MakeArray(FastSeq(I32(4)), TArray(TInt32)),
      MakeStruct(FastSeq("x" -> I32(4), "y" -> Str("foo"))),
      MakeTuple.ordered(FastSeq(I32(5), Str("bar"))),
    )

    allRequired.foreach { n =>
      nodes += Array(
        n,
        RequirednessSuite.deepInnerRequired(PType.canonical(n.typ, required), required),
      )
    }

    val bools = Array(true, false)
    for (r1 <- bools) {
      nodes += Array(int(r1), pint(r1))
      nodes += Array(nd(r1), pnd(r1))
      for (r2 <- bools) {
        nodes += Array(array(r2, r1), parray(r2, r1))
        for (r3 <- bools) {
          nodes += Array(nestedarray(r3, r2, r1), pnestedarray(r3, r2, r1))
          for (r4 <- bools) {
            nodes += Array(tuple(r4, r3, r2, r1), ptuple(r4, r3, r2, r1))
            nodes += Array(struct(r4, r3, r2, r1), pstruct(r4, r3, r2, r1))
          }
        }
      }
    }

    // test coalesce
    nodes += Array(
      Coalesce(FastSeq(
        array(required, optional),
        array(optional, required),
      )),
      parray(required, optional),
    )

    nodes += Array(
      Coalesce(FastSeq(
        array(optional, optional),
        array(optional, required),
      )),
      parray(optional, optional),
    )

    // test read/write
    val pDisc = PCanonicalStruct(
      required,
      "a" -> pint(optional),
      "b" -> parray(required, required),
      "c" -> PCanonicalArray(pstruct(required, required, optional, required), required),
    )

    val spec = TypedCodecSpec(ctx, pDisc, BufferSpec.default)
    val vr = ETypeValueWriter(spec)
    val pr = PartitionNativeReader(spec, "rowUID")
    val contextType = pr.contextType
    val rt1 = TStruct("a" -> TInt32, "b" -> TArray(TInt32))
    val rt2 = TStruct("a" -> TInt32, "c" -> TArray(TStruct("x" -> TInt32)))
    Array(
      Str("foo") -> pDisc,
      NA(TString) -> pDisc,
      Str("foo") -> pDisc.subsetTo(rt1),
      Str("foo") -> pDisc.subsetTo(rt2),
    ).foreach { case (path, pt: PStruct) =>
      nodes += Array(
        ReadPartition(
          if (path.isInstanceOf[Str])
            MakeStruct(Array("partitionIndex" -> I64(0), "partitionPath" -> path))
          else
            NA(contextType),
          pt.virtualType,
          pr,
        ),
        EmitType(SStream(EmitType(pt.sType, pt.required)), path.isInstanceOf[Str]),
      )
      nodes += Array(
        ReadValue(path, ETypeValueReader(spec), pt.virtualType),
        pt.setRequired(path.isInstanceOf[Str]),
      )
    }

    val value =
      Literal(pDisc.virtualType, Row(null, IndexedSeq(1), IndexedSeq(Row(1, IndexedSeq(1)))))
    nodes += Array(WriteValue(value, Str("foo"), vr), PCanonicalString(required))
    nodes += Array(WriteValue(NA(pDisc.virtualType), Str("foo"), vr), PCanonicalString(optional))
    nodes += Array(WriteValue(value, NA(TString), vr), PCanonicalString(optional))

    // test bindings
    nodes += Array(
      bindIR(nestedarray(required, optional, optional))(v => ArrayRef(v, I32(0))),
      PCanonicalArray(PInt32(optional), optional),
    )
    nodes += Array(
      IRBuilder.scoped { b =>
        val x0 = b.strictMemoize(int(optional))
        val x1 = b.strictMemoize(array(required, required))
        val x2 = b.strictMemoize(ToStream(x1))
        val x3 = b.strictMemoize(mapIR(x2)(x => x * x0))
        val x4 = b.strictMemoize(ToArray(x3))
        val x5 = int(required)
        ArrayRef(x4, x5)
      },
      pint(optional),
    )
    // filter
    nodes += Array(
      filterIR(stream(optional, optional))(_.ceq(0)),
      EmitType(SStream(EmitType(SInt32, optional)), optional),
    )
    // StreamFold
    nodes += Array(
      foldIR(nestedstream(optional, optional, optional), I32(0))((a, b) => ArrayRef(b, a)),
      PInt32(optional),
    )
    // StreamFold2
    nodes += Array(
      fold2IR(nestedstream(optional, optional, optional), I32(0)) {
        case (a, Seq(b)) =>
          ArrayRef(a, b)
      } { case Seq(b) => b },
      PInt32(optional),
    )
    // StreamScan
    nodes += Array(
      streamScanIR(nestedstream(optional, optional, optional), I32(0))((a, b) => ArrayRef(b, a)),
      EmitType(SStream(EmitType(SInt32, optional)), optional),
    )
    // TailLoop
    val loop = tailLoop(
      tnestedarray,
      array(required, required),
      int(required),
    ) { case (recur, Seq(param1, param2)) =>
      If(
        False(), // required
        MakeArray(FastSeq(param1), tnestedarray), // required
        If(
          param2 <= I32(1), // possibly missing
          recur(FastSeq(array(required, optional), int(required))),
          recur(FastSeq(array(optional, required), int(optional))),
        ),
      )
    }
    nodes += Array(loop, PCanonicalArray(PCanonicalArray(PInt32(optional), optional), optional))
    // Switch
    for (
      (x, d, cs, r) <- Array(
        (required, required, FastSeq(required), required),
        (optional, required, FastSeq(required), optional),
        (required, optional, FastSeq(required), optional),
        (required, required, FastSeq(optional), optional),
      )
    )
      nodes += Array(Switch(int(x), int(d), cs.map(int)), PInt32(r))
    // ArrayZip
    val notExtendNA = zipIR(
      FastSeq(stream(required, optional), stream(required, required)),
      ArrayZipBehavior.TakeMinLength,
    ) { case Seq(s1, s2) => s1 + s2 }
    val extendNA = zipIR(
      FastSeq(stream(required, required), stream(required, required)),
      ArrayZipBehavior.ExtendNA,
    ) { case Seq(s1, s2) => s1 + s2 }
    nodes += Array(notExtendNA, pstream(required, optional))
    nodes += Array(extendNA, pstream(required, optional))
    // ArraySort
    nodes += Array(
      sortIR(stream(optional, required))((_, _) => True()),
      parray(optional, required),
    )
    // CollectDistributedArray
    nodes += Array(
      cdaIR(stream(optional, required), int(optional), "test", NA(TString))(_ + _),
      parray(optional, optional),
    )

    // ApplyIR
    nodes += Array(
      invoke("argmin", TInt32, array(required, required)),
      pint(optional),
    )
    nodes += Array(
      invoke("argmin", TInt32, array(required, optional)),
      pint(optional),
    )
    nodes += Array(
      invoke("argmin", TInt32, array(optional, required)),
      pint(optional),
    )
    // Apply
    nodes += Array(
      invoke(
        "fisher_exact_test",
        fetStruct.virtualType,
        int(required),
        int(required),
        int(required),
        int(required),
      ),
      fetStruct.setRequired(required),
    )
    nodes += Array(
      invoke(
        "fisher_exact_test",
        fetStruct.virtualType,
        int(optional),
        int(required),
        int(required),
        int(required),
      ),
      fetStruct.setRequired(optional),
    )
    nodes += Array(
      invoke(
        "Interval",
        TInterval(TArray(TInt32)),
        array(required, optional),
        array(required, required),
        True(),
        NA(TBoolean),
      ),
      PCanonicalInterval(parray(required, optional), optional),
    )
    nodes.result()
  }

  @DataProvider(name = "tableIR")
  def tableIR(): Array[Array[Any]] = {
    val nodes = new BoxedArrayBuilder[Array[Any]](50)

    nodes += Array[Any](
      TableRange(1, 1),
      PCanonicalStruct(required, "idx" -> PInt32(required)),
      PCanonicalStruct.empty(required),
    )

    val table = TableParallelize(
      makestruct(
        "rows" -> MakeArray(makestruct(
          "a" -> nestedarray(optional, required, optional),
          "b" -> struct(required, required, required, optional),
          "c" -> nd(required),
        )),
        "global" -> makestruct(
          "x" -> array(required, optional),
          "y" -> int(optional),
          "z" -> struct(required, required, required, optional),
        ),
      ),
      None,
    )

    val rowType = PCanonicalStruct(
      required,
      "a" -> pnestedarray(optional, required, optional),
      "b" -> pstruct(required, required, required, optional),
      "c" -> pnd(required),
    )
    val globalType = PCanonicalStruct(
      required,
      "x" -> parray(required, optional),
      "y" -> pint(optional),
      "z" -> pstruct(required, required, required, optional),
    )

    def row = Ref(TableIR.rowName, table.typ.rowType)
    def global = Ref(TableIR.globalName, table.typ.globalType)

    // type-preserving
    nodes += Array(table, rowType, globalType)
    nodes += Array(TableKeyBy(table, FastSeq("b")), rowType, globalType)
    nodes += Array(TableFilter(table, GetField(global, "y") < 0), rowType, globalType)
    nodes += Array(TableHead(table, 5), rowType, globalType)
    nodes += Array(TableTail(table, 5), rowType, globalType)
    nodes += Array(TableRepartition(table, 5, RepartitionStrategy.SHUFFLE), rowType, globalType)
    nodes += Array(TableDistinct(table), rowType, globalType)
    nodes += Array(TableOrderBy(table, FastSeq()), rowType, globalType)
    nodes += Array(TableFilterIntervals(table, FastSeq(), true), rowType, globalType)

    val rMap = Map("a" -> "d", "c" -> "f")
    val gMap = Map("x" -> "foo", "y" -> "bar")
    nodes += Array(TableRename(table, rMap, gMap), rowType.rename(rMap), globalType.rename(gMap))

    nodes += Array(
      TableMapRows(
        table,
        insertIR(
          row,
          "a2" -> ApplyScanOp(Collect())(GetField(row, "a")),
          "x2" -> GetField(global, "x"),
        ),
      ),
      rowType.insertFields(FastSeq(
        "a2" -> PCanonicalArray(rowType.fieldType("a"), required),
        "x2" -> globalType.fieldType("x"),
      )),
      globalType,
    )

    nodes += Array(
      TableMapGlobals(table, insertIR(global, "x2" -> GetField(global, "x"))),
      rowType,
      globalType.insertFields(FastSeq("x2" -> globalType.fieldType("x"))),
    )

    nodes += Array(
      TableExplode(
        TableMapRows(
          table,
          insertIR(row, "e1" -> struct(r = optional, i = required, a = optional, elt = required)),
        ),
        FastSeq("e1", "y"),
      ),
      rowType.insertFields(FastSeq("e1" -> PCanonicalStruct(
        required,
        "x" -> pint(required),
        "y" -> pint(required),
      ))),
      globalType,
    )

    nodes += Array(
      TableUnion(FastSeq(
        table.deepCopy(),
        TableMapRows(table, insertIR(row, "a" -> nestedarray(optional, optional, required))),
      )),
      rowType.insertFields(FastSeq("a" -> pnestedarray(optional, optional, optional))),
      globalType,
    )

    val collect = ApplyAggOp(Collect())(GetField(row, "b"))
    val callstats = ApplyAggOp(CallStats(), int(optional))(call(required))
    val expr = makestruct("collect" -> collect, "callstats" -> callstats)

    nodes += Array(
      TableKeyByAndAggregate(table, expr, makestruct("a" -> GetField(row, "a")), None, 5),
      PCanonicalStruct(
        required,
        "a" -> rowType.fieldType("a"),
        "collect" -> PCanonicalArray(rowType.fieldType("b"), required),
        "callstats" -> CallStatsState.resultPType.setRequired(true),
      ),
      globalType,
    )

    nodes += Array(
      TableAggregateByKey(TableKeyBy(table, FastSeq("a")), expr),
      PCanonicalStruct(
        required,
        "a" -> rowType.fieldType("a"),
        "collect" -> PCanonicalArray(rowType.fieldType("b"), required),
        "callstats" -> CallStatsState.resultPType.setRequired(true),
      ),
      globalType,
    )

    val left = TableMapGlobals(
      TableKeyBy(
        TableMapRows(
          table.deepCopy(),
          makestruct(
            "a" -> nestedarray(required, optional, required),
            "b" -> GetField(row, "b"),
          ),
        ),
        FastSeq("a"),
      ),
      selectIR(global, "x"),
    )
    val right = TableMapGlobals(
      TableKeyBy(
        TableMapRows(
          table.deepCopy(),
          makestruct(
            "a" -> nestedarray(required, required, optional),
            "c" -> GetField(row, "c"),
          ),
        ),
        FastSeq("a"),
      ),
      selectIR(global, "y", "z"),
    )

    nodes += Array(
      TableJoin(left, right, "left", 1),
      PCanonicalStruct(
        required,
        "a" -> pnestedarray(required, optional, required),
        "b" -> rowType.fieldType("b"),
        "c" -> rowType.fieldType("c").setRequired(optional),
      ),
      globalType,
    )

    nodes += Array(
      TableJoin(left, right, "right", 1),
      PCanonicalStruct(
        required,
        "a" -> pnestedarray(required, required, optional),
        "b" -> rowType.fieldType("b").setRequired(optional),
        "c" -> rowType.fieldType("c"),
      ),
      globalType,
    )

    nodes += Array(
      TableJoin(left, right, "inner", 1),
      PCanonicalStruct(
        required,
        "a" -> pnestedarray(required, required, required),
        "b" -> rowType.fieldType("b"),
        "c" -> rowType.fieldType("c"),
      ),
      globalType,
    )

    nodes += Array(
      TableJoin(left, right, "outer", 1),
      PCanonicalStruct(
        required,
        "a" -> pnestedarray(required, optional, optional),
        "b" -> rowType.fieldType("b").setRequired(optional),
        "c" -> rowType.fieldType("c").setRequired(optional),
      ),
      globalType,
    )

    val intervalTable = TableKeyBy(
      TableMapRows(
        table.deepCopy(),
        makestruct(
          "a" -> interval(nestedarray(required, required, optional), required),
          "c" -> GetField(row, "c"),
        ),
      ),
      FastSeq("a"),
    )
    nodes += Array(
      TableIntervalJoin(left, intervalTable, "root", product = false),
      PCanonicalStruct(
        required,
        "a" -> pnestedarray(required, optional, required),
        "b" -> rowType.fieldType("b"),
        "root" -> PCanonicalStruct(optional, "c" -> rowType.fieldType("c")),
      ),
      globalType.selectFields(FastSeq("x")),
    )

    nodes += Array(
      TableIntervalJoin(left, intervalTable, "root", product = true),
      PCanonicalStruct(
        required,
        "a" -> pnestedarray(required, optional, required),
        "b" -> rowType.fieldType("b"),
        "root" -> PCanonicalArray(
          PCanonicalStruct(required, "c" -> rowType.fieldType("c")),
          optional,
        ),
      ),
      globalType.selectFields(FastSeq("x")),
    )

    nodes += Array(
      TableMultiWayZipJoin(
        FastSeq(
          TableKeyBy(
            TableMapRows(
              table.deepCopy(),
              insertIR(row, "a" -> nestedarray(required, optional, required)),
            ),
            FastSeq("a"),
          ),
          TableKeyBy(
            TableMapRows(
              table.deepCopy(),
              insertIR(row, "a" -> nestedarray(required, required, optional)),
            ),
            FastSeq("a"),
          ),
        ),
        "value",
        "global",
      ),
      PCanonicalStruct(
        required,
        "a" -> pnestedarray(required, optional, optional),
        "value" -> PCanonicalArray(
          PCanonicalStruct(optional, "b" -> rowType.fieldType("b"), "c" -> rowType.fieldType("c")),
          required,
        ),
      ),
      PCanonicalStruct(required, "global" -> PCanonicalArray(globalType, required)),
    )

    nodes += Array(
      TableLeftJoinRightDistinct(left, right, "root"),
      PCanonicalStruct(
        required,
        "a" -> pnestedarray(required, optional, required),
        "b" -> rowType.fieldType("b"),
        "root" -> PCanonicalStruct(optional, "c" -> rowType.fieldType("c")),
      ),
      globalType.selectFields(FastSeq("x")),
    )
    nodes.result()
  }

  @Test
  def testDataProviders(): scalatest.Assertion = {
    val s = new BoxedArrayBuilder[String]()
    valueIR().map(v => v(0) -> v(1)).foreach {
      case (n: IR, t: PType) =>
        if (n.typ != t.virtualType)
          s += s"${n.typ} != ${t.virtualType}: \n${Pretty(ctx, n)}"
      case (n: IR, et: EmitType) =>
        if (n.typ != et.virtualType)
          s += s"${n.typ} != ${et.virtualType}: \n${Pretty(ctx, n)}"
    }
    tableIR().map(v => (v(0), v(1), v(2))).foreach { case (n: TableIR, row: PType, global: PType) =>
      if (n.typ.rowType != row.virtualType || n.typ.globalType != global.virtualType)
        s +=
          s"""row: ${n.typ.rowType} vs ${row.virtualType}
             |global: ${n.typ.globalType} vs ${global.virtualType}:
             |${Pretty(ctx, n)}"
             |""".stripMargin
    }
    assert(s.size == 0, s.result().mkString("\n\n"))
  }

  def /**/ dump(m: Memo[BaseTypeWithRequiredness]): String =
    m.m.map { case (node, t) =>
      s"${Pretty(ctx, node.t)}: \n$t"
    }.mkString("\n\n")

  @Test(dataProvider = "valueIR")
  def testRequiredness(node: IR, expected: Any): scalatest.Assertion = {
    TypeCheck(ctx, node)
    val et = expected match {
      case pt: PType => EmitType(pt.sType, pt.required)
      case et: EmitType => et
    }
    val res = Requiredness.apply(node, ctx)
    val actual = res.r.lookup(node).asInstanceOf[TypeWithRequiredness]
    assert(
      actual.canonicalEmitType(node.typ) == et,
      s"\n\n${Pretty(ctx, node)}: \n$actual\n\n${dump(res.r)}",
    )
  }

  @Test def sharedNodesWorkCorrectly(): scalatest.Assertion = {
    val n2 = bindIR(I32(1))(x => MakeStruct(FastSeq("a" -> x, "b" -> x)))
    val node = InsertFields(n2, FastSeq("c" -> GetField(n2, "a"), "d" -> GetField(n2, "b")))
    val res = Requiredness.apply(node, ctx)
    val actual = tcoerce[TypeWithRequiredness](res.r.lookup(node)).canonicalPType(node.typ)
    assert(actual == PCanonicalStruct(
      required,
      "a" -> PInt32(required),
      "b" -> PInt32(required),
      "c" -> PInt32(required),
      "d" -> PInt32(required),
    ))
  }

  @Test(dataProvider = "tableIR")
  def testTableRequiredness(node: TableIR, row: PType, global: PType): scalatest.Assertion = {
    val res = Requiredness.apply(node, ctx)
    val actual = res.r.lookup(node).asInstanceOf[RTable]
    assert(
      actual.rowType.canonicalPType(node.typ.rowType) == row,
      s"\n\n${Pretty(ctx, node)}: \n$actual\n\n${dump(res.r)}",
    )
    assert(
      actual.globalType.canonicalPType(node.typ.globalType) == global,
      s"\n\n${Pretty(ctx, node)}: \n$actual\n\n${dump(res.r)}",
    )
  }

  @Test def testTableReader(): scalatest.Assertion = {
    val table = TableParallelize(
      makestruct(
        "rows" -> MakeArray(makestruct(
          "a" -> nestedarray(optional, required, optional),
          "b" -> struct(required, required, required, optional),
          "c" -> array(optional, required),
        )),
        "global" -> makestruct(
          "x" -> array(required, optional),
          "y" -> int(optional),
          "z" -> struct(required, required, required, optional),
        ),
      ),
      None,
    )

    val path = ctx.createTmpPath("test-table-requiredness", "ht")
    CompileAndEvaluate[Unit](
      ctx,
      TableWrite(table, TableNativeWriter(path, overwrite = true)),
      false,
    )

    val reader = TableNativeReader(fs, TableNativeReaderParameters(path, None))
    forAll(
      Array(
        table.typ,
        TableType(TStruct("a" -> tnestedarray), FastSeq(), TStruct("z" -> tstruct)),
      )
    ) { rType =>
      val row = reader.rowRequiredness(ctx, rType)
      val global = reader.globalRequiredness(ctx, rType)
      val node = TableRead(rType, dropRows = false, reader)
      val res = Requiredness.apply(node, ctx)
      val actual = res.r.lookup(node).asInstanceOf[RTable]
      assert(
        VirtualTypeWithReq(rType.rowType, actual.rowType) == row,
        s"\n\n${Pretty(ctx, node)}: \n$actual\n\n${dump(res.r)}",
      )
      assert(
        VirtualTypeWithReq(rType.globalType, actual.globalType) == global,
        s"\n\n${Pretty(ctx, node)}: \n$actual\n\n${dump(res.r)}",
      )
    }
  }

  @Test def testSubsettedTuple(): scalatest.Assertion = {
    val node = MakeTuple(FastSeq(0 -> I32(0), 4 -> NA(TInt32), 2 -> NA(TArray(TInt32))))
    val expected = PCanonicalTuple(
      FastSeq(
        PTupleField(0, PInt32(required)),
        PTupleField(4, PInt32(optional)),
        PTupleField(2, PCanonicalArray(PInt32(required), optional)),
      ),
      required,
    )
    val res = Requiredness.apply(node, ctx)
    val actual = tcoerce[TypeWithRequiredness](res.r.lookup(node)).canonicalPType(node.typ)
    assert(actual == expected)
  }
}

object RequirednessSuite {
  def deepInnerRequired(t: PType, required: Boolean): PType =
    t match {
      case t: PCanonicalArray => PCanonicalArray(deepInnerRequired(t.elementType, true), required)
      case t: PCanonicalSet => PCanonicalSet(deepInnerRequired(t.elementType, true), required)
      case t: PCanonicalDict => PCanonicalDict(
          deepInnerRequired(t.keyType, true),
          deepInnerRequired(t.valueType, true),
          required,
        )
      case t: PCanonicalStruct =>
        PCanonicalStruct(
          t.fields.map(f => PField(f.name, deepInnerRequired(f.typ, true), f.index)),
          required,
        )
      case t: PCanonicalTuple =>
        PCanonicalTuple(
          t._types.map(f => f.copy(typ = deepInnerRequired(f.typ, true))),
          required,
        )
      case t: PCanonicalInterval =>
        PCanonicalInterval(deepInnerRequired(t.pointType, true), required)
      case t =>
        t.setRequired(required)
    }
}
