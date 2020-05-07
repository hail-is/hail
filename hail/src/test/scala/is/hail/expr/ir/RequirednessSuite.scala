package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.expr.Nat
import is.hail.expr.types.encoded.EBaseStruct
import is.hail.expr.types.{BaseTypeWithRequiredness, TypeWithRequiredness}
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.stats.fetStruct
import is.hail.utils.{ArrayBuilder, FastIndexedSeq}
import is.hail.variant.Locus
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class RequirednessSuite extends HailSuite {
  val required: Boolean = true
  val optional: Boolean = false

  @DataProvider(name="valueIR")
  def valueIR(): Array[Array[Any]] = {
    val nodes = new ArrayBuilder[Array[Any]](50)

    val allRequired = Array(
      I32(5), I64(5), F32(3.14f), F64(3.14), Str("foo"), True(), False(),
      IsNA(I32(5)),
      Cast(I32(5), TFloat64),
      Die("mumblefoo", TFloat64),
      Literal(TStruct("x" -> TInt32), Row(1)),
      MakeArray(FastIndexedSeq(I32(4)), TArray(TInt32)),
      MakeStruct(FastIndexedSeq("x" -> I32(4), "y" -> Str("foo"))),
      MakeTuple.ordered(FastIndexedSeq(I32(5), Str("bar"))))

    allRequired.foreach { n =>
      nodes += Array(n, PType.canonical(n.typ, required).deepInnerRequired(required))
    }

    val tarray = TArray(TInt32)
    val tstream = TStream(TInt32)
    val tstruct = TStruct("x" -> TInt32, "y" -> TArray(TInt32))
    val ttuple = TTuple(TInt32, TArray(TInt32))
    val tnd = TNDArray(TInt32, Nat(2))
    val tnestednd = TNDArray(tarray, Nat(2))
    val tnestedarray = TArray(tarray)
    val tnestedstream = TStream(tarray)

    def int(r: Boolean): IR = if (r) I32(5) else NA(TInt32)

    def stream(r: Boolean, elt: Boolean): IR = {
      if (r)
        MakeStream(FastIndexedSeq(int(elt), int(required)), tstream)
      else
        mapIR(NA(tstream))(x => x + int(elt))
    }

    def array(r: Boolean, elt: Boolean): IR = ToArray(stream(r, elt))


    def struct(r: Boolean, i: Boolean, a: Boolean, elt: Boolean): IR = {
      val fields = FastIndexedSeq("x" -> int(i), "y" -> array(a, elt))
      if (r) MakeStruct(fields) else InsertFields(NA(tstruct), fields)
    }

    def tuple(r: Boolean, i: Boolean, a: Boolean, elt: Boolean): IR = {
      val t = MakeTuple.ordered(FastIndexedSeq(int(i), array(a, elt)))
      if (r) t else If(True(), NA(ttuple), t)
    }

    def nd(r: Boolean): IR =
      if (r) MakeNDArray.fill(int(optional), FastIndexedSeq(1, 2), True()) else NA(tnd)

    def nestednd(r: Boolean, aelt: Boolean): IR = {
      if (r)
        MakeNDArray.fill(array(optional, aelt), FastIndexedSeq(1, 2), True())
      else
        NDArrayMap(NA(tnestednd), genUID(), array(optional, aelt))
    }

    def nestedstream(r: Boolean, a: Boolean, aelt: Boolean): IR = {
      if (r)
        MakeStream(FastIndexedSeq(array(a, required), array(required, aelt)), tnestedstream)
      else
        mapIR(NA(tnestedstream))(x => array(a, aelt))
    }
    def nestedarray(r: Boolean, a: Boolean, aelt: Boolean): IR = ToArray(nestedstream(r, a, aelt))

    val bools = Array(true, false)
    for (r1 <- bools) {
      val pint = PInt32(r1)
      nodes += Array(int(r1), pint)
      nodes += Array(nd(r1), PCanonicalNDArray(PInt32(required), 2, r1))
      for (r2 <- bools) {
        val parray = PCanonicalArray(pint, r2)
        nodes += Array(array(r2, r1), parray)
        nodes += Array(nestednd(r2, r1), PCanonicalNDArray(PCanonicalArray(pint, required), 2, r2))
        for (r3 <- bools) {
          nodes += Array(nestedarray(r3, r2, r1), PCanonicalArray(parray, r3))
          for (r4 <- bools) {
            nodes += Array(tuple(r4, r3, r2, r1), PCanonicalTuple(r4, PInt32(r3), parray))
            nodes += Array(struct(r4, r3, r2, r1), PCanonicalStruct(r4, "x" -> PInt32(r3), "y" -> parray))
          }
        }
      }
    }

    // test coalesce
    nodes += Array(Coalesce(FastIndexedSeq(
      array(required, optional),
      array(optional, required))),
      PCanonicalArray(PInt32(optional), required))

    nodes += Array(Coalesce(FastIndexedSeq(
      array(optional, optional),
      array(optional, required))),
      PCanonicalArray(PInt32(optional), optional))

    // test read/write
    val pDisc = PCanonicalStruct(required,
      "a" -> PInt32(optional),
      "b" -> PCanonicalArray(PInt32(required), required),
      "c" -> PCanonicalArray(PCanonicalStruct(required,
        "x" -> PInt32(required),
        "y" -> PCanonicalArray(PInt32(required), optional)
      ), required)
    )

    val spec = TypedCodecSpec(pDisc, BufferSpec.default)
    val pr = PartitionNativeReader(spec)
    val rt1 = TStruct("a" -> TInt32, "b" -> TArray(TInt32))
    val rt2 = TStruct("a" -> TInt32, "c" -> TArray(TStruct("x" -> TInt32)))
    Array(Str("foo") -> pDisc,
      NA(TString) -> pDisc,
      Str("foo") -> pDisc.subsetTo(rt1),
      Str("foo") -> pDisc.subsetTo(rt2)).foreach { case (path, pt) =>
      nodes += Array(ReadPartition(path, pt.virtualType, pr), PCanonicalStream(pt).setRequired(path.isInstanceOf[Str]))
      nodes += Array(ReadValue(path, spec, pt.virtualType), pt.setRequired(path.isInstanceOf[Str]))
    }

    val value = Literal(pDisc.virtualType, Row(null, IndexedSeq(1), IndexedSeq(Row(1, IndexedSeq(1)))))
    nodes += Array(WriteValue(value, Str("foo"), spec), PCanonicalString(required))
    nodes += Array(WriteValue(NA(pDisc.virtualType), Str("foo"), spec), PCanonicalString(optional))
    nodes += Array(WriteValue(value, NA(TString), spec), PCanonicalString(optional))

    // test bindings
    nodes += Array(bindIR(nestedarray(required, optional, optional)) { v => ArrayRef(v, I32(0)) },
      PCanonicalArray(PInt32(optional), optional))
    // filter
    nodes += Array(StreamFilter(stream(optional, optional), "x", Ref("x", TInt32).ceq(0)),
      PCanonicalStream(PInt32(optional), optional))
    // StreamFold
    nodes += Array(StreamFold(
      nestedstream(optional, optional, optional),
      I32(0), "a", "b",
      ArrayRef(Ref("b", tarray), Ref("a", TInt32))
    ), PInt32(optional))
    // StreamFold2
    nodes += Array(StreamFold2(
      nestedstream(optional, optional, optional),
      FastIndexedSeq("b" -> I32(0)), "a",
      FastIndexedSeq(ArrayRef(Ref("a", tarray), Ref("b", TInt32))),
      Ref("b", TInt32)
    ), PInt32(optional))
    // StreamScan
      nodes += Array(StreamScan(
        nestedstream(optional, optional, optional),
        I32(0), "a", "b",
        ArrayRef(Ref("b", tarray), Ref("a", TInt32))
      ), PCanonicalStream(PInt32(optional), optional))
    // TailLoop
    val param1 = Ref(genUID(), tarray)
    val param2 = Ref(genUID(), TInt32)
    val loop = TailLoop("loop", FastIndexedSeq(
      param1.name -> array(required, required),
      param2.name -> int(required)),
      If(False(), // required
        MakeArray(FastIndexedSeq(param1), tnestedarray), // required
        If(param2 <= I32(1), // possibly missing
          Recur("loop", FastIndexedSeq(
            array(required, optional),
            int(required)), tnestedarray),
          Recur("loop", FastIndexedSeq(
            array(optional, required),
            int(optional)), tnestedarray))))
    nodes += Array(loop, PCanonicalArray(PCanonicalArray(PInt32(optional), optional), optional))
    // ArrayZip
    val s1 = Ref(genUID(), TInt32)
    val s2 = Ref(genUID(), TInt32)
    val notExtendNA = StreamZip(
      FastIndexedSeq(stream(required, optional), stream(required, required)),
      FastIndexedSeq(s1.name, s2.name),
      s1 + s2, ArrayZipBehavior.TakeMinLength)
    val extendNA = StreamZip(
      FastIndexedSeq(stream(required, required), stream(required, required)),
      FastIndexedSeq(s1.name, s2.name),
      s1 + s2, ArrayZipBehavior.ExtendNA)
    nodes += Array(notExtendNA, PCanonicalStream(PInt32(optional), required))
    nodes += Array(extendNA, PCanonicalStream(PInt32(optional), required))
    // ArraySort
    nodes += Array(ArraySort(stream(optional, required), s1.name, s2.name, True()), PCanonicalArray(PInt32(required), optional))
    // CollectDistributedArray
    nodes += Array(CollectDistributedArray(
      stream(optional, required),
      int(optional),
      s1.name, s2.name,
      s1 + s2), PCanonicalArray(PInt32(optional), optional))

    // ApplyIR
    nodes += Array(
      invoke("argmin", TInt32, array(required, required)),
      PInt32(optional))
    nodes += Array(
      invoke("argmin", TInt32, array(required, optional)),
      PInt32(optional))
    nodes += Array(
      invoke("argmin", TInt32, array(optional, required)),
      PInt32(optional))
    // Apply
    nodes += Array(
      invoke("fisher_exact_test", fetStruct.virtualType,
        int(required), int(required), int(required), int(required)),
      fetStruct.setRequired(required))
    nodes += Array(
      invoke("fisher_exact_test", fetStruct.virtualType,
        int(optional), int(required), int(required), int(required)),
      fetStruct.setRequired(optional))
    nodes += Array(
      invoke("Interval", TInterval(TArray(TInt32)),
        array(required, optional), array(required, required), True(), NA(TBoolean)),
      PCanonicalInterval(PCanonicalArray(PInt32(optional), required), optional))
    nodes.result()
  }

  @Test
  def testDataProviders(): Unit = {
    val s = new ArrayBuilder[String]()
    valueIR().map(v => v(0) -> v(1))foreach { case (n: IR, t: PType) =>
      if (n.typ != t.virtualType)
        s += s"${ n.typ } != ${ t.virtualType }: \n${ Pretty(n) }"
    }
    assert(s.size == 0, s.result().mkString("\n\n"))
  }

  def dump(m: Memo[BaseTypeWithRequiredness]): String = {
    m.m.map { case (node, t) =>
        s"${Pretty(node.t)}: \n$t"
    }.mkString("\n\n")
  }

  @Test(dataProvider = "valueIR")
  def testRequiredness(node: IR, expected: PType): Unit = {
    TypeCheck(node)
    val res = Requiredness.apply(node, ctx)
    val actual = res.r.lookup(node).asInstanceOf[TypeWithRequiredness]
    assert(actual.canonicalPType(node.typ) == expected, s"\n\n${Pretty(node)}: \n$actual\n\n${ dump(res.r) }")
  }
}
