package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.annotations._
import is.hail.check.{Gen, Prop}
import is.hail.asm4s._
import is.hail.TestUtils._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class OrderingSuite extends TestNGSuite {

  implicit val execStrats = ExecStrategy.values

  def recursiveSize(t: Type): Int = {
    val inner = t match {
      case ti: TInterval => recursiveSize(ti.pointType)
      case tc: TContainer => recursiveSize(tc.elementType)
      case tbs: TBaseStruct =>
        tbs.types.map { t => recursiveSize(t) }.sum
      case _ => 0
    }
    inner + 1
  }

  def getStagedOrderingFunction[T: TypeInfo](t: Type, comp: String): AsmFunction3[Region, Long, Long, T] = {
    val fb = EmitFunctionBuilder[Region, Long, Long, T]
    val stagedOrdering = t.physicalType.codeOrdering(fb.apply_method)
    val cregion: Code[Region] = fb.getArg[Region](1)
    val cv1 = coerce[stagedOrdering.T](cregion.getIRIntermediate(t)(fb.getArg[Long](2)))
    val cv2 = coerce[stagedOrdering.T](cregion.getIRIntermediate(t)(fb.getArg[Long](3)))
    comp match {
      case "compare" => fb.emit(stagedOrdering.compare(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "equiv" => fb.emit(stagedOrdering.equiv(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "lt" => fb.emit(stagedOrdering.lt(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "lteq" => fb.emit(stagedOrdering.lteq(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "gt" => fb.emit(stagedOrdering.gt(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "gteq" => fb.emit(stagedOrdering.gteq(cregion, (const(false), cv1), cregion, (const(false), cv2)))
    }
    fb.resultWithIndex()(0)
  }

  def addTupledArgsToRegion(region: Region, args: (Type, Annotation)*): Array[Long] = {
    val rvb = new RegionValueBuilder(region)
    args.map { case (t, a) =>
      rvb.start(TTuple(t).physicalType)
      rvb.startTuple()
      rvb.addAnnotation(t, a)
      rvb.endTuple()
      rvb.end()
    }.toArray
  }

  def getCompiledFunction(irFunction: Seq[IR] => IR, ts: Type*): (Region, Seq[Annotation]) => Annotation = {
    val args = ts.init
    val rt = ts.last
    val irs = args.zipWithIndex.map { case (t, i) => GetTupleElement(In(i, TTuple(t)), 0) }
    val ir = MakeTuple(Seq(irFunction(irs)))

    args.size match {
      case 1 =>
        val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]
        Emit(ir, fb)
        val f = fb.resultWithIndex()(0)
        val f2 = { (region: Region, as: Seq[Annotation]) =>
          val offs = addTupledArgsToRegion(region, args.zip(as): _*)
          SafeRow(TTuple(rt).physicalType, region, f(region, offs(0), false)).get(0)
        }
        f2
      case 2 =>
        val fb = EmitFunctionBuilder[Region, Long, Boolean, Long, Boolean, Long]
        Emit(ir, fb)
        val f = fb.resultWithIndex()(0)
        val f2 = { (region: Region, as: Seq[Annotation]) =>
          val offs = addTupledArgsToRegion(region, args.zip(as): _*)
          SafeRow(TTuple(rt).physicalType, region, f(region, offs(0), false, offs(1), false)).get(0)
        }
        f2
    }
  }

  @Test def testRandomOpsAgainstExtended() {
    val compareGen = for {
      t <- Type.genArb
      a1 <- t.genNonmissingValue
      a2 <- t.genNonmissingValue
    } yield (t, a1, a2)
    val p = Prop.forAll(compareGen) { case (t, a1, a2) =>
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(t.physicalType)
        rvb.addAnnotation(t, a1)
        val v1 = rvb.end()

        rvb.start(t.physicalType)
        rvb.addAnnotation(t, a2)
        val v2 = rvb.end()

        val compare = java.lang.Integer.signum(t.ordering.compare(a1, a2))
        val fcompare = getStagedOrderingFunction[Int](t, "compare")
        val result = java.lang.Integer.signum(fcompare(region, v1, v2))

        assert(result == compare, s"compare expected: $compare vs $result")


        val equiv = t.ordering.equiv(a1, a2)
        val fequiv = getStagedOrderingFunction[Boolean](t, "equiv")

        assert(fequiv(region, v1, v2) == equiv, s"equiv expected: $equiv")

        val lt = t.ordering.lt(a1, a2)
        val flt = getStagedOrderingFunction[Boolean](t, "lt")

        assert(flt(region, v1, v2) == lt, s"lt expected: $lt")

        val lteq = t.ordering.lteq(a1, a2)
        val flteq = getStagedOrderingFunction[Boolean](t, "lteq")

        assert(flteq(region, v1, v2) == lteq, s"lteq expected: $lteq")

        val gt = t.ordering.gt(a1, a2)
        val fgt = getStagedOrderingFunction[Boolean](t, "gt")

        assert(fgt(region, v1, v2) == gt, s"gt expected: $gt")

        val gteq = t.ordering.gteq(a1, a2)
        val fgteq = getStagedOrderingFunction[Boolean](t, "gteq")

        assert(fgteq(region, v1, v2) == gteq, s"gteq expected: $gteq")
      }

      true
    }
    p.check()
  }

  @Test def testSortOnRandomArray() {
    val compareGen = for {
      elt <- Type.genArb
      a <- TArray(elt).genNonmissingValue
      asc <- Gen.coin()
    } yield (elt, a, asc)
    val p = Prop.forAll(compareGen) { case (t, a: IndexedSeq[Any], asc: Boolean) =>
      val irF = { irs: Seq[IR] => ArraySort(irs(0), Literal.coerce(TBoolean(), asc)) }
      val f = getCompiledFunction(irF, TArray(t), TArray(t))
      val ord = if (asc) t.ordering.toOrdering else t.ordering.reverse.toOrdering

      Region.scoped { region =>
        val actual = f(region, Seq(a))
        val expected = a.sorted(ord)
        expected == actual
      }
    }
    p.check()
  }

  @Test def testToSetOnRandomDuplicatedArray() {
    val compareGen = for {
      elt <- Type.genArb
      a <- TArray(elt).genNonmissingValue
    } yield (elt, a)
    val p = Prop.forAll(compareGen) { case (t, a: IndexedSeq[Any]) =>
      val array = a ++ a
      val irF = { irs: Seq[IR] => ToSet(irs(0)) }
      val f = getCompiledFunction(irF, TArray(t), TArray(t))

      Region.scoped { region =>
        val actual = f(region, Seq(array))
        val expected = array.sorted(t.ordering.toOrdering).distinct
        expected == actual
      }
    }
    p.check()
  }

  @Test def testToDictOnRandomDuplicatedArray() {
    val compareGen = for {
      kt <- Type.genArb
      vt <- Type.genArb
      telt = TTuple(kt, vt)
      a <- TArray(telt).genNonmissingValue
    } yield (telt, a)
    val p = Prop.forAll(compareGen) { case (telt: TTuple, a: IndexedSeq[Row]@unchecked) =>
      val array: IndexedSeq[Row] = a ++ a
      val irF = { irs: Seq[IR] => ToDict(irs(0)) }
      val f = getCompiledFunction(irF, TArray(telt), TArray(+telt))

      Region.scoped { region =>
        val actual = f(region, Seq(array)).asInstanceOf[IndexedSeq[Row]]
        val actualKeys = actual.filter(_ != null).map { case Row(k, _) => k }
        val expectedMap = array.filter(_ != null).map { case Row(k, v) => (k, v) }.toMap
        val expectedKeys = expectedMap.keys.toFastIndexedSeq.sorted(telt.types(0).ordering.toOrdering)

        expectedKeys == actualKeys
      }
    }
    p.check()
  }

  @Test def testSortOnMissingArray() {
    val tarray = TArray(TStruct("key" -> TInt32(), "value" -> TInt32()))
    val irs: Array[IR => IR] = Array(ArraySort(_, True()), ToSet(_), ToDict(_))

    for (irF <- irs) {
      val ir = IsNA(irF(NA(tarray)))
      val fb = EmitFunctionBuilder[Region, Boolean]
      Emit(ir, fb)

      val f = fb.resultWithIndex()(0)
      Region.scoped { region =>
        assert(f(region))
      }
    }
  }

  @Test def testSetContainsOnRandomSet() {
    val compareGen = Type.genArb
      .flatMap(t => Gen.zip(Gen.const(TSet(t)), TSet(t).genNonmissingValue, t.genValue))
    val p = Prop.forAll(compareGen) { case (tset: TSet, set: Set[Any]@unchecked, test1) =>
      val telt = tset.elementType

      val ir = { irs: Seq[IR] => invoke("contains", irs(0), irs(1)) }
      val setcontainsF = getCompiledFunction(ir, tset, telt, TBoolean())

      Region.scoped { region =>
        if (set.nonEmpty) {
          val test2 = set.head
          val expected2 = set(test2)
          val actual2 = setcontainsF(region, Seq(set, test2))
          assert(expected2 == actual2)
        }

        val expected1 = set.contains(test1)
        val actual1 = setcontainsF(region, Seq(set, test1))

        expected1 == actual1
      }
    }
    p.check()
  }

  @Test def testDictGetOnRandomDict() {
    implicit val execStrats = ExecStrategy.javaOnly

    val compareGen = Gen.zip(Type.genArb, Type.genArb).flatMap {
      case (k, v) =>
        Gen.zip(Gen.const(TDict(k, v)), TDict(k, v).genNonmissingValue, k.genNonmissingValue)
    }
    val p = Prop.forAll(compareGen) { case (tdict: TDict, dict: Map[Any, Any]@unchecked, testKey1) =>
      assertEvalsTo(invoke("get", In(0, tdict), In(1, -tdict.keyType)),
        IndexedSeq(dict -> tdict,
          testKey1 -> -tdict.keyType),
        dict.getOrElse(testKey1, null))

      if (dict.nonEmpty) {
        val testKey2 = dict.keys.toSeq.head
        val expected2 = dict(testKey2)
        assertEvalsTo(invoke("get", In(0, tdict), In(1, -tdict.keyType)),
          IndexedSeq(dict -> tdict,
            testKey2 -> -tdict.keyType),
          expected2)
      }
      true
    }
    p.check()
  }

  @Test def testBinarySearchOnSet() {
    val compareGen = Type.genArb.flatMap(t => Gen.zip(Gen.const(t), TSet(t).genNonmissingValue, t.genNonmissingValue))
    val p = Prop.forAll(compareGen.filter { case (t, a, elem) => a.asInstanceOf[Set[Any]].nonEmpty }) { case (t, a, elem) =>
      val set = a.asInstanceOf[Set[Any]]
      val pset = PSet(t.physicalType)

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(pset)
        rvb.addAnnotation(pset.virtualType, set)
        val soff = rvb.end()

        rvb.start(TTuple(t).physicalType)
        rvb.addAnnotation(TTuple(t), Row(elem))
        val eoff = rvb.end()

        val fb = EmitFunctionBuilder[Region, Long, Long, Int]
        val cregion = fb.getArg[Region](1).load()
        val cset = fb.getArg[Long](2)
        val cetuple = fb.getArg[Long](3)

        val bs = new BinarySearch(fb.apply_method, pset, keyOnly = false)
        fb.emit(bs.getClosestIndex(cset, false, cregion.loadIRIntermediate(t)(TTuple(t).physicalType.fieldOffset(cetuple, 0))))

        val asArray = SafeIndexedSeq(TArray(t).physicalType, region, soff)

        val f = fb.resultWithIndex()(0)
        val closestI = f(region, soff, eoff)
        val maybeEqual = asArray(closestI)

        set.contains(elem) ==> (elem == maybeEqual) &&
          (t.ordering.compare(elem, maybeEqual) <= 0 || (closestI == set.size - 1))
      }
    }
    p.check()
  }

  @Test def testBinarySearchOnDict() {
    val compareGen = Gen.zip(Type.genArb, Type.genArb)
      .flatMap { case (k, v) => Gen.zip(Gen.const(TDict(k, v)), TDict(k, v).genNonmissingValue, k.genValue) }
    val p = Prop.forAll(compareGen.filter { case (tdict, a, key) => a.asInstanceOf[Map[Any, Any]].nonEmpty }) { case (tDict, a, key) =>
      val dict = a.asInstanceOf[Map[Any, Any]]
      val pDict = tDict.physicalType

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(pDict)
        rvb.addAnnotation(tDict, dict)
        val soff = rvb.end()

        val ptuple = PTuple(FastIndexedSeq(pDict.keyType))
        rvb.start(ptuple)
        rvb.addAnnotation(ptuple.virtualType, Row(key))
        val eoff = rvb.end()

        val fb = EmitFunctionBuilder[Region, Long, Long, Int]
        val cregion = fb.getArg[Region](1).load()
        val cdict = fb.getArg[Long](2)
        val cktuple = fb.getArg[Long](3)

        val bs = new BinarySearch(fb.apply_method, pDict, keyOnly = true)
        val m = ptuple.isFieldMissing(cregion, cktuple, 0)
        val v = cregion.loadIRIntermediate(pDict.keyType)(ptuple.fieldOffset(cktuple, 0))
        fb.emit(bs.getClosestIndex(cdict, m, v))

        val asArray = SafeIndexedSeq(PArray(pDict.elementType), region, soff)

        val f = fb.resultWithIndex()(0)
        val closestI = f(region, soff, eoff)

        def getKey(i: Int) = asArray(i).asInstanceOf[Row].get(0)

        val maybeEqual = getKey(closestI)

        val closestIIsClosest =
          (pDict.keyType.virtualType.ordering.compare(key, maybeEqual) <= 0 || closestI == dict.size - 1) &&
            (closestI == 0 || pDict.keyType.virtualType.ordering.compare(key, getKey(closestI - 1)) > 0)

        dict.contains(key) ==> (key == maybeEqual) && closestIIsClosest

      }
    }
    p.check()
  }

  @Test def testContainsWithArrayFold() {

    val set1 = ToSet(MakeArray(Seq(I32(1), I32(4)), TArray(TInt32())))
    val set2 = ToSet(MakeArray(Seq(I32(9), I32(1), I32(4)), TArray(TInt32())))
    val ir =
      ArrayFold(ToArray(set1), True(), "accumulator", "setelt",
        ApplySpecial("&&",
          FastSeq(
            Ref("accumulator", TBoolean()),
            invoke("contains", set2, Ref("setelt", TInt32())))))

    val fb = EmitFunctionBuilder[Region, Boolean]
    Emit(ir, fb)

    val f = fb.resultWithIndex()(0)
    Region.scoped { region =>
      assert(f(region))
    }
  }

  @DataProvider(name = "arrayDoubleOrderingData")
  def arrayDoubleOrderingData(): Array[Array[Any]] = {
    val xs = Array[Any](null, Double.NegativeInfinity, -0.0, 0.0, 1.0, Double.PositiveInfinity, Double.NaN)

    val as = Array(null: IndexedSeq[Any]) ++
      (for (x <- xs) yield IndexedSeq[Any](x))

    for (a <- as; a2 <- as)
      yield Array[Any](a, a2)
  }

  @Test(dataProvider = "arrayDoubleOrderingData")
  def testOrderingArrayDouble(
    a: IndexedSeq[Any], a2: IndexedSeq[Any]) {
    val t = TArray(TFloat64())

    val args = IndexedSeq(a -> t, a2 -> t)

    assertEvalSame(ApplyComparisonOp(EQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(EQWithNA(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQWithNA(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LT(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LTEQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GT(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GTEQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(Compare(t, t), In(0, t), In(1, t)), args)
  }

  @Test(dataProvider = "arrayDoubleOrderingData")
  def testOrderingSetDouble(
    a: IndexedSeq[Any], a2: IndexedSeq[Any]) {
    val t = TSet(TFloat64())

    val s = if (a != null) a.toSet else null
    val s2 = if (a2 != null) a2.toSet else null
    val args = IndexedSeq(s -> t, s2 -> t)

    assertEvalSame(ApplyComparisonOp(EQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(EQWithNA(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQWithNA(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LT(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LTEQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GT(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GTEQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(Compare(t, t), In(0, t), In(1, t)), args)
  }

  @DataProvider(name = "rowDoubleOrderingData")
  def rowDoubleOrderingData(): Array[Array[Any]] = {
    val xs = Array[Any](null, Double.NegativeInfinity, -0.0, 0.0, 1.0, Double.PositiveInfinity, Double.NaN)
    val as = Array(null: IndexedSeq[Any]) ++
      (for (x <- xs) yield IndexedSeq[Any](x))
    val ss = Array[Any](null, "a", "aa")

    val rs = for (x <- xs; s <- ss)
      yield Row(x, s)

    for (r <- rs; r2 <- rs)
      yield Array[Any](r, r2)
  }

  @Test(dataProvider = "rowDoubleOrderingData")
  def testOrderingRowDouble(
    r: Row, r2: Row) {
    val t = TStruct("x" -> TFloat64(), "s" -> TString())

    val args = IndexedSeq(r -> t, r2 -> t)

    assertEvalSame(ApplyComparisonOp(EQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(EQWithNA(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQWithNA(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LT(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LTEQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GT(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GTEQ(t, t), In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(Compare(t, t), In(0, t), In(1, t)), args)
  }
}
