package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.HailSuite
import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.check.{Gen, Prop}
import is.hail.asm4s._
import is.hail.TestUtils._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.rvd.RVDType
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.interfaces.SBaseStructValue
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class OrderingSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.values

  def sm = ctx.stateManager

  def recursiveSize(t: Type): Int = {
    val inner = t match {
      case ti: TInterval => recursiveSize(ti.pointType)
      case tc: TIterable => recursiveSize(tc.elementType)
      case tbs: TBaseStruct =>
        tbs.types.map { t => recursiveSize(t) }.sum
      case _ => 0
    }
    inner + 1
  }

  def getStagedOrderingFunction(
    t: PType,
    op: CodeOrdering.Op,
    r: Region,
    sortOrder: SortOrder = Ascending
  ): AsmFunction3[Region, Long, Long, op.ReturnType] = {
    implicit val x = op.rtti
    val fb = EmitFunctionBuilder[Region, Long, Long, op.ReturnType](ctx, "lifted")
    fb.emitWithBuilder { cb =>
      val cv1 = t.loadCheapSCode(cb, fb.getCodeParam[Long](2))
      val cv2 = t.loadCheapSCode(cb, fb.getCodeParam[Long](3))
      fb.ecb.getOrderingFunction(cv1.st, cv2.st, op)
          .apply(cb, EmitValue.present(cv1), EmitValue.present(cv2))
    }
    fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)
  }

  @Test def testMissingNonequalComparisons() {
    def getStagedOrderingFunctionWithMissingness(
      t: PType,
      op: CodeOrdering.Op,
      r: Region,
      sortOrder: SortOrder = Ascending
    ): AsmFunction5[Region, Boolean, Long, Boolean, Long, op.ReturnType] = {
      implicit val x = op.rtti
      val fb = EmitFunctionBuilder[Region, Boolean, Long, Boolean, Long, op.ReturnType](ctx, "lifted")
      fb.emitWithBuilder { cb =>
        val m1 = fb.getCodeParam[Boolean](2)
        val cv1 = t.loadCheapSCode(cb, fb.getCodeParam[Long](3))
        val m2 = fb.getCodeParam[Boolean](4)
        val cv2 = t.loadCheapSCode(cb, fb.getCodeParam[Long](5))
        val ev1 = EmitValue(Some(m1), cv1)
        val ev2 = EmitValue(Some(m2), cv2)
        fb.ecb.getOrderingFunction(ev1.st, ev2.st, op)
          .apply(cb, ev1, ev2)
      }
      fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)
    }

    val compareGen = for {
      t <- Type.genStruct
      a <- t.genNonmissingValue(sm)
    } yield (t, a)
    val p = Prop.forAll(compareGen) { case (t, a) => pool.scopedRegion { region =>
      val pType = PType.canonical(t).asInstanceOf[PStruct]
      val rvb = new RegionValueBuilder(sm, region)

      val v = pType.unstagedStoreJavaObject(sm, a, region)

      val eordME = t.mkOrdering(sm)
      val eordMNE = t.mkOrdering(sm, missingEqual = false)

      def checkCompare(compResult: Int, expected: Int) {
        assert(java.lang.Integer.signum(compResult) == expected,
               s"compare expected: $expected vs $compResult\n  t=${t.parsableString()}\n  v=$a")
      }

      val fcompareME = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Compare(), region)

      checkCompare(fcompareME(region, true, v, true, v), 0)
      checkCompare(fcompareME(region, true, v, false, v), 1)
      checkCompare(fcompareME(region, false, v, true, v), -1)

      checkCompare(eordME.compare(null, null), 0)
      checkCompare(eordME.compare(null, a), 1)
      checkCompare(eordME.compare(a, null), -1)

      val fcompareMNE = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Compare(false), region)

      checkCompare(fcompareMNE(region, true, v, true, v), -1)
      checkCompare(fcompareMNE(region, true, v, false, v), 1)
      checkCompare(fcompareMNE(region, false, v, true, v), -1)

      checkCompare(eordMNE.compare(null, null), -1)
      checkCompare(eordMNE.compare(null, a), 1)
      checkCompare(eordMNE.compare(a, null), -1)

      def check(result: Boolean, expected: Boolean) {
        assert(result == expected, s"t=${t.parsableString()}\n  v=$a")
      }

      val fequivME = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Equiv(), region)

      check(fequivME(region, true, v, true, v), true)
      check(fequivME(region, true, v, false, v), false)
      check(fequivME(region, false, v, true, v), false)

      check(eordME.equiv(null, null), true)
      check(eordME.equiv(null, a), false)
      check(eordME.equiv(a, null), false)

      val fequivMNE = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Equiv(false), region)

      check(fequivMNE(region, true, v, true, v), false)
      check(fequivMNE(region, true, v, false, v), false)
      check(fequivMNE(region, false, v, true, v), false)

      check(eordMNE.equiv(null, null), false)
      check(eordMNE.equiv(null, a), false)
      check(eordMNE.equiv(a, null), false)

      val fltME = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Lt(), region)

      check(fltME(region, true, v, true, v), false)
      check(fltME(region, true, v, false, v), false)
      check(fltME(region, false, v, true, v), true)

      check(eordME.lt(null, null), false)
      check(eordME.lt(null, a), false)
      check(eordME.lt(a, null), true)

      val fltMNE = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Lt(false), region)

      check(fltMNE(region, true, v, true, v), true)
      check(fltMNE(region, true, v, false, v), false)
      check(fltMNE(region, false, v, true, v), true)

      check(eordMNE.lt(null, null), true)
      check(eordMNE.lt(null, a), false)
      check(eordMNE.lt(a, null), true)

      val flteqME = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Lteq(), region)

      check(flteqME(region, true, v, true, v), true)
      check(flteqME(region, true, v, false, v), false)
      check(flteqME(region, false, v, true, v), true)

      check(eordME.lteq(null, null), true)
      check(eordME.lteq(null, a), false)
      check(eordME.lteq(a, null), true)

      val flteqMNE = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Lteq(false), region)

      check(flteqMNE(region, true, v, true, v), true)
      check(flteqMNE(region, true, v, false, v), false)
      check(flteqMNE(region, false, v, true, v), true)

      check(eordMNE.lteq(null, null), true)
      check(eordMNE.lteq(null, a), false)
      check(eordMNE.lteq(a, null), true)

      val fgtME = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Gt(), region)

      check(fgtME(region, true, v, true, v), false)
      check(fgtME(region, true, v, false, v), true)
      check(fgtME(region, false, v, true, v), false)

      check(eordME.gt(null, null), false)
      check(eordME.gt(null, a), true)
      check(eordME.gt(a, null), false)

      val fgtMNE = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Gt(false), region)

      check(fgtMNE(region, true, v, true, v), false)
      check(fgtMNE(region, true, v, false, v), true)
      check(fgtMNE(region, false, v, true, v), false)

      check(eordMNE.gt(null, null), false)
      check(eordMNE.gt(null, a), true)
      check(eordMNE.gt(a, null), false)

      val fgteqME = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Gteq(), region)

      check(fgteqME(region, true, v, true, v), true)
      check(fgteqME(region, true, v, false, v), true)
      check(fgteqME(region, false, v, true, v), false)

      check(eordME.gteq(null, null), true)
      check(eordME.gteq(null, a), true)
      check(eordME.gteq(a, null), false)

      val fgteqMNE = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Gt(false), region)

      check(fgteqMNE(region, true, v, true, v), false)
      check(fgteqMNE(region, true, v, false, v), true)
      check(fgteqMNE(region, false, v, true, v), false)

      check(eordMNE.gteq(null, null), false)
      check(eordMNE.gteq(null, a), true)
      check(eordMNE.gteq(a, null), false)

      true
    }}

    p.check()
  }

  @Test def testRandomOpsAgainstExtended() {
    val compareGen = for {
      t <- Type.genArb
      a1 <- t.genNonmissingValue(sm)
      a2 <- t.genNonmissingValue(sm)
    } yield (t, a1, a2)
    val p = Prop.forAll(compareGen) { case (t, a1, a2) =>
      pool.scopedRegion { region =>
        val pType = PType.canonical(t)
        val rvb = new RegionValueBuilder(sm, region)

        val v1 = pType.unstagedStoreJavaObject(sm, a1, region)

        val v2 = pType.unstagedStoreJavaObject(sm, a2, region)

        val compare = java.lang.Integer.signum(t.ordering(sm).compare(a1, a2))
        val fcompare = getStagedOrderingFunction(pType, CodeOrdering.Compare(), region)
        val result = java.lang.Integer.signum(fcompare(region, v1, v2))

        assert(result == compare, s"compare expected: $compare vs $result\n  t=${t.parsableString()}\n  v1=${a1}\n  v2=$a2")

        val equiv = t.ordering(sm).equiv(a1, a2)
        val fequiv = getStagedOrderingFunction(pType, CodeOrdering.Equiv(), region)

        assert(fequiv(region, v1, v2) == equiv, s"equiv expected: $equiv")

        val lt = t.ordering(sm).lt(a1, a2)
        val flt = getStagedOrderingFunction(pType, CodeOrdering.Lt(), region)

        assert(flt(region, v1, v2) == lt, s"lt expected: $lt")

        val lteq = t.ordering(sm).lteq(a1, a2)
        val flteq = getStagedOrderingFunction(pType, CodeOrdering.Lteq(), region)

        assert(flteq(region, v1, v2) == lteq, s"lteq expected: $lteq")

        val gt = t.ordering(sm).gt(a1, a2)
        val fgt = getStagedOrderingFunction(pType, CodeOrdering.Gt(), region)

        assert(fgt(region, v1, v2) == gt, s"gt expected: $gt")

        val gteq = t.ordering(sm).gteq(a1, a2)
        val fgteq = getStagedOrderingFunction(pType, CodeOrdering.Gteq(), region)

        assert(fgteq(region, v1, v2) == gteq, s"gteq expected: $gteq")
      }

      true
    }
    p.check()
  }

  @Test def testReverseIsSwappedArgumentsOfExtendedOrdering() {
    val compareGen = for {
      t <- Type.genArb
      a1 <- t.genNonmissingValue(sm)
      a2 <- t.genNonmissingValue(sm)
    } yield (t, a1, a2)
    val p = Prop.forAll(compareGen) { case (t, a1, a2) =>
      pool.scopedRegion { region =>
        val pType = PType.canonical(t)
        val rvb = new RegionValueBuilder(sm, region)

        val v1 = pType.unstagedStoreJavaObject(sm, a1, region)

        val v2 = pType.unstagedStoreJavaObject(sm, a2, region)

        val reversedExtendedOrdering = t.ordering(sm).reverse

        val compare = java.lang.Integer.signum(reversedExtendedOrdering.compare(a2, a1))
        val fcompare = getStagedOrderingFunction(pType, CodeOrdering.Compare(), region, Descending)
        val result = java.lang.Integer.signum(fcompare(region, v1, v2))

        assert(result == compare, s"compare expected: $compare vs $result")


        val equiv = reversedExtendedOrdering.equiv(a2, a1)
        val fequiv = getStagedOrderingFunction(pType, CodeOrdering.Equiv(), region, Descending)

        assert(fequiv(region, v1, v2) == equiv, s"equiv expected: $equiv")

        val lt = reversedExtendedOrdering.lt(a2, a1)
        val flt = getStagedOrderingFunction(pType, CodeOrdering.Lt(), region, Descending)

        assert(flt(region, v1, v2) == lt, s"lt expected: $lt")

        val lteq = reversedExtendedOrdering.lteq(a2, a1)
        val flteq = getStagedOrderingFunction(pType, CodeOrdering.Lteq(), region, Descending)

        assert(flteq(region, v1, v2) == lteq, s"lteq expected: $lteq")

        val gt = reversedExtendedOrdering.gt(a2, a1)
        val fgt = getStagedOrderingFunction(pType, CodeOrdering.Gt(), region, Descending)

        assert(fgt(region, v1, v2) == gt, s"gt expected: $gt")

        val gteq = reversedExtendedOrdering.gteq(a2, a1)
        val fgteq = getStagedOrderingFunction(pType, CodeOrdering.Gteq(), region, Descending)

        assert(fgteq(region, v1, v2) == gteq, s"gteq expected: $gteq")
      }

      true
    }
    p.check()
  }

  @Test def testSortOnRandomArray() {
    implicit val execStrats = ExecStrategy.javaOnly
    val compareGen = for {
      elt <- Type.genArb
      a <- TArray(elt).genNonmissingValue(sm)
      asc <- Gen.coin()
    } yield (elt, a, asc)
    val p = Prop.forAll(compareGen) { case (t, a: IndexedSeq[Any], asc: Boolean) =>
      val ord = if (asc) t.ordering(sm).toOrdering else t.ordering(sm).reverse.toOrdering
      assertEvalsTo(ArraySort(ToStream(In(0, TArray(t))), Literal.coerce(TBoolean, asc)),
        FastIndexedSeq(a -> TArray(t)),
        expected = a.sorted(ord))
      true
    }
    p.check()
  }

  def testToSetOnRandomDuplicatedArray() {
    implicit val execStrats = ExecStrategy.javaOnly
    val compareGen = for {
      elt <- Type.genArb
      a <- TArray(elt).genNonmissingValue(sm)
    } yield (elt, a)
    val p = Prop.forAll(compareGen) { case (t, a: IndexedSeq[Any]) =>
      val array = a ++ a
      assertEvalsTo(ToArray(ToSet(In(0, TArray(t)))),
        FastIndexedSeq(array -> TArray(t)),
        expected = array.sorted(t.ordering(sm).toOrdering).distinct)
      true
    }
    p.check()
  }

  def testToDictOnRandomDuplicatedArray() {
    implicit val execStrats = ExecStrategy.javaOnly
    val compareGen = for {
      kt <- Type.genArb
      vt <- Type.genArb
      telt = TTuple(kt, vt)
      a <- TArray(telt).genNonmissingValue(sm)
    } yield (telt, a)
    val p = Prop.forAll(compareGen) { case (telt: TTuple, a: IndexedSeq[Row]@unchecked) =>
      val tdict = TDict(telt.types(0), telt.types(1))
      val array: IndexedSeq[Row] = a ++ a
      val expectedMap = array.filter(_ != null).map { case Row(k, v) => (k, v) }.toMap
      assertEvalsTo(
        ToArray(StreamMap(ToStream(In(0, TArray(telt))),
        "x", GetField(Ref("x", tdict.elementType), "key"))),
        FastIndexedSeq(array -> TArray(telt)),
        expected = expectedMap.keys.toFastIndexedSeq.sorted(telt.types(0).ordering(sm).toOrdering))
      true
    }
    p.check()
  }

  @Test def testSortOnMissingArray() {
    implicit val execStrats = ExecStrategy.javaOnly
    val ts = TStream(TStruct("key" -> TInt32, "value" -> TInt32))
    val irs: Array[IR => IR] = Array(ArraySort(_, True()), ToSet(_), ToDict(_))

    for (irF <- irs) { assertEvalsTo(IsNA(irF(NA(ts))), true) }
  }

  @Test def testSetContainsOnRandomSet() {
    implicit val execStrats = ExecStrategy.javaOnly
    val compareGen = Type.genArb
      .flatMap(t => Gen.zip(Gen.const(TSet(t)), TSet(t).genNonmissingValue(sm), t.genValue(sm)))
    val p = Prop.forAll(compareGen) { case (tset: TSet, set: Set[Any]@unchecked, test1) =>
      val telt = tset.elementType

      if (set.nonEmpty) {
        assertEvalsTo(
          invoke("contains", TBoolean, In(0, tset), In(1, telt)),
          FastIndexedSeq(set -> tset, set.head -> telt),
          expected = true)
      }

      assertEvalsTo(
        invoke("contains", TBoolean, In(0, tset), In(1, telt)),
        FastIndexedSeq(set -> tset, test1 -> telt),
        expected = set.contains(test1))
      true
    }
    p.check()
  }

  def testDictGetOnRandomDict() {
    implicit val execStrats = ExecStrategy.javaOnly

    val compareGen = Gen.zip(Type.genArb, Type.genArb).flatMap {
      case (k, v) =>
        Gen.zip(Gen.const(TDict(k, v)), TDict(k, v).genNonmissingValue(sm), k.genNonmissingValue(sm))
    }
    val p = Prop.forAll(compareGen) { case (tdict: TDict, dict: Map[Any, Any]@unchecked, testKey1) =>
      assertEvalsTo(invoke("get", tdict.valueType, In(0, tdict), In(1, tdict.keyType)),
        FastIndexedSeq(dict -> tdict,
          testKey1 -> tdict.keyType),
        dict.getOrElse(testKey1, null))

      if (dict.nonEmpty) {
        val testKey2 = dict.keys.toSeq.head
        val expected2 = dict(testKey2)
        assertEvalsTo(invoke("get", tdict.valueType, In(0, tdict), In(1, tdict.keyType)),
          FastIndexedSeq(dict -> tdict,
            testKey2 -> tdict.keyType),
          expected2)
      }
      true
    }
    p.check()
  }

  def testBinarySearchOnSet() {
    val compareGen = Type.genArb.flatMap(t => Gen.zip(Gen.const(t), TSet(t).genNonmissingValue(sm), t.genNonmissingValue(sm)))
    val p = Prop.forAll(compareGen.filter { case (t, a, elem) => a.asInstanceOf[Set[Any]].nonEmpty }) { case (t, a, elem) =>
      val set = a.asInstanceOf[Set[Any]]
      val pt = PType.canonical(t)
      val pset = PCanonicalSet(pt)

      val pTuple = PCanonicalTuple(false, pt)
      val pArray = PCanonicalArray(pt)

      pool.scopedRegion { region =>
        val rvb = new RegionValueBuilder(sm, region)

        val soff = pset.unstagedStoreJavaObject(sm, set, region)

        val eoff = pTuple.unstagedStoreJavaObject(sm, Row(elem), region)

        val fb = EmitFunctionBuilder[Region, Long, Long, Int](ctx, "binary_search")
        val cregion = fb.getCodeParam[Region](1).load()
        val cset = fb.getCodeParam[Long](2)
        val cetuple = fb.getCodeParam[Long](3)

        val bs = new BinarySearch(fb.apply_method, pset.sType, EmitType(pset.elementType.sType, true), {
          (cb, elt) => elt
        })
        fb.emitWithBuilder(cb =>
          bs.search(cb, pset.loadCheapSCode(cb, cset),
            EmitCode.fromI(fb.apply_method)(cb => IEmitCode.present(cb, pt.loadCheapSCode(cb, pTuple.loadField(cetuple, 0))))))

        val asArray = SafeIndexedSeq(pArray, soff)

        val f = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, region)
        val closestI = f(region, soff, eoff)
        val maybeEqual = asArray(closestI)

        set.contains(elem) ==> (elem == maybeEqual) &&
          (t.ordering(sm).compare(elem, maybeEqual) <= 0 || (closestI == set.size - 1))
      }
    }
    p.check()
  }

  @Test def testBinarySearchOnDict() {
    val compareGen = Gen.zip(Type.genArb, Type.genArb)
      .flatMap { case (k, v) => Gen.zip(Gen.const(TDict(k, v)), TDict(k, v).genNonmissingValue(sm), k.genValue(sm)) }
    val p = Prop.forAll(compareGen.filter { case (tdict, a, key) => a.asInstanceOf[Map[Any, Any]].nonEmpty }) { case (tDict, a, key) =>
      val dict = a.asInstanceOf[Map[Any, Any]]
      val pDict = PType.canonical(tDict).asInstanceOf[PDict]

      pool.scopedRegion { region =>
        val soff = pDict.unstagedStoreJavaObject(sm, dict, region)

        val ptuple = PCanonicalTuple(false, FastIndexedSeq(pDict.keyType): _*)
        val eoff = ptuple.unstagedStoreJavaObject(sm, Row(key), region)

        val fb = EmitFunctionBuilder[Region, Long, Long, Int](ctx, "binary_search_dict")
        val cdict = fb.getCodeParam[Long](2)
        val cktuple = fb.getCodeParam[Long](3)

        val bs = new BinarySearch(fb.apply_method, pDict.sType, EmitType(pDict.keyType.sType, false), { (cb, elt) =>
          cb.memoize(elt.toI(cb).flatMap(cb) {
            case x: SBaseStructValue =>
              x.loadField(cb, 0)
          })
        })

        fb.emitWithBuilder(cb =>
          bs.search(cb, pDict.loadCheapSCode(cb, cdict),
            EmitCode.fromI(fb.apply_method)(cb => IEmitCode.present(cb, pDict.keyType.loadCheapSCode(cb, ptuple.loadField(cktuple, 0))))))

        val asArray = SafeIndexedSeq(PCanonicalArray(pDict.elementType), soff)

        val f = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, region)
        val closestI = f(region, soff, eoff)

        if (closestI == asArray.length) {
          !dict.contains(key) ==> asArray.forall { keyI =>
            val otherKey = keyI.asInstanceOf[Row].get(0)
            pDict.keyType.virtualType.ordering(sm).compare(key, otherKey) > 0
          }
        } else {
          def getKey(i: Int) = asArray(i).asInstanceOf[Row].get(0)
          val maybeEqual = getKey(closestI)
          val closestIIsClosest =
            (pDict.keyType.virtualType.ordering(sm).compare(key, maybeEqual) <= 0 || closestI == dict.size - 1) &&
              (closestI == 0 || pDict.keyType.virtualType.ordering(sm).compare(key, getKey(closestI - 1)) > 0)

          // FIXME: -0.0 and 0.0 count as the same in scala Map, but not off-heap Hail data structures
          val kord = tDict.keyType.ordering(sm)
          (dict.contains(key) && dict.keysIterator.exists(kord.compare(_, key) == 0)) ==> (key == maybeEqual) && closestIIsClosest
        }
      }
    }
    p.check()
  }

  @Test def testContainsWithArrayFold() {
    implicit val execStrats = ExecStrategy.javaOnly
    val set1 = ToSet(MakeStream(IndexedSeq(I32(1), I32(4)), TStream(TInt32)))
    val set2 = ToSet(MakeStream(IndexedSeq(I32(9), I32(1), I32(4)), TStream(TInt32)))
    assertEvalsTo(StreamFold(ToStream(set1), True(), "accumulator", "setelt",
        ApplySpecial("land",
          FastSeq(),
          FastSeq(
            Ref("accumulator", TBoolean),
            invoke("contains", TBoolean, set2, Ref("setelt", TInt32))), TBoolean, ErrorIDs.NO_ERROR)), true)
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
    val t = TArray(TFloat64)

    val args = FastIndexedSeq(a -> t, a2 -> t)

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
    val t = TSet(TFloat64)

    val s = if (a != null) a.toSet else null
    val s2 = if (a2 != null) a2.toSet else null
    val args = FastIndexedSeq(s -> t, s2 -> t)

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
      (for (x <- xs) yield FastIndexedSeq[Any](x))
    val ss = Array[Any](null, "a", "aa")

    val rs = for (x <- xs; s <- ss)
      yield Row(x, s)

    for (r <- rs; r2 <- rs)
      yield Array[Any](r, r2)
  }

  @Test(dataProvider = "rowDoubleOrderingData")
  def testOrderingRowDouble(
    r: Row, r2: Row) {
    val t = TStruct("x" -> TFloat64, "s" -> TString)

    val args = FastIndexedSeq(r -> t, r2 -> t)

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
