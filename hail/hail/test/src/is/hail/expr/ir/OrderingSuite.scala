package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.defs.{
  ApplyComparisonOp, ApplySpecial, ArraySort, ErrorIDs, GetField, I32, In, IsNA, Literal,
  MakeStream, NA, ToArray, ToDict, ToSet, ToStream, True,
}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.scalacheck._
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.interfaces.SBaseStructValue
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.compat._
import is.hail.utils.compat.immutable.ArraySeq

import is.hail
import org.apache.spark.sql.Row
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen
import org.scalatest
import org.scalatestplus.scalacheck.CheckerAsserting.assertingNatureOfAssertion
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks
import org.testng.annotations.{DataProvider, Test}

class OrderingSuite extends HailSuite with ScalaCheckDrivenPropertyChecks {

  implicit val execStrats: hail.ExecStrategy.ValueSet = ExecStrategy.values

  def sm = ctx.stateManager

  def genTypeNonMissingVal2: Gen[(Type, Annotation, Annotation)] =
    for {
      typ <- scale(0.3, arbitrary[Type])
      a <- genNonMissing(ctx, typ)
      b <- genNonMissing(ctx, typ)
    } yield (typ, a, b)

  def recursiveSize(t: Type): Int = {
    val inner = t match {
      case ti: TInterval => recursiveSize(ti.pointType)
      case tc: TIterable => recursiveSize(tc.elementType)
      case tbs: TBaseStruct =>
        tbs.types.map(t => recursiveSize(t)).sum
      case _ => 0
    }
    inner + 1
  }

  def getStagedOrderingFunction(
    t: PType,
    op: CodeOrdering.Op,
    r: Region,
    sortOrder: SortOrder = Ascending,
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

  @Test def testMissingNonequalComparisons(): Unit = {
    def getStagedOrderingFunctionWithMissingness(
      t: PType,
      op: CodeOrdering.Op,
      r: Region,
      sortOrder: SortOrder = Ascending,
    ): AsmFunction5[Region, Boolean, Long, Boolean, Long, op.ReturnType] = {
      implicit val x = op.rtti
      val fb =
        EmitFunctionBuilder[Region, Boolean, Long, Boolean, Long, op.ReturnType](ctx, "lifted")
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

    forAll(genTypeVal[TStruct](ctx)) { case (t, a) =>
      pool.scopedRegion { region =>
        val pType = PType.canonical(t).asInstanceOf[PStruct]

        val v = pType.unstagedStoreJavaObject(sm, a, region)

        val eordME = t.mkOrdering(sm)
        val eordMNE = t.mkOrdering(sm, missingEqual = false)

        def checkCompare(compResult: Int, expected: Int): Unit =
          assert(
            java.lang.Integer.signum(compResult) == expected,
            s"compare expected: $expected vs $compResult\n  t=${t.parsableString()}\n  v=$a",
          )

        val fcompareME =
          getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Compare(), region)

        checkCompare(fcompareME(region, true, v, true, v), 0)
        checkCompare(fcompareME(region, true, v, false, v), 1)
        checkCompare(fcompareME(region, false, v, true, v), -1)

        checkCompare(eordME.compare(null, null), 0)
        checkCompare(eordME.compare(null, a), 1)
        checkCompare(eordME.compare(a, null), -1)

        val fcompareMNE =
          getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Compare(false), region)

        checkCompare(fcompareMNE(region, true, v, true, v), -1)
        checkCompare(fcompareMNE(region, true, v, false, v), 1)
        checkCompare(fcompareMNE(region, false, v, true, v), -1)

        checkCompare(eordMNE.compare(null, null), -1)
        checkCompare(eordMNE.compare(null, a), 1)
        checkCompare(eordMNE.compare(a, null), -1)

        def check(result: Boolean, expected: Boolean): Unit =
          assert(result == expected, s"t=${t.parsableString()}\n  v=$a")

        val fequivME = getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Equiv(), region)

        check(fequivME(region, true, v, true, v), true)
        check(fequivME(region, true, v, false, v), false)
        check(fequivME(region, false, v, true, v), false)

        check(eordME.equiv(null, null), true)
        check(eordME.equiv(null, a), false)
        check(eordME.equiv(a, null), false)

        val fequivMNE =
          getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Equiv(false), region)

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

        val flteqMNE =
          getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Lteq(false), region)

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

        val fgteqMNE =
          getStagedOrderingFunctionWithMissingness(pType, CodeOrdering.Gt(false), region)

        check(fgteqMNE(region, true, v, true, v), false)
        check(fgteqMNE(region, true, v, false, v), true)
        check(fgteqMNE(region, false, v, true, v), false)

        check(eordMNE.gteq(null, null), false)
        check(eordMNE.gteq(null, a), true)
        check(eordMNE.gteq(a, null), false)
      }
    }
  }

  @Test def testRandomOpsAgainstExtended(): Unit =
    forAll(genTypeNonMissingVal2) { case (t, a1, a2) =>
      pool.scopedRegion { region =>
        val pType = PType.canonical(t)

        val v1 = pType.unstagedStoreJavaObject(sm, a1, region)

        val v2 = pType.unstagedStoreJavaObject(sm, a2, region)

        val compare = java.lang.Integer.signum(t.ordering(sm).compare(a1, a2))
        val fcompare = getStagedOrderingFunction(pType, CodeOrdering.Compare(), region)
        val result = java.lang.Integer.signum(fcompare(region, v1, v2))

        assert(
          result == compare,
          s"compare expected: $compare vs $result\n  t=${t.parsableString()}\n  v1=$a1\n  v2=$a2",
        )

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
    }

  @Test def testReverseIsSwappedArgumentsOfExtendedOrdering(): Unit =
    forAll(genTypeNonMissingVal2) { case (t, a1, a2) =>
      pool.scopedRegion { region =>
        val pType = PType.canonical(t)

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
    }

  @Test def testSortOnRandomArray(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly

    forAll(genTypeVal[TArray](ctx), arbitrary[Boolean]) {
      case ((tarray, a: IndexedSeq[Any]), asc) =>
        val ord = tarray.elementType.ordering(sm)
        assertEvalsTo(
          ArraySort(ToStream(In(0, tarray)), Literal.coerce(TBoolean, asc)),
          FastSeq(a -> tarray),
          expected = a.sorted((if (asc) ord else ord.reverse).toOrdering),
        )
    }
  }

  @Test def testToSetOnRandomDuplicatedArray(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly

    forAll(genTypeVal[TArray](ctx)) { case (tarray, a: IndexedSeq[Any]) =>
      val array = a ++ a
      assertEvalsTo(
        ToArray(ToStream(ToSet(ToStream(In(0, tarray))))),
        FastSeq(array -> tarray),
        expected = array.sorted(tarray.elementType.ordering(sm).toOrdering).distinct,
      )
    }
  }

  @Test def testToDictOnRandomDuplicatedArray(): Unit = {
    implicit val execStrats: Set[ExecStrategy] =
      ExecStrategy.javaOnly

    val compareGen: Gen[(TTuple, IndexedSeq[Annotation])] =
      for {
        pkey <- smaller[PType]
        if pkey.virtualType.ordering(sm) != null
        pval <- smaller[PType]
        pelt = PCanonicalTuple(false, pkey.setRequired(true), pval)
        a <- genVal(ctx, PCanonicalArray(pelt, required = true))
        array = a.asInstanceOf[IndexedSeq[Annotation]]
      } yield (pelt.virtualType, array ++ array)

    forAll(compareGen) { case (telt, array) =>
      assertEvalsTo(
        ToArray(mapIR(ToStream(ToDict(ToStream(In(0, TArray(telt))))))(GetField(_, "key"))),
        FastSeq(array -> TArray(telt)),
        expected =
          array
            .filter(_ != null)
            .map { case Row(k, _) => k }
            .to(ArraySeq)
            .distinct
            .sorted(telt.types(0).ordering(sm).toOrdering),
      )
    }
  }

  @Test def testSortOnMissingArray(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly
    val ts = TStream(TStruct("key" -> TInt32, "value" -> TInt32))
    val irs: Array[IR => IR] = Array(ArraySort(_, True()), ToSet(_), ToDict(_))

    scalatest.Inspectors.forAll(irs)(irF => assertEvalsTo(IsNA(irF(NA(ts))), true))
  }

  @Test def testSetContainsOnRandomSet(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly
    val compareGen =
      for {
        t <- scale(0.3, arbitrary[TSet])
        s <- genNullable(ctx, t)
        a <- genNullable(ctx, t.elementType)
        if s != null && a != null
      } yield (t, s, a)

    forAll(compareGen) { case (tset: TSet, set: Set[Any] @unchecked, test1) =>
      val telt = tset.elementType

      if (set.nonEmpty) {
        assertEvalsTo(
          invoke("contains", TBoolean, In(0, tset), In(1, telt)),
          FastSeq(set -> tset, set.head -> telt),
          expected = true,
        )
      }

      assertEvalsTo(
        invoke("contains", TBoolean, In(0, tset), In(1, telt)),
        FastSeq(set -> tset, test1 -> telt),
        expected = set.contains(test1),
      )
    }
  }

  @Test def testDictGetOnRandomDict(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly

    val compareGen =
      for {
        tdict <- arbitrary[TDict]
        dict <- genNullable(ctx, tdict)
        key <- genNullable(ctx, tdict.keyType)
        if dict != null && key != null
      } yield (tdict, dict, key)

    forAll(compareGen) { case (tdict: TDict, dict: Map[Any, Any] @unchecked, testKey1) =>
      assertEvalsTo(
        invoke("get", tdict.valueType, In(0, tdict), In(1, tdict.keyType)),
        FastSeq(dict -> tdict, testKey1 -> tdict.keyType),
        dict.getOrElse(testKey1, null),
      )

      if (dict.isEmpty) scalatest.Succeeded
      else {
        val testKey2 = dict.keys.toSeq.head
        assertEvalsTo(
          invoke("get", tdict.valueType, In(0, tdict), In(1, tdict.keyType)),
          FastSeq(dict -> tdict, testKey2 -> tdict.keyType),
          expected = dict(testKey2),
        )
      }
    }
  }

  @Test def testBinarySearchOnSet(): Unit = {
    val compareGen =
      for {
        elt <- arbitrary[Type]
        set: Set[Annotation] <- genNonMissingT(ctx, TSet(elt))
        v <- genNonMissing(ctx, elt)
      } yield (elt, set, v)

    forAll(compareGen) { case (t, set, elem) =>
      val pt = PType.canonical(t)
      val pset = PCanonicalSet(pt)

      val pTuple = PCanonicalTuple(false, pt)
      val pArray = PCanonicalArray(pt)

      pool.scopedRegion { region =>
        val soff = pset.unstagedStoreJavaObject(sm, set, region)

        val eoff = pTuple.unstagedStoreJavaObject(sm, Row(elem), region)

        val fb = EmitFunctionBuilder[Region, Long, Long, Int](ctx, "binary_search")
        val cset = fb.getCodeParam[Long](2)
        val cetuple = fb.getCodeParam[Long](3)

        val bs = new BinarySearch(
          fb.apply_method,
          pset.sType,
          EmitType(pset.elementType.sType, true),
          {
            (cb, elt) => elt
          },
        )
        fb.emitWithBuilder(cb =>
          bs.search(
            cb,
            pset.loadCheapSCode(cb, cset),
            EmitCode.fromI(fb.apply_method)(cb =>
              IEmitCode.present(cb, pt.loadCheapSCode(cb, pTuple.loadField(cetuple, 0)))
            ),
          )
        )

        val asArray = SafeIndexedSeq(pArray, soff)

        val f = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, region)
        val i = f(region, soff, eoff)
        val ordering = t.ordering(sm)

        // if i-1 is in bounds, then asArray(i) < elem
        // if i is in bounds, then elem <= asArray(i)
        assert(((i - 1 < 0) || ordering.compare(asArray(i - 1), elem) < 0) &&
          ((i >= set.size) || ordering.compare(elem, asArray(i)) <= 0))
      }
    }
  }

  @Test def testBinarySearchOnDict(): Unit = {
    val compareGen =
      for {
        tdict <- arbitrary[TDict]
        dict: Map[Annotation, Annotation] <- genNonMissingT(ctx, tdict, innerRequired = false)
        key <- genNonMissing(ctx, tdict.keyType, innerRequired = false)
      } yield (tdict, dict, key)

    forAll(compareGen) { case (tDict, dict, key) =>
      val pDict = PType.canonical(tDict).asInstanceOf[PDict]

      pool.scopedRegion { region =>
        val soff = pDict.unstagedStoreJavaObject(sm, dict, region)

        val ptuple = PCanonicalTuple(false, FastSeq(pDict.keyType): _*)
        val eoff = ptuple.unstagedStoreJavaObject(sm, Row(key), region)

        val fb = EmitFunctionBuilder[Region, Long, Long, Int](ctx, "binary_search_dict")
        val cdict = fb.getCodeParam[Long](2)
        val cktuple = fb.getCodeParam[Long](3)

        val bs = new BinarySearch(
          fb.apply_method,
          pDict.sType,
          EmitType(pDict.keyType.sType, false),
          { (cb, elt) =>
            cb.memoize(elt.toI(cb).flatMap(cb) {
              case x: SBaseStructValue =>
                x.loadField(cb, 0)
            })
          },
        )

        fb.emitWithBuilder(cb =>
          bs.search(
            cb,
            pDict.loadCheapSCode(cb, cdict),
            EmitCode.fromI(fb.apply_method)(cb =>
              IEmitCode.present(cb, pDict.keyType.loadCheapSCode(cb, ptuple.loadField(cktuple, 0)))
            ),
          )
        )

        val asArray =
          SafeIndexedSeq(PCanonicalArray(pDict.elementType), soff).map(_.asInstanceOf[Row])

        val f = fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, region)
        val i = f(region, soff, eoff)
        val ordering = pDict.keyType.virtualType.ordering(sm)

        // if i-1 is in bounds, then asArray(i).getKey < key
        // if i is in bounds, then key <= asArray(i).getKey
        assert(((i - 1 < 0) || ordering.compare(asArray(i - 1).get(0), key) < 0) &&
          ((i >= asArray.size) || ordering.compare(key, asArray(i).get(0)) <= 0))
      }
    }
  }

  @Test def testContainsWithArrayFold(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly
    val set1 = ToSet(MakeStream(IndexedSeq(I32(1), I32(4)), TStream(TInt32)))
    val set2 = ToSet(MakeStream(IndexedSeq(I32(9), I32(1), I32(4)), TStream(TInt32)))
    assertEvalsTo(
      foldIR(ToStream(set1), True()) { (acc, elt) =>
        ApplySpecial(
          "land",
          FastSeq(),
          FastSeq(
            acc,
            invoke("contains", TBoolean, set2, elt),
          ),
          TBoolean,
          ErrorIDs.NO_ERROR,
        )
      },
      true,
    )
  }

  @DataProvider(name = "arrayDoubleOrderingData")
  def arrayDoubleOrderingData(): Array[Array[Any]] = {
    val xs =
      Array[Any](null, Double.NegativeInfinity, -0.0, 0.0, 1.0, Double.PositiveInfinity, Double.NaN)

    val as = Array(null: IndexedSeq[Any]) ++
      (for (x <- xs) yield IndexedSeq[Any](x))

    for {
      a <- as
      a2 <- as
    } yield Array[Any](a, a2)
  }

  @Test(dataProvider = "arrayDoubleOrderingData")
  def testOrderingArrayDouble(
    a: IndexedSeq[Any],
    a2: IndexedSeq[Any],
  ): Unit = {
    val t = TArray(TFloat64)

    val args = FastSeq(a -> t, a2 -> t)

    assertEvalSame(ApplyComparisonOp(EQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(EQWithNA, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQWithNA, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LT, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LTEQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GT, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GTEQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(Compare, In(0, t), In(1, t)), args)
  }

  @Test(dataProvider = "arrayDoubleOrderingData")
  def testOrderingSetDouble(
    a: IndexedSeq[Any],
    a2: IndexedSeq[Any],
  ): Unit = {
    val t = TSet(TFloat64)

    val s = if (a != null) a.toSet else null
    val s2 = if (a2 != null) a2.toSet else null
    val args = FastSeq(s -> t, s2 -> t)

    assertEvalSame(ApplyComparisonOp(EQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(EQWithNA, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQWithNA, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LT, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LTEQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GT, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GTEQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(Compare, In(0, t), In(1, t)), args)
  }

  @DataProvider(name = "rowDoubleOrderingData")
  def rowDoubleOrderingData(): Array[Array[Any]] = {
    val xs =
      Array[Any](null, Double.NegativeInfinity, -0.0, 0.0, 1.0, Double.PositiveInfinity, Double.NaN)
    val ss = Array[Any](null, "a", "aa")

    val rs = for {
      x <- xs
      s <- ss
    } yield Row(x, s)

    for {
      r <- rs
      r2 <- rs
    } yield Array[Any](r, r2)
  }

  @Test(dataProvider = "rowDoubleOrderingData")
  def testOrderingRowDouble(
    r: Row,
    r2: Row,
  ): Unit = {
    val t = TStruct("x" -> TFloat64, "s" -> TString)

    val args = FastSeq(r -> t, r2 -> t)

    assertEvalSame(ApplyComparisonOp(EQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(EQWithNA, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(NEQWithNA, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LT, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(LTEQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GT, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(GTEQ, In(0, t), In(1, t)), args)
    assertEvalSame(ApplyComparisonOp(Compare, In(0, t), In(1, t)), args)
  }
}
