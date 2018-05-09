package is.hail.expr.ir

import is.hail.annotations._
import is.hail.check.{Gen, Prop}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class OrderingSuite {

  def recursiveSize(t: Type): Int = {
    val inner = t match {
      case ti: TInterval => recursiveSize(ti.pointType)
      case tc: TContainer => recursiveSize(tc.elementType)
      case tbs: TBaseStruct =>
        tbs.types.map{ t => recursiveSize(t) }.sum
      case _ => 0
    }
    inner + 1
  }

  def getStagedOrderingFunction[T: TypeInfo](t: Type, comp: String): AsmFunction3[Region, Long, Long, T] = {
    val fb = EmitFunctionBuilder[Region, Long, Long, T]
    val stagedOrdering = t.codeOrdering(fb.apply_method)
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

    fb.result()()
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

        rvb.start(t)
        rvb.addAnnotation(t, a1)
        val v1 = rvb.end()

        rvb.start(t)
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
    } yield (elt, a)
    val p = Prop.forAll(compareGen) { case (t, a) =>
      val ir = ArraySort(GetTupleElement(In(0, TTuple(TArray(t))), 0))
      val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]

      Emit(ir, fb)

      val f = fb.result()()

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(TTuple(TArray(t)))
        rvb.startTuple()
        rvb.addAnnotation(TArray(t), a)
        rvb.endTuple()
        val off = rvb.end()

        val res = f(region, off, false)
        val actual = SafeIndexedSeq(TArray(t), region, res)
        val expected = a.asInstanceOf[IndexedSeq[Any]].sorted(t.ordering.toOrdering)
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
    val p = Prop.forAll(compareGen) { case (t, a) =>
      val array = a.asInstanceOf[IndexedSeq[Any]] ++ a.asInstanceOf[IndexedSeq[Any]]
      val ir = ToSet(GetTupleElement(In(0, TTuple(TArray(t))), 0))
      val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]

      Emit(ir, fb)

      val f = fb.result()()

      Region.scoped[Boolean] { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(TTuple(TArray(t)))
        rvb.startTuple()
        rvb.addAnnotation(TArray(t), array)
        rvb.endTuple()
        val off = rvb.end()

        val res = f(region, off, false)
        val actual = SafeIndexedSeq(TArray(t), region, res)
        val expected = a.asInstanceOf[IndexedSeq[Any]].sorted(t.ordering.toOrdering).distinct

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
    val p = Prop.forAll(compareGen) { case (telt: TTuple, a) =>
      val array: IndexedSeq[Row] = a.asInstanceOf[IndexedSeq[Row]] ++ a.asInstanceOf[IndexedSeq[Row]]
      val ir = ToDict(GetTupleElement(In(0, TTuple(TArray(telt))), 0))
      val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]

      Emit(ir, fb)

      val f = fb.result()()

      Region.scoped[Boolean] { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(TTuple(TArray(telt)))
        rvb.startTuple()
        rvb.addAnnotation(TArray(telt), array)
        rvb.endTuple()
        val off = rvb.end()

        val res = f(region, off, false)
        val actual = SafeIndexedSeq(TArray(telt), region, res)
        val actualKeys = actual.filter(_ != null).map { case Row(k, _) => k }
        val expectedMap = array.filter(_ != null).map { case Row(k, v) => (k, v) }.toMap
        val expectedKeys = expectedMap.keys.toIndexedSeq.sorted(telt.types(0).ordering.toOrdering)

        expectedKeys == actualKeys
      }
    }
    p.check()
  }

  @Test def testSortOnMissingArray() {
    val tarray = TArray(TStruct("key" -> TInt32(), "value" -> TInt32()))
    val irs: Array[IR => IR] = Array(ArraySort(_), ToSet(_), ToDict(_))

    for (irF <- irs) {
      val ir = IsNA(irF(NA(tarray)))
      val fb = EmitFunctionBuilder[Region, Long, Boolean, Boolean]
      Emit(ir, fb)

      val f = fb.result()()
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(tarray)
        rvb.startArray(0)
        rvb.endArray()
        val off = rvb.end()

        assert(f(region, off, false))
      }
    }
  }

  @Test def testSetContainsOnRandomSet() {
    val compareGen = Type.genArb
      .flatMap(t => Gen.zip(Gen.const(TSet(t)), TSet(t).genNonmissingValue, t.genValue))
    val p = Prop.forAll(compareGen) { case (tset, a, testElem1) =>
      val telt = tset.elementType
      val set: Set[Any] = a.asInstanceOf[Set[Any]]
      val ir = SetContains(GetTupleElement(In(0, TTuple(tset)), 0),
          GetTupleElement(In(1, TTuple(telt)), 0))
      val fb = EmitFunctionBuilder[Region, Long, Boolean, Long, Boolean, Boolean]

      Emit(ir, fb)

      val f = fb.result()()

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(TTuple(tset))
        rvb.startTuple()
        rvb.addAnnotation(tset, set)
        rvb.endTuple()
        val doff = rvb.end()

        rvb.start(TTuple(telt))
        rvb.startTuple()
        rvb.addAnnotation(telt, testElem1)
        rvb.endTuple()
        val k1off = rvb.end()

        if (set.nonEmpty) {
          val testElem2 = set.head

          rvb.start(TTuple(telt))
          rvb.startTuple()
          rvb.addAnnotation(telt, testElem2)
          rvb.endTuple()
          val k2off = rvb.end()

          val expected2 = set.contains(testElem2)
          val actual2 = f(region, doff, false, k2off, false)
          assert(expected2 == actual2)
        }

        val expected1 = set.contains(testElem1)
        val actual1 = f(region, doff, false, k1off, false)

        expected1 == actual1
      }
    }
    p.check()
  }

  @Test def testDictGetOnRandomDict() {
    val compareGen = Gen.zip(Type.genArb, Type.genArb).flatMap {
      case (k, v) => Gen.zip(Gen.const(TDict(k, v)), TDict(k, v).genNonmissingValue, k.genValue)
    }
    val p = Prop.forAll(compareGen) { case (tdict, a, testKey1) =>
      val telt = coerce[TBaseStruct](tdict.elementType)
      val dict: Map[Any, Any] = a.asInstanceOf[Map[Any, Any]]
      val ir = MakeTuple(Seq(
        DictGet(GetTupleElement(In(0, TTuple(tdict)), 0),
          GetTupleElement(In(1, TTuple(telt.types(0))), 0))))
      val fb = EmitFunctionBuilder[Region, Long, Boolean, Long, Boolean, Long]

      Emit(ir, fb)

      val f = fb.result()()

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(TTuple(tdict))
        rvb.startTuple()
        rvb.addAnnotation(tdict, dict)
        rvb.endTuple()
        val doff = rvb.end()

        rvb.start(TTuple(telt.types(0)))
        rvb.startTuple()
        rvb.addAnnotation(telt.types(0), testKey1)
        rvb.endTuple()
        val k1off = rvb.end()

        if (dict.nonEmpty) {
          val testKey2 = dict.keys.toSeq.head

          rvb.start(TTuple(telt.types(0)))
          rvb.startTuple()
          rvb.addAnnotation(telt.types(0), testKey2)
          rvb.endTuple()
          val k2off = rvb.end()

          val expected2 = dict(testKey2)
          val actual2 = SafeRow(TTuple(-telt.types(1)), region, f(region, doff, false, k2off, false)).get(0)
          assert(expected2 == actual2)
        }

        val expected1 = dict.getOrElse(testKey1, null)
        val actual1 = SafeRow(TTuple(-telt.types(1)), region, f(region, doff, false, k1off, false)).get(0)

        expected1 == actual1
      }
    }
    p.check()
  }

  @Test def testBinarySearchOnSet() {
    val compareGen = Type.genArb.flatMap(t => Gen.zip(Gen.const(t), TSet(t).genNonmissingValue, t.genNonmissingValue))
    val p = Prop.forAll(compareGen.filter { case (t, a, elem) => a.asInstanceOf[Set[Any]].nonEmpty }) { case (t, a, elem) =>
      val set = a.asInstanceOf[Set[Any]]
      val tset = TSet(t)

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(tset)
        rvb.addAnnotation(tset, set)
        val soff = rvb.end()

        rvb.start(TTuple(t))
        rvb.addAnnotation(TTuple(t), Row(elem))
        val eoff = rvb.end()

        val fb = EmitFunctionBuilder[Region, Long, Long, Int]
        val cregion = fb.getArg[Region](1).load()
        val cset = fb.getArg[Long](2)
        val cetuple = fb.getArg[Long](3)

        val bs = new BinarySearch(fb.apply_method, tset, keyOnly = false)
        fb.emit(bs.getClosestIndex(cset, false, cregion.loadIRIntermediate(t)(TTuple(t).fieldOffset(cetuple, 0))))

        val asArray = SafeIndexedSeq(TArray(t), region, soff)

        val f = fb.result()()
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
    val p = Prop.forAll(compareGen.filter { case (tdict, a, key) => a.asInstanceOf[Map[Any, Any]].nonEmpty }) { case (tdict, a, key) =>
      val dict = a.asInstanceOf[Map[Any, Any]]

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(tdict)
        rvb.addAnnotation(tdict, dict)
        val soff = rvb.end()

        rvb.start(TTuple(tdict.keyType))
        rvb.addAnnotation(TTuple(tdict.keyType), Row(key))
        val eoff = rvb.end()

        val fb = EmitFunctionBuilder[Region, Long, Long, Int]
        val cregion = fb.getArg[Region](1).load()
        val cdict = fb.getArg[Long](2)
        val cktuple = fb.getArg[Long](3)

        val bs = new BinarySearch(fb.apply_method, tdict, keyOnly = true)
        val m = TTuple(tdict.keyType).isFieldMissing(cregion, cktuple, 0)
        val v = cregion.loadIRIntermediate(tdict.keyType)(TTuple(tdict.keyType).fieldOffset(cktuple, 0))
        fb.emit(bs.getClosestIndex(cdict, m, v))

        val asArray = SafeIndexedSeq(TArray(tdict.elementType), region, soff)

        val f = fb.result()()
        val closestI = f(region, soff, eoff)
        def getKey(i: Int) = asArray(i).asInstanceOf[Row].get(0)
        val maybeEqual = getKey(closestI)

        val closestIIsClosest =
          (tdict.keyType.ordering.compare(key, maybeEqual) <= 0 || closestI == dict.size - 1) &&
            (closestI == 0 || tdict.keyType.ordering.compare(key, getKey(closestI - 1)) > 0)

        dict.contains(key) ==> (key == maybeEqual) && closestIIsClosest

      }
    }
    p.check()
  }
}
