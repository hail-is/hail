package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitFunctionBuilder, defaultValue, typeToTypeInfo}
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

object TakeByRVAS {
  val END_SERIALIZATION: Int = 0x1324
}

class TakeByRVAS(val valueType: PType, val keyType: PType, val resultType: PArray, val cb: EmitClassBuilder[_]) extends AggregatorState {
  private val r: Settable[Region] = cb.genFieldThisRef[Region]("takeby_region")

  val region: Value[Region] = r

  private val indexedKeyType = PCanonicalTuple(true, keyType, PInt64Required)
  private val eltTuple = PCanonicalTuple(true, indexedKeyType, valueType)
  val ab = new StagedArrayBuilder(eltTuple, cb, region)

  private val maxIndex = cb.genFieldThisRef[Long]("max_index")
  private val maxSize = cb.genFieldThisRef[Int]("max_size")
  private val staging = cb.genFieldThisRef[Long]("staging")
  private val keyStage = cb.genFieldThisRef[Long]("key_stage")
  private val tempPtr = cb.genFieldThisRef[Long]("tmp_ptr")

  private val canHaveGarbage = eltTuple.containsPointers
  private val (garbage, maxGarbage) = if (canHaveGarbage) (cb.genFieldThisRef[Int](), cb.genFieldThisRef[Int]()) else (null, null)

  private val garbageFields: IndexedSeq[(String, PType)] = if (canHaveGarbage)
    FastIndexedSeq(("current_garbage", PInt32Required), ("max_garbage", PInt32Required))
  else
    FastIndexedSeq()

  val storageType: PStruct =
    PStruct(true,
      Array(("state", ab.stateType),
        ("staging", PInt64Required),
        ("key_stage", PInt64Required),
        ("max_index", PInt64Required),
        ("max_size", PInt32Required)) ++ garbageFields: _*
    )

  private val compareKey: ((Code[Boolean], Code[_]), (Code[Boolean], Code[_])) => Code[Int] = {
    val keyInfo = typeToTypeInfo(keyType.virtualType)
    val cmp = cb.genEmitMethod("compare", Array[TypeInfo[_]](BooleanInfo, keyInfo, BooleanInfo, keyInfo), IntInfo)
    val ord = keyType.codeOrdering(cmp)
    val k1m = cmp.getArg[Boolean](1)
    val k1 = cmp.getArg(2)(keyInfo)
    val k2m = cmp.getArg[Boolean](3)
    val k2 = cmp.getArg(4)(keyInfo)

    cmp.emit(
      ord.compare((k1m, asm4s.coerce[ord.T](k1)), (k2m, asm4s.coerce[ord.T](k2)))
    );

    def wrappedValue(missingBit: Code[Boolean], value: Code[_]): Code[_] = {
      missingBit.mux(defaultValue(keyType), value)
    }

    (k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])) => {
      Code.memoize(k1._1, "tba_comp_key_k1m",
        k2._1, "tba_comp_key_k2m") { (k1m, k2m) =>
        cmp.invoke[Int](k1m, wrappedValue(k1m, k1._2), k2m, wrappedValue(k2m, k2._2))
      }
    }
  }

  private val compareIndexedKey: (Code[Long], Code[Long]) => Code[Int] = {
    val indexedkeyTypeTypeInfo = typeToTypeInfo(indexedKeyType.virtualType)
    val cmp = cb.genEmitMethod("take_by_compare", Array[TypeInfo[_]](indexedkeyTypeTypeInfo, indexedkeyTypeTypeInfo), IntInfo)
    val ord = indexedKeyType.codeOrdering(cmp)
    val k1 = cmp.getArg(1)(indexedkeyTypeTypeInfo)
    val k2 = cmp.getArg(2)(indexedkeyTypeTypeInfo)

    cmp.emit(ord.compare((false, asm4s.coerce[ord.T](k1)), (false, asm4s.coerce[ord.T](k2))))

    cmp.invoke(_, _)
  }

  private def maybeGCCode(alwaysRun: Code[Unit]*)(runIfGarbage: => Array[Code[Unit]], runBefore: Boolean = false): Code[Unit] = {
    val gcCodes = (if (canHaveGarbage) runIfGarbage else Array[Code[Unit]]())
    val allCode = if (runBefore) (gcCodes ++ alwaysRun) else (alwaysRun.toArray ++ gcCodes)
    Code(allCode)
  }

  def newState(off: Code[Long]): Code[Unit] = region.getNewRegion(regionSize)

  def createState: Code[Unit] = region.isNull.mux(Code(r := Region.stagedCreate(regionSize), region.invalidate()), Code._empty)

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code(
      regionLoader(r),
      loadFields(src))

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    region.isValid.orEmpty(
      Code(
        regionStorer(region),
        region.invalidate(),
        storeFields(dest)))

  private def initStaging(): Code[Unit] = Code(
    staging := eltTuple.allocate(region),
    keyStage := indexedKeyType.allocate(region)
  )

  def initialize(_maxSize: Code[Int]): Code[Unit] = {
    maybeGCCode(
      maxIndex := 0L,
      maxSize := _maxSize,
      (maxSize < 0).orEmpty(Code._fatal[Unit](const("'take': 'n' cannot be negative, found '").concat(maxSize.toS))),
      initStaging(),
      ab.initialize()
    )(Array(
      garbage := 0,
      maxGarbage := Code.invokeStatic[Math, Int, Int, Int]("max", maxSize * 2, 256)
    ))
  }

  private def storeFields(dest: Code[Long]): Code[Unit] = {
    Code.memoize(dest, "tba_store_fields_dest") { dest =>
      maybeGCCode(
        ab.storeTo(storageType.fieldOffset(dest, 0)),
        Region.storeAddress(storageType.fieldOffset(dest, 1), staging),
        Region.storeAddress(storageType.fieldOffset(dest, 2), keyStage),
        Region.storeLong(storageType.fieldOffset(dest, 3), maxIndex),
        Region.storeInt(storageType.fieldOffset(dest, 4), maxSize)
      )(Array(
        Region.storeInt(storageType.fieldOffset(dest, 5), garbage),
        Region.storeInt(storageType.fieldOffset(dest, 6), maxGarbage)))
    }
  }

  private def loadFields(src: Code[Long]): Code[Unit] =
    Code.memoize(src, "takeby_rvas_load_fields_src") { src =>
      maybeGCCode(
        ab.loadFrom(storageType.fieldOffset(src, 0)),
        staging := Region.loadAddress(storageType.fieldOffset(src, 1)),
        keyStage := Region.loadAddress(storageType.fieldOffset(src, 2)),
        maxIndex := Region.loadLong(storageType.fieldOffset(src, 3)),
        maxSize := Region.loadInt(storageType.fieldOffset(src, 4))
      )(Array(
        garbage := Region.loadInt(storageType.fieldOffset(src, 5)),
        maxGarbage := Region.loadInt(storageType.fieldOffset(src, 6))
      ))
    }

  def copyFrom(src: Code[Long]): Code[Unit] = {
    Code.memoize(src, "tba_copy_from_src") { src =>
      maybeGCCode(
        initStaging(),
        ab.copyFrom(storageType.fieldOffset(src, 0)),
        maxIndex := Region.loadLong(storageType.fieldOffset(src, 3)),
        maxSize := Region.loadInt(storageType.fieldOffset(src, 4)))(
        Array(
          maxGarbage := Region.loadInt(storageType.fieldOffset(src, 4))))
    }
  }

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    { ob: Value[OutputBuffer] =>
      maybeGCCode(
        ob.writeLong(maxIndex),
        ob.writeInt(maxSize),
        ab.serialize(codec)(ob),
        ob.writeInt(const(TakeByRVAS.END_SERIALIZATION))
      )(Array(
        ob.writeInt(maxGarbage)
      ), runBefore = true)
    }
  }

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    { (ib: Value[InputBuffer]) =>
      maybeGCCode(
        maxIndex := ib.readLong(),
        maxSize := ib.readInt(),
        ab.deserialize(codec)(ib),
        initStaging(),
        ib.readInt()
          .cne(const(TakeByRVAS.END_SERIALIZATION))
          .orEmpty(Code._fatal[Unit](s"StagedSizedKeyValuePriorityQueue serialization failed"))
      )(Array(
        maxGarbage := ib.readInt(),
        garbage := 0
      ), runBefore = true)
    }
  }

  //  NOTE: these print methods are unused but helpful for debugging, should the need arise:
  //
  //  def indexedKeyRepElt(o: Code[Long]): Code[String] = indexedKeyRep(eltTuple.fieldOffset(o, 0))
  //
  //  def indexedKeyRep(o: Code[Long]): Code[String] = {
  //    val kr = keyRep(loadKeyValue(o), keyIsMissing(o))
  //    val idx = Region.loadLong(indexedKeyType.fieldOffset(o, 1)).toS
  //    kr.concat(" [").concat(idx).concat("]")
  //
  //  }
  //
  //  def keyRep(key: Code[_], keyM: Code[Boolean]): Code[String] = {
  //    keyType match {
  //      case _: PInt32 => keyM.mux(const("NA"), key.asInstanceOf[Code[Int]].toS)
  //      case _: PFloat32 => keyM.mux(const("NA"), key.asInstanceOf[Code[Float]].toS)
  //      case _: PFloat64 => keyM.mux(const("NA"), key.asInstanceOf[Code[Double]].toS)
  //      case _: PInt64 => keyM.mux(const("NA"), key.asInstanceOf[Code[Long]].toS)
  //      case _ => keyM.mux(const("NA"), const("??"))
  //    }
  //  }
  //
  //  def dump(): Code[Unit] = {
  //    val x = fb.newField[Int]
  //    Code(
  //      Code._println(const(s"Dumping heap of size=").concat(ab.size.toS)),
  //      x := 0,
  //      Code.whileLoop(x < ab.size,
  //        Code._println(const("  at idx ").concat(x.toS).concat(", key=")
  //          .concat(indexedKeyRep(eltTuple.loadField(elementOffset(x), 0)))),
  //        x := x + 1
  //      )
  //    )
  //  }

  private def elementOffset(i: Value[Int]): Code[Long] = asm4s.coerce[Long](ab.elementOffset(i)._2)

  private def keyIsMissing(offset: Code[Long]): Code[Boolean] = indexedKeyType.isFieldMissing(offset, 0)

  private def loadKeyValue(offset: Code[Long]): Code[_] = Region.loadIRIntermediate(keyType)(indexedKeyType.fieldOffset(offset, 0))

  private def loadKey(offset: Value[Long]): (Code[Boolean], Code[_]) = (keyIsMissing(offset), loadKeyValue(offset))

  private val compareElt: (Code[Long], Code[Long]) => Code[Int] = {
    val mb = cb.genEmitMethod("i_gt_j", Array[TypeInfo[_]](LongInfo, LongInfo), IntInfo)
    val i = mb.getArg[Long](1)
    val j = mb.getArg[Long](2)

    mb.emit(compareIndexedKey(eltTuple.fieldOffset(i, 0), eltTuple.fieldOffset(j, 0)))

    mb.invoke(_, _)
  }

  private val swap: (Code[Long], Code[Long]) => Code[Unit] = {
    val mb = cb.genEmitMethod("swap", Array[TypeInfo[_]](LongInfo, LongInfo), UnitInfo)
    val i = mb.getArg[Long](1)
    val j = mb.getArg[Long](2)

    mb.emit(
      Code(
        Region.copyFrom(i, staging, eltTuple.byteSize),
        Region.copyFrom(j, i, eltTuple.byteSize),
        Region.copyFrom(staging, j, eltTuple.byteSize))
    )
    mb.invoke(_, _)
  }


  private val rebalanceUp: Code[Int] => Code[Unit] = {
    val mb = cb.genEmitMethod("rebalance_up", Array[TypeInfo[_]](IntInfo), UnitInfo)
    val idx = mb.getArg[Int](1)

    val ii = mb.newLocal[Long]("rebalance_up_ii")
    val jj = mb.newLocal[Long]("rebalance_up_jj")

    val parent = mb.newLocal[Int]("parent")

    mb.emit(
      (idx > 0).orEmpty(
        Code(
          parent := (idx + 1) / 2 - 1,
          ii := elementOffset(idx),
          jj := elementOffset(parent),
          (compareElt(ii, jj) > 0).orEmpty(
            Code(
              swap(ii, jj),
              mb.invoke(parent))
          ))))

    mb.invoke(_)
  }

  private val rebalanceDown: Code[Int] => Code[Unit] = {
    val mb = cb.genEmitMethod("rebalance_down", Array[TypeInfo[_]](IntInfo), UnitInfo)
    val idx = mb.getArg[Int](1)

    val child1 = mb.newLocal[Int]("child_1")
    val child2 = mb.newLocal[Int]("child_2")
    val minChild = mb.newLocal[Int]("min_child")
    val ii = mb.newLocal[Long]("ii")
    val jj = mb.newLocal[Long]("jj")

    mb.emit(Code(
      child1 := (idx + 1) * 2 - 1,
      child2 := child1 + 1,
      (child1 < ab.size).orEmpty(
        Code(
          minChild := (child2 >= ab.size || compareElt(elementOffset(child1), elementOffset(child2)) > 0).mux(child1, child2),
          ii := elementOffset(minChild),
          jj := elementOffset(idx),
          (compareElt(ii, jj) > 0).mux(
            Code(
              swap(ii, jj),
              mb.invoke(minChild)
            ),
            Code._empty
          )))))
    mb.invoke(_)
  }

  private lazy val gc: () => Code[Unit] = {
    if (canHaveGarbage) {
      val mb = cb.genEmitMethod("take_by_garbage_collect", Array[TypeInfo[_]](), UnitInfo)
      val oldRegion = mb.newLocal[Region]("old_region")
      mb.emit(
        Code(
          garbage := garbage + 1,
          (garbage >= maxGarbage).orEmpty(Code(
            oldRegion := region,
            r := Region.stagedCreate(regionSize),
            ab.reallocateData(),
            initStaging(),
            garbage := 0,
            oldRegion.invoke[Unit]("invalidate")
          ))
        ))
      () => mb.invoke()
    } else
      () => Code._empty
  }


  private def stageAndIndexKey(km: Code[Boolean], k: Code[_]): Code[Unit] = Code(
    if (keyType.required)
      Region.storeIRIntermediate(keyType)(indexedKeyType.fieldOffset(keyStage, 0), k)
    else
      km.mux(
        indexedKeyType.setFieldMissing(keyStage, 0),
        Code(
          indexedKeyType.setFieldPresent(keyStage, 0),
          Region.storeIRIntermediate(keyType)(indexedKeyType.fieldOffset(keyStage, 0), k)
        )),
    Region.storeLong(indexedKeyType.fieldOffset(keyStage, 1), maxIndex),
    maxIndex := maxIndex + 1L
  )

  private def copyElementToStaging(o: Code[Long]): Code[Unit] = Region.copyFrom(o, staging, eltTuple.byteSize)

  private def copyToStaging(value: Code[_], valueM: Code[Boolean], indexedKey: Code[Long]): Code[Unit] = {
    Code(
      staging.ceq(0L).orEmpty(Code._fatal[Unit]("staging is 0")),
      Region.copyFrom(indexedKey, eltTuple.fieldOffset(staging, 0), indexedKeyType.byteSize),
      if (valueType.required)
        Region.storeIRIntermediate(valueType)(eltTuple.fieldOffset(staging, 1), value)
      else
        valueM.mux(
          eltTuple.setFieldMissing(staging, 1),
          Code(
            eltTuple.setFieldPresent(staging, 1),
            Region.storeIRIntermediate(valueType)(eltTuple.fieldOffset(staging, 1), value)
          ))
    )
  }

  private def swapStaging(): Code[Unit] = {
    Code(
      StagedRegionValueBuilder.deepCopy(cb, region, eltTuple, staging, ab.elementOffset(0)._2),
      rebalanceDown(0)
    )
  }

  private def enqueueStaging(): Code[Unit] = {
    Code(
      ab.append(Region.loadIRIntermediate(eltTuple)(staging)),
      rebalanceUp(ab.size - 1))
  }

  val seqOp: (Code[Boolean], Code[_], Code[Boolean], Code[_]) => Code[Unit] = {
    val ki = typeToTypeInfo(keyType)
    val vi = typeToTypeInfo(valueType)
    val mb = cb.genEmitMethod("take_by_seqop",
      Array[TypeInfo[_]](BooleanInfo, vi, BooleanInfo, ki),
      UnitInfo)

    val valueM = mb.getArg[Boolean](1)
    val value = mb.getArg(2)(vi)
    val keyM = mb.getArg[Boolean](3)
    val key = mb.getArg(4)(ki)

    mb.emit(
      (maxSize > 0).orEmpty(
        (ab.size < maxSize).mux(
          Code(
            stageAndIndexKey(keyM, key),
            copyToStaging(value, valueM, keyStage),
            enqueueStaging()),
          Code(
            tempPtr := eltTuple.loadField(elementOffset(0), 0),
            (compareKey((keyM, key), loadKey(tempPtr)) < 0)
              .orEmpty(Code(
                stageAndIndexKey(keyM, key),
                copyToStaging(value, valueM, keyStage),
                swapStaging(),
                gc()))))))

    val kmVar = cb.genFieldThisRef[Boolean]("km")
    val vmVar = cb.genFieldThisRef[Boolean]("vm")

    { (vm: Code[Boolean], v: Code[_], km: Code[Boolean], k: Code[_]) =>
      Code(
        vmVar := vm,
        kmVar := km,
        mb.invoke(vmVar, vmVar.mux(defaultValue(valueType), v), kmVar, kmVar.mux(defaultValue(keyType), k))
      )
    }
  }

  def combine(other: TakeByRVAS, dummy: Boolean): Code[Unit] = {
    val mb = cb.genEmitMethod("take_by_combop", Array[TypeInfo[_]](), UnitInfo)

    val i = mb.newLocal[Int]("combine_i")
    val offset = mb.newLocal[Long]("combine_offset")
    val indexOffset = mb.newLocal[Long]("index_offset")

    mb.emit(Code(
      i := 0,
      Code.whileLoop(i < other.ab.size,
        offset := other.elementOffset(i),
        indexOffset := indexedKeyType.fieldOffset(eltTuple.loadField(offset, 0), 1),
        Region.storeLong(indexOffset, Region.loadLong(indexOffset) + maxIndex),
        (maxSize > 0).orEmpty(
          (ab.size < maxSize).mux(
            Code(
              copyElementToStaging(offset),
              enqueueStaging()),
            Code(
              tempPtr := elementOffset(0),
              (compareElt(offset, tempPtr) < 0)
                .orEmpty(Code(
                  copyElementToStaging(offset),
                  swapStaging(),
                  gc()))))),
        i := i + 1),
      maxIndex := maxIndex + other.maxIndex))

    mb.invoke()
  }

  def result(_r: Code[Region], resultType: PArray): Code[Long] = {
    val mb = cb.genEmitMethod("take_by_result", Array[TypeInfo[_]](classInfo[Region]), LongInfo)

    val quickSort: (Code[Long], Code[Int], Code[Int]) => Code[Unit] = {
      val mb = cb.genEmitMethod("result_quicksort", Array[TypeInfo[_]](LongInfo, IntInfo, IntInfo), UnitInfo)
      val indices = mb.getArg[Long](1)
      val low = mb.getArg[Int](2)
      val high = mb.getArg[Int](3)

      val pivotIndex = mb.newLocal[Int]("pivotIdx")

      val swap: (Code[Long], Code[Long]) => Code[Unit] = {
        val mb = cb.genEmitMethod("quicksort_swap", Array[TypeInfo[_]](LongInfo, LongInfo), UnitInfo)
        val i = mb.getArg[Long](1)
        val j = mb.getArg[Long](2)

        val tmp = mb.newLocal[Int]("swap_tmp")

        mb.emit(
          Code(
            tmp := Region.loadInt(i),
            Region.storeInt(i, Region.loadInt(j)),
            Region.storeInt(j, tmp)
          )
        )
        mb.invoke(_, _)
      }

      val partition: (Code[Long], Code[Int], Code[Int]) => Code[Int] = {
        val mb = cb.genEmitMethod("quicksort_partition", Array[TypeInfo[_]](LongInfo, IntInfo, IntInfo), IntInfo)

        val indices = mb.getArg[Long](1)
        val low = mb.getArg[Int](2)
        val high = mb.getArg[Int](3)

        val pivotIndex = mb.newLocal[Int]("pivotIndex")
        val pivotOffset = mb.newLocal[Long]("pivot")
        val tmpOffset = mb.newLocal[Long]("tmpOffset")
        val continue = mb.newLocal[Boolean]("continue")

        def indexOffset(idx: Code[Int]): Code[Long] = indices + idx.toL * 4L

        def indexAt(idx: Code[Int]): Code[Int] = Region.loadInt(indexOffset(idx))

        mb.emit(Code(
          low.ceq(high).orEmpty(Code._return(low)),
          pivotIndex := (low + high) / 2,
          pivotOffset := Code.memoize(indexAt(pivotIndex), "tba_qsort_pivot") { i => elementOffset(i) },
          continue := true,
          Code.whileLoop(continue,
            Code.whileLoop(
              Code(
                tmpOffset := Code.memoize(indexAt(low), "tba_qsort_pivot") { i => elementOffset(i) },
                compareElt(tmpOffset, pivotOffset) < 0),
              low := low + 1
            ),
            Code.whileLoop(
              Code(
                tmpOffset := Code.memoize(indexAt(high), "tba_qsort_pivot") { i => elementOffset(i) },
                compareElt(tmpOffset, pivotOffset) > 0),
              high := high - 1
            ),
            (low >= high).mux(
              continue := false,
              Code(
                swap(indexOffset(low), indexOffset(high)),
                low := low + 1,
                high := high - 1))),
          high
        )
        )
        mb.invoke(_, _, _)
      }

      mb.emit(
        (low < high).orEmpty(
          Code(
            pivotIndex := partition(indices, low, high),
            mb.invoke(indices, low, pivotIndex),
            mb.invoke(indices, pivotIndex + 1, high))))
      mb.invoke(_, _, _)
    }

    val r = mb.getArg[Region](1)
    val indicesToSort = mb.newLocal[Long]("indices_to_sort")
    val i = mb.newLocal[Int]("i")
    val o = mb.newLocal[Long]("i")

    def indexOffset(idx: Code[Int]): Code[Long] = indicesToSort + idx.toL * 4L

    def indexAt(idx: Code[Int]): Code[Int] = Region.loadInt(indexOffset(idx))

    val srvb = (new StagedRegionValueBuilder(mb, resultType, r))
    mb.emit(Code(
      indicesToSort := r.load().allocate(4L, ab.size.toL * 4L),
      i := 0,
      Code.whileLoop(i < ab.size,
        Region.storeInt(indicesToSort + i.toL * 4L, i),
        i := i + 1),
      quickSort(indicesToSort, 0, ab.size - 1),
      srvb.start(ab.size),
      i := 0,
      Code.whileLoop(i < ab.size,
        o := Code.memoize(indexAt(i), "tba_qsort_i") { i => elementOffset(i) },
        eltTuple.isFieldDefined(o, 1).mux(
          srvb.addWithDeepCopy(valueType, Region.loadIRIntermediate(valueType)(eltTuple.fieldOffset(o, 1))),
          srvb.setMissing()
        ),
        srvb.advance(),
        i := i + 1
      ),
      srvb.end()
    ))
    mb.invoke(_r)
  }

}

class TakeByAggregator(valueType: PType, keyType: PType) extends StagedAggregator {

  type State = TakeByRVAS

  val resultType: PArray = PArray(valueType)

  def createState(cb: EmitClassBuilder[_]): State =
    new TakeByRVAS(valueType, keyType, resultType, cb)

  def initOp(state: State, init: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    Code(
      sizeTriplet.setup,
      sizeTriplet.m.orEmpty(Code._fatal[Unit](s"argument 'n' for 'hl.agg.take' may not be missing")),
      state.initialize(coerce[Int](sizeTriplet.v))
    )
  }

  def seqOp(state: State, seq: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(value: EmitCode, key: EmitCode) = seq
    assert(value.pv.pt == valueType)
    assert(key.pv.pt == keyType)
    Code(
      value.setup,
      key.setup,
      state.seqOp(value.m, value.v, key.m, key.v)
    )
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = state.combine(other, dummy)

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    srvb.addIRIntermediate(resultType)(state.result(srvb.region, resultType))
}
