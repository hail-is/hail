package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{Ascending, EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitMethodBuilder, ParamType, SortOrder}
import is.hail.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

object TakeByRVAS {
  val END_SERIALIZATION: Int = 0x1324
}

class TakeByRVAS(val valueType: PType, val keyType: PType, val resultType: PArray, val kb: EmitClassBuilder[_], so: SortOrder = Ascending) extends AggregatorState {
  private val r: Settable[Region] = kb.genFieldThisRef[Region]("takeby_region")

  val region: Value[Region] = r

  private val indexedKeyType = PCanonicalTuple(true, keyType, PInt64Required)
  private val eltTuple = PCanonicalTuple(true, indexedKeyType, valueType)
  val ab = new StagedArrayBuilder(eltTuple, kb, region)

  private val maxIndex = kb.genFieldThisRef[Long]("max_index")
  private val maxSize = kb.genFieldThisRef[Int]("max_size")
  private val staging = kb.genFieldThisRef[Long]("staging")
  private val keyStage = kb.genFieldThisRef[Long]("key_stage")
  private val tempPtr = kb.genFieldThisRef[Long]("tmp_ptr")

  private val canHaveGarbage = eltTuple.containsPointers
  private val (garbage, maxGarbage) = if (canHaveGarbage) (kb.genFieldThisRef[Int](), kb.genFieldThisRef[Int]()) else (null, null)

  private val garbageFields: IndexedSeq[(String, PType)] = if (canHaveGarbage)
    FastIndexedSeq(("current_garbage", PInt32Required), ("max_garbage", PInt32Required))
  else
    FastIndexedSeq()

  val storageType: PStruct =
    PCanonicalStruct(true,
      Array(("state", ab.stateType),
        ("staging", PInt64Required),
        ("key_stage", PInt64Required),
        ("max_index", PInt64Required),
        ("max_size", PInt32Required)) ++ garbageFields: _*
    )

  private val _compareKey: EmitMethodBuilder[_] = {
    implicit val line = LineNumber.none
    val cmp = kb.genEmitMethod("compare", FastIndexedSeq[ParamType](keyType.asEmitParam, keyType.asEmitParam), IntInfo)
    val ord = keyType.codeOrdering(cmp, so)
    val k1 = cmp.getEmitParam(1)
    val k2 = cmp.getEmitParam(2)

    cmp.emit(
      ord.compare(k1.m -> coerce(k1.v), k2.m -> coerce(k2.v))
    )

    cmp
  }

  private def compareKey(cb: EmitCodeBuilder, k1: EmitCode, k2: EmitCode)(implicit line: LineNumber): Code[Int] =
    cb.invokeCode(_compareKey, k1, k2)

  private val _compareIndexedKey: EmitMethodBuilder[_] = {
    implicit val line = LineNumber.none
    val indexedkeyTypeTypeInfo = typeToTypeInfo(indexedKeyType)
    val cmp = kb.genEmitMethod("take_by_compare", FastIndexedSeq[ParamType](indexedkeyTypeTypeInfo, indexedkeyTypeTypeInfo), IntInfo)
    val ord = indexedKeyType.codeOrdering(cmp, indexedKeyType, Array(so, Ascending), true)
    val k1 = cmp.getCodeParam(1)(indexedkeyTypeTypeInfo)
    val k2 = cmp.getCodeParam(2)(indexedkeyTypeTypeInfo)

    cmp.emit(ord.compare((false, asm4s.coerce[ord.T](k1)), (false, asm4s.coerce[ord.T](k2))))

    cmp
  }

  private def compareIndexedKey(l: Code[Long], r: Code[Long])(implicit line: LineNumber): Code[Int] = {
    _compareIndexedKey.invokeCode(l, r)
  }

  private def maybeGCCode(alwaysRun: Code[Unit]*)(runIfGarbage: => Array[Code[Unit]], runBefore: Boolean = false)(implicit line: LineNumber): Code[Unit] = {
    val gcCodes = (if (canHaveGarbage) runIfGarbage else Array[Code[Unit]]())
    val allCode = if (runBefore) (gcCodes ++ alwaysRun) else (alwaysRun.toArray ++ gcCodes)
    Code(allCode)
  }

  def newState(off: Code[Long])(implicit line: LineNumber): Code[Unit] =
    region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit = {
    implicit val line = cb.lineNumber
    cb.ifx(region.isNull, {
      cb.assign(r, Region.stagedCreate(regionSize))
      cb += region.invalidate()
    })
  }

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long])(implicit line: LineNumber): Code[Unit] =
    Code(
      regionLoader(r),
      loadFields(src))

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long])(implicit line: LineNumber): Code[Unit] =
    region.isValid.orEmpty(
      Code(
        regionStorer(region),
        region.invalidate(),
        storeFields(dest)))

  private def initStaging()(implicit line: LineNumber): Code[Unit] = Code(
    staging := eltTuple.allocate(region),
    keyStage := indexedKeyType.allocate(region)
  )

  def initialize(_maxSize: Code[Int])(implicit line: LineNumber): Code[Unit] = {
    maybeGCCode(
      maxIndex := 0L,
      maxSize := _maxSize,
      (maxSize < 0).orEmpty(Code._fatal[Unit](const("'take': 'n' cannot be negative, found '").concat(maxSize.toS))),
      initStaging(),
      ab.initialize()
    )(Array(
      garbage := 0,
      maxGarbage := Code.invokeStatic2[Math, Int, Int, Int]("max", maxSize * 2, 256)
    ))
  }

  private def storeFields(dest: Code[Long])(implicit line: LineNumber): Code[Unit] = {
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

  private def loadFields(src: Code[Long])(implicit line: LineNumber): Code[Unit] =
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

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    implicit val line = cb.lineNumber
    cb += Code.memoize(src, "tba_copy_from_src") { src =>
      maybeGCCode(
        initStaging(),
        ab.copyFrom(storageType.fieldOffset(src, 0)),
        maxIndex := Region.loadLong(storageType.fieldOffset(src, 3)),
        maxSize := Region.loadInt(storageType.fieldOffset(src, 4)))(
        Array(
          maxGarbage := Region.loadInt(storageType.fieldOffset(src, 4))))
    }
  }

  def serialize(codec: BufferSpec)(implicit line: LineNumber): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      cb += maybeGCCode(
        ob.writeLong(maxIndex),
        ob.writeInt(maxSize),
        ab.serialize(codec, ob),
        ob.writeInt(const(TakeByRVAS.END_SERIALIZATION))
      )(Array(
        ob.writeInt(maxGarbage)
      ), runBefore = true)
    }
  }

  def deserialize(codec: BufferSpec)(implicit line: LineNumber): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      cb += maybeGCCode(
        maxIndex := ib.readLong(),
        maxSize := ib.readInt(),
        ab.deserialize(codec, ib),
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

  private def elementOffset(i: Value[Int])(implicit line: LineNumber): Code[Long] = ab.elementOffset(i)

  private def keyIsMissing(offset: Code[Long])(implicit line: LineNumber): Code[Boolean] = indexedKeyType.isFieldMissing(offset, 0)

  private def loadKeyValue(cb: EmitCodeBuilder, offset: Code[Long])(implicit line: LineNumber): PCode =
    keyType.loadCheapPCode(cb, indexedKeyType.loadField(offset, 0))

  private def loadKey(cb: EmitCodeBuilder, offset: Value[Long])(implicit line: LineNumber): EmitCode =
    EmitCode(Code._empty, keyIsMissing(offset), loadKeyValue(cb, offset))

  private val _compareElt: EmitMethodBuilder[_] = {
    implicit val line = LineNumber.none
    val mb = kb.genEmitMethod("i_gt_j", FastIndexedSeq[ParamType](LongInfo, LongInfo), IntInfo)
    val i = mb.getCodeParam[Long](1)
    val j = mb.getCodeParam[Long](2)

    mb.emit(compareIndexedKey(eltTuple.fieldOffset(i, 0), eltTuple.fieldOffset(j, 0)))

    mb
  }

  private def compareElt(l: Code[Long], r: Code[Long])(implicit line: LineNumber): Code[Int] =
    _compareElt.invokeCode(l, r)

  private val _swap: EmitMethodBuilder[_] = {
    implicit val line = LineNumber.none
    val mb = kb.genEmitMethod("swap", FastIndexedSeq[ParamType](LongInfo, LongInfo), UnitInfo)
    val i = mb.getCodeParam[Long](1)
    val j = mb.getCodeParam[Long](2)

    mb.emit(
      Code(
        Region.copyFrom(i, staging, eltTuple.byteSize),
        Region.copyFrom(j, i, eltTuple.byteSize),
        Region.copyFrom(staging, j, eltTuple.byteSize)))

    mb
  }

  private def swap(l: Code[Long], r: Code[Long])(implicit line: LineNumber): Code[Unit] =
    _swap.invokeCode(l, r)

  private val _rebalanceUp: EmitMethodBuilder[_] = {
    implicit val line = LineNumber.none
    val mb = kb.genEmitMethod("rebalance_up", FastIndexedSeq[ParamType](IntInfo), UnitInfo)
    val idx = mb.getCodeParam[Int](1)

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
              mb.invokeCode(parent))
          ))))

    mb
  }

  private def rebalanceUp(idx: Code[Int])(implicit line: LineNumber): Code[Unit] =
    _rebalanceUp.invokeCode(idx)

  private val _rebalanceDown: EmitMethodBuilder[_] = {
    implicit val line = LineNumber.none
    val mb = kb.genEmitMethod("rebalance_down", FastIndexedSeq[ParamType](IntInfo), UnitInfo)
    val idx = mb.getCodeParam[Int](1)

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
              mb.invokeCode(minChild)
            ),
            Code._empty
          )))))

    mb
  }

  private def rebalanceDown(idx: Code[Int])(implicit line: LineNumber): Code[Unit] =
    _rebalanceDown.invokeCode(idx)

  private lazy val gc: () => Code[Unit] = {
    implicit val line = LineNumber.none
    if (canHaveGarbage) {
      val mb = kb.genEmitMethod("take_by_garbage_collect", FastIndexedSeq[ParamType](), UnitInfo)
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
      () => mb.invokeCode()
    } else
      () => Code._empty
  }


  private def stageAndIndexKey(km: Code[Boolean], k: Code[_])(implicit line: LineNumber): Code[Unit] = Code(
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

  private def copyElementToStaging(o: Code[Long])(implicit line: LineNumber): Code[Unit] =
    Region.copyFrom(o, staging, eltTuple.byteSize)

  private def copyToStaging(value: Code[_], valueM: Code[Boolean], indexedKey: Code[Long])(implicit line: LineNumber): Code[Unit] = {
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

  private def swapStaging()(implicit line: LineNumber): Code[Unit] = {
    Code(
      StagedRegionValueBuilder.deepCopy(kb, region, eltTuple, staging, ab.elementOffset(0)),
      rebalanceDown(0)
    )
  }

  private def enqueueStaging()(implicit line: LineNumber): Code[Unit] = {
    Code(
      ab.append(Region.loadIRIntermediate(eltTuple)(staging)),
      rebalanceUp(ab.size - 1))
  }

  val _seqOp: EmitMethodBuilder[_] = {
    implicit val line = LineNumber.none
    val mb = kb.genEmitMethod("take_by_seqop",
      FastIndexedSeq[ParamType](valueType.asEmitParam, keyType.asEmitParam),
      UnitInfo)

    val value = mb.getEmitParam(1)
    val key = mb.getEmitParam(2)

    mb.emitWithBuilder { cb =>
      cb.ifx(maxSize > 0, {
        cb.ifx(ab.size < maxSize, {
          cb += stageAndIndexKey(key.m, key.v)
          cb += copyToStaging(value.v, value.m, keyStage)
          cb += enqueueStaging()
        }, {
          cb.assign(tempPtr, eltTuple.loadField(elementOffset(0), 0))
          cb.ifx(compareKey(cb, key, loadKey(cb, tempPtr)) < 0, {
            cb += stageAndIndexKey(key.m, key.v)
            cb += copyToStaging(value.v, value.m, keyStage)
            cb += swapStaging()
            cb += gc()
          })
        })
      })
      Code._empty
    }

    mb
  }

  def seqOp(cb: EmitCodeBuilder, v: EmitCode, k: EmitCode): Unit =
    cb.invokeVoid(_seqOp, v, k)

  // for tests
  def seqOp(cb: EmitCodeBuilder, vm: Code[Boolean], v: Code[_], km: Code[Boolean], k: Code[_])(implicit line: LineNumber): Unit = {
    val vec = EmitCode(Code._empty, vm, PCode(valueType, v))
    val kec = EmitCode(Code._empty, km, PCode(keyType, k))
    seqOp(cb, vec, kec)
  }

  def combine(other: TakeByRVAS)(implicit line: LineNumber): Code[Unit] = {
    val mb = kb.genEmitMethod("take_by_combop", FastIndexedSeq[ParamType](), UnitInfo)

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

    mb.invokeCode()
  }

  def result(_r: Code[Region], resultType: PArray)(implicit line: LineNumber): Code[Long] = {
    val mb = kb.genEmitMethod("take_by_result", FastIndexedSeq[ParamType](classInfo[Region]), LongInfo)

    val quickSort: (Code[Long], Code[Int], Code[Int]) => Code[Unit] = {
      val mb = kb.genEmitMethod("result_quicksort", FastIndexedSeq[ParamType](LongInfo, IntInfo, IntInfo), UnitInfo)
      val indices = mb.getCodeParam[Long](1)
      val low = mb.getCodeParam[Int](2)
      val high = mb.getCodeParam[Int](3)

      val pivotIndex = mb.newLocal[Int]("pivotIdx")

      val swap: (Code[Long], Code[Long]) => Code[Unit] = {
        val mb = kb.genEmitMethod("quicksort_swap", FastIndexedSeq[ParamType](LongInfo, LongInfo), UnitInfo)
        val i = mb.getCodeParam[Long](1)
        val j = mb.getCodeParam[Long](2)

        val tmp = mb.newLocal[Int]("swap_tmp")

        mb.emit(
          Code(
            tmp := Region.loadInt(i),
            Region.storeInt(i, Region.loadInt(j)),
            Region.storeInt(j, tmp)
          )
        )
        mb.invokeCode(_, _)
      }

      val partition: (Code[Long], Code[Int], Code[Int]) => Code[Int] = {
        val mb = kb.genEmitMethod("quicksort_partition", FastIndexedSeq[ParamType](LongInfo, IntInfo, IntInfo), IntInfo)

        val indices = mb.getCodeParam[Long](1)
        val low = mb.getCodeParam[Int](2)
        val high = mb.getCodeParam[Int](3)

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
        mb.invokeCode(_, _, _)
      }

      mb.emit(
        (low < high).orEmpty(
          Code(
            pivotIndex := partition(indices, low, high),
            mb.invokeCode(indices, low, pivotIndex),
            mb.invokeCode(indices, pivotIndex + 1, high))))
      mb.invokeCode(_, _, _)
    }

    val r = mb.getCodeParam[Region](1)
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
    mb.invokeCode(_r)
  }

}

class TakeByAggregator(valueType: PType, keyType: PType) extends StagedAggregator {

  assert(valueType.isCanonical)
  assert(keyType.isCanonical)
  type State = TakeByRVAS

  val resultType: PArray = PCanonicalArray(valueType, true)
  val initOpTypes: Seq[PType] = Array(PInt32(true))
  val seqOpTypes: Seq[PType] = Array(valueType, keyType)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    implicit val line = cb.lineNumber
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    cb += Code(
      sizeTriplet.setup,
      sizeTriplet.m.orEmpty(Code._fatal[Unit](s"argument 'n' for 'hl.agg.take' may not be missing")),
      state.initialize(coerce[Int](sizeTriplet.v))
    )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    implicit val line = cb.lineNumber
    val Array(value: EmitCode, key: EmitCode) = seq
    assert(value.pv.pt == valueType)
    assert(key.pv.pt == keyType)
    state.seqOp(cb, value, key)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    implicit val line = cb.lineNumber
    cb += state.combine(other)
  }

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    implicit val line = cb.lineNumber
    cb += srvb.addIRIntermediate(resultType)(state.result(srvb.region, resultType))
  }
}
