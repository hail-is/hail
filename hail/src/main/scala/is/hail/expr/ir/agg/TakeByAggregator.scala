package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.expr.ir.{Ascending, EmitClassBuilder, EmitCode, EmitCodeBuilder, ParamType, SortOrder}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SBaseStructPointerCode, SIndexablePointerCode}
import is.hail.types.physical.stypes.interfaces.SBaseStruct
import is.hail.types.virtual.{TInt32, Type}
import is.hail.utils._

object TakeByRVAS {
  val END_SERIALIZATION: Int = 0x1324
}

class TakeByRVAS(val valueVType: VirtualTypeWithReq, val keyVType: VirtualTypeWithReq, val kb: EmitClassBuilder[_], so: SortOrder = Ascending) extends AggregatorState {
  private val r: Settable[Region] = kb.genFieldThisRef[Region]("takeby_region")

  val valueType: PType = valueVType.canonicalPType
  val keyType: PType = keyVType.canonicalPType

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

  def compareKey(cb: EmitCodeBuilder, k1: EmitCode, k2: EmitCode): Code[Int] = {
    val ord = cb.emb.ecb.getOrdering(k1.st, k2.st, so)
    ord.compare(cb, k1, k2, true)
  }

  private def compareIndexedKey(cb: EmitCodeBuilder, k1: PCode, k2: PCode): Code[Int] = {
    val ord = StructOrdering.make(k1.st.asInstanceOf[SBaseStruct], k2.st.asInstanceOf[SBaseStruct], cb.emb.ecb, Array(so, Ascending), true)
    ord.compareNonnull(cb, k1, k2)
  }

  private def maybeGCCode(cb: EmitCodeBuilder, alwaysRun: EmitCodeBuilder => Unit)(runIfGarbage: EmitCodeBuilder => Unit, runBefore: Boolean = false): Unit = {
    val gc = (if (canHaveGarbage) runIfGarbage else (cb: EmitCodeBuilder) => ())
    if (runBefore) {
      gc(cb)
      alwaysRun(cb)
    } else {
      alwaysRun(cb)
      gc(cb)
    }
  }

  def newState(cb: EmitCodeBuilder, off: Code[Long]): Unit = cb += region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.ifx(region.isNull, {
      cb.assign(r, Region.stagedCreate(regionSize, kb.pool()))
      cb += region.invalidate()
    })

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, srcc: Code[Long]): Unit = {
    regionLoader(cb, r)
    loadFields(cb, srcc)
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, destc: Code[Long]): Unit = {
    cb.ifx(region.isValid,
      {
        regionStorer(cb, region)
        cb += region.invalidate()
        storeFields(cb, destc)
      })
  }

  private def initStaging(cb: EmitCodeBuilder): Unit = {
    cb.assign(staging, eltTuple.allocate(region))
    cb.assign(keyStage, indexedKeyType.allocate(region))
  }

  def initialize(cb: EmitCodeBuilder, _maxSize: Code[Int]): Unit = {
    maybeGCCode(cb,
      { cb =>
        cb.assign(maxIndex, 0L)
        cb.assign(maxSize, _maxSize)
        cb.ifx(maxSize < 0,
          cb += Code._fatal[Unit](const("'take': 'n' cannot be negative, found '").concat(maxSize.toS)))
        initStaging(cb)
        ab.initialize(cb)
      })({ cb =>
      cb.assign(garbage, 0)
      cb.assign(maxGarbage, Code.invokeStatic2[Math, Int, Int, Int]("max", maxSize * 2, 256))
    })
  }

  private def storeFields(cb: EmitCodeBuilder, destc: Code[Long]): Unit = {
    val dest = cb.newLocal("tba_store_fields_dest", destc)
    maybeGCCode(cb,
      { cb =>
        ab.storeTo(cb, storageType.fieldOffset(dest, 0))
        cb += Region.storeAddress(storageType.fieldOffset(dest, 1), staging)
        cb += Region.storeAddress(storageType.fieldOffset(dest, 2), keyStage)
        cb += Region.storeLong(storageType.fieldOffset(dest, 3), maxIndex)
        cb += Region.storeInt(storageType.fieldOffset(dest, 4), maxSize)
      }
    )({ cb =>
      cb += Region.storeInt(storageType.fieldOffset(dest, 5), garbage)
      cb += Region.storeInt(storageType.fieldOffset(dest, 6), maxGarbage)
    })
  }

  private def loadFields(cb: EmitCodeBuilder, srcc: Code[Long]): Unit = {
    val src = cb.newLocal("takeby_rvas_load_fields_src", srcc)
    maybeGCCode(cb,
      { cb =>
        ab.loadFrom(cb, storageType.fieldOffset(src, 0))
        cb.assign(staging, Region.loadAddress(storageType.fieldOffset(src, 1)))
        cb.assign(keyStage, Region.loadAddress(storageType.fieldOffset(src, 2)))
        cb.assign(maxIndex, Region.loadLong(storageType.fieldOffset(src, 3)))
        cb.assign(maxSize, Region.loadInt(storageType.fieldOffset(src, 4)))
      }
    )({ cb =>
      cb.assign(garbage, Region.loadInt(storageType.fieldOffset(src, 5)))
      cb.assign(maxGarbage, Region.loadInt(storageType.fieldOffset(src, 6)))
    }
    )
  }

  def copyFrom(cb: EmitCodeBuilder, srcc: Code[Long]): Unit = {
    val src = cb.newLocal("tba_copy_from_src", srcc)
    maybeGCCode(cb,
      { cb =>
        initStaging(cb)
        ab.copyFrom(cb, storageType.fieldOffset(src, 0))
        cb.assign(maxIndex, Region.loadLong(storageType.fieldOffset(src, 3)))
        cb.assign(maxSize, Region.loadInt(storageType.fieldOffset(src, 4)))
      })({ cb =>
      cb.assign(maxGarbage, Region.loadInt(storageType.fieldOffset(src, 4)))
    })
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      maybeGCCode(cb,
        { cb =>
          cb += ob.writeLong(maxIndex)
          cb += ob.writeInt(maxSize)
          ab.serialize(codec)(cb, ob)
          cb += ob.writeInt(const(TakeByRVAS.END_SERIALIZATION))
        }
      )({ cb =>
        cb += ob.writeInt(maxGarbage)
      }, runBefore = true)
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      maybeGCCode(cb,
        { cb =>
          cb.assign(maxIndex, ib.readLong())
          cb.assign(maxSize, ib.readInt())
          ab.deserialize(codec)(cb, ib)
          initStaging(cb)
          cb += ib.readInt()
            .cne(const(TakeByRVAS.END_SERIALIZATION))
            .orEmpty(Code._fatal[Unit](s"StagedSizedKeyValuePriorityQueue serialization failed"))
        }
      )({ cb =>
        cb.assign(maxGarbage, ib.readInt())
        cb.assign(garbage, 0)
      }, runBefore = true)
    }
  }

  private def elementOffset(i: Value[Int]): Code[Long] = ab.elementOffset(i)

  private def keyIsMissing(offset: Code[Long]): Code[Boolean] = indexedKeyType.isFieldMissing(offset, 0)

  private def loadKeyValue(cb: EmitCodeBuilder, offset: Code[Long]): PCode = keyType.loadCheapPCode(cb, indexedKeyType.loadField(offset, 0))

  private def loadKey(cb: EmitCodeBuilder, offset: Value[Long]): EmitCode =
    EmitCode(Code._empty, keyIsMissing(offset), loadKeyValue(cb, offset))

  private val compareElt: (Code[Long], Code[Long]) => Code[Int] = {
    val mb = kb.genEmitMethod("i_gt_j", FastIndexedSeq[ParamType](LongInfo, LongInfo), IntInfo)
    val i = mb.getCodeParam[Long](1)
    val j = mb.getCodeParam[Long](2)

    mb.emitWithBuilder(cb => compareIndexedKey(cb,
      indexedKeyType.loadCheapPCode(cb, eltTuple.fieldOffset(i, 0)),
      indexedKeyType.loadCheapPCode(cb, eltTuple.fieldOffset(j, 0))))

    mb.invokeCode(_, _)
  }

  private val swap: (EmitCodeBuilder, Code[Long], Code[Long]) => Unit = {
    val mb = kb.genEmitMethod("swap", FastIndexedSeq[ParamType](LongInfo, LongInfo), UnitInfo)
    val i = mb.getCodeParam[Long](1)
    val j = mb.getCodeParam[Long](2)

    mb.voidWithBuilder({ cb =>
      cb += Region.copyFrom(i, staging, eltTuple.byteSize)
      cb += Region.copyFrom(j, i, eltTuple.byteSize)
      cb += Region.copyFrom(staging, j, eltTuple.byteSize)
    })

    (cb: EmitCodeBuilder, x: Code[Long], y: Code[Long]) => cb.invokeVoid(mb, x, y)
  }


  private val rebalanceUp: (EmitCodeBuilder, Code[Int]) => Unit = {
    val mb = kb.genEmitMethod("rebalance_up", FastIndexedSeq[ParamType](IntInfo), UnitInfo)
    val idx = mb.getCodeParam[Int](1)

    val ii = mb.newLocal[Long]("rebalance_up_ii")
    val jj = mb.newLocal[Long]("rebalance_up_jj")

    val parent = mb.newLocal[Int]("parent")

    mb.voidWithBuilder { cb =>
      cb.ifx(idx > 0,
        {
          cb.assign(parent, (idx + 1) / 2 - 1)
          cb.assign(ii, elementOffset(idx))
          cb.assign(jj, elementOffset(parent))
          cb.ifx(compareElt(ii, jj) > 0,
            {
              swap(cb, ii, jj)
              cb.invokeVoid(mb, parent)
            })
        })
    }

    (cb: EmitCodeBuilder, x: Code[Int]) => cb.invokeVoid(mb, x)
  }

  private val rebalanceDown: (EmitCodeBuilder, Code[Int]) => Unit = {
    val mb = kb.genEmitMethod("rebalance_down", FastIndexedSeq[ParamType](IntInfo), UnitInfo)
    val idx = mb.getCodeParam[Int](1)

    val child1 = mb.newLocal[Int]("child_1")
    val child2 = mb.newLocal[Int]("child_2")
    val minChild = mb.newLocal[Int]("min_child")
    val ii = mb.newLocal[Long]("ii")
    val jj = mb.newLocal[Long]("jj")

    mb.voidWithBuilder { cb =>
      cb.assign(child1, (idx + 1) * 2 - 1)
      cb.assign(child2, child1 + 1)
      cb.ifx(child1 < ab.size,
        {
          cb.assign(minChild, (child2 >= ab.size || compareElt(elementOffset(child1), elementOffset(child2)) > 0).mux(child1, child2))
          cb.assign(ii, elementOffset(minChild))
          cb.assign(jj, elementOffset(idx))
          cb.ifx(compareElt(ii, jj) > 0,
            {
              swap(cb, ii, jj)
              cb.invokeVoid(mb, minChild)
            })
        })
    }
    (cb: EmitCodeBuilder, x: Code[Int]) => cb.invokeVoid(mb, x)
  }

  private lazy val gc: EmitCodeBuilder => Unit = {
    if (canHaveGarbage) {
      val mb = kb.genEmitMethod("take_by_garbage_collect", FastIndexedSeq[ParamType](), UnitInfo)
      val oldRegion = mb.newLocal[Region]("old_region")
      mb.voidWithBuilder { cb =>
        cb.assign(garbage, garbage + 1)
        cb.ifx(garbage >= maxGarbage,
          {
            cb.assign(oldRegion, region)
            cb.assign(r, Region.stagedCreate(regionSize, kb.pool()))
            ab.reallocateData(cb)
            initStaging(cb)
            cb.assign(garbage, 0)
            cb += oldRegion.invoke[Unit]("invalidate")
          })
      }
      (cb: EmitCodeBuilder) => cb.invokeVoid(mb)
    } else
      (_: EmitCodeBuilder) => ()
  }


  private def stageAndIndexKey(cb: EmitCodeBuilder, k: EmitCode): Unit = {
    k.toI(cb)
      .consume(cb,
        {
          cb += indexedKeyType.setFieldMissing(keyStage, 0)
        },
        { sc =>
          cb += indexedKeyType.setFieldPresent(keyStage, 0)
          keyType.storeAtAddress(cb, indexedKeyType.fieldOffset(keyStage, 0), region, sc, deepCopy = false)
        }
      )
    cb += Region.storeLong(indexedKeyType.fieldOffset(keyStage, 1), maxIndex)
    cb.assign(maxIndex, maxIndex + 1L)
  }

  private def copyElementToStaging(cb: EmitCodeBuilder, o: Code[Long]): Unit = cb += Region.copyFrom(o, staging, eltTuple.byteSize)

  private def copyToStaging(cb: EmitCodeBuilder, value: EmitCode, indexedKey: Code[Long]): Unit = {
    cb.ifx(staging.ceq(0L), cb += Code._fatal[Unit]("staging is 0"))
    indexedKeyType.storeAtAddress(cb,
      eltTuple.fieldOffset(staging, 0),
      region,
      indexedKeyType.loadCheapPCode(cb, indexedKey),
      deepCopy = false)
    value.toI(cb)
      .consume(cb,
        {
          cb += eltTuple.setFieldMissing(staging, 1)
        },
        { v =>
          cb += eltTuple.setFieldPresent(staging, 1)
          valueType.storeAtAddress(cb, eltTuple.fieldOffset(staging, 1), region, v, deepCopy = false)
        })
  }

  private def swapStaging(cb: EmitCodeBuilder): Unit = {
    eltTuple.storeAtAddress(cb, ab.elementOffset(0), region, eltTuple.loadCheapPCode(cb, staging), true)
    rebalanceDown(cb, 0)
  }

  private def enqueueStaging(cb: EmitCodeBuilder): Unit = {
    ab.append(cb, eltTuple.loadCheapPCode(cb, staging))
    rebalanceUp(cb, ab.size - 1)
  }

  def seqOp(cb: EmitCodeBuilder, v: EmitCode, k: EmitCode): Unit = {
    val mb = kb.genEmitMethod("take_by_seqop",
      FastIndexedSeq[ParamType](v.emitParamType, k.emitParamType),
      UnitInfo)

    mb.voidWithBuilder { cb =>
      val value = mb.getEmitParam(1, null) // don't need region
      val key = mb.getEmitParam(2, null) // don't need region

      cb.ifx(maxSize > 0, {
        cb.ifx(ab.size < maxSize, {
          stageAndIndexKey(cb, key)
          copyToStaging(cb, value, keyStage)
          enqueueStaging(cb)
        }, {
          cb.assign(tempPtr, eltTuple.loadField(elementOffset(0), 0))
          cb.ifx(compareKey(cb, key, loadKey(cb, tempPtr)) < 0, {
            stageAndIndexKey(cb, key)
            copyToStaging(cb, value, keyStage)
            swapStaging(cb)
            gc(cb)
          })
        })
      })
    }

    cb.invokeVoid(mb, v, k)
  }

  // for tests
  def seqOp(cb: EmitCodeBuilder, vm: Code[Boolean], v: Code[_], km: Code[Boolean], k: Code[_]): Unit = {
    val vec = EmitCode(Code._empty, vm, PCode(valueType, v))
    val kec = EmitCode(Code._empty, km, PCode(keyType, k))
    seqOp(cb, vec, kec)
  }

  def combine(cb: EmitCodeBuilder, other: TakeByRVAS): Unit = {
    val mb = kb.genEmitMethod("take_by_combop", FastIndexedSeq[ParamType](), UnitInfo)


    mb.voidWithBuilder { cb =>
      val i = cb.newLocal[Int]("combine_i")
      val offset = cb.newLocal[Long]("combine_offset")
      val indexOffset = cb.newLocal[Long]("index_offset")
      cb.forLoop(
        cb.assign(i, 0),
        i < other.ab.size,
        cb.assign(i, i + 1),
        {
          cb.assign(offset, other.elementOffset(i))
          cb.assign(indexOffset, indexedKeyType.fieldOffset(eltTuple.loadField(offset, 0), 1))
          cb += Region.storeLong(indexOffset, Region.loadLong(indexOffset) + maxIndex)
          cb.ifx(maxSize > 0,
            cb.ifx(ab.size < maxSize,
              {
                copyElementToStaging(cb, offset)
                enqueueStaging(cb)
              },
              {
                cb.assign(tempPtr, elementOffset(0))
                cb.ifx(compareElt(offset, tempPtr) < 0,
                  {
                    copyElementToStaging(cb, offset)
                    swapStaging(cb)
                    gc(cb)
                  })
              }
            ))
        })
      cb.assign(maxIndex, maxIndex + other.maxIndex)
    }

    cb.invokeVoid(mb)
  }

  def result(cb: EmitCodeBuilder, _r: Code[Region], resultType: PCanonicalArray): SIndexablePointerCode = {
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

    mb.emitWithBuilder[Long] { cb =>
      val r = mb.getCodeParam[Region](1)

      val indicesToSort = cb.newLocal[Long]("indices_to_sort",
        r.load().allocate(4L, ab.size.toL * 4L))

      val i = cb.newLocal[Int]("i", 0)

      def indexOffset(idx: Code[Int]): Code[Long] = indicesToSort + idx.toL * 4L

      cb.whileLoop(i < ab.size, {
        cb += Region.storeInt(indexOffset(i), i)
        cb.assign(i, i + 1)
      })

      cb += quickSort(indicesToSort, 0, ab.size - 1)

      resultType.constructFromElements(cb, r, ab.size, deepCopy = true) { case (cb, idx) =>
        val sortedIdx = cb.newLocal[Int]("tba_result_sortedidx", Region.loadInt(indexOffset(idx)))
        ab.loadElement(cb, sortedIdx).toI(cb)
          .flatMap(cb) { case pct: SBaseStructPointerCode =>
            pct.memoize(cb, "takeby_result_tuple").loadField(cb, 1).typecast[PCode]
          }
      }.a
    }
    resultType.loadCheapPCode(cb, cb.invokeCode[Long](mb, _r))
  }
}

class TakeByAggregator(valueType: VirtualTypeWithReq, keyType: VirtualTypeWithReq) extends StagedAggregator {

  type State = TakeByRVAS

  val resultType: PCanonicalArray = PCanonicalArray(valueType.canonicalPType, true)
  val initOpTypes: Seq[Type] = Array(TInt32)
  val seqOpTypes: Seq[Type] = Array(valueType.t, keyType.t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    sizeTriplet.toI(cb)
      .consume(cb,
        cb += Code._fatal[Unit](s"argument 'n' for 'hl.agg.take' may not be missing"),
        sc => state.initialize(cb, sc.asInt.intCode(cb)))
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(value: EmitCode, key: EmitCode) = seq
    state.seqOp(cb, value, key)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = state.combine(cb, other)


  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    // deepCopy is false because state.result does a deep copy
    pt.storeAtAddress(cb, addr, region, state.result(cb, region, resultType), deepCopy = false)
  }
}
