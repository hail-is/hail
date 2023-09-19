package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.expr.ir.{Ascending, EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitMethodBuilder, EmitValue, IEmitCode, ParamType, SortOrder}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SBaseStructPointerValue, SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.{EmitType, SCode, SValue}
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

  def compareKey(cb: EmitCodeBuilder, k1: EmitValue, k2: EmitValue): Code[Int] = {
    val ord = cb.emb.ecb.getOrdering(k1.st, k2.st, so)
    ord.compare(cb, k1, k2, true)
  }

  private def compareIndexedKey(cb: EmitCodeBuilder, k1: SValue, k2: SValue): Value[Int] = {
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

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit = cb += region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.ifx(region.isNull, {
      cb.assign(r, Region.stagedCreate(regionSize, kb.pool()))
      cb += region.invalidate()
    })

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit = {
    regionLoader(cb, r)
    loadFields(cb, src)
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Value[Long]): Unit = {
    cb.ifx(region.isValid,
      {
        regionStorer(cb, region)
        cb += region.invalidate()
        storeFields(cb, dest)
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

  def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
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

  private def elementOffset(cb: EmitCodeBuilder, i: Value[Int]): Value[Long] =
    ab.elementOffset(cb, i)

  private def keyIsMissing(cb: EmitCodeBuilder, offset: Code[Long]): Value[Boolean] =
    indexedKeyType.isFieldMissing(cb, offset, 0)

  private def loadKeyValue(cb: EmitCodeBuilder, offset: Code[Long]): SValue =
    keyType.loadCheapSCode(cb, indexedKeyType.loadField(offset, 0))

  private def loadKey(cb: EmitCodeBuilder, offset: Value[Long]): EmitValue =
    cb.memoize(IEmitCode(cb, keyIsMissing(cb, offset), loadKeyValue(cb, offset)))

  private val compareElt: EmitMethodBuilder[_] =
    kb.defineEmitMethod(genName("m", "i_gt_j"), FastSeq(LongInfo, LongInfo), IntInfo) { mb =>
      val i = mb.getCodeParam[Long](1)
      val j = mb.getCodeParam[Long](2)

      mb.emitWithBuilder { cb =>
        compareIndexedKey(cb,
          indexedKeyType.loadCheapSCode(cb, eltTuple.fieldOffset(i, 0)),
          indexedKeyType.loadCheapSCode(cb, eltTuple.fieldOffset(j, 0))
        )
      }
    }

  private val swap: EmitMethodBuilder[_] =
    kb.defineEmitMethod(genName("m", "swap"), FastSeq(LongInfo, LongInfo), UnitInfo) { mb =>
      val i = mb.getCodeParam[Long](1)
      val j = mb.getCodeParam[Long](2)

      mb.voidWithBuilder { cb =>
        cb += Region.copyFrom(i, staging, eltTuple.byteSize)
        cb += Region.copyFrom(j, i, eltTuple.byteSize)
        cb += Region.copyFrom(staging, j, eltTuple.byteSize)
      }
    }

  private val rebalanceUp: EmitMethodBuilder[_] =
    kb.defineEmitMethod(genName("m", "rebalance_up"), FastSeq(IntInfo), UnitInfo) { mb =>
      val idx = mb.getCodeParam[Int](1)

      mb.voidWithBuilder { cb =>
        cb.ifx(idx > 0, {
          val parent = cb.memoize((idx + 1) / 2 - 1)
          val ii = elementOffset(cb, idx)
          val jj = elementOffset(cb, parent)
          cb.ifx(cb.invokeCode[Int](compareElt, cb._this, ii, jj) > 0, {
            cb.invokeVoid(swap, cb._this, ii, jj)
            cb.invokeVoid(mb, cb._this, parent)
          })
        })
      }
    }

  private val rebalanceDown: EmitMethodBuilder[_] =
    kb.defineEmitMethod(genName("m", "rebalance_down"), FastSeq(IntInfo), UnitInfo) { mb =>
      val idx = mb.getCodeParam[Int](1)

      mb.voidWithBuilder { cb =>
        val child1 = cb.newLocal[Int]("child_1")
        val child2 = cb.newLocal[Int]("child_2")
        val minChild = cb.newLocal[Int]("min_child")
        val ii = cb.newLocal[Long]("ii")
        val jj = cb.newLocal[Long]("jj")


        cb.assign(child1, (idx + 1) * 2 - 1)
        cb.assign(child2, child1 + 1)
        cb.ifx(child1 < ab.size, {
          cb.ifx(
            child2 >= ab.size,
            cb.assign(minChild, child1),
            {
              val o1 = elementOffset(cb, child1)
              val o2 = elementOffset(cb, child2)
              cb.ifx(
                cb.invokeCode[Int](compareElt, cb._this, o1, o2) > 0,
                cb.assign(minChild, child1),
                cb.assign(minChild, child2)
              )
            }
          )
          cb.assign(ii, elementOffset(cb, minChild))
          cb.assign(jj, elementOffset(cb, idx))
          cb.ifx(cb.invokeCode[Int](compareElt, cb._this, ii, jj) > 0, {
            cb.invokeVoid(swap, cb._this, ii, jj)
            cb.invokeVoid(mb, cb._this, minChild)
          })
        })
      }
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
      (cb: EmitCodeBuilder) => cb.invokeVoid(mb, cb._this)
    } else
      (_: EmitCodeBuilder) => ()
  }


  private def stageAndIndexKey(cb: EmitCodeBuilder, k: EmitCode): Unit = {
    k.toI(cb)
      .consume(cb,
        {
          indexedKeyType.setFieldMissing(cb, keyStage, 0)
        },
        { sc =>
          indexedKeyType.setFieldPresent(cb, keyStage, 0)
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
      indexedKeyType.loadCheapSCode(cb, indexedKey),
      deepCopy = false)
    value.toI(cb)
      .consume(cb,
        {
          eltTuple.setFieldMissing(cb, staging, 1)
        },
        { v =>
          eltTuple.setFieldPresent(cb, staging, 1)
          valueType.storeAtAddress(cb, eltTuple.fieldOffset(staging, 1), region, v, deepCopy = false)
        })
  }

  private def swapStaging(cb: EmitCodeBuilder): Unit = {
    eltTuple.storeAtAddress(cb, ab.elementOffset(cb, 0), region, eltTuple.loadCheapSCode(cb, staging), true)
    cb.invokeVoid(rebalanceDown, cb._this, const(0))
  }

  private def enqueueStaging(cb: EmitCodeBuilder): Unit = {
    ab.append(cb, eltTuple.loadCheapSCode(cb, staging))
    cb.invokeVoid(rebalanceUp, cb._this, cb.memoize(ab.size - 1))
  }

  def seqOp(cb: EmitCodeBuilder, v: EmitCode, k: EmitCode): Unit = {
    val mb = cb.emb.ecb.genEmitMethod("take_by_seqop",
      FastIndexedSeq[ParamType](v.emitParamType, k.emitParamType),
      UnitInfo
    )

    mb.voidWithBuilder { cb =>
      val value = mb.getEmitParam(cb, 1)
      val key = mb.getEmitParam(cb, 2)

      cb.ifx(maxSize > 0, {
        cb.ifx(ab.size < maxSize, {
          stageAndIndexKey(cb, key)
          copyToStaging(cb, value, keyStage)
          enqueueStaging(cb)
        }, {
          cb.assign(tempPtr, eltTuple.loadField(elementOffset(cb, 0), 0))
          cb.ifx(compareKey(cb, key, loadKey(cb, tempPtr)) < 0, {
            stageAndIndexKey(cb, key)
            copyToStaging(cb, value, keyStage)
            swapStaging(cb)
            gc(cb)
          })
        })
      })
    }

    cb.invokeVoid(mb, cb._this, v, k)
  }

  // for tests
  def seqOp(cb: EmitCodeBuilder, vm: Code[Boolean], v: Value[_], km: Code[Boolean], k: Value[_]): Unit = {
    val vec = EmitCode(Code._empty, vm, if (valueType.isPrimitive) primitive(valueType.virtualType, v) else valueType.loadCheapSCode(cb, coerce[Long](v)))
    val kec = EmitCode(Code._empty, km, if (keyType.isPrimitive) primitive(keyType.virtualType, k) else keyType.loadCheapSCode(cb, coerce[Long](k)))
    seqOp(cb, vec, kec)
  }

  def combine(cb: EmitCodeBuilder, other: TakeByRVAS): Unit = {
    val mb = cb.emb.ecb.genEmitMethod("take_by_combop", FastIndexedSeq[ParamType](), UnitInfo)

    mb.voidWithBuilder { cb =>
      val i = cb.newLocal[Int]("combine_i")
      cb.forLoop(cb.assign(i, 0), i < other.ab.size, cb.assign(i, i + 1), {
        val offset = other.elementOffset(cb, i)
        val indexOffset = cb.memoize(indexedKeyType.fieldOffset(eltTuple.loadField(offset, 0), 1))
        cb += Region.storeLong(indexOffset, Region.loadLong(indexOffset) + maxIndex)
        cb.ifx(maxSize > 0,
          cb.ifx(ab.size < maxSize,
            {
              copyElementToStaging(cb, offset)
              enqueueStaging(cb)
            },
            {
              cb.assign(tempPtr, elementOffset(cb, 0))
              cb.ifx(cb.invokeCode[Int](compareElt, cb._this, offset, tempPtr) < 0,
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

    cb.invokeVoid(mb, cb._this)
  }

  def result(cb: EmitCodeBuilder, _r: Value[Region], resultType: PCanonicalArray): SIndexablePointerValue = {

    val swap: EmitMethodBuilder[_] =
      cb.emb.ecb.defineEmitMethod(genName("m", "quicksort_swap"),
        FastIndexedSeq[ParamType](LongInfo, LongInfo),
        UnitInfo
      ) { mb =>

        val i = mb.getCodeParam[Long](1)
        val j = mb.getCodeParam[Long](2)

        mb.voidWithBuilder { cb =>
          val tmp = mb.newLocal[Int]("swap_tmp")
          cb.assign(tmp, Region.loadInt(i))
          cb += Region.storeInt(i, Region.loadInt(j))
          cb += Region.storeInt(j, tmp)
        }
      }

    val partition: EmitMethodBuilder[_] =
      cb.emb.ecb.defineEmitMethod(genName("m", "quicksort_partition"),
        FastSeq(LongInfo, IntInfo, IntInfo),
        IntInfo
      ) { mb =>

        val indices = mb.getCodeParam[Long](1)
        val low = mb.getCodeParam[Int](2)
        val high = mb.getCodeParam[Int](3)

        mb.emitWithBuilder[Int] { cb =>
          val pivotIndex = cb.newLocal[Int]("pivotIndex")
          val pivotOffset = cb.newLocal[Long]("pivot")
          val tmpOffset = cb.newLocal[Long]("tmpOffset")
          val continue = cb.newLocal[Boolean]("continue")

          def indexOffset(idx: Value[Int]): Value[Long] =
            cb.memoize(indices + idx.toL * 4L)

          def indexAt(idx: Value[Int]): Value[Int] =
            cb.memoize(Region.loadInt(indexOffset(idx)))

          cb.ifx(low.ceq(high), cb.append(Code._return(low)))
          cb.assign(pivotIndex, (low + high) / 2)
          cb.assign(pivotOffset, elementOffset(cb, indexAt(pivotIndex)))
          cb.assign(continue, true)
          cb.whileLoop(continue, {
            cb.whileLoop({
              cb.assign(tmpOffset, elementOffset(cb, indexAt(low)))
              cb.invokeCode[Int](compareElt, cb._this, tmpOffset, pivotOffset) < 0
            }, {
              cb.assign(low, low + 1)
            })
            cb.whileLoop({
              cb.assign(tmpOffset, elementOffset(cb, indexAt(high)))
              cb.invokeCode[Int](compareElt, cb._this, tmpOffset, pivotOffset) > 0
            }, {
              cb.assign(high, high - 1)
            })
            cb.ifx(low >= high, {
              cb.assign(continue, false)
            }, {
              cb.invokeVoid(swap, cb._this, indexOffset(low), indexOffset(high))
              cb.assign(low, low + 1)
              cb.assign(high, high - 1)
            })
          })
          high
        }
      }

    val quickSort: EmitMethodBuilder[_] =
      cb.emb.ecb.defineEmitMethod(genName("m", "result_quicksort"),
        FastSeq(LongInfo, IntInfo, IntInfo),
        UnitInfo
      ) { mb =>

        val indices = mb.getCodeParam[Long](1)
        val low = mb.getCodeParam[Int](2)
        val high = mb.getCodeParam[Int](3)

        mb.voidWithBuilder { cb =>
          val pivotIndex = cb.newLocal[Int]("pivotIdx")

          cb.ifx(low < high, {
            cb.assign(pivotIndex, cb.invokeCode[Int](partition, cb._this, indices, low, high))
            cb.invokeVoid(mb, cb._this, indices, low, pivotIndex)
            cb.invokeVoid(mb, cb._this, indices, cb.memoize(pivotIndex + 1), high)
          })
        }
      }

    val result: EmitMethodBuilder[_] =
      cb.emb.ecb.defineEmitMethod(genName("m", "take_by_result"),
        FastSeq(classInfo[Region]),
        LongInfo
      ) { mb =>

        val r = mb.getCodeParam[Region](1)

        mb.emitWithBuilder[Long] { cb =>
          val indicesToSort = cb.newLocal[Long]("indices_to_sort",
            r.load().allocate(4L, ab.size.toL * 4L)
          )

          val i = cb.newLocal[Int]("i", 0)

          def indexOffset(idx: Code[Int]): Code[Long] =
            indicesToSort + idx.toL * 4L

          cb.whileLoop(i < ab.size, {
            cb += Region.storeInt(indexOffset(i), i)
            cb.assign(i, i + 1)
          })

          cb.invokeVoid(quickSort, cb._this, indicesToSort, const(0), cb.memoize(ab.size - 1))

          resultType.constructFromElements(cb, r, ab.size, deepCopy = true) { case (cb, idx) =>
            val sortedIdx = cb.newLocal[Int]("tba_result_sortedidx", Region.loadInt(indexOffset(idx)))
            ab.loadElement(cb, sortedIdx).toI(cb)
              .flatMap(cb) { case pct: SBaseStructPointerValue =>
                pct.loadField(cb, 1)
              }
          }.a
        }
      }

    resultType.loadCheapSCode(cb, cb.invokeCode[Long](result, cb._this, _r))
  }
}

class TakeByAggregator(valueType: VirtualTypeWithReq, keyType: VirtualTypeWithReq) extends StagedAggregator {

  type State = TakeByRVAS

  val resultEmitType: EmitType = EmitType(SIndexablePointer(PCanonicalArray(valueType.canonicalPType)), true)
  val initOpTypes: Seq[Type] = Array(TInt32)
  val seqOpTypes: Seq[Type] = Array(valueType.t, keyType.t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    sizeTriplet.toI(cb)
      .consume(cb,
        cb += Code._fatal[Unit](s"argument 'n' for 'hl.agg.take' may not be missing"),
        sc => state.initialize(cb, sc.asInt.value))
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(value: EmitCode, key: EmitCode) = seq
    state.seqOp(cb, value, key)
  }

  protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, state: TakeByRVAS, other: TakeByRVAS): Unit = state.combine(cb, other)


  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    // state.result does a deep copy
    IEmitCode.present(cb, state.result(cb, region, resultEmitType.storageType.asInstanceOf[PCanonicalArray]))
  }
}
