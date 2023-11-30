package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitRegion, EmitValue, IEmitCode, ParamType}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces.SBaseStructValue
import is.hail.types.physical.stypes.{EmitType, SingleCodeSCode}
import is.hail.types.virtual._
import is.hail.utils._


class DownsampleBTreeKey(binType: PBaseStruct, pointType: PBaseStruct, kb: EmitClassBuilder[_], region: Code[Region]) extends BTreeKey {
  override val storageType: PCanonicalStruct = PCanonicalStruct(required = true,
    "bin" -> binType,
    "point" -> pointType,
    "empty" -> PBooleanRequired)

  override val compType: PType = binType
  private val kcomp = kb.getOrderingFunction(binType.sType, CodeOrdering.Compare())

  override def isEmpty(cb: EmitCodeBuilder, off: Code[Long]): Value[Boolean] =
    PBooleanRequired.loadCheapSCode(cb, storageType.loadField(off, "empty")).value

  override def initializeEmpty(cb: EmitCodeBuilder, off: Code[Long]): Unit = cb += Region.storeBoolean(storageType.fieldOffset(off, "empty"), true)

  override def copy(cb: EmitCodeBuilder, src: Code[Long], dest: Code[Long]): Unit = cb += Region.copyFrom(src, dest, storageType.byteSize)

  override def deepCopy(cb: EmitCodeBuilder, er: EmitRegion, srcc: Code[Long], dest: Code[Long]): Unit = {
    val src = cb.newLocal[Long]("dsa_deep_copy_src", srcc)
    cb.if_(Region.loadBoolean(storageType.loadField(src, "empty")),
      cb += Code._fatal[Unit]("key empty!"))
    storageType.storeAtAddress(cb, dest, er.region, storageType.loadCheapSCode(cb, src), deepCopy = true)
  }

  override def compKeys(cb: EmitCodeBuilder, k1: EmitValue, k2: EmitValue): Value[Int] =
    kcomp(cb, k1, k2)

  override def loadCompKey(cb: EmitCodeBuilder, off: Value[Long]): EmitValue =
    EmitValue.present(binType.loadCheapSCode(cb, storageType.loadField(off, "bin")))
}


object DownsampleState {
  val serializationEndMarker: Int = 883625255
}

class DownsampleState(val kb: EmitClassBuilder[_], labelType: VirtualTypeWithReq, maxBufferSize: Int = 256) extends AggregatorState {
  private val labelPType = labelType.canonicalPType
  val r: Settable[Region] = kb.genFieldThisRef[Region]("region")
  val region: Value[Region] = r

  val oldRegion: Settable[Region] = kb.genFieldThisRef[Region]("old_region")

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit = cb += region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.if_(region.isNull, cb.assign(r, Region.stagedCreate(regionSize, kb.pool())))

  val binType = PCanonicalStruct(required = true, "x" -> PInt32Required, "y" -> PInt32Required)
  val pointType = PCanonicalStruct(required = true, "x" -> PFloat64Required, "y" -> PFloat64Required, "label" -> labelPType)

  private val binET = EType.defaultFromPType(binType)
  private val pointET = EType.defaultFromPType(pointType)

  private val root: Settable[Long] = kb.genFieldThisRef[Long]("root")
  private val oldRoot: Settable[Long] = kb.genFieldThisRef[Long]("old_root")

  val key = new DownsampleBTreeKey(binType, pointType, kb, region)
  val tree = new AppendOnlyBTree(kb, key, region, root)
  val buffer = new StagedArrayBuilder(pointType, kb, region, initialCapacity = maxBufferSize)
  val oldRootBTree = new AppendOnlyBTree(kb, key, region, oldRoot)

  private val off: Settable[Long] = kb.genFieldThisRef[Long]("offset")
  private val nDivisions: Settable[Int] = kb.genFieldThisRef[Int]("n_divisions")
  private val left: Settable[Double] = kb.genFieldThisRef[Double]("left")
  private val right: Settable[Double] = kb.genFieldThisRef[Double]("right")
  private val bottom: Settable[Double] = kb.genFieldThisRef[Double]("bottom")
  private val top: Settable[Double] = kb.genFieldThisRef[Double]("top")
  private val bufferLeft: Settable[Double] = kb.genFieldThisRef[Double]("buffer_left")
  private val bufferRight: Settable[Double] = kb.genFieldThisRef[Double]("buffer_right")
  private val bufferBottom: Settable[Double] = kb.genFieldThisRef[Double]("buffer_bottom")
  private val bufferTop: Settable[Double] = kb.genFieldThisRef[Double]("buffer_top")
  private val treeSize: Settable[Int] = kb.genFieldThisRef[Int]("treeSize")

  val storageType = PCanonicalStruct(required = true,
    "nDivisions" -> PInt32Required,
    "treeSize" -> PInt32Required,
    "left" -> PFloat64Required,
    "right" -> PFloat64Required,
    "bottom" -> PFloat64Required,
    "top" -> PFloat64Required,
    "bufferLeft" -> PFloat64Required,
    "bufferRight" -> PFloat64Required,
    "bufferBottom" -> PFloat64Required,
    "bufferTop" -> PFloat64Required,
    "buffer" -> buffer.stateType,
    "tree" -> PInt64Required,
    "binStaging" -> binType, // used as scratch space
    "pointStaging" -> pointType // used as scratch space
  )

  override val regionSize: Int = Region.SMALL

  def allocateSpace(cb: EmitCodeBuilder): Unit =
    cb.assign(off, region.allocate(storageType.alignment, storageType.byteSize))

  def init(cb: EmitCodeBuilder, nDivisions: Value[Int]): Unit = {
    val mb = kb.genEmitMethod("downsample_init", FastSeq[ParamType](IntInfo), UnitInfo)
    mb.voidWithBuilder { cb =>
      allocateSpace(cb)
      cb.assign(this.nDivisions, mb.getCodeParam[Int](1).load())

      cb.if_(this.nDivisions < 4, cb += Code._fatal[Unit](const("downsample: require n_divisions >= 4, found ").concat(this.nDivisions.toS)))

      cb.assign(left, 0d)
      cb.assign(right, 0d)
      cb.assign(bottom, 0d)
      cb.assign(top, 0d)
      cb.assign(treeSize, 0)
      tree.init(cb)
      buffer.initialize(cb)
    }
    cb.invokeVoid(mb, cb.this_, nDivisions)
  }

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit = {
    val mb = kb.genEmitMethod("downsample_load", FastSeq[ParamType](), UnitInfo)
    mb.voidWithBuilder { cb =>

      cb.assign(nDivisions, Region.loadInt(storageType.loadField(off, "nDivisions")))
      cb.assign(treeSize, Region.loadInt(storageType.loadField(off, "treeSize")))
      cb.assign(left, Region.loadDouble(storageType.loadField(off, "left")))
      cb.assign(right, Region.loadDouble(storageType.loadField(off, "right")))
      cb.assign(bottom, Region.loadDouble(storageType.loadField(off, "bottom")))
      cb.assign(top, Region.loadDouble(storageType.loadField(off, "top")))
      cb.assign(bufferLeft, Region.loadDouble(storageType.loadField(off, "bufferLeft")))
      cb.assign(bufferRight, Region.loadDouble(storageType.loadField(off, "bufferRight")))
      cb.assign(bufferBottom, Region.loadDouble(storageType.loadField(off, "bufferBottom")))
      cb.assign(bufferTop, Region.loadDouble(storageType.loadField(off, "bufferTop")))
      buffer.loadFrom(cb, storageType.fieldOffset(off, "buffer"))
      cb.assign(root, Region.loadAddress(storageType.fieldOffset(off, "tree")))
    }
    cb.assign(off, src)
    regionLoader(cb, r)
    cb.invokeVoid(mb, cb.this_)
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Value[Long]): Unit = {
    val mb = kb.genEmitMethod("downsample_store", FastSeq[ParamType](), UnitInfo)
    mb.voidWithBuilder { cb =>
      cb += Region.storeInt(storageType.fieldOffset(off, "nDivisions"), nDivisions)
      cb += Region.storeInt(storageType.fieldOffset(off, "treeSize"), treeSize)
      cb += Region.storeDouble(storageType.fieldOffset(off, "left"), left)
      cb += Region.storeDouble(storageType.fieldOffset(off, "right"), right)
      cb += Region.storeDouble(storageType.fieldOffset(off, "bottom"), bottom)
      cb += Region.storeDouble(storageType.fieldOffset(off, "top"), top)
      cb += Region.storeDouble(storageType.fieldOffset(off, "bufferLeft"), bufferLeft)
      cb += Region.storeDouble(storageType.fieldOffset(off, "bufferRight"), bufferRight)
      cb += Region.storeDouble(storageType.fieldOffset(off, "bufferBottom"), bufferBottom)
      cb += Region.storeDouble(storageType.fieldOffset(off, "bufferTop"), bufferTop)
      buffer.storeTo(cb, storageType.fieldOffset(off, "buffer"))
      cb += Region.storeAddress(storageType.fieldOffset(off, "tree"), root)
    }

    cb.assign(off, dest)
    cb.invokeVoid(mb, cb.this_)
    cb.if_(region.isValid,
      {
        regionStorer(cb, region)
        cb += region.invalidate()
      })
  }

  def copyFrom(cb: EmitCodeBuilder, _src: Value[Long]): Unit = {
    val mb = kb.genEmitMethod("downsample_copy", FastSeq[ParamType](LongInfo), UnitInfo)

    val src = mb.getCodeParam[Long](1)
    mb.voidWithBuilder { cb =>
      allocateSpace(cb)
      cb.assign(nDivisions, Region.loadInt(storageType.loadField(src, "nDivisions")))
      cb.assign(treeSize, Region.loadInt(storageType.loadField(src, "treeSize")))
      cb.assign(left, Region.loadDouble(storageType.loadField(src, "left")))
      cb.assign(right, Region.loadDouble(storageType.loadField(src, "right")))
      cb.assign(bottom, Region.loadDouble(storageType.loadField(src, "top")))
      cb.assign(top, Region.loadDouble(storageType.loadField(src, "bottom")))
      cb.assign(treeSize, Region.loadInt(storageType.loadField(src, "treeSize")))
      tree.deepCopy(cb, cb.memoize(Region.loadAddress(storageType.loadField(src, "tree"))))
      buffer.copyFrom(cb, storageType.loadField(src, "buffer"))
    }
    cb.invokeVoid(mb, cb.this_, _src)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      val mb = kb.genEmitMethod("downsample_serialize", FastSeq[ParamType](typeInfo[OutputBuffer]), UnitInfo)
      mb.emitWithBuilder { cb =>
        val ob = mb.getCodeParam[OutputBuffer](1)
        dumpBuffer(cb)
        cb += ob.writeInt(nDivisions)
        cb += ob.writeInt(treeSize)
        cb += ob.writeDouble(left)
        cb += ob.writeDouble(right)
        cb += ob.writeDouble(bottom)
        cb += ob.writeDouble(top)
        cb += ob.writeInt(treeSize)
        tree.bulkStore(cb, ob) { (cb, ob, srcCode) =>
          val src = cb.newLocal("downsample_state_ser_src", srcCode)
          cb.if_(Region.loadBoolean(key.storageType.loadField(src, "empty")), cb._fatal("bad"))
          val binCode = binType.loadCheapSCode(cb, key.storageType.loadField(src, "bin"))
          binET.buildEncoder(binCode.st, kb).apply(cb, binCode, ob)

          val pointCode = pointType.loadCheapSCode(cb, key.storageType.loadField(src, "point"))
          pointET.buildEncoder(pointCode.st, kb).apply(cb, pointCode, ob)
        }
        cb += ob.writeInt(DownsampleState.serializationEndMarker)
        Code._empty
      }

      cb.invokeVoid(mb, cb.this_, ob)
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val binDec = binET.buildInplaceDecoderMethod(binType, kb)
    val pointDec = pointET.buildInplaceDecoderMethod(pointType, kb)

    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      val mb = kb.genEmitMethod("downsample_deserialize", FastSeq[ParamType](typeInfo[InputBuffer]), UnitInfo)
      mb.emitWithBuilder { cb =>
        val ib = cb.emb.getCodeParam[InputBuffer](1)
        val serializationEndTag = cb.newLocal[Int]("de_end_tag")
        allocateSpace(cb)
        cb.assign(nDivisions, ib.readInt())
        cb.assign(treeSize, ib.readInt())
        cb.assign(left, ib.readDouble())
        cb.assign(right, ib.readDouble())
        cb.assign(bottom, ib.readDouble())
        cb.assign(top, ib.readDouble())
        cb.assign(bufferLeft, left)
        cb.assign(bufferRight, right)
        cb.assign(bufferBottom, bottom)
        cb.assign(bufferTop, top)
        cb.assign(treeSize, ib.readInt())
        tree.init(cb)
        tree.bulkLoad(cb, ib) { (cb, ib, destCode) =>
          val dest = cb.newLocal("dss_deser_dest", destCode)
          cb.invokeCode(binDec, cb.this_, region, cb.memoize(key.storageType.fieldOffset(dest, "bin")), ib)
          cb.invokeCode(pointDec, cb.this_, region, cb.memoize(key.storageType.fieldOffset(dest, "point")), ib)
          cb += Region.storeBoolean(key.storageType.fieldOffset(dest, "empty"), false)
        }
        buffer.initialize(cb)
        cb.assign(serializationEndTag, ib.readInt())
        cb.if_(serializationEndTag.cne(DownsampleState.serializationEndMarker), {
          cb._fatal("downsample aggregator failed to serialize!")
        })
        Code._empty
      }
      cb.invokeVoid(mb, cb.this_, ib)
    }
  }

  val xBinCoordinate: (EmitCodeBuilder, Value[Double]) => Value[Int] = {
    val mb = kb.genEmitMethod("downsample_x_bin_coordinate", FastSeq[ParamType](DoubleInfo), IntInfo)
    val x = mb.getCodeParam[Double](1)
    mb.emit(right.ceq(left).mux(0, (((x - left) / (right - left)) * nDivisions.toD).toI))
    (cb, x) => cb.invokeCode(mb, cb.this_, x)
  }

  val yBinCoordinate: (EmitCodeBuilder, Value[Double]) => Value[Int] = {
    val mb = kb.genEmitMethod("downsample_y_bin_coordinate", FastSeq[ParamType](DoubleInfo), IntInfo)
    val y = mb.getCodeParam[Double](1)
    mb.emit(top.ceq(bottom).mux(0, (((y - bottom) / (top - bottom)) * nDivisions.toD).toI))
    (cb, y) => cb.invokeCode(mb, cb.this_, y)
  }

  def insertIntoTree(cb: EmitCodeBuilder, binX: Value[Int], binY: Value[Int], point: Value[Long], deepCopy: Boolean): Unit = {
    val name = s"downsample_insert_into_tree${ if (deepCopy) "_deep_copy" else "" }"
    val mb = kb.getOrDefineEmitMethod(name, FastSeq[ParamType](IntInfo, IntInfo, LongInfo), UnitInfo) { mb =>
      val binX = mb.getCodeParam[Int](1)
      val binY = mb.getCodeParam[Int](2)
      val point = mb.getCodeParam[Long](3)
      val insertOffset = mb.newLocal[Long]("insert_offset")
      val binOffset = mb.newLocal[Long]("bin_offset")
      val insertedPointOffset = mb.newLocal[Long]("inserted_point_offset")
      val binStaging = mb.newLocal[Long]("binStaging")

      mb.voidWithBuilder { cb =>
        cb.assign(binStaging, storageType.loadField(off, "binStaging"))
        cb += Region.storeInt(binType.fieldOffset(binStaging, "x"), binX)
        cb += Region.storeInt(binType.fieldOffset(binStaging, "y"), binY)
        cb.assign(insertOffset,
          tree.getOrElseInitialize(cb, EmitCode.present(cb.emb, storageType.fieldType("binStaging").loadCheapSCode(cb, binStaging))))
        cb.if_(key.isEmpty(cb, insertOffset), {
          cb.assign(binOffset, key.storageType.loadField(insertOffset, "bin"))
          cb += Region.storeInt(binType.loadField(binOffset, "x"), binX)
          cb += Region.storeInt(binType.loadField(binOffset, "y"), binY)
          cb.assign(insertedPointOffset, key.storageType.loadField(insertOffset, "point"))
          pointType.storeAtAddress(cb, insertedPointOffset, region, pointType.loadCheapSCode(cb, point), deepCopy = deepCopy)
          cb += Region.storeBoolean(key.storageType.loadField(insertOffset, "empty"), false)
          cb.assign(treeSize, treeSize + 1)
        })
      }
    }

    cb.invokeVoid(mb, cb.this_, binX, binY, point)
  }

  def copyFromTree(cb: EmitCodeBuilder, other: AppendOnlyBTree): Unit = {
    val mb = kb.genEmitMethod("downsample_copy_from_tree", FastSeq[ParamType](), UnitInfo)

    mb.voidWithBuilder { cb =>
      other.foreach(cb) { (cb, v) =>
        val mb = kb.genEmitMethod("downsample_copy_from_tree_foreach", FastSeq[ParamType](LongInfo), UnitInfo)
        val value = mb.getCodeParam[Long](1)
        mb.voidWithBuilder { cb =>
          val point = cb.memoize(key.storageType.loadField(value, "point"))
          val pointX = cb.memoize(Region.loadDouble(pointType.loadField(point, "x")))
          val pointY = cb.memoize(Region.loadDouble(pointType.loadField(point, "y")))
          insertIntoTree(cb, xBinCoordinate(cb, pointX), yBinCoordinate(cb, pointY), point, deepCopy = true)
        }
        cb.invokeVoid(mb, cb.this_, v)
      }
    }

    cb.invokeVoid(mb, cb.this_)
  }

  def min(a: Code[Double], b: Code[Double]): Code[Double] =
    Code.invokeStatic2[java.lang.Double, Double, Double, Double]("min", a, b)

  def intMax(a: Code[Int], b: Code[Int]): Code[Int] =
    Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("max", a, b)

  def max(a: Code[Double], b: Code[Double]): Code[Double] =
    Code.invokeStatic2[java.lang.Double, Double, Double, Double]("max", a, b)

  def isFinite(a: Code[Double]): Code[Boolean] = Code.invokeStatic1[java.lang.Double, Double, Boolean]("isFinite", a)

  def dumpBuffer(cb: EmitCodeBuilder): Unit = {
    val name = "downsample_dump_buffer"
    val mb = kb.getOrDefineEmitMethod(name, FastSeq(), UnitInfo) { mb =>
      val i = mb.newLocal[Int]("i")
      mb.voidWithBuilder { cb =>
        cb.if_(buffer.size.ceq(0), cb += Code._return[Unit](Code._empty))
        cb.assign(left, min(left, bufferLeft))
        cb.assign(right, max(right, bufferRight))
        cb.assign(bottom, min(bottom, bufferBottom))
        cb.assign(top, max(top, bufferTop))
        cb.assign(oldRegion, region)
        cb.assign(oldRoot, root)
        cb.assign(r, Region.stagedCreate(regionSize, region.getPool()))
        cb.assign(treeSize, 0)
        tree.init(cb)
        copyFromTree(cb, oldRootBTree)
        cb.assign(i, 0)
        cb.while_(i < buffer.size,
          {
            buffer.loadElement(cb, i).toI(cb).consume(cb, {}, { case point: SBaseStructValue =>
              val x = point.loadField(cb, "x").get(cb).asFloat64.value
              val y = point.loadField(cb, "y").get(cb).asFloat64.value
              val pointc = coerce[Long](SingleCodeSCode.fromSCode(cb, point, region).code)
              insertIntoTree(cb, xBinCoordinate(cb, x), yBinCoordinate(cb, y), pointc, deepCopy = true)
            })
            cb.assign(i, i + 1)
          })
        buffer.initialize(cb)
        cb += oldRegion.invalidate()
        allocateSpace(cb)
      }
    }

    cb.invokeVoid(mb, cb.this_)
  }

  def insertPointIntoBuffer(cb: EmitCodeBuilder, x: Value[Double], y: Value[Double], point: Value[Long], deepCopy: Boolean): Unit = {
    val name = s"downsample_insert_into_buffer${if (deepCopy) "_deep_copy" else ""}"
    val mb = kb.getOrDefineEmitMethod(name, FastSeq[ParamType](DoubleInfo, DoubleInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getCodeParam[Double](1)
      val y = mb.getCodeParam[Double](2)
      val point = mb.getCodeParam[Long](3)

      mb.voidWithBuilder { cb =>
        cb.assign(bufferLeft, min(bufferLeft, x))
        cb.assign(bufferRight, max(bufferRight, x))
        cb.assign(bufferBottom, min(bufferBottom, y))
        cb.assign(bufferTop, max(bufferTop, y))
        buffer.append(cb, pointType.loadCheapSCode(cb, point))
        cb.if_(buffer.size >= maxBufferSize, dumpBuffer(cb))
      }
    }

    cb.invokeVoid(mb, cb.this_, x, y, point)
  }

  def checkBounds(cb: EmitCodeBuilder, xBin: Value[Int], yBin: Value[Int]): Value[Boolean] = {
    val name = "downsample_check_bounds"
    val mb = kb.getOrDefineEmitMethod(name, FastSeq[ParamType](IntInfo, IntInfo), BooleanInfo) { mb =>
      val xBin = mb.getCodeParam[Int](1)
      val yBin = mb.getCodeParam[Int](2)
      val factor = mb.newLocal[Int]("factor")
      mb.emit(Code(
        factor := nDivisions >> 2,
        treeSize.ceq(0)
          || (xBin < -factor)
          || (xBin > nDivisions + factor)
          || (yBin < -factor)
          || (yBin > nDivisions + factor)))
    }

    cb.invokeCode(mb, cb.this_, xBin, yBin)
  }

  def binAndInsert(cb: EmitCodeBuilder, x: Value[Double], y: Value[Double], point: Value[Long], deepCopy: Boolean): Unit = {
    val name = s"downsample_bin_and_insert${if (deepCopy) "_deep_copy" else ""}"
    val mb = kb.getOrDefineEmitMethod(name, FastSeq[ParamType](DoubleInfo, DoubleInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getCodeParam[Double](1)
      val y = mb.getCodeParam[Double](2)
      val point = mb.getCodeParam[Long](3)

      val binX = mb.newLocal[Int]("bin_x")
      val binY = mb.newLocal[Int]("bin_y")

      mb.voidWithBuilder { cb =>
        cb.assign(binX, xBinCoordinate(cb, x))
        cb.assign(binY, yBinCoordinate(cb, y))
        cb.if_(checkBounds(cb, binX, binY),
          insertPointIntoBuffer(cb, x, y, point, deepCopy = deepCopy),
          insertIntoTree(cb, binX, binY, point, deepCopy = deepCopy)
        )
      }
    }
    cb.invokeVoid(mb, cb.this_, x, y, point)
  }

  def insert(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, l: EmitCode): Unit = {
    val name = "downsample_insert"
    val mb = kb.getOrDefineEmitMethod(name, FastSeq[ParamType](x.st.paramType, y.st.paramType, l.emitParamType), UnitInfo) { mb =>

      val pointStaging = mb.newLocal[Long]("pointStaging")
      mb.voidWithBuilder { cb =>
        val x = mb.getSCodeParam(1)
        val y = mb.getSCodeParam(2)
        val l = mb.getEmitParam(cb, 3)

        def xx = x.asDouble.value

        def yy = y.asDouble.value

        cb.if_((!(isFinite(xx) && isFinite(yy))), cb += Code._return[Unit](Code._empty))
        cb.assign(pointStaging, storageType.loadField(off, "pointStaging"))
        pointType.fieldType("x").storeAtAddress(cb, pointType.fieldOffset(pointStaging, "x"), region, x, deepCopy = true)
        pointType.fieldType("y").storeAtAddress(cb, pointType.fieldOffset(pointStaging, "y"), region, y, deepCopy = true)
        l.toI(cb)
          .consume(cb,
            pointType.setFieldMissing(cb, pointStaging, "label"),
            { sc =>
              pointType.setFieldPresent(cb, pointStaging, "label")
              pointType.fieldType("label").storeAtAddress(cb, pointType.fieldOffset(pointStaging, "label"), region, sc, deepCopy = true)
            }
          )
        binAndInsert(cb, xx, yy, pointStaging, deepCopy = false)
      }
    }

    x.toI(cb)
      .consume(cb,
        {
          /* do nothing if x is missing */
        },
        { xcode =>
          y.toI(cb)
            .consume(cb,
              {
                /* do nothing if y is missing */
              },
              ycode => cb.invokeVoid(mb, cb.this_, xcode, ycode, l)
            )
        })
  }

  def deepCopyAndInsertPoint(cb: EmitCodeBuilder, point: Value[Long]): Unit = {
    val name = "downsample_deep_copy_insert_point"
    val mb = kb.getOrDefineEmitMethod(name, IndexedSeq[ParamType](LongInfo), UnitInfo) { mb =>
      val point = mb.getCodeParam[Long](1)

      val x = mb.newLocal[Double]("x")
      val y = mb.newLocal[Double]("y")

      mb.voidWithBuilder { cb =>
        cb.assign(x, Region.loadDouble(pointType.loadField(point, "x")))
        cb.assign(y, Region.loadDouble(pointType.loadField(point, "y")))
        binAndInsert(cb, x, y, point, deepCopy = true)
      }
    }

    cb.invokeVoid(mb, cb.this_, point)
  }

  def merge(cb: EmitCodeBuilder, other: DownsampleState): Unit = {
    val mb = kb.genEmitMethod("downsample_insert_from", FastSeq[ParamType](), UnitInfo)

    val i = mb.newLocal[Int]("i")
    mb.emitWithBuilder { cb =>
      cb.assign(i, 0)
      cb.while_(i < other.buffer.size, {
        val point = SingleCodeSCode.fromSCode(cb, other.buffer.loadElement(cb, i).pv, region)
        deepCopyAndInsertPoint(cb, coerce[Long](point.code))
        cb.assign(i, i + 1)
      })
      other.tree.foreach(cb) { (cb, value) =>
        deepCopyAndInsertPoint(cb, cb.memoize(key.storageType.loadField(value, "point")))
      }
      Code._empty
    }
    cb.invokeVoid(mb, cb.this_)
  }

  def resultArray(cb: EmitCodeBuilder, region: Value[Region], resType: PCanonicalArray): SIndexablePointerValue = {
    // dump all elements into tree for simplicity
    dumpBuffer(cb)

    val (pushElement, finish) = resType.constructFromFunctions(cb, region, treeSize, deepCopy = true)
    cb.if_(treeSize > 0, {
      tree.foreach(cb) { (cb, tv) =>
        val pointCode = pointType.loadCheapSCode(cb, key.storageType.loadField(tv, "point"))
        pushElement(cb, IEmitCode.present(cb, pointCode))
      }
    })
    finish(cb)
  }
}

object DownsampleAggregator {
  val resultType: TArray = TArray(TTuple(TFloat64, TFloat64, TArray(TString)))
}

class DownsampleAggregator(arrayType: VirtualTypeWithReq) extends StagedAggregator {
  type State = DownsampleState

  val resultPType: PCanonicalArray = PCanonicalArray(PCanonicalTuple(required = true, PFloat64(true), PFloat64(true), arrayType.canonicalPType))
  val resultEmitType = EmitType(SIndexablePointer(resultPType), true)

  val initOpTypes: Seq[Type] = Array(TInt32)
  val seqOpTypes: Seq[Type] = Array(TFloat64, TFloat64, arrayType.t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(nDivisions) = init
    nDivisions.toI(cb)
      .consume(cb,
        cb += Code._fatal[Unit]("downsample: n_divisions may not be missing"),
        sc => state.init(cb, sc.asInt.value)
      )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(x, y, label) = seq

    state.insert(cb, x, y, label)
  }

  protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, state: DownsampleState, other: DownsampleState): Unit = state.merge(cb, other)

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    // deepCopy is handled by state.resultArray
    IEmitCode.present(cb, state.resultArray(cb, region, resultPType))
  }
}
