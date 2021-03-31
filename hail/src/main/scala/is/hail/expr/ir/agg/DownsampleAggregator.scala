package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitParamType, EmitRegion, IEmitCode, PCodeEmitParamType, ParamType}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.SIndexablePointerCode
import is.hail.types.virtual._
import is.hail.utils._


class DownsampleBTreeKey(binType: PBaseStruct, pointType: PBaseStruct, kb: EmitClassBuilder[_], region: Code[Region]) extends BTreeKey {
  val storageType: PCanonicalStruct = PCanonicalStruct(required = true,
    "bin" -> binType,
    "point" -> pointType,
    "empty" -> PBooleanRequired)

  val compType: PType = binType
  private val kcomp = kb.getOrderingFunction(binType.sType, CodeOrdering.Compare())

  def isEmpty(cb: EmitCodeBuilder, off: Code[Long]): Code[Boolean] = PBooleanRequired.loadCheapPCode(cb, storageType.loadField(off, "empty")).boolCode(cb)

  def initializeEmpty(cb: EmitCodeBuilder, off: Code[Long]): Unit = cb += Region.storeBoolean(storageType.fieldOffset(off, "empty"), true)

  def copy(cb: EmitCodeBuilder, src: Code[Long], dest: Code[Long]): Unit = cb += Region.copyFrom(src, dest, storageType.byteSize)

  def deepCopy(cb: EmitCodeBuilder, er: EmitRegion, srcc: Code[Long], dest: Code[Long]): Unit = {
    val src = cb.newLocal[Long]("dsa_deep_copy_src", srcc)
    cb.ifx(Region.loadBoolean(storageType.loadField(src, "empty")),
      cb += Code._fatal[Unit]("key empty!"))
    storageType.storeAtAddress(cb, dest, er.region, storageType.loadCheapPCode(cb, src), deepCopy = true)
  }

  def compKeys(cb: EmitCodeBuilder, k1: EmitCode, k2: EmitCode): Code[Int] = kcomp(cb, k1, k2)

  def loadCompKey(cb: EmitCodeBuilder, off: Value[Long]): EmitCode = EmitCode.present(cb.emb, binType.loadCheapPCode(cb, storageType.loadField(off, "bin")))
}


object DownsampleState {
  val serializationEndMarker: Int = 883625255
}

class DownsampleState(val kb: EmitClassBuilder[_], labelType: VirtualTypeWithReq, maxBufferSize: Int = 256) extends AggregatorState {
  private val labelPType = labelType.canonicalPType
  val r: Settable[Region] = kb.genFieldThisRef[Region]("region")
  val region: Value[Region] = r

  val oldRegion: Settable[Region] = kb.genFieldThisRef[Region]("old_region")

  def newState(cb: EmitCodeBuilder, off: Code[Long]): Unit = cb += region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.ifx(region.isNull, cb.assign(r, Region.stagedCreate(regionSize, kb.pool())))

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

  def init(cb: EmitCodeBuilder, nDivisions: Code[Int]): Unit = {
    val mb = kb.genEmitMethod("downsample_init", FastIndexedSeq[ParamType](IntInfo), UnitInfo)
    mb.voidWithBuilder { cb =>
      allocateSpace(cb)
      cb.assign(this.nDivisions, mb.getCodeParam[Int](1).load())

      cb.ifx(this.nDivisions < 4, cb += Code._fatal[Unit](const("downsample: require n_divisions >= 4, found ").concat(this.nDivisions.toS)))

      cb.assign(left, 0d)
      cb.assign(right, 0d)
      cb.assign(bottom, 0d)
      cb.assign(top, 0d)
      cb.assign(treeSize, 0)
      tree.init(cb)
      buffer.initialize(cb)
    }
    cb.invokeVoid(mb, nDivisions)
  }

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, srcc: Code[Long]): Unit = {
    val mb = kb.genEmitMethod("downsample_load", FastIndexedSeq[ParamType](), UnitInfo)
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
    cb.assign(off, srcc)
    regionLoader(cb, r)
    cb.invokeVoid(mb)
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, destc: Code[Long]): Unit = {
    val mb = kb.genEmitMethod("downsample_store", FastIndexedSeq[ParamType](), UnitInfo)
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

    cb.assign(off, destc)
    cb.invokeVoid(mb)
    cb.ifx(region.isValid,
      {
        regionStorer(cb, region)
        cb += region.invalidate()
      })
  }

  def copyFrom(cb: EmitCodeBuilder, _src: Code[Long]): Unit = {
    val mb = kb.genEmitMethod("downsample_copy", FastIndexedSeq[ParamType](LongInfo), UnitInfo)

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
      tree.deepCopy(cb, Region.loadAddress(storageType.loadField(src, "tree")))
      buffer.copyFrom(cb, storageType.loadField(src, "buffer"))
    }
    cb.invokeVoid(mb, _src)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      val mb = kb.genEmitMethod("downsample_serialize", FastIndexedSeq[ParamType](typeInfo[OutputBuffer]), UnitInfo)
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
          cb += Region.loadBoolean(key.storageType.loadField(src, "empty")).orEmpty(Code._fatal[Unit]("bad"))
          val binCode = binType.loadCheapPCode(cb, key.storageType.loadField(src, "bin"))
          binET.buildEncoder(binCode.st, kb).apply(cb, binCode, ob)

          val pointCode = pointType.loadCheapPCode(cb, key.storageType.loadField(src, "point"))
          pointET.buildEncoder(pointCode.st, kb).apply(cb, pointCode, ob)
        }
        cb += ob.writeInt(DownsampleState.serializationEndMarker)
        Code._empty
      }

      cb.invokeVoid(mb, ob)
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val binDec = binET.buildInplaceDecoderMethod(binType, kb)
    val pointDec = pointET.buildInplaceDecoderMethod(pointType, kb)

    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      val mb = kb.genEmitMethod("downsample_deserialize", FastIndexedSeq[ParamType](typeInfo[InputBuffer]), UnitInfo)
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
          cb += binDec.invokeCode(region, key.storageType.fieldOffset(dest, "bin"), ib)
          cb += pointDec.invokeCode(region, key.storageType.fieldOffset(dest, "point"), ib)
          cb += Region.storeBoolean(key.storageType.fieldOffset(dest, "empty"), false)
        }
        buffer.initialize(cb)
        cb.assign(serializationEndTag, ib.readInt())
        cb.ifx(serializationEndTag.cne(DownsampleState.serializationEndMarker), {
          cb._fatal("downsample aggregator failed to serialize!")
        })
        Code._empty
      }
      cb.invokeVoid(mb, ib)
    }
  }

  val xBinCoordinate: Code[Double] => Code[Int] = {
    val mb = kb.genEmitMethod("downsample_x_bin_coordinate", FastIndexedSeq[ParamType](DoubleInfo), IntInfo)
    val x = mb.getCodeParam[Double](1)
    mb.emit(right.ceq(left).mux(0, (((x - left) / (right - left)) * nDivisions.toD).toI))
    mb.invokeCode(_)
  }

  val yBinCoordinate: Code[Double] => Code[Int] = {
    val mb = kb.genEmitMethod("downsample_y_bin_coordinate", FastIndexedSeq[ParamType](DoubleInfo), IntInfo)
    val y = mb.getCodeParam[Double](1)
    mb.emit(top.ceq(bottom).mux(0, (((y - bottom) / (top - bottom)) * nDivisions.toD).toI))
    mb.invokeCode(_)
  }

  def insertIntoTree(cb: EmitCodeBuilder, binX: Code[Int], binY: Code[Int], point: Code[Long], deepCopy: Boolean): Unit = {
    val name = s"downsample_insert_into_tree_${ deepCopy.toString }"
    val mb = kb.getOrGenEmitMethod(name, (this, name, deepCopy), FastIndexedSeq[ParamType](IntInfo, IntInfo, LongInfo), UnitInfo) { mb =>
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
          tree.getOrElseInitialize(cb, EmitCode.present(cb.emb, storageType.fieldType("binStaging").loadCheapPCode(cb, binStaging))))
        cb.ifx(key.isEmpty(cb, insertOffset), {
          cb.assign(binOffset, key.storageType.loadField(insertOffset, "bin"))
          cb += Region.storeInt(binType.loadField(binOffset, "x"), binX)
          cb += Region.storeInt(binType.loadField(binOffset, "y"), binY)
          cb.assign(insertedPointOffset, key.storageType.loadField(insertOffset, "point"))
          pointType.storeAtAddress(cb, insertedPointOffset, region, pointType.loadCheapPCode(cb, point), deepCopy = deepCopy)
          cb += Region.storeBoolean(key.storageType.loadField(insertOffset, "empty"), false)
          cb.assign(treeSize, treeSize + 1)
        })
      }
    }

    cb.invokeVoid(mb, binX, binY, point)
  }

  def copyFromTree(cb: EmitCodeBuilder, other: AppendOnlyBTree): Unit = {
    val mb = kb.genEmitMethod("downsample_copy_from_tree", FastIndexedSeq[ParamType](), UnitInfo)

    mb.voidWithBuilder { cb =>
      other.foreach(cb) { (cb, v) =>
        val mb = kb.genEmitMethod("downsample_copy_from_tree_foreach", FastIndexedSeq[ParamType](LongInfo), UnitInfo)
        val value = mb.getCodeParam[Long](1)
        val point = mb.newLocal[Long]("point_offset")
        val pointX = mb.newLocal[Double]("point_x")
        val pointY = mb.newLocal[Double]("point_y")
        val lm = mb.newLocal[Boolean]("lm")
        mb.voidWithBuilder { cb =>
          cb.assign(point, key.storageType.loadField(value, "point"))
          cb.assign(pointX, Region.loadDouble(pointType.loadField(point, "x")))
          cb.assign(pointY, Region.loadDouble(pointType.loadField(point, "y")))
          cb.assign(lm, pointType.isFieldMissing(point, "label"))
          insertIntoTree(cb, xBinCoordinate(pointX), yBinCoordinate(pointY), point, deepCopy = true)
        }
        cb.invokeVoid(mb, v)
      }
    }

    cb.invokeVoid(mb)
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
    val mb = kb.getOrGenEmitMethod(name, (this, name), FastIndexedSeq[ParamType](), UnitInfo) { mb =>
      val i = mb.newLocal[Int]("i")
      val point = mb.newLocal[Long]("elt")
      val x = mb.newLocal[Double]("x")
      val y = mb.newLocal[Double]("y")
      mb.voidWithBuilder { cb =>
        cb.ifx(buffer.size.ceq(0), cb += Code._return[Unit](Code._empty))
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
        cb.whileLoop(i < buffer.size,
          {
            cb.assign(point, buffer.loadElement(cb, i).value)
            cb.assign(x, Region.loadDouble(pointType.loadField(point, "x")))
            cb.assign(y, Region.loadDouble(pointType.loadField(point, "y")))
            insertIntoTree(cb, xBinCoordinate(x), yBinCoordinate(y), point, deepCopy = true)
            cb.assign(i, i + 1)
          })
        buffer.initialize(cb)
        cb += oldRegion.invalidate()
        allocateSpace(cb)
      }
    }

    cb.invokeVoid(mb)
  }

  def insertPointIntoBuffer(cb: EmitCodeBuilder, x: Code[Double], y: Code[Double], point: Code[Long], deepCopy: Boolean): Unit = {
    val name = "downsample_insert_into_buffer"
    val mb = kb.getOrGenEmitMethod(name, (this, name, deepCopy), FastIndexedSeq[ParamType](DoubleInfo, DoubleInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getCodeParam[Double](1)
      val y = mb.getCodeParam[Double](2)
      val point = mb.getCodeParam[Long](3)

      mb.voidWithBuilder { cb =>
        cb.assign(bufferLeft, min(bufferLeft, x))
        cb.assign(bufferRight, max(bufferRight, x))
        cb.assign(bufferBottom, min(bufferBottom, y))
        cb.assign(bufferTop, max(bufferTop, y))
        buffer.append(cb, pointType.loadCheapPCode(cb, point))
        cb.ifx(buffer.size >= maxBufferSize, dumpBuffer(cb))
      }
    }

    cb.invokeVoid(mb, x, y, point)
  }

  def checkBounds(xBin: Code[Int], yBin: Code[Int]): Code[Boolean] = {
    val name = "downsample_check_bounds"
    val mb = kb.getOrGenEmitMethod(name, (this, name), FastIndexedSeq[ParamType](IntInfo, IntInfo), BooleanInfo) { mb =>
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

    mb.invokeCode(xBin, yBin)
  }

  def binAndInsert(cb: EmitCodeBuilder, x: Code[Double], y: Code[Double], point: Code[Long], deepCopy: Boolean): Unit = {
    val name = "downsample_bin_and_insert"
    val mb = kb.getOrGenEmitMethod(name, (this, name, deepCopy), FastIndexedSeq[ParamType](DoubleInfo, DoubleInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getCodeParam[Double](1)
      val y = mb.getCodeParam[Double](2)
      val point = mb.getCodeParam[Long](3)

      val binX = mb.newLocal[Int]("bin_x")
      val binY = mb.newLocal[Int]("bin_y")

      mb.voidWithBuilder { cb =>
        cb.assign(binX, xBinCoordinate(x))
        cb.assign(binY, yBinCoordinate(y))
        cb.ifx(checkBounds(binX, binY),
          insertPointIntoBuffer(cb, x, y, point, deepCopy = deepCopy),
          insertIntoTree(cb, binX, binY, point, deepCopy = deepCopy)
        )
      }
    }
    cb.invokeVoid(mb, x, y, point)
  }

  def insert(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, l: EmitCode): Unit = {
    val name = "downsample_insert"
    val mb = kb.getOrGenEmitMethod(name, (this, name), FastIndexedSeq[ParamType](x.pv.st.pType.asParam, y.pv.st.pType.asParam, PCodeEmitParamType(l.pv.st.pType)), UnitInfo) { mb =>

      val pointStaging = mb.newLocal[Long]("pointStaging")
      mb.voidWithBuilder { cb =>
        val x = mb.getPCodeParam(1)
          .memoize(cb, "downsample_insert_x")
        val y = mb.getPCodeParam(2)
          .memoize(cb, "downsample_insert_y")
        val l = mb.getEmitParam(3, region)

        def xx = x.asDouble.doubleCode(cb)

        def yy = y.asDouble.doubleCode(cb)

        cb.ifx((!(isFinite(xx) && isFinite(yy))), cb += Code._return[Unit](Code._empty))
        cb.assign(pointStaging, storageType.loadField(off, "pointStaging"))
        pointType.fieldType("x").storeAtAddress(cb, pointType.fieldOffset(pointStaging, "x"), region, x, deepCopy = true)
        pointType.fieldType("y").storeAtAddress(cb, pointType.fieldOffset(pointStaging, "y"), region, y, deepCopy = true)
        l.toI(cb)
          .consume(cb,
            cb += pointType.setFieldMissing(pointStaging, "label"),
            { sc =>
              cb += pointType.setFieldPresent(pointStaging, "label")
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
              ycode => cb.invokeVoid(mb, xcode, ycode, l)
            )
        })
  }

  def deepCopyAndInsertPoint(cb: EmitCodeBuilder, point: Code[Long]): Unit = {
    val name = "downsample_deep_copy_insert_point"
    val mb = kb.getOrGenEmitMethod(name, (this, name), IndexedSeq[ParamType](LongInfo), UnitInfo) { mb =>
      val point = mb.getCodeParam[Long](1)

      val x = mb.newLocal[Double]("x")
      val y = mb.newLocal[Double]("y")

      mb.voidWithBuilder { cb =>
        cb.assign(x, Region.loadDouble(pointType.loadField(point, "x")))
        cb.assign(y, Region.loadDouble(pointType.loadField(point, "y")))
        binAndInsert(cb, x, y, point, deepCopy = true)
      }
    }

    cb.invokeVoid(mb, point)
  }

  def merge(cb: EmitCodeBuilder, other: DownsampleState): Unit = {
    val mb = kb.genEmitMethod("downsample_insert_from", FastIndexedSeq[ParamType](), UnitInfo)

    val i = mb.newLocal[Int]("i")
    mb.emitWithBuilder { cb =>
      cb.assign(i, 0)
      cb.whileLoop(i < other.buffer.size, {
        deepCopyAndInsertPoint(cb, other.buffer.loadElement(cb, i).value)
        cb.assign(i, i + 1)
      })
      other.tree.foreach(cb) { (cb, value) =>
        deepCopyAndInsertPoint(cb, key.storageType.loadField(value, "point"))
      }
      Code._empty
    }
    cb.invokeVoid(mb)
  }

  def resultArray(cb: EmitCodeBuilder, region: Value[Region], resType: PCanonicalArray): SIndexablePointerCode = {
    // dump all elements into tree for simplicity
    dumpBuffer(cb)

    val eltType = resType.elementType.asInstanceOf[PCanonicalBaseStruct]

    val (pushElement, finish) = resType.constructFromFunctions(cb, region, treeSize, deepCopy = true)
    cb.ifx(treeSize > 0, {
      tree.foreach(cb) { (cb, tv) =>
        val pointCode = pointType.loadCheapPCode(cb, key.storageType.loadField(tv, "point"))
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

  val resultType: PCanonicalArray = PCanonicalArray(PCanonicalTuple(required = true, PFloat64(true), PFloat64(true), arrayType.canonicalPType))

  val initOpTypes: Seq[Type] = Array(TInt32)
  val seqOpTypes: Seq[Type] = Array(TFloat64, TFloat64, arrayType.t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(nDivisions) = init
    nDivisions.toI(cb)
      .consume(cb,
        cb += Code._fatal[Unit]("downsample: n_divisions may not be missing"),
        sc => state.init(cb, sc.asInt.intCode(cb))
      )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(x, y, label) = seq

    state.insert(cb, x, y, label)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = state.merge(cb, other)

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    assert(pt == resultType)
    // deepCopy is handled by state.resultArray
    pt.storeAtAddress(cb, addr, region, state.resultArray(cb, region, resultType), deepCopy = false)
  }
}
