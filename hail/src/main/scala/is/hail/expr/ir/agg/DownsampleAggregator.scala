package is.hail.expr.ir.agg

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitMethodBuilder, EmitRegion, ParamType}
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.utils._


class DownsampleBTreeKey(binType: PBaseStruct, pointType: PBaseStruct, kb: EmitClassBuilder[_], region: Code[Region]) extends BTreeKey {
  val storageType: PCanonicalStruct = PCanonicalStruct(required = true,
    "bin" -> binType,
    "point" -> pointType,
    "empty" -> PBooleanRequired)

  val compType: PType = binType
  private val kcomp = kb.getCodeOrdering(binType, CodeOrdering.Compare(), ignoreMissingness = false)

  def isEmpty(off: Code[Long]): Code[Boolean] = coerce[Boolean](Region.loadIRIntermediate(PBooleanRequired)(storageType.fieldOffset(off, "empty")))

  def initializeEmpty(off: Code[Long]): Code[Unit] = Region.storeBoolean(storageType.fieldOffset(off, "empty"), true)

  def copy(src: Code[Long], dest: Code[Long]): Code[Unit] = Region.copyFrom(src, dest, storageType.byteSize)

  def deepCopy(er: EmitRegion, src: Code[Long], dest: Code[Long]): Code[Unit] =
    Code.memoize(src, "dsa_deep_copy_src") { src =>
      Code(
        Region.loadBoolean(storageType.loadField(src, "empty")).orEmpty(Code._fatal[Unit]("key empty!!")),
        StagedRegionValueBuilder.deepCopy(er, storageType, src, dest))
    }

  def compKeys(k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])): Code[Int] = kcomp(k1, k2)

  def loadCompKey(off: Value[Long]): (Code[Boolean], Code[_]) = (const(false), storageType.loadField(off, "bin"))
}


object DownsampleState {
  val serializationEndMarker: Int = 883625255
}

class DownsampleState(val kb: EmitClassBuilder[_], labelType: PArray, maxBufferSize: Int = 256) extends AggregatorState {
  val r: Settable[Region] = kb.genFieldThisRef[Region]("region")
  val region: Value[Region] = r

  val oldRegion: Settable[Region] = kb.genFieldThisRef[Region]("old_region")

  def newState(off: Code[Long]): Code[Unit] = region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.ifx(region.isNull, cb.assign(r, Region.stagedCreate(regionSize)))

  val binType = PCanonicalStruct(required = true, "x" -> PInt32Required, "y" -> PInt32Required)
  val pointType = PCanonicalStruct(required = true, "x" -> PFloat64Required, "y" -> PFloat64Required, "label" -> labelType)

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

  def allocateSpace(): Code[Unit] =
    off := region.allocate(storageType.alignment, storageType.byteSize)

  def init(nDivisions: Code[Int]): Code[Unit] = {
    val mb = kb.genEmitMethod("downsample_init", FastIndexedSeq[ParamType](IntInfo), UnitInfo)
    mb.emit(Code(FastIndexedSeq(
      allocateSpace(),
      this.nDivisions := mb.getCodeParam[Int](1),
      (this.nDivisions < 4).orEmpty(Code._fatal[Unit](const("downsample: require n_divisions >= 4, found ").concat(this.nDivisions.toS))),
      left := 0d,
      right := 0d,
      bottom := 0d,
      top := 0d,
      treeSize := 0,
      tree.init,
      buffer.initialize())))
    mb.invokeCode(nDivisions)
  }

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] = {
    val mb = kb.genEmitMethod("downsample_load", FastIndexedSeq[ParamType](), UnitInfo)
    mb.emit(
      Code(FastIndexedSeq(
        off := src,
        nDivisions := Region.loadInt(storageType.loadField(off, "nDivisions")),
        treeSize := Region.loadInt(storageType.loadField(off, "treeSize")),
        left := Region.loadDouble(storageType.loadField(off, "left")),
        right := Region.loadDouble(storageType.loadField(off, "right")),
        bottom := Region.loadDouble(storageType.loadField(off, "bottom")),
        top := Region.loadDouble(storageType.loadField(off, "top")),
        bufferLeft := Region.loadDouble(storageType.loadField(off, "bufferLeft")),
        bufferRight := Region.loadDouble(storageType.loadField(off, "bufferRight")),
        bufferBottom := Region.loadDouble(storageType.loadField(off, "bufferBottom")),
        bufferTop := Region.loadDouble(storageType.loadField(off, "bufferTop")),
        buffer.loadFrom(storageType.fieldOffset(off, "buffer")),
        root := Region.loadAddress(storageType.fieldOffset(off, "tree"))
      )))
    Code(regionLoader(r), mb.invokeCode())
  }

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] = {
    val mb = kb.genEmitMethod("downsample_store", FastIndexedSeq[ParamType](), UnitInfo)
    mb.emit(Code(FastIndexedSeq(
      off := dest,
      Region.storeInt(storageType.fieldOffset(off, "nDivisions"), nDivisions),
      Region.storeInt(storageType.fieldOffset(off, "treeSize"), treeSize),
      Region.storeDouble(storageType.fieldOffset(off, "left"), left),
      Region.storeDouble(storageType.fieldOffset(off, "right"), right),
      Region.storeDouble(storageType.fieldOffset(off, "bottom"), bottom),
      Region.storeDouble(storageType.fieldOffset(off, "top"), top),
      Region.storeDouble(storageType.fieldOffset(off, "bufferLeft"), bufferLeft),
      Region.storeDouble(storageType.fieldOffset(off, "bufferRight"), bufferRight),
      Region.storeDouble(storageType.fieldOffset(off, "bufferBottom"), bufferBottom),
      Region.storeDouble(storageType.fieldOffset(off, "bufferTop"), bufferTop),
      buffer.storeTo(storageType.fieldOffset(off, "buffer")),
      Region.storeAddress(storageType.fieldOffset(off, "tree"), root)
    )))

    Code(
      mb.invokeCode(),
      region.isValid.orEmpty(Code(regionStorer(region), region.invalidate())))
  }

  def copyFrom(cb: EmitCodeBuilder, _src: Code[Long]): Unit = {
    val mb = kb.genEmitMethod("downsample_copy", FastIndexedSeq[ParamType](LongInfo), UnitInfo)

    val src = mb.getCodeParam[Long](1)
    mb.emit(Code(FastIndexedSeq(
      allocateSpace(),
      nDivisions := Region.loadInt(storageType.loadField(src, "nDivisions")),
      treeSize := Region.loadInt(storageType.loadField(src, "treeSize")),
      left := Region.loadDouble(storageType.loadField(src, "left")),
      right := Region.loadDouble(storageType.loadField(src, "right")),
      bottom := Region.loadDouble(storageType.loadField(src, "top")),
      top := Region.loadDouble(storageType.loadField(src, "bottom")),
      treeSize := Region.loadInt(storageType.loadField(src, "treeSize")),
      tree.deepCopy(Region.loadAddress(storageType.loadField(src, "tree"))),
      buffer.copyFrom(storageType.loadField(src, "buffer")))))
    cb += mb.invokeCode(_src)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val binEnc = binET.buildEncoderMethod(binType, kb)
    val pointEnc = pointET.buildEncoderMethod(pointType, kb)

    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      val mb = kb.genEmitMethod("downsample_serialize", FastIndexedSeq[ParamType](typeInfo[OutputBuffer]), UnitInfo)
      mb.emitWithBuilder { cb =>
        val ob = mb.getCodeParam[OutputBuffer](1)
        cb += dumpBuffer()
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
          cb += binEnc.invokeCode(key.storageType.loadField(src, "bin"), ob)
          cb += pointEnc.invokeCode(key.storageType.loadField(src, "point"), ob)
        }
        cb += ob.writeInt(DownsampleState.serializationEndMarker)
        Code._empty
      }

      cb += mb.invokeCode(ob)
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
        cb += allocateSpace()
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
        cb += tree.init
        tree.bulkLoad(cb, ib) { (cb, ib, destCode) =>
          val dest = cb.newLocal("dss_deser_dest", destCode)
          cb += binDec.invokeCode(region, key.storageType.fieldOffset(dest, "bin"), ib)
          cb += pointDec.invokeCode(region, key.storageType.fieldOffset(dest, "point"), ib)
          cb += Region.storeBoolean(key.storageType.fieldOffset(dest, "empty"), false)
        }
        cb += buffer.initialize()
        cb.assign(serializationEndTag, ib.readInt())
        cb.ifx(serializationEndTag.cne(DownsampleState.serializationEndMarker), {
          cb._fatal("downsample aggregator failed to serialize!")
        })
        Code._empty
      }
      cb += mb.invokeCode(ib)
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

  def insertIntoTree(binX: Code[Int], binY: Code[Int], point: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val name = s"downsample_insert_into_tree_${ deepCopy.toString }"
    val mb = kb.getOrGenEmitMethod(name, (this, name, deepCopy), FastIndexedSeq[ParamType](IntInfo, IntInfo, LongInfo), UnitInfo) { mb =>
      val binX = mb.getCodeParam[Int](1)
      val binY = mb.getCodeParam[Int](2)
      val point = mb.getCodeParam[Long](3)
      val insertOffset = mb.newLocal[Long]("insert_offset")
      val binOffset = mb.newLocal[Long]("bin_offset")
      val insertedPointOffset = mb.newLocal[Long]("inserted_point_offset")
      val binStaging = mb.newLocal[Long]("binStaging")

      mb.emit(Code(
        binStaging := storageType.loadField(off, "binStaging"),
        Region.storeInt(binType.fieldOffset(binStaging, "x"), binX),
        Region.storeInt(binType.fieldOffset(binStaging, "y"), binY),
        insertOffset := tree.getOrElseInitialize(false, binStaging),
        key.isEmpty(insertOffset).orEmpty(
          Code(
            binOffset := key.storageType.loadField(insertOffset, "bin"),
            Region.storeInt(binType.loadField(binOffset, "x"), binX),
            Region.storeInt(binType.loadField(binOffset, "y"), binY),
            insertedPointOffset := key.storageType.loadField(insertOffset, "point"),
            (if (deepCopy)
              StagedRegionValueBuilder.deepCopy(kb, region, pointType, point, insertedPointOffset)
            else
              Region.copyFrom(point, insertedPointOffset, pointType.byteSize)),
            Region.storeBoolean(key.storageType.loadField(insertOffset, "empty"), false),
            treeSize := treeSize + 1
          )
        )
      ))
    }

    mb.invokeCode(binX, binY, point)
  }

  def copyFromTree(other: AppendOnlyBTree): Code[Unit] = {
    val mb = kb.genEmitMethod("downsample_copy_from_tree", FastIndexedSeq[ParamType](), UnitInfo)

    mb.emitWithBuilder { cb =>
        other.foreach(cb) { (cb, v) =>
        val mb = kb.genEmitMethod("downsample_copy_from_tree_foreach", FastIndexedSeq[ParamType](LongInfo), UnitInfo)
        val value = mb.getCodeParam[Long](1)
        val point = mb.newLocal[Long]("point_offset")
        val pointX = mb.newLocal[Double]("point_x")
        val pointY = mb.newLocal[Double]("point_y")
        val lm = mb.newLocal[Boolean]("lm")
        mb.emit(Code(
          point := key.storageType.loadField(value, "point"),
          pointX := Region.loadDouble(pointType.loadField(point, "x")),
          pointY := Region.loadDouble(pointType.loadField(point, "y")),
          lm := pointType.isFieldMissing(point, "label"),
          insertIntoTree(xBinCoordinate(pointX), yBinCoordinate(pointY), point, deepCopy = true)
        ))
        cb += mb.invokeCode(v)
      }
      Code._empty
    }

    mb.invokeCode()
  }

  def min(a: Code[Double], b: Code[Double]): Code[Double] =
    Code.invokeStatic2[java.lang.Double, Double, Double, Double]("min", a, b)

  def intMax(a: Code[Int], b: Code[Int]): Code[Int] =
    Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("max", a, b)

  def max(a: Code[Double], b: Code[Double]): Code[Double] =
    Code.invokeStatic2[java.lang.Double, Double, Double, Double]("max", a, b)

  def isFinite(a: Code[Double]): Code[Boolean] = Code.invokeStatic1[java.lang.Double, Double, Boolean]("isFinite", a)

  def dumpBuffer(): Code[Unit] = {
    val name = "downsample_dump_buffer"
    val mb = kb.getOrGenEmitMethod(name, (this, name), FastIndexedSeq[ParamType](), UnitInfo) { mb =>
      val i = mb.newLocal[Int]("i")
      val point = mb.newLocal[Long]("elt")
      val x = mb.newLocal[Double]("x")
      val y = mb.newLocal[Double]("y")
      mb.emit(Code(FastIndexedSeq(
        buffer.size.ceq(0).orEmpty(Code._return(Code._empty)),
        left := min(left, bufferLeft),
        right := max(right, bufferRight),
        bottom := min(bottom, bufferBottom),
        top := max(top, bufferTop),
        oldRegion := region,
        oldRoot := root,
        r := Region.stagedCreate(regionSize),
        treeSize := 0,
        tree.init,
        copyFromTree(oldRootBTree),
        i := 0,
        Code.whileLoop(i < buffer.size,
          point := coerce[Long](buffer.loadElement(i)._2),
          x := Region.loadDouble(pointType.loadField(point, "x")),
          y := Region.loadDouble(pointType.loadField(point, "y")),
          insertIntoTree(xBinCoordinate(x), yBinCoordinate(y), point, deepCopy = true),
          i := i + 1),
        buffer.initialize(),
        oldRegion.invalidate(),
        allocateSpace()
      )))
    }

    mb.invokeCode()
  }

  def insertPointIntoBuffer(x: Code[Double], y: Code[Double], point: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val name = "downsample_insert_into_buffer"
    val mb = kb.getOrGenEmitMethod(name, (this, name, deepCopy), FastIndexedSeq[ParamType](DoubleInfo, DoubleInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getCodeParam[Double](1)
      val y = mb.getCodeParam[Double](2)
      val point = mb.getCodeParam[Long](3)

      mb.emit(Code(
        bufferLeft := min(bufferLeft, x),
        bufferRight := max(bufferRight, x),
        bufferBottom := min(bufferBottom, y),
        bufferTop := max(bufferTop, y),
        buffer.append(point, deepCopy = deepCopy),
        (buffer.size >= maxBufferSize).orEmpty(dumpBuffer())
      ))
    }

    mb.invokeCode(x, y, point)
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

  def binAndInsert(x: Code[Double], y: Code[Double], point: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val name = "downsample_bin_and_insert"
    val mb = kb.getOrGenEmitMethod(name, (this, name, deepCopy), FastIndexedSeq[ParamType](DoubleInfo, DoubleInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getCodeParam[Double](1)
      val y = mb.getCodeParam[Double](2)
      val point = mb.getCodeParam[Long](3)

      val binX = mb.newLocal[Int]("bin_x")
      val binY = mb.newLocal[Int]("bin_y")

      mb.emit(Code(
        binX := xBinCoordinate(x),
        binY := yBinCoordinate(y),
        checkBounds(binX, binY).mux(
          insertPointIntoBuffer(x, y, point, deepCopy = deepCopy),
          insertIntoTree(binX, binY, point, deepCopy = deepCopy))))
    }
    mb.invokeCode(x, y, point)
  }

  def insert(x: Code[Double], y: Code[Double], lm: Code[Boolean], l: Code[Long]): Code[Unit] = {
    val name = "downsample_insert"
    val mb = kb.getOrGenEmitMethod(name, (this, name), FastIndexedSeq[ParamType](DoubleInfo, DoubleInfo, BooleanInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getCodeParam[Double](1)
      val y = mb.getCodeParam[Double](2)
      val lm = mb.getCodeParam[Boolean](3)
      val l = mb.getCodeParam[Long](4)

      val pointStaging = mb.newLocal[Long]("pointStaging")
      mb.emit(Code(
        (!(isFinite(x) && isFinite(y))).orEmpty(Code._return[Unit](Code._empty)),
        pointStaging := storageType.loadField(off, "pointStaging"),
        Region.storeDouble(pointType.fieldOffset(pointStaging, "x"), x),
        Region.storeDouble(pointType.fieldOffset(pointStaging, "y"), y),
        (if (labelType.required)
          StagedRegionValueBuilder.deepCopy(kb, region, labelType, l, pointType.fieldOffset(pointStaging, "label"))
        else
          lm.mux(
            pointType.setFieldMissing(pointStaging, "label"),
            Code(
              pointType.setFieldPresent(pointStaging, "label"),
              StagedRegionValueBuilder.deepCopy(kb, region, labelType, l, pointType.fieldOffset(pointStaging, "label"))))),
        binAndInsert(x, y, pointStaging, deepCopy = false)))
    }

    val lmField = kb.genFieldThisRef[Boolean]("lm_field")

    Code(
      lmField := lm,
      mb.invokeCode(x, y, lmField, lmField.mux(0L, l))
    )
  }

  def deepCopyAndInsertPoint(point: Code[Long]): Code[Unit] = {
    val name = "downsample_deep_copy_insert_point"
    val mb = kb.getOrGenEmitMethod(name, (this, name), IndexedSeq[ParamType](LongInfo), UnitInfo) { mb =>
      val point = mb.getCodeParam[Long](1)

      val x = mb.newLocal[Double]("x")
      val y = mb.newLocal[Double]("y")

      mb.emit(Code(
        x := Region.loadDouble(pointType.loadField(point, "x")),
        y := Region.loadDouble(pointType.loadField(point, "y")),
        binAndInsert(x, y, point, deepCopy = true)
      ))
    }

    mb.invokeCode(point)
  }

  def merge(cb: EmitCodeBuilder, other: DownsampleState): Unit = {
    val mb = kb.genEmitMethod("downsample_insert_from", FastIndexedSeq[ParamType](), UnitInfo)

    val i = mb.newLocal[Int]("i")
    mb.emitWithBuilder { cb =>
      cb.assign(i, 0)
      cb.whileLoop(i < other.buffer.size, {
        cb += deepCopyAndInsertPoint(coerce[Long](other.buffer.loadElement(i)._2))
        cb.assign(i, i + 1)
      })
      other.tree.foreach(cb) { (cb, value) =>
        cb += deepCopyAndInsertPoint(key.storageType.loadField(value, "point"))
      }
      Code._empty
    }
    cb += mb.invokeCode()
  }

  def result(mb: EmitMethodBuilder[_], srvb: StagedRegionValueBuilder, resultType: PArray): Code[Unit] = {
    val eltType = resultType.elementType.asInstanceOf[PBaseStruct]
    Code(
      dumpBuffer(),
      srvb.addArray(resultType, { srvb => EmitCodeBuilder.scopedVoid(mb) { cb =>
          cb += srvb.start(treeSize)
          cb.ifx(treeSize > 0, {
            tree.foreach(cb) { (cb, tv) =>
              val point = mb.newLocal[Long]("point_offset")
              cb += Code(
                point := key.storageType.loadField(tv, "point"),
                srvb.addBaseStruct(eltType, { srvb =>
                  Code(
                    srvb.start(),
                    srvb.addDouble(Region.loadDouble(pointType.loadField(point, "x"))),
                    srvb.advance(),
                    srvb.addDouble(Region.loadDouble(pointType.loadField(point, "y"))),
                    srvb.advance(),
                    pointType.isFieldDefined(point, "label").mux(
                      srvb.addWithDeepCopy(labelType, pointType.loadField(point, "label")),
                      srvb.setMissing()
                    )
                  )
                }),
                srvb.advance())
            }
          })
      }}))
  }
}

object DownsampleAggregator {
  val resultType: TArray = TArray(TTuple(TFloat64, TFloat64, TArray(TString)))
}

class DownsampleAggregator(arrayType: PArray) extends StagedAggregator {
  type State = DownsampleState

  val resultType: PArray = PCanonicalArray(PCanonicalTuple(required = true, PFloat64(true), PFloat64(true), PType.canonical(arrayType)))

  val initOpTypes: Seq[PType] = Array(PInt32(true))
  val seqOpTypes: Seq[PType] = Array(PFloat64(), PFloat64(), PType.canonical(arrayType))

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(nDivisions) = init
    cb += Code(
      nDivisions.setup,
      nDivisions.m.mux(
        Code._fatal[Unit]("downsample: n_divisions may not be missing"),
        state.init(coerce[Int](nDivisions.v))
      )
    )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(x, y, label) = seq

    cb += Code(
      x.setup,
      y.setup,
      label.setup,
      (!(x.m || y.m)).orEmpty(
        state.insert(coerce[Double](x.v), coerce[Double](y.v), label.m, coerce[Long](label.v))
      )
    )
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = state.merge(cb, other)

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit =
    cb += state.result(cb.emb, srvb, resultType)
}
