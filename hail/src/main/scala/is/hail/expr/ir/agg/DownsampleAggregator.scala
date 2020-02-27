package is.hail.expr.ir.agg

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitFunctionBuilder, EmitRegion, EmitTriplet}
import is.hail.expr.types.encoded.EType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.utils._


class DownsampleBTreeKey(binType: PBaseStruct, pointType: PBaseStruct, fb: EmitFunctionBuilder[_], region: Code[Region]) extends BTreeKey {
  val storageType: PStruct = PStruct(required = true,
    "bin" -> binType,
    "point" -> pointType,
    "empty" -> PBooleanRequired)

  val compType: PType = binType
  private val kcomp = fb.getCodeOrdering(binType, CodeOrdering.compare, ignoreMissingness = false)

  def isEmpty(off: Code[Long]): Code[Boolean] = coerce[Boolean](Region.loadIRIntermediate(PBooleanRequired)(storageType.fieldOffset(off, "empty")))

  def initializeEmpty(off: Code[Long]): Code[Unit] = Region.storeBoolean(storageType.fieldOffset(off, "empty"), true)

  def copy(src: Code[Long], dest: Code[Long]): Code[Unit] = Region.copyFrom(src, dest, storageType.byteSize)

  def deepCopy(er: EmitRegion, src: Code[Long], dest: Code[Long]): Code[Unit] =
    Code(
      Region.loadBoolean(storageType.loadField(src, "empty")).orEmpty(Code._fatal("key empty!!")),
      StagedRegionValueBuilder.deepCopy(er, storageType, src, dest)
    )

  def compKeys(k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])): Code[Int] = kcomp(k1, k2)

  def loadCompKey(off: Code[Long]): (Code[Boolean], Code[_]) = (const(false), storageType.loadField(off, "bin"))
}


object DownsampleState {
  val serializationEndMarker: Int = 883625255
}

class DownsampleState(val fb: EmitFunctionBuilder[_], labelType: PArray, maxBufferSize: Int = 256) extends AggregatorState {
  val r: ClassFieldRef[Region] = fb.newField[Region]("region")
  val region: Code[Region] = r.load()

  val oldRegion: ClassFieldRef[Region] = fb.newField[Region]("old_region")

  def newState(off: Code[Long]): Code[Unit] = region.getNewRegion(regionSize)

  def createState: Code[Unit] = region.isNull.mux(r := Region.stagedCreate(regionSize), Code._empty)

  val binType = PStruct(required = true, "x" -> PInt32Required, "y" -> PInt32Required)
  val pointType = PStruct(required = true, "x" -> PFloat64Required, "y" -> PFloat64Required, "label" -> labelType)

  private val binET = EType.defaultFromPType(binType)
  private val pointET = EType.defaultFromPType(pointType)

  private val root: ClassFieldRef[Long] = fb.newField[Long]("root")
  private val oldRoot: ClassFieldRef[Long] = fb.newField[Long]("old_root")

  val key = new DownsampleBTreeKey(binType, pointType, fb, region)
  val tree = new AppendOnlyBTree(fb, key, region, root)
  val buffer = new StagedArrayBuilder(pointType, fb, region, initialCapacity = maxBufferSize)
  val oldRootBTree = new AppendOnlyBTree(fb, key, region, oldRoot)

  private val off: ClassFieldRef[Long] = fb.newField[Long]("offset")
  private val nDivisions: ClassFieldRef[Int] = fb.newField[Int]("n_divisions")
  private val left: ClassFieldRef[Double] = fb.newField[Double]("left")
  private val right: ClassFieldRef[Double] = fb.newField[Double]("right")
  private val bottom: ClassFieldRef[Double] = fb.newField[Double]("bottom")
  private val top: ClassFieldRef[Double] = fb.newField[Double]("top")
  private val bufferLeft: ClassFieldRef[Double] = fb.newField[Double]("buffer_left")
  private val bufferRight: ClassFieldRef[Double] = fb.newField[Double]("buffer_right")
  private val bufferBottom: ClassFieldRef[Double] = fb.newField[Double]("buffer_bottom")
  private val bufferTop: ClassFieldRef[Double] = fb.newField[Double]("buffer_top")
  private val treeSize: ClassFieldRef[Int] = fb.newField[Int]("treeSize")

  val storageType = PStruct(required = true,
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
    val mb = fb.newMethod("downsample_init", Array[TypeInfo[_]](IntInfo), UnitInfo)
    mb.emit(Code(
      allocateSpace(),
      this.nDivisions := mb.getArg[Int](1),
      (this.nDivisions < 4).orEmpty(Code._fatal(const("downsample: require n_divisions >= 4, found ").concat(this.nDivisions.toS))),
      left := 0d,
      right := 0d,
      bottom := 0d,
      top := 0d,
      treeSize := 0,
      tree.init,
      buffer.initialize()))
    mb.invoke(nDivisions)
  }

  override def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] = {
    val mb = fb.newMethod("downsample_load", Array[TypeInfo[_]](), UnitInfo)
    mb.emit(
      Code(
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
      )

    )
    Code(regionLoader(r), mb.invoke())
  }

  override def store(regionStorer: Code[Region] => Code[Unit], dest: Code[Long]): Code[Unit] = {
    val mb = fb.newMethod("downsample_store", Array[TypeInfo[_]](), UnitInfo)
    mb.emit(Code(
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
    ))

    Code(
      mb.invoke(),
      region.isValid.orEmpty(Code(regionStorer(region), region.invalidate())))
  }

  def copyFrom(_src: Code[Long]): Code[Unit] = {
    val mb = fb.newMethod("downsample_copy", Array[TypeInfo[_]](LongInfo), UnitInfo)

    val src = mb.getArg[Long](1)
    mb.emit(Code(
      allocateSpace(),
      nDivisions := Region.loadInt(storageType.loadField(src, "nDivisions")),
      treeSize := Region.loadInt(storageType.loadField(src, "treeSize")),
      left := Region.loadDouble(storageType.loadField(src, "left")),
      right := Region.loadDouble(storageType.loadField(src, "right")),
      bottom := Region.loadDouble(storageType.loadField(src, "top")),
      top := Region.loadDouble(storageType.loadField(src, "bottom")),
      treeSize := Region.loadInt(storageType.loadField(src, "treeSize")),
      tree.deepCopy(src),
      buffer.copyFrom(storageType.loadField(src, "buffer"))))
    mb.invoke(_src)
  }

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit] = {
    val binEnc = binET.buildEncoderMethod(binType, fb)
    val pointEnc = pointET.buildEncoderMethod(pointType, fb)

    { _ob: Code[OutputBuffer] =>
      val mb = fb.newMethod("downsample_serialize", Array[TypeInfo[_]](typeInfo[OutputBuffer]), UnitInfo)
      val ob = mb.getArg[OutputBuffer](1).load()

      mb.emit(Code(
        dumpBuffer(),
        ob.writeInt(nDivisions),
        ob.writeInt(treeSize),
        ob.writeDouble(left),
        ob.writeDouble(right),
        ob.writeDouble(bottom),
        ob.writeDouble(top),
        ob.writeInt(treeSize),
        tree.bulkStore(ob) { (ob, src) =>
          Code(
            Region.loadBoolean(key.storageType.loadField(src, "empty")).orEmpty(Code._fatal("bad")),
            binEnc.invoke(key.storageType.loadField(src, "bin"), ob),
            pointEnc.invoke(key.storageType.loadField(src, "point"), ob))
        },
        ob.writeInt(DownsampleState.serializationEndMarker)
      ))
      mb.invoke(_ob)
    }
  }

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit] = {
    val binDec = binET.buildInplaceDecoderMethod(binType, fb)
    val pointDec = pointET.buildInplaceDecoderMethod(pointType, fb)

    { _ib: Code[InputBuffer] =>
      val mb = fb.newMethod("downsample_deserialize", Array[TypeInfo[_]](typeInfo[InputBuffer]), UnitInfo)
      val ib = mb.getArg[InputBuffer](1).load()
      val serializationEndTag = mb.newLocal[Int]
      mb.emit(
        Code(
          allocateSpace(),
          nDivisions := ib.readInt(),
          treeSize := ib.readInt(),
          left := ib.readDouble(),
          right := ib.readDouble(),
          bottom := ib.readDouble(),
          top := ib.readDouble(),
          bufferLeft := left,
          bufferRight := right,
          bufferBottom := bottom,
          bufferTop := top,
          treeSize := ib.readInt(),
          tree.init,
          tree.bulkLoad(ib) { (ib, dest) =>
            Code(
              binDec.invoke(region, key.storageType.fieldOffset(dest, "bin"), ib),
              pointDec.invoke(region, key.storageType.fieldOffset(dest, "point"), ib),
              Region.storeBoolean(key.storageType.fieldOffset(dest, "empty"), false))
          },
          buffer.initialize(),
          serializationEndTag := ib.readInt(),
          serializationEndTag.cne(DownsampleState.serializationEndMarker).orEmpty(Code._fatal("downsample aggregator failed to serialize!"))
        )
      )
      mb.invoke(_ib)
    }
  }

  val xBinCoordinate: Code[Double] => Code[Int] = {
    val mb = fb.newMethod("downsample_x_bin_coordinate", Array[TypeInfo[_]](DoubleInfo), IntInfo)
    val x = mb.getArg[Double](1)
    mb.emit(right.ceq(left).mux(0, (((x - left) / (right - left)) * nDivisions.toD).toI))
    mb.invoke(_)
  }

  val yBinCoordinate: Code[Double] => Code[Int] = {
    val mb = fb.newMethod("downsample_y_bin_coordinate", Array[TypeInfo[_]](DoubleInfo), IntInfo)
    val y = mb.getArg[Double](1)
    mb.emit(top.ceq(bottom).mux(0, (((y - bottom) / (top - bottom)) * nDivisions.toD).toI))
    mb.invoke(_)
  }

  def insertIntoTree(binX: Code[Int], binY: Code[Int], point: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val name = s"downsample_insert_into_tree_${ deepCopy.toString }"
    val mb = fb.getOrDefineMethod(name, (this, name, deepCopy), Array[TypeInfo[_]](IntInfo, IntInfo, LongInfo), UnitInfo) { mb =>
      val binX = mb.getArg[Int](1)
      val binY = mb.getArg[Int](2)
      val point = mb.getArg[Long](3)
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
              StagedRegionValueBuilder.deepCopy(fb, region, pointType, point, insertedPointOffset)
            else
              Region.copyFrom(point, insertedPointOffset, pointType.byteSize)),
            Region.storeBoolean(key.storageType.loadField(insertOffset, "empty"), false),
            treeSize := treeSize + 1
          )
        )
      ))
    }

    mb.invoke(binX, binY, point)
  }

  def copyFromTree(other: AppendOnlyBTree): Code[Unit] = {
    val mb = fb.newMethod("downsample_copy_from_tree", Array[TypeInfo[_]](), UnitInfo)

    mb.emit(Code(
      other.foreach {
        val mb = fb.newMethod("downsample_copy_from_tree_foreach", Array[TypeInfo[_]](LongInfo), UnitInfo)
        val value = mb.getArg[Long](1)
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
        mb.invoke(_)
      }
    ))

    mb.invoke()
  }

  def min(a: Code[Double], b: Code[Double]): Code[Double] =
    Code.invokeStatic[java.lang.Double, Double, Double, Double]("min", a, b)

  def intMax(a: Code[Int], b: Code[Int]): Code[Int] =
    Code.invokeStatic[java.lang.Integer, Int, Int, Int]("max", a, b)

  def max(a: Code[Double], b: Code[Double]): Code[Double] =
    Code.invokeStatic[java.lang.Double, Double, Double, Double]("max", a, b)

  def isFinite(a: Code[Double]): Code[Boolean] = Code.invokeStatic[java.lang.Double, Double, Boolean]("isFinite", a)

  def dumpBuffer(): Code[Unit] = {
    val name = "downsample_dump_buffer"
    val mb = fb.getOrDefineMethod(name, (this, name), Array[TypeInfo[_]](), UnitInfo) { mb =>
      val i = mb.newLocal[Int]("i")
      val point = mb.newLocal[Long]("elt")
      val x = mb.newLocal[Double]("x")
      val y = mb.newLocal[Double]("y")
      mb.emit(Code(
        buffer.size.ceq(0).orEmpty(Code._return(Code._empty[Unit])),
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
        oldRegion.load().invalidate(),
        allocateSpace()
      ))
    }

    mb.invoke()
  }

  def insertPointIntoBuffer(x: Code[Double], y: Code[Double], point: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val name = "downsample_insert_into_buffer"
    val mb = fb.getOrDefineMethod(name, (this, name, deepCopy), Array[TypeInfo[_]](DoubleInfo, DoubleInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getArg[Double](1)
      val y = mb.getArg[Double](2)
      val point = mb.getArg[Long](3)

      mb.emit(Code(
        bufferLeft := min(bufferLeft, x),
        bufferRight := max(bufferRight, x),
        bufferBottom := min(bufferBottom, y),
        bufferTop := max(bufferTop, y),
        buffer.append(point, deepCopy = deepCopy),
        (buffer.size >= maxBufferSize).orEmpty(dumpBuffer())
      ))
    }

    mb.invoke(x, y, point)
  }

  def checkBounds(xBin: Code[Int], yBin: Code[Int]): Code[Boolean] = {
    val name = "downsample_check_bounds"
    val mb = fb.getOrDefineMethod(name, (this, name), Array[TypeInfo[_]](IntInfo, IntInfo), BooleanInfo) { mb =>
      val xBin = mb.getArg[Int](1)
      val yBin = mb.getArg[Int](2)
      val factor = mb.newLocal[Int]("factor")
      mb.emit(Code(
        factor := nDivisions >> 2,
        treeSize.ceq(0)
          || (xBin < -factor)
          || (xBin > nDivisions + factor)
          || (yBin < -factor)
          || (yBin > nDivisions + factor)))
    }

    mb.invoke(xBin, yBin)
  }

  def binAndInsert(x: Code[Double], y: Code[Double], point: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val name = "downsample_bin_and_insert"
    val mb = fb.getOrDefineMethod(name, (this, name, deepCopy), Array[TypeInfo[_]](DoubleInfo, DoubleInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getArg[Double](1)
      val y = mb.getArg[Double](2)
      val point = mb.getArg[Long](3)

      val binX = mb.newLocal[Int]("bin_x")
      val binY = mb.newLocal[Int]("bin_y")

      mb.emit(Code(
        binX := xBinCoordinate(x),
        binY := yBinCoordinate(y),
        checkBounds(binX, binY).mux(
          insertPointIntoBuffer(x, y, point, deepCopy = deepCopy),
          insertIntoTree(binX, binY, point, deepCopy = deepCopy))))
    }
    mb.invoke(x, y, point)
  }

  def insert(x: Code[Double], y: Code[Double], lm: Code[Boolean], l: Code[Long]): Code[Unit] = {
    val name = "downsample_insert"
    val mb = fb.getOrDefineMethod(name, (this, name), Array[TypeInfo[_]](DoubleInfo, DoubleInfo, BooleanInfo, LongInfo), UnitInfo) { mb =>
      val x = mb.getArg[Double](1)
      val y = mb.getArg[Double](2)
      val lm = mb.getArg[Boolean](3)
      val l = mb.getArg[Long](4)

      val pointStaging = mb.newLocal[Long]("pointStaging")
      mb.emit(Code(
        (!(isFinite(x) && isFinite(y))).orEmpty(Code._return[Unit](Code._empty)),
        pointStaging := storageType.loadField(off, "pointStaging"),
        Region.storeDouble(pointType.fieldOffset(pointStaging, "x"), x),
        Region.storeDouble(pointType.fieldOffset(pointStaging, "y"), y),
        (if (labelType.required)
          StagedRegionValueBuilder.deepCopy(fb, region, labelType, l, pointType.fieldOffset(pointStaging, "label"))
        else
          lm.mux(
            pointType.setFieldMissing(pointStaging, "label"),
            Code(
              pointType.setFieldPresent(pointStaging, "label"),
              StagedRegionValueBuilder.deepCopy(fb, region, labelType, l, pointType.fieldOffset(pointStaging, "label"))))),
        binAndInsert(x, y, pointStaging, deepCopy = false)))
    }

    val lmField = fb.newField[Boolean]("lm_field")

    Code(
      lmField := lm,
      mb.invoke(x, y, lmField, lmField.mux(0L, l))
    )
  }

  def deepCopyAndInsertPoint(point: Code[Long]): Code[Unit] = {
    val name = "downsample_deep_copy_insert_point"
    val mb = fb.getOrDefineMethod(name, (this, name), Array[TypeInfo[_]](LongInfo), UnitInfo) { mb =>
      val point = mb.getArg[Long](1)

      val x = mb.newLocal[Double]("x")
      val y = mb.newLocal[Double]("y")

      mb.emit(Code(
        x := Region.loadDouble(pointType.loadField(point, "x")),
        y := Region.loadDouble(pointType.loadField(point, "y")),
        binAndInsert(x, y, point, deepCopy = true)
      ))
    }

    mb.invoke(point)
  }

  def merge(other: DownsampleState): Code[Unit] = {
    val mb = fb.newMethod("downsample_insert_from", Array[TypeInfo[_]](), UnitInfo)

    val i = mb.newLocal[Int]("i")
    mb.emit(Code(
      i := 0,
      Code.whileLoop(i < other.buffer.size,
        deepCopyAndInsertPoint(coerce[Long](other.buffer.loadElement(i)._2)),
        i := i + 1),
      other.tree.foreach { value => deepCopyAndInsertPoint(key.storageType.loadField(value, "point")) }))
    mb.invoke()
  }

  def result(srvb: StagedRegionValueBuilder, resultType: PArray): Code[Unit] = {
    val mb = fb.newMethod("downsample_result", Array[TypeInfo[_]](), UnitInfo)
    val eltType = resultType.elementType.asInstanceOf[PBaseStruct]
    mb.emit(Code(
      dumpBuffer(),
      srvb.addArray(resultType, { srvb =>
        Code(
          srvb.start(treeSize),
          (treeSize > 0).orEmpty(tree.foreach {
            val mb = fb.newMethod("downsample_result_foreach", Array[TypeInfo[_]](LongInfo), UnitInfo)
            val value = mb.getArg[Long](1)
            val point = mb.newLocal[Long]("point_offset")

            mb.emit(Code(
              point := key.storageType.loadField(value, "point"),
              srvb.addBaseStruct(eltType, { srvb =>
                Code(
                  srvb.start(),
                  srvb.addDouble(Region.loadDouble(pointType.loadField(point, "x"))),
                  srvb.advance(),
                  srvb.addDouble(Region.loadDouble(pointType.loadField(point, "y"))),
                  srvb.advance(),
                  pointType.isFieldDefined(point, "label").mux(
                    srvb.addIRIntermediate(labelType)(pointType.loadField(point, "label")),
                    srvb.setMissing()
                  )
                )
              }),
              srvb.advance()))
            mb.invoke(_)
          }))
      })))

    mb.invoke()
  }
}

object DownsampleAggregator {
  val resultType: TArray = TArray(TTuple(TFloat64(), TFloat64(), TArray(TString())))
}

class DownsampleAggregator(arrayType: PArray) extends StagedAggregator {
  type State = DownsampleState

  val resultType: PArray = PArray(PTuple(PFloat64(), PFloat64(), PType.canonical(arrayType)))

  def createState(fb: EmitFunctionBuilder[_]): State = new DownsampleState(fb, arrayType)

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(nDivisions) = init
    Code(
      nDivisions.setup,
      nDivisions.m.mux(
        Code._fatal("downsample: n_divisions may not be missing"),
        state.init(coerce[Int](nDivisions.v))
      )
    )
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(x, y, label) = seq

    Code(
      x.setup,
      y.setup,
      label.setup,
      (!(x.m || y.m)).orEmpty(
        state.insert(coerce[Double](x.v), coerce[Double](y.v), label.m, coerce[Long](label.v))
      )
    )
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = state.merge(other)

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = state.result(srvb, resultType)
}
