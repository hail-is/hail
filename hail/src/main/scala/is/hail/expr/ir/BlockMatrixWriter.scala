package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.Nat
import is.hail.expr.ir.lowering.{BlockMatrixStage2, LowererUnsupportedOperation}
import is.hail.io.{StreamBufferSpec, TypedCodecSpec}
import is.hail.io.fs.FS
import is.hail.linalg.{BlockMatrix, BlockMatrixMetadata}
import is.hail.types.{BlockMatrixType, TypeWithRequiredness}
import is.hail.types.encoded.{EBlockMatrixNDArray, ENumpyBinaryNDArray, EType}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.richUtils.RichDenseMatrixDouble

import org.json4s.{jackson, DefaultFormats, Formats, ShortTypeHints}

import java.io.DataOutputStream

object BlockMatrixWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(
        classOf[BlockMatrixNativeWriter],
        classOf[BlockMatrixBinaryWriter],
        classOf[BlockMatrixRectanglesWriter],
        classOf[BlockMatrixBinaryMultiWriter],
        classOf[BlockMatrixTextMultiWriter],
        classOf[BlockMatrixPersistWriter],
        classOf[BlockMatrixNativeMultiWriter],
      ),
      typeHintFieldName = "name",
    )
  }
}

abstract class BlockMatrixWriter {
  def pathOpt: Option[String]
  def apply(ctx: ExecuteContext, bm: BlockMatrix): Any
  def loweredTyp: Type

  def lower(
    ctx: ExecuteContext,
    s: BlockMatrixStage2,
    evalCtx: IRBuilder,
    eltR: TypeWithRequiredness,
  ): IR =
    throw new LowererUnsupportedOperation(s"unimplemented writer: \n${this.getClass}")
}

case class BlockMatrixNativeWriter(
  path: String,
  overwrite: Boolean,
  forceRowMajor: Boolean,
  stageLocally: Boolean,
) extends BlockMatrixWriter {
  def pathOpt: Option[String] = Some(path)

  def apply(ctx: ExecuteContext, bm: BlockMatrix): Unit =
    bm.write(ctx, path, overwrite, forceRowMajor, stageLocally)

  def loweredTyp: Type = TVoid

  override def lower(
    ctx: ExecuteContext,
    s: BlockMatrixStage2,
    evalCtx: IRBuilder,
    eltR: TypeWithRequiredness,
  ): IR = {
    val etype = EBlockMatrixNDArray(
      EType.fromTypeAndAnalysis(s.typ.elementType, eltR),
      encodeRowMajor = forceRowMajor,
      required = true,
    )
    val spec = TypedCodecSpec(etype, TNDArray(s.typ.elementType, Nat(2)), BlockMatrix.bufferSpec)
    val writer = ETypeValueWriter(spec)

    val paths = s.collectBlocks(evalCtx, "block_matrix_native_writer") { (_, idx, block) =>
      val suffix = strConcat("parts/part-", idx, UUID4())
      val filepath = strConcat(s"$path/", suffix)
      WriteValue(
        block,
        filepath,
        writer,
        if (stageLocally) Some(strConcat(s"${ctx.localTmpdir}/", suffix)) else None,
      )
    }
    RelationalWriter.scoped(path, overwrite, None)(WriteMetadata(
      paths,
      BlockMatrixNativeMetadataWriter(path, stageLocally, s.typ),
    ))
  }
}

case class BlockMatrixNativeMetadataWriter(
  path: String,
  stageLocally: Boolean,
  typ: BlockMatrixType,
) extends MetadataWriter {

  case class BMMetadataHelper(
    path: String,
    blockSize: Int,
    nRows: Long,
    nCols: Long,
    partIdxToBlockIdx: Option[IndexedSeq[Int]],
  ) {
    def write(fs: FS, rawPartFiles: Array[String]): Unit = {
      val partFiles = rawPartFiles.map(_.split('/').last)
      using(new DataOutputStream(fs.create(s"$path/metadata.json"))) { os =>
        implicit val formats = defaultJSONFormats
        jackson.Serialization.write(
          BlockMatrixMetadata(blockSize, nRows, nCols, partIdxToBlockIdx, partFiles),
          os,
        )
      }
      val nBlocks = partIdxToBlockIdx.map(_.length).getOrElse {
        BlockMatrixType.numBlocks(nRows, blockSize) * BlockMatrixType.numBlocks(nCols, blockSize)
      }
      assert(nBlocks == partFiles.length, s"$nBlocks vs ${partFiles.mkString(", ")}")

      info(s"wrote matrix with $nRows ${plural(nRows, "row")} " +
        s"and $nCols ${plural(nCols, "column")} " +
        s"as $nBlocks ${plural(nBlocks, "block")} " +
        s"of size $blockSize to $path")
    }
  }

  def annotationType: Type = TArray(TString)

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    val metaHelper =
      BMMetadataHelper(path, typ.blockSize, typ.nRows, typ.nCols, typ.linearizedDefinedBlocks)

    val pc = writeAnnotations.get(cb, "write annotations can't be missing!").asIndexable
    val partFiles = cb.newLocal[Array[String]]("partFiles")
    val n = cb.newLocal[Int]("n", pc.loadLength())
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partFiles, Code.newArray[String](n))
    cb.while_(
      i < n, {
        val s = pc.loadElement(cb, i).get(cb, "file name can't be missing!").asString
        cb += partFiles.update(i, s.loadString(cb))
        cb.assign(i, i + 1)
      },
    )
    cb += cb.emb.getObject(metaHelper).invoke[FS, Array[String], Unit](
      "write",
      cb.emb.getFS,
      partFiles,
    )
  }

  def loweredTyp: Type = TVoid
}

case class BlockMatrixBinaryWriter(path: String) extends BlockMatrixWriter {
  def pathOpt: Option[String] = Some(path)

  def apply(ctx: ExecuteContext, bm: BlockMatrix): String = {
    RichDenseMatrixDouble.exportToDoubles(ctx.fs, path, bm.toBreezeMatrix(), forceRowMajor = true)
    path
  }

  def loweredTyp: Type = TString

  override def lower(
    ctx: ExecuteContext,
    s: BlockMatrixStage2,
    evalCtx: IRBuilder,
    eltR: TypeWithRequiredness,
  ): IR = {
    val nd = s.collectLocal(evalCtx, "block_matrix_binary_writer")

    // FIXME remove numpy encoder
    val etype = ENumpyBinaryNDArray(s.typ.nRows, s.typ.nCols, true)
    val spec = TypedCodecSpec(etype, TNDArray(s.typ.elementType, Nat(2)), new StreamBufferSpec())
    val writer = ETypeValueWriter(spec)
    WriteValue(nd, Str(path), writer)
  }
}

case class BlockMatrixPersistWriter(id: String, storageLevel: String) extends BlockMatrixWriter {
  def pathOpt: Option[String] = None

  def apply(ctx: ExecuteContext, bm: BlockMatrix): Unit =
    HailContext.backend.persist(ctx.backendContext, id, bm, storageLevel)

  def loweredTyp: Type = TVoid
}

case class BlockMatrixRectanglesWriter(
  path: String,
  rectangles: Array[Array[Long]],
  delimiter: String,
  binary: Boolean,
) extends BlockMatrixWriter {

  def pathOpt: Option[String] = Some(path)

  def apply(ctx: ExecuteContext, bm: BlockMatrix): Unit =
    bm.exportRectangles(ctx, path, rectangles, delimiter, binary)

  def loweredTyp: Type = TVoid
}

abstract class BlockMatrixMultiWriter {
  def apply(ctx: ExecuteContext, bms: IndexedSeq[BlockMatrix]): Unit
}

case class BlockMatrixBinaryMultiWriter(
  prefix: String,
  overwrite: Boolean,
) extends BlockMatrixMultiWriter {

  def apply(ctx: ExecuteContext, bms: IndexedSeq[BlockMatrix]): Unit =
    BlockMatrix.binaryWriteBlockMatrices(ctx.fs, bms, prefix, overwrite)

  def loweredTyp: Type = TVoid
}

case class BlockMatrixTextMultiWriter(
  prefix: String,
  overwrite: Boolean,
  delimiter: String,
  header: Option[String],
  addIndex: Boolean,
  compression: Option[String],
  customFilenames: Option[Array[String]],
) extends BlockMatrixMultiWriter {

  def apply(ctx: ExecuteContext, bms: IndexedSeq[BlockMatrix]): Unit =
    BlockMatrix.exportBlockMatrices(
      ctx.fs,
      bms,
      prefix,
      overwrite,
      delimiter,
      header,
      addIndex,
      compression,
      customFilenames,
    )

  def loweredTyp: Type = TVoid
}

case class BlockMatrixNativeMultiWriter(
  prefix: String,
  overwrite: Boolean,
  forceRowMajor: Boolean,
) extends BlockMatrixMultiWriter {

  def apply(ctx: ExecuteContext, bms: IndexedSeq[BlockMatrix]): Unit =
    BlockMatrix.writeBlockMatrices(ctx, bms, prefix, overwrite, forceRowMajor)

  def loweredTyp: Type = TVoid
}
