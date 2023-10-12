package is.hail.expr.ir.analyses

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.{TableCalculateNewPartitions, WrappedMatrixToValueFunction}
import is.hail.expr.ir.{MatrixRangeReader, _}
import is.hail.io.fs.FS
import is.hail.io.vcf.MatrixVCFReader
import is.hail.methods._
import is.hail.types.virtual._
import is.hail.utils.{Logging, TreeTraversal, toRichBoolean}
import org.apache.commons.codec.digest.MurmurHash3

import java.nio.ByteBuffer
import scala.collection.mutable
import scala.language.implicitConversions

case object SemanticHash extends Logging {
  type Type = Int

  // Picked from https://softwareengineering.stackexchange.com/a/145633
  def extend(x: Type, bytes: Array[Byte]): Type =
    MurmurHash3.hash32x86(bytes, 0, bytes.length, x)


  def apply(ctx: ExecuteContext)(root: BaseIR): Option[Type] =
    ctx.timer.time("SemanticHash") {

      // Running the algorithm on the name-normalised IR
      // removes sensitivity to compiler-generated names
      val nameNormalizedIR = ctx.timer.time("NormalizeNames") {
        new NormalizeNames(_.toString, allowFreeVariables = true)(root)
      }

      val semhash = ctx.timer.time("Hash") {
        def go: Option[Int] = {
          var hash: Type =
            MurmurHash3.DEFAULT_SEED

          // Include an encoding of a node's position in the parent's child array
          // to differentiate between IR trees that look identical when flattened
          for ((ir, index) <- levelOrder(nameNormalizedIR)) {
            encode(ctx.fs, ir, index) match {
              case Right(bytes) =>
                hash = extend(hash, bytes)
              case Left(failure) =>
                log.warn(s"Failed to compute SemanticHash: $failure")
                return None
            }
          }

          Some(hash)
        }

        go
      }

      log.info(s"IR Semantic Hash: $semhash")
      semhash
    }

  private def encode(fs: FS, ir: BaseIR, index: Int)
  : Either[String, Array[Byte]] = {
    val buffer: mutable.ArrayBuilder[Byte] =
      Array.newBuilder[Byte] ++=
        Bytes.fromClass(ir.getClass) ++=
        Bytes.fromInt(index)

    ir match {
      case a: AggExplode =>
        buffer += a.isScan.toByte

      case a: AggFilter =>
        buffer += a.isScan.toByte

      case a: AggFold =>
        buffer += a.isScan.toByte

      case a: AggGroupBy =>
        buffer += a.isScan.toByte

      case a: AggLet =>
        buffer += a.isScan.toByte

      case a: AggArrayPerElement =>
        buffer += a.isScan.toByte

      case Apply(fname, tyArgs, _, retTy, _) =>
        buffer ++= fname.getBytes
        tyArgs.foreach(buffer ++= EncodeTypename(_))
        buffer ++= EncodeTypename(retTy)

      case ApplyAggOp(_, _, AggSignature(op, initOpTys, seqOpTys)) =>
        buffer ++= Bytes.fromClass(op.getClass)
        initOpTys.foreach(buffer ++= EncodeTypename(_))
        seqOpTys.foreach(buffer ++= EncodeTypename(_))

      case ApplyBinaryPrimOp(op, _, _) =>
        buffer ++= Bytes.fromClass(op.getClass)

      case ApplyComparisonOp(op, _, _) =>
        buffer ++= Bytes.fromClass(op.getClass)

      case ApplyIR(fname, tyArgs, _, _) =>
        buffer ++= fname.getBytes
        tyArgs.foreach(buffer ++= EncodeTypename(_))

      case ApplySeeded(fname, _, _, staticUID, retTy) =>
        buffer ++=
          fname.getBytes ++=
          Bytes.fromLong(staticUID) ++=
          EncodeTypename(retTy)

      case ApplySpecial(fname, tyArgs, _, retTy, _) =>
        buffer ++= fname.getBytes
        tyArgs.foreach(buffer ++= EncodeTypename(_))
        buffer ++= EncodeTypename(retTy)

      case ApplyUnaryPrimOp(op, _) =>
        buffer ++= Bytes.fromClass(op.getClass)

      case BlockMatrixToTableApply(_, _, function) =>
        function match {
          case PCRelate(maf, blockSize, kinship, stats) =>
            buffer ++=
              Bytes.fromClass(classOf[PCRelate]) ++=
              Bytes.fromDouble(maf) ++=
              Bytes.fromInt(blockSize) ++=
              kinship.fold(Array.empty[Byte])(Bytes.fromDouble) ++=
              Bytes.fromInt(stats)
        }

      case BlockMatrixRead(reader) =>
        buffer ++= Bytes.fromClass(reader.getClass)
        reader match {
          case _: BlockMatrixNativeReader =>
            reader
              .pathsUsed
              .flatMap(p => fs.glob(p + "/**").filter(_.isFile))
              .foreach(g => buffer ++= getFileHash(fs)(g.getPath))

          case BlockMatrixBinaryReader(path, _, _) =>
            buffer ++= getFileHash(fs)(path)

          case _ =>
            return Left(s"SemanticHash unknown: ${reader.getClass.getName}")
        }

      case Cast(_, typ) =>
        buffer ++= EncodeTypename(typ)

      case EncodedLiteral(_, bytes) =>
        bytes.ba.foreach(buffer ++= _)

      case GetField(struct, name) =>
        buffer ++= Bytes.fromInt(struct.typ.asInstanceOf[TStruct].fieldIdx(name))

      case GetTupleElement(_, idx) =>
        buffer ++= Bytes.fromInt(idx)

      case Literal(typ, value) =>
        buffer ++= EncodeTypename(typ) ++= typ.toJSON(value).toString.getBytes

      case MakeTuple(fields) =>
        fields.foreach { case (index, _) => buffer ++= Bytes.fromInt(index) }

      case MatrixLiteral(ty, tLiteral) =>
        encode(fs, tLiteral, 0) match {
          case Right(bytes) =>
            buffer ++=
              EncodeTypename(ty.globalType) ++=
              EncodeTypename(ty.rowType) ++=
              EncodeTypename(ty.colType) ++=
              EncodeTypename(ty.entryType) ++=
              bytes

          case failure =>
            return failure
        }

      case MatrixRead(_, _, _, reader) =>
        buffer ++= Bytes.fromClass(reader.getClass)
        reader match {
          case MatrixRangeReader(params, nPartitionsAdj) =>
            buffer ++=
              Bytes.fromInt(params.nRows) ++=
              Bytes.fromInt(params.nCols) ++=
              params.nPartitions.fold(Array.empty[Byte])(Bytes.fromInt) ++=
              Bytes.fromInt(nPartitionsAdj)


          case _: MatrixNativeReader =>
            reader
              .pathsUsed
              .flatMap(p => fs.glob(p + "/**").filter(_.isFile))
              .foreach(g => buffer ++= getFileHash(fs)(g.getPath))

          case _: MatrixVCFReader =>
            reader.pathsUsed.foreach(buffer ++= getFileHash(fs)(_))

          case _ =>
            return Left(s"SemanticHash unknown: ${reader.getClass.getName}")
        }

      case MatrixWrite(_, writer) =>
        buffer ++= writer.path.getBytes

      case NDArrayReindex(_, indices) =>
        indices.foreach(buffer ++= Bytes.fromInt(_))

      case Ref(name, _) =>
        buffer ++= name.getBytes

      case RelationalRef(name, _) =>
        buffer ++= name.getBytes

      case SelectFields(struct, names) =>
        val getFieldIndex = struct.typ.asInstanceOf[TStruct].fieldIdx
        names.map(getFieldIndex).foreach(buffer ++= Bytes.fromInt(_))

      case StreamZip(_, _, _, behaviour, _) =>
        buffer ++= Bytes.fromInt(behaviour.id)

      case TableKeyBy(table, keys, _) =>
        val getFieldIndex = table.typ.rowType.fieldIdx
        keys.map(getFieldIndex).foreach(buffer ++= Bytes.fromInt(_))

      case TableJoin(_, _, joinop, key) =>
        buffer ++= joinop.getBytes ++= Bytes.fromInt(key)

      case TableParallelize(_, nPartitions) =>
        nPartitions.foreach(buffer ++= Bytes.fromInt(_))

      case TableRange(count, numPartitions) =>
        buffer ++= Bytes.fromInt(count) ++= Bytes.fromInt(numPartitions)

      case TableRead(_, dropRows, reader) =>
        buffer += dropRows.toByte ++= Bytes.fromClass(reader.getClass)

        reader match {
          case StringTableReader(_, fileStatuses) =>
            fileStatuses.foreach(s => buffer ++= getFileHash(fs)(s.getPath))

          case reader@(_: TableNativeReader | _: TableNativeZippedReader) =>
            reader
              .pathsUsed
              .flatMap(p => fs.glob(p + "/**").filter(_.isFile))
              .foreach(g => buffer ++= getFileHash(fs)(g.getPath))

          case _ =>
            return Left(s"SemanticHash unknown: ${reader.getClass.getName}")
        }

      case TableToValueApply(_, op) =>
        buffer ++= Bytes.fromClass(op.getClass)
        op match {
          case TableCalculateNewPartitions(nPartitions) =>
            buffer ++= Bytes.fromInt(nPartitions)

          case WrappedMatrixToValueFunction(op, _, _, _) =>
            op match {
              case _: ForceCountMatrixTable | _: NPartitionsMatrixTable =>
                buffer ++= Bytes.fromClass(op.getClass)

              case _: MatrixExportEntriesByCol =>
                return Left("SemanticHash unknown: MatrixExportEntriesByCol")
            }

          case _: ForceCountTable | _: NPartitionsTable =>
            buffer ++= Bytes.fromClass(op.getClass)
        }

      case TableWrite(_, writer) =>
        buffer ++= writer.path.getBytes

      // The following are parameterized entirely by the operation's input and the operation itself
      case _: ArrayLen |
           _: ArrayRef |
           _: ArraySlice |
           _: ArraySort |
           _: ArrayZeros |
           _: Begin |
           _: BlockMatrixCollect |
           _: CastToArray |
           _: Coalesce |
           _: CollectDistributedArray |
           _: ConsoleLog |
           _: Consume |
           _: Die |
           _: GroupByKey |
           _: If |
           _: InsertFields |
           _: IsNA |
           _: Let |
           _: LiftMeOut |
           _: MakeArray |
           _: MakeNDArray |
           _: MakeStream |
           _: MakeStruct |
           _: MatrixAggregate |
           _: MatrixColsTable |
           _: MatrixCount |
           _: MatrixMapGlobals |
           _: MatrixMapRows |
           _: MatrixFilterRows |
           _: MatrixMapCols |
           _: MatrixFilterCols |
           _: MatrixMapEntries |
           _: MatrixFilterEntries |
           _: MatrixDistinctByRow |
           _: NDArrayShape |
           _: NDArraySlice |
           _: NDArrayReshape |
           _: NDArrayWrite |
           _: RelationalLet |
           _: RNGSplit |
           _: RNGStateLiteral |
           _: StreamAgg |
           _: StreamDrop |
           _: StreamDropWhile |
           _: StreamFilter |
           _: StreamFlatMap |
           _: StreamFold |
           _: StreamFold2 |
           _: StreamFor |
           _: StreamIota |
           _: StreamLen |
           _: StreamMap |
           _: StreamRange |
           _: StreamTake |
           _: StreamTakeWhile |
           _: TableGetGlobals |
           _: TableAggregate |
           _: TableCollect |
           _: TableCount |
           _: TableDistinct |
           _: TableFilter |
           _: TableMapGlobals |
           _: TableMapRows |
           _: TableRename |
           _: ToArray |
           _: ToDict |
           _: ToSet |
           _: ToStream |
           _: Trap =>
        ()

      // Discrete values
      case _: Void | _: True | _: False =>
        ()

      // integral constants
      case I32(x) => buffer ++= Bytes.fromInt(x)
      case I64(x) => buffer ++= Bytes.fromLong(x)
      case F32(x) => buffer ++= Bytes.fromFloat(x)
      case F64(x) => buffer ++= Bytes.fromDouble(x)
      case NA(typ) => buffer ++= EncodeTypename(typ)
      case Str(x) => buffer ++= x.getBytes

      // In these cases, just return None meaning that two
      // invocations will never return the same thing.
      case _ =>
        return Left(s"SemanticHash unknown: ${ir.getClass.getName}")
    }

    Right(buffer.result())
  }

  def getFileHash(fs: FS)(path: String): Array[Byte] =
    fs.eTag(path) match {
      case Some(etag) =>
        etag.getBytes
      case None =>
        path.getBytes ++ Bytes.fromLong(fs.fileListEntry(path).getModificationTime)
    }


  def levelOrder(root: BaseIR): Iterator[(BaseIR, Int)] = {
    val adj: ((BaseIR, Int)) => Iterator[(BaseIR, Int)] =
      Function.tupled((ir, _) => ir.children.zipWithIndex.iterator)

    TreeTraversal.levelOrder(adj)((root, 0))
  }

  object Bytes {
    def fromInt(x: Int): Array[Byte] =
      ByteBuffer.allocate(4).putInt(x).array()

    def fromLong(x: Long): Array[Byte] =
      ByteBuffer.allocate(8).putLong(x).array()

    def fromFloat(x: Float): Array[Byte] =
      ByteBuffer.allocate(4).putFloat(x).array()

    def fromDouble(x: Double): Array[Byte] =
      ByteBuffer.allocate(8).putDouble(x).array()

    def fromClass(clz: Class[_]): Array[Byte] =
      fromInt(clz.hashCode())
  }

  object CodeGenSupport {
    def lift(hash: SemanticHash.Type): Option[SemanticHash.Type] =
      Some(hash)
  }

}

case object EncodeTypename {
  // Encode `t` as a byte array, excluding field-names
  def apply(t: Type): Array[Byte] = {
    val builder = Array.newBuilder[Byte]

    def go(typ: Type): Unit = {
      builder ++= typ.getClass.getSimpleName.getBytes
      val children = typ.children
      if (children.nonEmpty) {
        builder += '['
        go(children.head)

        children.tail.foreach { t =>
          builder += ','
          go(t)
        }

        builder += ']'
      }
    }

    go(t)

    builder.result()
  }
}
