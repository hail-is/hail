package is.hail.io.columnar

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.expr.ir.defs.PartitionReader
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.FS
import is.hail.types.{RStruct, TypeWithRequiredness}
import is.hail.types.arrow.ArrowSchemaConversion
import is.hail.types.physical._
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SCanonicalCallValue, SPackedLocus, SPackedLocusValue}
import is.hail.types.physical.stypes.interfaces.{primitive, SIndexableValue, SStreamValue}
import is.hail.types.virtual._

import scala.collection.mutable

import org.apache.parquet.schema.{GroupType, MessageType, Type => ParquetType}
import org.apache.parquet.schema.Type.Repetition
import org.json4s.{Extraction, JValue}

/** Reads one Parquet data file of the columnar format as a stream of Hail structs.
  *
  * `fileType` is the Hail type of the file's fields, as converted from its Arrow schema by
  * [[ArrowSchemaConversion]], and `arrowSchema` that schema as a base64 encapsulated IPC message.
  * The file's footer must hold the same Arrow schema (spec section 3).
  *
  * The row's UID, `uidFieldName`, is the `rowIdField` field if there is one, and otherwise the
  * row's position in the file. The context is `{path}`, the file's absolute path.
  *
  * Only the column chunks of requested fields are read, at any depth of nesting.
  */
case class ColumnarPartitionReader(
  fileType: PStruct,
  arrowSchema: String,
  rowIdField: Option[String],
  uidFieldName: String,
) extends PartitionReader {

  override def contextType: TStruct = TStruct("path" -> TString)

  private def dataFields: IndexedSeq[PField] =
    fileType.fields.filterNot(f => rowIdField.contains(f.name))

  override def fullRowType: TStruct =
    TStruct(dataFields.map(f => f.name -> f.typ.virtualType) :+ (uidFieldName -> TInt64): _*)

  /** The Hail type of `requestedType`'s fields as read. */
  def rowPType(requestedType: TStruct): PCanonicalStruct =
    PCanonicalStruct(
      required = true,
      requestedType.fields.map { f =>
        f.name -> (
          if (f.name == uidFieldName) PInt64(true)
          else ColumnarReadPlan.subset(fileType.field(f.name).typ, f.typ)
        )
      }: _*
    )

  override def rowRequiredness(requestedType: TStruct): RStruct = {
    val r = TypeWithRequiredness(requestedType).asInstanceOf[RStruct]
    r.fromPType(rowPType(requestedType))
    r
  }

  override def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct,
  ): IEmitCode =
    context.toI(cb).map(cb) { ctxValue =>
      val path = cb.memoizeField(
        ctxValue.asBaseStruct.loadField(cb, "path").getOrAssert(cb).asString.loadString(cb),
        "columnar_path",
      )

      val parquetSchema = ColumnarSchema.toParquet(ColumnarFileSink.decodeSchema(arrowSchema))
      val plan = ColumnarReadPlan(fileType, parquetSchema, rowIdField, uidFieldName, requestedType)
      val spec = ColumnarSourceSpec(arrowSchema, plan.leafPaths)

      val source = mb.genFieldThisRef[ColumnarFileSource]("columnar_source")
      val leaves = plan.leafPaths.indices.map(i =>
        mb.genFieldThisRef[ColumnarLeaf](s"columnar_leaf$i")
      )
      val position = mb.genFieldThisRef[Long]("columnar_position")
      val region = mb.genFieldThisRef[Region]("columnar_region")

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb

        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
          cb.assign(
            source,
            cb.emb.getObject(spec).invoke[FS, String, ColumnarFileSource](
              "open",
              cb.emb.getFS,
              path,
            ),
          )
          leaves.zipWithIndex.foreach { case (l, i) =>
            cb.assign(l, source.invoke[Int, ColumnarLeaf]("leaf", i))
          }
          cb.assign(position, -1L)
        }

        override val elementRegion: Settable[Region] = region

        override val requiresMemoryManagementPerElement: Boolean = true

        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.if_(!source.invoke[Boolean]("nextRecord"), cb.goto(LendOfStream))
          cb.assign(position, position + 1L)
          cb.goto(LproduceElementDone)
        }

        override val element: EmitCode = EmitCode.fromI(mb) { cb =>
          val emitter = new ColumnarValueEmitter(leaves, region)
          IEmitCode.present(
            cb,
            plan.rowPType.constructFromFields(
              cb,
              region,
              plan.fields.map {
                case None => EmitCode.present(mb, primitive(position))
                case Some(p) => EmitCode.fromI(mb)(cb => emitter.read(cb, p))
              },
              deepCopy = false,
            ),
          )
        }

        override def close(cb: EmitCodeBuilder): Unit = cb += source.invoke[Unit]("close")
      }

      SStreamValue(producer)
    }

  override def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

/** How to assemble one Hail value from the buffered leaves of a record.
  *
  * `pt` is the value's Hail type. `leaves` are the indices of the leaves read under it, never
  * empty. `definitionLevel` is the definition level of a present value, and `repetitionLevel` the
  * repetition level of the node holding it. `optional` is whether the Parquet node may be null.
  */
sealed abstract private class ReadNode {
  def pt: PType

  def leaves: IndexedSeq[Int]

  def definitionLevel: Int

  def repetitionLevel: Int

  def optional: Boolean
}

private case class LeafNode(
  pt: PType,
  leaf: Int,
  definitionLevel: Int,
  repetitionLevel: Int,
  optional: Boolean,
) extends ReadNode {
  override def leaves: IndexedSeq[Int] = FastSeq(leaf)
}

/** A struct or tuple. `structural` are leaves outside every requested field, read only to tell
  * whether the struct is present when no requested field can.
  */
private case class StructNode(
  pt: PCanonicalBaseStruct,
  fields: IndexedSeq[ReadNode],
  structural: IndexedSeq[Int],
  definitionLevel: Int,
  repetitionLevel: Int,
  optional: Boolean,
) extends ReadNode {
  override def leaves: IndexedSeq[Int] = fields.flatMap(_.leaves) ++ structural
}

private case class IntervalNode(
  pt: PCanonicalInterval,
  start: ReadNode,
  end: ReadNode,
  includesStart: ReadNode,
  includesEnd: ReadNode,
  definitionLevel: Int,
  repetitionLevel: Int,
  optional: Boolean,
) extends ReadNode {
  override def leaves: IndexedSeq[Int] =
    FastSeq(start, end, includesStart, includesEnd).flatMap(_.leaves)
}

/** A three-level `LIST` or a `MAP`, read as an array of `element`, whose node repeats at
  * `repetitionLevel + 1`. A dict's elements are its key-value structs.
  */
private case class ListNode(
  pt: PContainer,
  element: ReadNode,
  definitionLevel: Int,
  repetitionLevel: Int,
  optional: Boolean,
) extends ReadNode {
  override def leaves: IndexedSeq[Int] = element.leaves

  def arrayType: PCanonicalArray = pt match {
    case t: PCanonicalArray => t
    case t: PCanonicalSet => t.arrayRep
    case t: PCanonicalDict => t.arrayRep
  }
}

private case class NDArrayNode(
  pt: PCanonicalNDArray,
  data: ListNode,
  shape: ListNode,
  definitionLevel: Int,
  repetitionLevel: Int,
  optional: Boolean,
) extends ReadNode {
  override def leaves: IndexedSeq[Int] = data.leaves ++ shape.leaves
}

/** The plan for reading `requestedType` from a data file: the Parquet paths of the leaves to read,
  * and for each requested field, how to assemble it, or `None` for a UID that is the row's
  * position.
  */
private case class ColumnarReadPlan(
  rowPType: PCanonicalStruct,
  fields: IndexedSeq[Option[ReadNode]],
  leafPaths: IndexedSeq[IndexedSeq[String]],
)

private object ColumnarReadPlan {

  /** `pt` restricted to the fields of `t`, in `t`'s order. */
  def subset(pt: PType, t: Type): PType = (pt, t) match {
    case (pt: PCanonicalStruct, t: TStruct) =>
      PCanonicalStruct(
        pt.required,
        t.fields.map(f => f.name -> subset(pt.field(f.name).typ, f.typ)): _*
      )
    case (pt: PCanonicalTuple, t: TTuple) =>
      PCanonicalTuple(
        t._types.map(f => PTupleField(f.index, subset(pt.types(pt.fieldIndex(f.index)), f.typ))),
        pt.required,
      )
    case (pt: PCanonicalArray, t: TArray) =>
      PCanonicalArray(subset(pt.elementType, t.elementType), pt.required)
    case (pt: PCanonicalSet, t: TSet) =>
      PCanonicalSet(subset(pt.elementType, t.elementType), pt.required)
    case (pt: PCanonicalDict, t: TDict) =>
      PCanonicalDict(subset(pt.keyType, t.keyType), subset(pt.valueType, t.valueType), pt.required)
    case _ =>
      assert(pt.virtualType == t, s"$pt is not a supertype of $t")
      pt
  }

  def apply(
    fileType: PStruct,
    schema: MessageType,
    rowIdField: Option[String],
    uidFieldName: String,
    requestedType: TStruct,
  ): ColumnarReadPlan = {
    val leafPaths = mutable.ArrayBuffer[IndexedSeq[String]]()
    val leafIndex = mutable.Map[IndexedSeq[String], Int]()

    def leaf(path: IndexedSeq[String]): Int =
      leafIndex.getOrElseUpdate(path, { leafPaths += path; leafPaths.length - 1 })

    def firstLeaf(t: ParquetType, path: IndexedSeq[String]): IndexedSeq[String] =
      if (t.isPrimitive) path
      else {
        val f = t.asGroupType.getType(0)
        firstLeaf(f, path :+ f.getName)
      }

    def child(g: ParquetType, name: String): ParquetType = g.asGroupType.getType(name)

    /** `pq` is the node at `path` holding a value of type `pt`; `d` and `r` are the definition and
      * repetition levels of its parent.
      */
    def plan(pt: PType, t: Type, pq: ParquetType, path: IndexedSeq[String], d: Int, r: Int)
      : ReadNode = {
      val optional = pq.isRepetition(Repetition.OPTIONAL)
      val dl = if (optional) d + 1 else d

      def list(pt: PContainer, element: (ParquetType, IndexedSeq[String], Int, Int) => ReadNode)
        : ListNode = {
        val repeated = pq.asGroupType.getType(0)
        ListNode(pt, element(repeated, path :+ repeated.getName, dl + 1, r + 1), dl, r, optional)
      }

      def listOf(pt: PContainer, elementType: Type): ListNode =
        list(
          pt,
          (repeated, rpath, d, r) => {
            val e = child(repeated, ArrowSchemaConversion.ElementFieldName)
            plan(pt.elementType, elementType, e, rpath :+ e.getName, d, r)
          },
        )

      (pt, t) match {
        case (pt: PCanonicalBaseStruct, _) if pt.size == 0 && pq.isPrimitive =>
          LeafNode(pt, leaf(path), dl, r, optional)
        case (pt: PCanonicalInterval, t: TInterval) =>
          val IndexedSeq(start, end, inclStart, inclEnd) =
            ArrowSchemaConversion.IntervalFieldNames.map { name =>
              val c = child(pq, name)
              val cpt = if (name.startsWith("includes")) PBoolean(true) else pt.pointType
              val ct = if (name.startsWith("includes")) TBoolean else t.pointType
              plan(cpt, ct, c, path :+ name, dl, r)
            }
          IntervalNode(pt, start, end, inclStart, inclEnd, dl, r, optional)
        case (pt: PCanonicalNDArray, _) =>
          val IndexedSeq(dataName, shapeName) = ArrowSchemaConversion.TensorFieldNames
          def part(name: String, ept: PType) = {
            val c = child(pq, name)
            plan(
              PCanonicalArray(ept, required = true),
              TArray(ept.virtualType),
              c,
              path :+ name,
              dl,
              r,
            )
              .asInstanceOf[ListNode]
          }
          NDArrayNode(
            pt,
            part(dataName, pt.elementType),
            part(shapeName, PInt32(true)),
            dl,
            r,
            optional,
          )
        case (pt: PCanonicalDict, t: TDict) =>
          list(
            pt,
            (repeated, rpath, d, r) => {
              val kv = pt.arrayRep.elementType.asInstanceOf[PCanonicalStruct]
              val k = child(repeated, ArrowSchemaConversion.KeyFieldName)
              val v = child(repeated, ArrowSchemaConversion.ValueFieldName)
              StructNode(
                kv,
                FastSeq(
                  plan(pt.keyType, t.keyType, k, rpath :+ k.getName, d, r),
                  plan(pt.valueType, t.valueType, v, rpath :+ v.getName, d, r),
                ),
                FastSeq(),
                d,
                r,
                optional = false,
              )
            },
          )
        case (pt: PContainer, t: TContainer) => listOf(pt, t.elementType)
        case (pt: PCanonicalBaseStruct, t: TBaseStruct) =>
          val fields = pt.fields.zip(t.types).map { case (f, ft) =>
            val c = child(pq, f.name)
            plan(f.typ, ft, c, path :+ f.name, dl, r)
          }
          val structural =
            if (fields.nonEmpty) FastSeq()
            else FastSeq(leaf(firstLeaf(pq, path)))
          StructNode(pt, fields, structural, dl, r, optional)
        case _ =>
          LeafNode(pt, leaf(path), dl, r, optional)
      }
    }

    val rowPTypeFields = mutable.ArrayBuffer[(String, PType)]()
    val fields = requestedType.fields.map { f =>
      val fileField = if (f.name == uidFieldName) rowIdField else Some(f.name)
      fileField match {
        case Some(name) =>
          val pt = subset(fileType.field(name).typ, f.typ)
          rowPTypeFields += f.name -> pt
          Some(plan(pt, f.typ, (schema: GroupType).getType(name), FastSeq(name), 0, 0))
        case None =>
          rowPTypeFields += f.name -> PInt64(true)
          None
      }
    }

    ColumnarReadPlan(
      PCanonicalStruct(true, rowPTypeFields.toSeq: _*),
      fields,
      leafPaths.toIndexedSeq,
    )
  }
}

/** Emits code that assembles Hail values in `region` from the buffered `leaves` of a record,
  * following a [[ReadNode]].
  */
private class ColumnarValueEmitter(leaves: IndexedSeq[Value[ColumnarLeaf]], region: Value[Region]) {

  private def skip(cb: EmitCodeBuilder, node: ReadNode): Unit =
    node.leaves.foreach(i => cb += leaves(i).invoke[Int, Unit]("skip", node.repetitionLevel))

  /** The value of `node` at the cursor, which is moved past it. */
  def read(cb: EmitCodeBuilder, node: ReadNode): IEmitCode =
    if (!node.optional) IEmitCode.present(cb, readPresent(cb, node))
    else {
      val missing = cb.memoize(
        leaves(node.leaves.head).invoke[Int]("definitionLevel") < node.definitionLevel
      )
      cb.if_(missing, skip(cb, node))
      IEmitCode(cb, missing, readPresent(cb, node))
    }

  private def readPresent(cb: EmitCodeBuilder, node: ReadNode): SValue = node match {
    case n: LeafNode => readLeaf(cb, n)
    case n: StructNode =>
      val s = n.pt.constructFromFields(
        cb,
        region,
        n.fields.map(f => EmitCode.fromI(cb.emb)(cb => read(cb, f))),
        deepCopy = false,
      )
      n.structural.foreach(i => cb += leaves(i).invoke[Int, Unit]("skip", n.repetitionLevel))
      s
    case n: IntervalNode =>
      def bool(f: ReadNode): Value[Boolean] =
        cb.memoize(read(cb, f).getOrAssert(cb).asBoolean.value)
      val start = cb.memoize(read(cb, n.start))
      val end = cb.memoize(read(cb, n.end))
      n.pt.constructFromCodes(
        cb,
        region,
        start.load,
        end.load,
        bool(n.includesStart),
        bool(n.includesEnd),
      )
    case n: ListNode => readList(cb, n)
    case n: NDArrayNode => readNDArray(cb, n)
  }

  private def readList(cb: EmitCodeBuilder, n: ListNode): SIndexableValue = {
    val first = leaves(n.element.leaves.head)
    val length = cb.newLocal[Int]("columnar_list_length")
    cb.if_(
      first.invoke[Int]("definitionLevel") <= n.definitionLevel, {
        // Present and empty: one triple per leaf.
        skip(cb, n)
        cb.assign(length, 0)
      },
      cb.assign(length, first.invoke[Int, Int]("countElements", n.repetitionLevel + 1)),
    )
    val array = n.arrayType.constructFromElements(cb, region, length, deepCopy = false) {
      (cb, _) => read(cb, n.element)
    }
    // An SType's PType is never required.
    n.pt.setRequired(false) match {
      case t: PCanonicalSet => t.construct(array)
      case t: PCanonicalDict => t.construct(array)
      case _ => array
    }
  }

  private def readNDArray(cb: EmitCodeBuilder, n: NDArrayNode): SValue = {
    val shapeArray = read(cb, n.shape).getOrFatal(
      cb,
      "columnar reader: a present ndarray has a null shape",
    ).asIndexable
    cb.if_(
      shapeArray.loadLength cne n.pt.nDims,
      cb._fatal(s"columnar reader: an ndarray's shape does not have ${n.pt.nDims} dimensions"),
    )
    val shape = (0 until n.pt.nDims).map { i =>
      cb.memoize(shapeArray.loadElement(cb, i).getOrAssert(cb).asInt.value.toL)
    }
    val data = read(cb, n.data).getOrAssert(cb).asIndexable
    val size = shape.foldLeft[Code[Long]](1L)(_ * _)
    cb.if_(
      data.loadLength.toL cne size,
      cb._fatal("columnar reader: an ndarray's data does not match its shape"),
    )
    n.pt.constructByCopyingArray(shape, n.pt.makeRowMajorStrides(shape, cb), data, cb, region)
  }

  private def readLeaf(cb: EmitCodeBuilder, n: LeafNode): SValue = {
    val l = leaves(n.leaf)
    val v: SValue = n.pt match {
      case _: PBoolean => primitive(cb.memoize(l.invoke[Boolean]("getBoolean")))
      case _: PInt32 => primitive(cb.memoize(l.invoke[Int]("getInt")))
      case _: PInt64 => primitive(cb.memoize(l.invoke[Long]("getLong")))
      case _: PFloat32 => primitive(cb.memoize(l.invoke[Float]("getFloat")))
      case _: PFloat64 => primitive(cb.memoize(l.invoke[Double]("getDouble")))
      case t: PCanonicalString =>
        val bytes = cb.memoize(l.invoke[Array[Byte]]("getBinary"))
        val addr = cb.memoize(t.binaryRepresentation.allocate(region, bytes.length()))
        t.binaryRepresentation.store(cb, addr, bytes)
        t.loadCheapSCode(cb, addr)
      case t: PCanonicalBinary =>
        t.constructFromByteArray(cb, region, l.invoke[Array[Byte]]("getBinary"))
      case _: PCanonicalCall => new SCanonicalCallValue(cb.memoize(l.invoke[Int]("getInt")))
      case t: PPackedLocus =>
        new SPackedLocusValue(SPackedLocus(t.rg), cb.memoize(l.invoke[Long]("getLong")))
      case t: PCanonicalBaseStruct =>
        assert(t.size == 0)
        cb.if_(
          !l.invoke[Boolean]("getBoolean"),
          cb._fatal("columnar reader: an empty struct or tuple is stored as false"),
        )
        t.constructFromFields(cb, region, FastSeq(), deepCopy = false)
    }
    cb += l.invoke[Int, Unit]("skip", n.repetitionLevel)
    v
  }
}
