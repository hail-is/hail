package is.hail.expr.ir

import is.hail.annotations.{Annotation, Region}
import is.hail.asm4s.Value
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.ArrayZipBehavior.ArrayZipBehavior
import is.hail.expr.ir.agg.{AggStateSig, PhysicalAggSig}
import is.hail.expr.ir.functions._
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, TypedCodecSpec}
import is.hail.io.avro.{AvroPartitionReader, AvroSchemaSerializer}
import is.hail.io.bgen.BgenPartitionReader
import is.hail.io.vcf.{GVCFPartitionReader, VCFHeaderInfo}
import is.hail.rvd.RVDSpecMaker
import is.hail.types.{tcoerce, RIterable, RStruct, TypeWithRequiredness}
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete.SJavaString
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._
import is.hail.utils._

import java.io.OutputStream

import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}
import org.json4s.JsonAST.{JNothing, JString}

sealed trait IR extends BaseIR {
  private var _typ: Type = null

  def typ: Type = {
    if (_typ == null)
      try
        _typ = InferType(this)
      catch {
        case e: Throwable => throw new RuntimeException(s"typ: inference failure:", e)
      }
    _typ
  }

  protected lazy val childrenSeq: IndexedSeq[BaseIR] =
    Children(this)

  override protected def copy(newChildren: IndexedSeq[BaseIR]): IR =
    Copy(this, newChildren)

  override def mapChildren(f: BaseIR => BaseIR): IR = super.mapChildren(f).asInstanceOf[IR]

  override def mapChildrenWithIndex(f: (BaseIR, Int) => BaseIR): IR =
    super.mapChildrenWithIndex(f).asInstanceOf[IR]

  override def deepCopy(): this.type = {

    val cp = super.deepCopy()
    if (_typ != null)
      cp._typ = _typ
    cp
  }

  lazy val size: Int = 1 + children.map {
    case x: IR => x.size
    case _ => 0
  }.sum

  private[this] def _unwrap: IR => IR = {
    case node: ApplyIR => MapIR(_unwrap)(node.explicitNode)
    case node => MapIR(_unwrap)(node)
  }

  def unwrap: IR = _unwrap(this)
}

sealed trait TypedIR[T <: Type] extends IR {
  override def typ: T = tcoerce[T](super.typ)
}

// Mark Refs and constants as IRs that are safe to duplicate
sealed trait TrivialIR extends IR

object Literal {
  def coerce(t: Type, x: Any): IR = {
    if (x == null)
      return NA(t)
    t match {
      case TInt32 => I32(x.asInstanceOf[Number].intValue())
      case TInt64 => I64(x.asInstanceOf[Number].longValue())
      case TFloat32 => F32(x.asInstanceOf[Number].floatValue())
      case TFloat64 => F64(x.asInstanceOf[Number].doubleValue())
      case TBoolean => if (x.asInstanceOf[Boolean]) True() else False()
      case TString => Str(x.asInstanceOf[String])
      case _ => Literal(t, x)
    }
  }
}

final case class Literal(_typ: Type, value: Annotation) extends IR {
  require(!CanEmit(_typ))
  require(value != null)
  // expensive, for debugging
//   require(SafeRow.isSafe(value))
//   assert(_typ.typeCheck(value), s"literal invalid:\n  ${_typ}\n  $value")
}

object EncodedLiteral {
  def apply(codec: AbstractTypedCodecSpec, value: Array[Array[Byte]]): EncodedLiteral =
    EncodedLiteral(codec, new WrappedByteArrays(value))

  def fromPTypeAndAddress(pt: PType, addr: Long, ctx: ExecuteContext): IR = {
    pt match {
      case _: PInt32 => I32(Region.loadInt(addr))
      case _: PInt64 => I64(Region.loadLong(addr))
      case _: PFloat32 => F32(Region.loadFloat(addr))
      case _: PFloat64 => F64(Region.loadDouble(addr))
      case _: PBoolean => if (Region.loadBoolean(addr)) True() else False()
      case ts: PString => Str(ts.loadString(addr))
      case _ =>
        val etype = EType.defaultFromPType(pt)
        val codec = TypedCodecSpec(etype, pt.virtualType, BufferSpec.wireSpec)
        val bytes = codec.encodeArrays(ctx, pt, addr)
        EncodedLiteral(codec, bytes)
    }
  }
}

final case class EncodedLiteral(codec: AbstractTypedCodecSpec, value: WrappedByteArrays)
    extends IR {
  require(!CanEmit(codec.encodedVirtualType))
  require(value != null)
}

class WrappedByteArrays(val ba: Array[Array[Byte]]) {
  override def hashCode(): Int =
    ba.foldLeft(31)((h, b) => 37 * h + java.util.Arrays.hashCode(b))

  override def equals(obj: Any): Boolean = {
    this.eq(obj.asInstanceOf[AnyRef]) || {
      if (!obj.isInstanceOf[WrappedByteArrays]) {
        false
      } else {
        val other = obj.asInstanceOf[WrappedByteArrays]
        ba.length == other.ba.length && (ba, other.ba).zipped.forall(java.util.Arrays.equals)
      }
    }
  }
}

final case class I32(x: Int) extends IR with TrivialIR
final case class I64(x: Long) extends IR with TrivialIR
final case class F32(x: Float) extends IR with TrivialIR
final case class F64(x: Double) extends IR with TrivialIR

final case class Str(x: String) extends IR with TrivialIR {
  override def toString(): String = s"""Str("${StringEscapeUtils.escapeString(x)}")"""
}

final case class True() extends IR with TrivialIR
final case class False() extends IR with TrivialIR
final case class Void() extends IR with TrivialIR

object UUID4 {
  def apply(): UUID4 = UUID4(genUID())
}

// WARNING! This node can only be used when trying to append a one-off,
// random string that will not be reused elsewhere in the pipeline.
// Any other uses will need to write and then read again; this node is
// non-deterministic and will not e.g. exhibit the correct semantics when
// self-joining on streams.
final case class UUID4(id: String) extends IR

final case class Cast(v: IR, _typ: Type) extends IR
final case class CastRename(v: IR, _typ: Type) extends IR

final case class NA(_typ: Type) extends IR with TrivialIR
final case class IsNA(value: IR) extends IR

final case class Coalesce(values: Seq[IR]) extends IR {
  require(values.nonEmpty)
}

final case class Consume(value: IR) extends IR

final case class If(cond: IR, cnsq: IR, altr: IR) extends IR

final case class Switch(x: IR, default: IR, cases: IndexedSeq[IR]) extends IR {
  override lazy val size: Int =
    2 + cases.length
}

final case class AggLet(name: String, value: IR, body: IR, isScan: Boolean) extends IR

final case class Let(bindings: IndexedSeq[(String, IR)], body: IR) extends IR {
  override lazy val size: Int =
    bindings.length + 1
}

object Let {
  case class Extract(p: ((String, IR)) => Boolean) {
    def unapply(bindings: IndexedSeq[(String, IR)])
      : Option[(IndexedSeq[(String, IR)], IndexedSeq[(String, IR)])] = {
      val idx = bindings.indexWhere(p)
      if (idx == -1) None else Some(bindings.splitAt(idx))
    }
  }

  object Nested extends Extract(_._2.isInstanceOf[Let])
  object Insert extends Extract(_._2.isInstanceOf[InsertFields])

}

sealed abstract class BaseRef extends IR with TrivialIR {
  def name: String
  def _typ: Type
}

final case class Ref(name: String, var _typ: Type) extends BaseRef {
  override def typ: Type = {
    assert(_typ != null)
    _typ
  }
}

// Recur can't exist outside of loop
// Loops can be nested, but we can't call outer loops in terms of inner loops so there can only be one loop "active" in a given context
final case class TailLoop(
  name: String,
  params: IndexedSeq[(String, IR)],
  resultType: Type,
  body: IR,
) extends IR {
  lazy val paramIdx: Map[String, Int] = params.map(_._1).zipWithIndex.toMap
}

final case class Recur(name: String, args: IndexedSeq[IR], var _typ: Type) extends BaseRef

final case class RelationalLet(name: String, value: IR, body: IR) extends IR
final case class RelationalRef(name: String, _typ: Type) extends BaseRef

final case class ApplyBinaryPrimOp(op: BinaryOp, l: IR, r: IR) extends IR
final case class ApplyUnaryPrimOp(op: UnaryOp, x: IR) extends IR
final case class ApplyComparisonOp(var op: ComparisonOp[_], l: IR, r: IR) extends IR

object MakeArray {
  def apply(args: IR*): MakeArray = {
    assert(args.nonEmpty)
    MakeArray(args.toArray, TArray(args.head.typ))
  }

  def unify(ctx: ExecuteContext, args: IndexedSeq[IR], requestedType: TArray = null): MakeArray = {
    assert(requestedType != null || args.nonEmpty)

    if (args.nonEmpty)
      if (args.forall(_.typ == args.head.typ))
        return MakeArray(args, TArray(args.head.typ))

    MakeArray(
      args.map { arg =>
        val upcast = PruneDeadFields.upcast(ctx, arg, requestedType.elementType)
        assert(upcast.typ == requestedType.elementType)
        upcast
      },
      requestedType,
    )
  }
}

final case class MakeArray(args: IndexedSeq[IR], _typ: TArray) extends IR

object MakeStream {
  def unify(
    ctx: ExecuteContext,
    args: IndexedSeq[IR],
    requiresMemoryManagementPerElement: Boolean = false,
    requestedType: TStream = null,
  ): MakeStream = {
    assert(requestedType != null || args.nonEmpty)

    if (args.nonEmpty)
      if (args.forall(_.typ == args.head.typ))
        return MakeStream(args, TStream(args.head.typ), requiresMemoryManagementPerElement)

    MakeStream(
      args.map { arg =>
        val upcast = PruneDeadFields.upcast(ctx, arg, requestedType.elementType)
        assert(upcast.typ == requestedType.elementType)
        upcast
      },
      requestedType,
      requiresMemoryManagementPerElement,
    )
  }
}

final case class MakeStream(
  args: IndexedSeq[IR],
  _typ: TStream,
  requiresMemoryManagementPerElement: Boolean = false,
) extends IR

object ArrayRef {
  def apply(a: IR, i: IR): ArrayRef = ArrayRef(a, i, ErrorIDs.NO_ERROR)
}

final case class ArrayRef(a: IR, i: IR, errorID: Int) extends IR

final case class ArraySlice(
  a: IR,
  start: IR,
  stop: Option[IR],
  step: IR = I32(1),
  errorID: Int = ErrorIDs.NO_ERROR,
) extends IR

final case class ArrayLen(a: IR) extends IR
final case class ArrayZeros(length: IR) extends IR

final case class ArrayMaximalIndependentSet(edges: IR, tieBreaker: Option[(String, String, IR)])
    extends IR

/** [[StreamIota]] is an infinite stream producer, whose element is an integer starting at `start`,
  * updated by `step` at each iteration. The name comes from APL:
  * [[https://stackoverflow.com/questions/9244879/what-does-iota-of-stdiota-stand-for]]
  */
final case class StreamIota(
  start: IR,
  step: IR,
  requiresMemoryManagementPerElement: Boolean = false,
) extends IR

final case class StreamRange(
  start: IR,
  stop: IR,
  step: IR,
  requiresMemoryManagementPerElement: Boolean = false,
  errorID: Int = ErrorIDs.NO_ERROR,
) extends IR

object ArraySort {
  def apply(a: IR, ascending: IR = True(), onKey: Boolean = false): ArraySort = {
    val l = genUID()
    val r = genUID()
    val atyp = tcoerce[TStream](a.typ)
    val compare = if (onKey) {
      val elementType = atyp.elementType.asInstanceOf[TBaseStruct]
      elementType match {
        case _: TStruct =>
          val elt = tcoerce[TStruct](atyp.elementType)
          ApplyComparisonOp(
            Compare(elt.types(0)),
            GetField(Ref(l, elt), elt.fieldNames(0)),
            GetField(Ref(r, atyp.elementType), elt.fieldNames(0)),
          )
        case _: TTuple =>
          val elt = tcoerce[TTuple](atyp.elementType)
          ApplyComparisonOp(
            Compare(elt.types(0)),
            GetTupleElement(Ref(l, elt), elt.fields(0).index),
            GetTupleElement(Ref(r, atyp.elementType), elt.fields(0).index),
          )
      }
    } else {
      ApplyComparisonOp(
        Compare(atyp.elementType),
        Ref(l, atyp.elementType),
        Ref(r, atyp.elementType),
      )
    }

    ArraySort(a, l, r, If(ascending, compare < 0, compare > 0))
  }
}

final case class ArraySort(a: IR, left: String, right: String, lessThan: IR) extends IR
final case class ToSet(a: IR) extends IR
final case class ToDict(a: IR) extends IR
final case class ToArray(a: IR) extends IR
final case class CastToArray(a: IR) extends IR
final case class ToStream(a: IR, requiresMemoryManagementPerElement: Boolean = false) extends IR

final case class StreamBufferedAggregate(
  streamChild: IR,
  initAggs: IR,
  newKey: IR,
  seqOps: IR,
  name: String,
  aggSignatures: IndexedSeq[PhysicalAggSig],
  bufferSize: Int,
) extends IR

final case class LowerBoundOnOrderedCollection(orderedCollection: IR, elem: IR, onKey: Boolean)
    extends IR

final case class GroupByKey(collection: IR) extends IR

final case class RNGStateLiteral() extends IR

final case class RNGSplit(state: IR, dynBitstring: IR) extends IR

final case class StreamLen(a: IR) extends IR

final case class StreamGrouped(a: IR, groupSize: IR) extends IR
final case class StreamGroupByKey(a: IR, key: IndexedSeq[String], missingEqual: Boolean) extends IR

final case class StreamMap(a: IR, name: String, body: IR) extends TypedIR[TStream] {
  def elementTyp: Type = typ.elementType
}

final case class StreamTakeWhile(a: IR, elementName: String, body: IR) extends IR

final case class StreamDropWhile(a: IR, elementName: String, body: IR) extends IR

final case class StreamTake(a: IR, num: IR) extends IR
final case class StreamDrop(a: IR, num: IR) extends IR

// Generate, in ascending order, a uniform random sample, without replacement, of numToSample integers in the range [0, totalRange)
final case class SeqSample(
  totalRange: IR,
  numToSample: IR,
  rngState: IR,
  requiresMemoryManagementPerElement: Boolean,
) extends IR

// Take the child stream and sort each element into buckets based on the provided pivots. The first and last elements of
// pivots are the endpoints of the first and last interval respectively, should not be contained in the dataset.
final case class StreamDistribute(
  child: IR,
  pivots: IR,
  path: IR,
  comparisonOp: ComparisonOp[_],
  spec: AbstractTypedCodecSpec,
) extends IR

// "Whiten" a stream of vectors by regressing out from each vector all components
// in the direction of vectors in the preceding window. For efficiency, takes
// a stream of "chunks" of vectors.
// Takes a stream of structs, with two designated fields: `prevWindow` is the
// previous window (e.g. from the previous partition), if there is one, and
// `newChunk` is the new chunk to whiten.
final case class StreamWhiten(
  stream: IR,
  newChunk: String,
  prevWindow: String,
  vecSize: Int,
  windowSize: Int,
  chunkSize: Int,
  blockSize: Int,
  normalizeAfterWhiten: Boolean,
) extends IR

object ArrayZipBehavior extends Enumeration {
  type ArrayZipBehavior = Value
  val AssumeSameLength: Value = Value(0)
  val AssertSameLength: Value = Value(1)
  val TakeMinLength: Value = Value(2)
  val ExtendNA: Value = Value(3)
}

final case class StreamZip(
  as: IndexedSeq[IR],
  names: IndexedSeq[String],
  body: IR,
  behavior: ArrayZipBehavior,
  errorID: Int = ErrorIDs.NO_ERROR,
) extends TypedIR[TStream]

final case class StreamMultiMerge(as: IndexedSeq[IR], key: IndexedSeq[String])
    extends TypedIR[TStream]

final case class StreamZipJoinProducers(
  contexts: IR,
  ctxName: String,
  makeProducer: IR,
  key: IndexedSeq[String],
  curKey: String,
  curVals: String,
  joinF: IR,
) extends TypedIR[TStream]

/** The StreamZipJoin node assumes that input streams have distinct keys. If input streams do not
  * have distinct keys, the key that is included in the result is undefined, but is likely the last.
  */
final case class StreamZipJoin(
  as: IndexedSeq[IR],
  key: IndexedSeq[String],
  curKey: String,
  curVals: String,
  joinF: IR,
) extends TypedIR[TStream]

final case class StreamFilter(a: IR, name: String, cond: IR) extends TypedIR[TStream]
final case class StreamFlatMap(a: IR, name: String, body: IR) extends TypedIR[TStream]

final case class StreamFold(a: IR, zero: IR, accumName: String, valueName: String, body: IR)
    extends IR

object StreamFold2 {
  def apply(a: StreamFold): StreamFold2 =
    StreamFold2(
      a.a,
      FastSeq((a.accumName, a.zero)),
      a.valueName,
      FastSeq(a.body),
      Ref(a.accumName, a.zero.typ),
    )
}

final case class StreamFold2(
  a: IR,
  accum: IndexedSeq[(String, IR)],
  valueName: String,
  seq: IndexedSeq[IR],
  result: IR,
) extends IR {
  assert(accum.length == seq.length)
  val nameIdx: Map[String, Int] = accum.map(_._1).zipWithIndex.toMap
}

final case class StreamScan(a: IR, zero: IR, accumName: String, valueName: String, body: IR)
    extends IR

final case class StreamFor(a: IR, valueName: String, body: IR) extends IR

final case class StreamAgg(a: IR, name: String, query: IR) extends IR
final case class StreamAggScan(a: IR, name: String, query: IR) extends IR

object StreamJoin {
  def apply(
    left: IR,
    right: IR,
    lKey: IndexedSeq[String],
    rKey: IndexedSeq[String],
    l: String,
    r: String,
    joinF: IR,
    joinType: String,
    requiresMemoryManagement: Boolean,
    rightKeyIsDistinct: Boolean = false,
  ): IR = {
    val lType = tcoerce[TStream](left.typ)
    val rType = tcoerce[TStream](right.typ)
    val lEltType = tcoerce[TStruct](lType.elementType)
    val rEltType = tcoerce[TStruct](rType.elementType)
    assert(lEltType.typeAfterSelectNames(lKey) isJoinableWith rEltType.typeAfterSelectNames(rKey))

    if (!rightKeyIsDistinct) {
      val rightGroupedStream = StreamGroupByKey(right, rKey, missingEqual = false)
      val groupField = genUID()

      // stream of {key, groupField}, where 'groupField' is an array of all rows
      // in 'right' with key 'key'
      val rightGrouped = mapIR(rightGroupedStream) { group =>
        bindIR(ToArray(group)) { array =>
          bindIR(ArrayRef(array, 0)) { head =>
            MakeStruct(rKey.map(key => key -> GetField(head, key)) :+ groupField -> array)
          }
        }
      }

      val rElt = Ref(genUID(), tcoerce[TStream](rightGrouped.typ).elementType)
      val lElt = Ref(genUID(), lEltType)
      val makeTupleFromJoin = MakeStruct(FastSeq("left" -> lElt, "rightGroup" -> rElt))
      val joined = StreamJoinRightDistinct(
        left,
        rightGrouped,
        lKey,
        rKey,
        lElt.name,
        rElt.name,
        makeTupleFromJoin,
        joinType,
      )

      // joined is a stream of {leftElement, rightGroup}
      bindIR(MakeArray(NA(rEltType))) { missingSingleton =>
        flatMapIR(joined) { x =>
          Let(
            FastSeq(l -> GetField(x, "left")),
            bindIR(GetField(GetField(x, "rightGroup"), groupField)) { rightElts =>
              joinType match {
                case "left" | "outer" => StreamMap(
                    ToStream(
                      If(IsNA(rightElts), missingSingleton, rightElts),
                      requiresMemoryManagement,
                    ),
                    r,
                    joinF,
                  )
                case "right" | "inner" =>
                  StreamMap(ToStream(rightElts, requiresMemoryManagement), r, joinF)
              }
            },
          )
        }
      }
    } else {
      val rElt = Ref(r, rEltType)
      val lElt = Ref(l, lEltType)
      StreamJoinRightDistinct(left, right, lKey, rKey, lElt.name, rElt.name, joinF, joinType)
    }
  }
}

final case class StreamLeftIntervalJoin(
  // input streams
  left: IR,
  right: IR,

  // names for joiner
  lKeyFieldName: String,
  rIntervalFieldName: String,

  // how to combine records
  lname: String,
  rname: String,
  body: IR,
) extends IR {
  override protected lazy val childrenSeq: IndexedSeq[BaseIR] =
    FastSeq(left, right, body)
}

final case class StreamJoinRightDistinct(
  left: IR,
  right: IR,
  lKey: IndexedSeq[String],
  rKey: IndexedSeq[String],
  l: String,
  r: String,
  joinF: IR,
  joinType: String,
) extends IR {
  def isIntervalJoin: Boolean = {
    if (rKey.size != 1) return false
    val lKeyTyp = tcoerce[TStruct](tcoerce[TStream](left.typ).elementType).fieldType(lKey(0))
    val rKeyTyp = tcoerce[TStruct](tcoerce[TStream](right.typ).elementType).fieldType(rKey(0))

    rKeyTyp.isInstanceOf[TInterval] && lKeyTyp != rKeyTyp
  }
}

final case class StreamLocalLDPrune(
  child: IR,
  r2Threshold: IR,
  windowSize: IR,
  maxQueueSize: IR,
  nSamples: IR,
) extends IR

sealed trait NDArrayIR extends TypedIR[TNDArray] {
  def elementTyp: Type = typ.elementType
}

object MakeNDArray {
  def fill(elt: IR, shape: IndexedSeq[IR], rowMajor: IR): MakeNDArray = {
    val flatSize: IR = if (shape.nonEmpty)
      shape.reduce((l, r) => l * r)
    else
      0L
    MakeNDArray(
      ToArray(mapIR(rangeIR(flatSize.toI))(_ => elt)),
      MakeTuple.ordered(shape),
      rowMajor,
      ErrorIDs.NO_ERROR,
    )
  }
}

final case class MakeNDArray(data: IR, shape: IR, rowMajor: IR, errorId: Int) extends NDArrayIR

final case class NDArrayShape(nd: IR) extends IR

final case class NDArrayReshape(nd: IR, shape: IR, errorID: Int) extends NDArrayIR

final case class NDArrayConcat(nds: IR, axis: Int) extends NDArrayIR

final case class NDArrayRef(nd: IR, idxs: IndexedSeq[IR], errorId: Int) extends IR
final case class NDArraySlice(nd: IR, slices: IR) extends NDArrayIR
final case class NDArrayFilter(nd: IR, keep: IndexedSeq[IR]) extends NDArrayIR

final case class NDArrayMap(nd: IR, valueName: String, body: IR) extends NDArrayIR

final case class NDArrayMap2(l: IR, r: IR, lName: String, rName: String, body: IR, errorID: Int)
    extends NDArrayIR

final case class NDArrayReindex(nd: IR, indexExpr: IndexedSeq[Int]) extends NDArrayIR
final case class NDArrayAgg(nd: IR, axes: IndexedSeq[Int]) extends IR
final case class NDArrayWrite(nd: IR, path: IR) extends IR

final case class NDArrayMatMul(l: IR, r: IR, errorID: Int) extends NDArrayIR

object NDArrayQR {
  def pType(mode: String, req: Boolean): PType = {
    mode match {
      case "r" => PCanonicalNDArray(PFloat64Required, 2, req)
      case "raw" => PCanonicalTuple(
          req,
          PCanonicalNDArray(PFloat64Required, 2, true),
          PCanonicalNDArray(PFloat64Required, 1, true),
        )
      case "reduced" => PCanonicalTuple(
          req,
          PCanonicalNDArray(PFloat64Required, 2, true),
          PCanonicalNDArray(PFloat64Required, 2, true),
        )
      case "complete" => PCanonicalTuple(
          req,
          PCanonicalNDArray(PFloat64Required, 2, true),
          PCanonicalNDArray(PFloat64Required, 2, true),
        )
    }
  }
}

object NDArraySVD {
  def pTypes(computeUV: Boolean, req: Boolean): PType = {
    if (computeUV) {
      PCanonicalTuple(
        req,
        PCanonicalNDArray(PFloat64Required, 2, true),
        PCanonicalNDArray(PFloat64Required, 1, true),
        PCanonicalNDArray(PFloat64Required, 2, true),
      )
    } else {
      PCanonicalNDArray(PFloat64Required, 1, req)
    }
  }
}

object NDArrayInv {
  val pType = PCanonicalNDArray(PFloat64Required, 2)
}

final case class NDArrayQR(nd: IR, mode: String, errorID: Int) extends IR

final case class NDArraySVD(nd: IR, fullMatrices: Boolean, computeUV: Boolean, errorID: Int)
    extends IR

object NDArrayEigh {
  def pTypes(eigvalsOnly: Boolean, req: Boolean): PType =
    if (eigvalsOnly) {
      PCanonicalNDArray(PFloat64Required, 1, req)
    } else {
      PCanonicalTuple(
        req,
        PCanonicalNDArray(PFloat64Required, 1, true),
        PCanonicalNDArray(PFloat64Required, 2, true),
      )
    }
}

final case class NDArrayEigh(nd: IR, eigvalsOnly: Boolean, errorID: Int) extends IR

final case class NDArrayInv(nd: IR, errorID: Int) extends IR

final case class AggFilter(cond: IR, aggIR: IR, isScan: Boolean) extends IR

final case class AggExplode(array: IR, name: String, aggBody: IR, isScan: Boolean) extends IR

final case class AggGroupBy(key: IR, aggIR: IR, isScan: Boolean) extends IR

final case class AggArrayPerElement(
  a: IR,
  elementName: String,
  indexName: String,
  aggBody: IR,
  knownLength: Option[IR],
  isScan: Boolean,
) extends IR

object ApplyAggOp {
  def apply(op: AggOp, initOpArgs: IR*)(seqOpArgs: IR*): ApplyAggOp =
    ApplyAggOp(
      initOpArgs.toIndexedSeq,
      seqOpArgs.toIndexedSeq,
      AggSignature(op, initOpArgs.map(_.typ), seqOpArgs.map(_.typ)),
    )
}

final case class ApplyAggOp(
  initOpArgs: IndexedSeq[IR],
  seqOpArgs: IndexedSeq[IR],
  aggSig: AggSignature,
) extends IR {

  def nSeqOpArgs = seqOpArgs.length

  def nInitArgs = initOpArgs.length

  def op: AggOp = aggSig.op
}

object AggFold {

  def min(element: IR, sortFields: IndexedSeq[SortField]): IR = {
    val elementType = element.typ.asInstanceOf[TStruct]
    val keyType = elementType.select(sortFields.map(_.field))._1
    minAndMaxHelper(element, keyType, StructLT(keyType, sortFields))
  }

  def max(element: IR, sortFields: IndexedSeq[SortField]): IR = {
    val elementType = element.typ.asInstanceOf[TStruct]
    val keyType = elementType.select(sortFields.map(_.field))._1
    minAndMaxHelper(element, keyType, StructGT(keyType, sortFields))
  }

  def all(element: IR): IR =
    aggFoldIR(True(), element) { case (accum, element) =>
      ApplySpecial("land", Seq.empty[Type], Seq(accum, element), TBoolean, ErrorIDs.NO_ERROR)
    } { case (accum1, accum2) =>
      ApplySpecial("land", Seq.empty[Type], Seq(accum1, accum2), TBoolean, ErrorIDs.NO_ERROR)
    }

  private def minAndMaxHelper(element: IR, keyType: TStruct, comp: ComparisonOp[Boolean]): IR = {
    val keyFields = keyType.fields.map(_.name)

    val minAndMaxZero = NA(keyType)
    val aggFoldMinAccumName1 = genUID()
    val aggFoldMinAccumName2 = genUID()
    val aggFoldMinAccumRef1 = Ref(aggFoldMinAccumName1, keyType)
    val aggFoldMinAccumRef2 = Ref(aggFoldMinAccumName2, keyType)
    val minSeq = bindIR(SelectFields(element, keyFields)) { keyOfCurElementRef =>
      If(
        IsNA(aggFoldMinAccumRef1),
        keyOfCurElementRef,
        If(
          ApplyComparisonOp(comp, aggFoldMinAccumRef1, keyOfCurElementRef),
          aggFoldMinAccumRef1,
          keyOfCurElementRef,
        ),
      )
    }
    val minComb =
      If(
        IsNA(aggFoldMinAccumRef1),
        aggFoldMinAccumRef2,
        If(
          ApplyComparisonOp(comp, aggFoldMinAccumRef1, aggFoldMinAccumRef2),
          aggFoldMinAccumRef1,
          aggFoldMinAccumRef2,
        ),
      )

    AggFold(minAndMaxZero, minSeq, minComb, aggFoldMinAccumName1, aggFoldMinAccumName2, false)
  }
}

final case class AggFold(
  zero: IR,
  seqOp: IR,
  combOp: IR,
  accumName: String,
  otherAccumName: String,
  isScan: Boolean,
) extends IR

object ApplyScanOp {
  def apply(op: AggOp, initOpArgs: IR*)(seqOpArgs: IR*): ApplyScanOp =
    ApplyScanOp(
      initOpArgs.toIndexedSeq,
      seqOpArgs.toIndexedSeq,
      AggSignature(op, initOpArgs.map(_.typ), seqOpArgs.map(_.typ)),
    )
}

final case class ApplyScanOp(
  initOpArgs: IndexedSeq[IR],
  seqOpArgs: IndexedSeq[IR],
  aggSig: AggSignature,
) extends IR {

  def nSeqOpArgs = seqOpArgs.length

  def nInitArgs = initOpArgs.length

  def op: AggOp = aggSig.op
}

final case class InitOp(i: Int, args: IndexedSeq[IR], aggSig: PhysicalAggSig) extends IR
final case class SeqOp(i: Int, args: IndexedSeq[IR], aggSig: PhysicalAggSig) extends IR
final case class CombOp(i1: Int, i2: Int, aggSig: PhysicalAggSig) extends IR

object ResultOp {
  def makeTuple(aggs: IndexedSeq[PhysicalAggSig]) =
    MakeTuple.ordered(aggs.zipWithIndex.map { case (aggSig, index) =>
      ResultOp(index, aggSig)
    })
}

final case class ResultOp(idx: Int, aggSig: PhysicalAggSig) extends IR

final private case class CombOpValue(i: Int, value: IR, aggSig: PhysicalAggSig) extends IR
final case class AggStateValue(i: Int, aggSig: AggStateSig) extends IR
final case class InitFromSerializedValue(i: Int, value: IR, aggSig: AggStateSig) extends IR

final case class SerializeAggs(
  startIdx: Int,
  serializedIdx: Int,
  spec: BufferSpec,
  aggSigs: IndexedSeq[AggStateSig],
) extends IR

final case class DeserializeAggs(
  startIdx: Int,
  serializedIdx: Int,
  spec: BufferSpec,
  aggSigs: IndexedSeq[AggStateSig],
) extends IR

final case class RunAgg(body: IR, result: IR, signature: IndexedSeq[AggStateSig]) extends IR

final case class RunAggScan(
  array: IR,
  name: String,
  init: IR,
  seqs: IR,
  result: IR,
  signature: IndexedSeq[AggStateSig],
) extends IR

object Begin {
  def apply(xs: IndexedSeq[IR]): IR =
    if (xs.isEmpty)
      Void()
    else
      Let(xs.init.map(x => ("__void", x)), xs.last)
}

final case class Begin(xs: IndexedSeq[IR]) extends IR
final case class MakeStruct(fields: IndexedSeq[(String, IR)]) extends IR
final case class SelectFields(old: IR, fields: IndexedSeq[String]) extends IR

object InsertFields {
  def apply(old: IR, fields: Seq[(String, IR)]): InsertFields = InsertFields(old, fields, None)
}

final case class InsertFields(
  old: IR,
  fields: Seq[(String, IR)],
  fieldOrder: Option[IndexedSeq[String]],
) extends TypedIR[TStruct]

object GetFieldByIdx {
  def apply(s: IR, field: Int): IR =
    (s.typ: @unchecked) match {
      case t: TStruct => GetField(s, t.fieldNames(field))
      case _: TTuple => GetTupleElement(s, field)
    }
}

final case class GetField(o: IR, name: String) extends IR

object MakeTuple {
  def ordered(types: IndexedSeq[IR]): MakeTuple = MakeTuple(types.zipWithIndex.map { case (ir, i) =>
    (i, ir)
  })
}

final case class MakeTuple(fields: IndexedSeq[(Int, IR)]) extends IR
final case class GetTupleElement(o: IR, idx: Int) extends IR

object In {
  def apply(i: Int, typ: Type): In = In(
    i,
    SingleCodeEmitParamType(
      false,
      typ match {
        case TInt32 => Int32SingleCodeType
        case TInt64 => Int64SingleCodeType
        case TFloat32 => Float32SingleCodeType
        case TFloat64 => Float64SingleCodeType
        case TBoolean => BooleanSingleCodeType
        case _: TStream => throw new UnsupportedOperationException
        case t => PTypeReferenceSingleCodeType(PType.canonical(t))
      },
    ),
  )
}

// Function Input
final case class In(i: Int, _typ: EmitParamType) extends IR

// FIXME: should be type any
object Die {
  def apply(message: String, typ: Type): Die = Die(Str(message), typ, ErrorIDs.NO_ERROR)
  def apply(message: String, typ: Type, errorId: Int): Die = Die(Str(message), typ, errorId)
}

/** the Trap node runs the `child` node with an exception handler. If the child throws a
  * HailException (user exception), then we return the tuple ((msg, errorId), NA). If the child
  * throws any other exception, we raise that exception. If the child does not throw, then we return
  * the tuple (NA, child value).
  */
final case class Trap(child: IR) extends IR
final case class Die(message: IR, _typ: Type, errorId: Int) extends IR
final case class ConsoleLog(message: IR, result: IR) extends IR

final case class ApplyIR(
  function: String,
  typeArgs: Seq[Type],
  args: Seq[IR],
  returnType: Type,
  errorID: Int,
) extends IR {
  var conversion: (Seq[Type], Seq[IR], Int) => IR = _
  var inline: Boolean = _

  private lazy val refs = args.map(a => Ref(genUID(), a.typ)).toArray
  lazy val body: IR = conversion(typeArgs, refs, errorID).deepCopy()
  lazy val refIdx: Map[String, Int] = refs.map(_.name).zipWithIndex.toMap

  lazy val explicitNode: IR = {
    val ir = Let(refs.map(_.name).zip(args), body)
    assert(ir.typ == returnType)
    ir
  }
}

sealed abstract class AbstractApplyNode[F <: JVMFunction] extends IR {
  def function: String
  def args: Seq[IR]
  def returnType: Type
  def typeArgs: Seq[Type]
  def argTypes: Seq[Type] = args.map(_.typ)

  lazy val implementation: F =
    IRFunctionRegistry.lookupFunctionOrFail(function, returnType, typeArgs, argTypes)
      .asInstanceOf[F]
}

final case class Apply(
  function: String,
  typeArgs: Seq[Type],
  args: Seq[IR],
  returnType: Type,
  errorID: Int,
) extends AbstractApplyNode[UnseededMissingnessObliviousJVMFunction]

final case class ApplySeeded(
  function: String,
  _args: Seq[IR],
  rngState: IR,
  staticUID: Long,
  returnType: Type,
) extends AbstractApplyNode[UnseededMissingnessObliviousJVMFunction] {
  val args = rngState +: _args
  val typeArgs: Seq[Type] = Seq.empty[Type]
}

final case class ApplySpecial(
  function: String,
  typeArgs: Seq[Type],
  args: Seq[IR],
  returnType: Type,
  errorID: Int,
) extends AbstractApplyNode[UnseededMissingnessAwareJVMFunction]

final case class LiftMeOut(child: IR) extends IR
final case class TableCount(child: TableIR) extends IR
final case class MatrixCount(child: MatrixIR) extends IR
final case class TableAggregate(child: TableIR, query: IR) extends IR
final case class MatrixAggregate(child: MatrixIR, query: IR) extends IR

final case class TableWrite(child: TableIR, writer: TableWriter) extends IR

final case class TableMultiWrite(
  _children: IndexedSeq[TableIR],
  writer: WrappedMatrixNativeMultiWriter,
) extends IR

final case class TableGetGlobals(child: TableIR) extends IR
final case class TableCollect(child: TableIR) extends IR

final case class MatrixWrite(child: MatrixIR, writer: MatrixWriter) extends IR

final case class MatrixMultiWrite(_children: IndexedSeq[MatrixIR], writer: MatrixNativeMultiWriter)
    extends IR

final case class TableToValueApply(child: TableIR, function: TableToValueFunction) extends IR
final case class MatrixToValueApply(child: MatrixIR, function: MatrixToValueFunction) extends IR

final case class BlockMatrixToValueApply(child: BlockMatrixIR, function: BlockMatrixToValueFunction)
    extends IR

final case class BlockMatrixCollect(child: BlockMatrixIR) extends NDArrayIR

final case class BlockMatrixWrite(child: BlockMatrixIR, writer: BlockMatrixWriter) extends IR

final case class BlockMatrixMultiWrite(
  blockMatrices: IndexedSeq[BlockMatrixIR],
  writer: BlockMatrixMultiWriter,
) extends IR

final case class CollectDistributedArray(
  contexts: IR,
  globals: IR,
  cname: String,
  gname: String,
  body: IR,
  dynamicID: IR,
  staticID: String,
  tsd: Option[TableStageDependency] = None,
) extends IR

object PartitionReader {
  implicit val formats: Formats =
    new DefaultFormats() {
      override val typeHints = ShortTypeHints(
        List(
          classOf[PartitionRVDReader],
          classOf[PartitionNativeReader],
          classOf[PartitionNativeReaderIndexed],
          classOf[PartitionNativeIntervalReader],
          classOf[PartitionZippedNativeReader],
          classOf[PartitionZippedIndexedNativeReader],
          classOf[BgenPartitionReader],
          classOf[GVCFPartitionReader],
          classOf[TextInputFilterAndReplace],
          classOf[VCFHeaderInfo],
          classOf[AbstractTypedCodecSpec],
          classOf[TypedCodecSpec],
          classOf[AvroPartitionReader],
        ),
        typeHintFieldName = "name",
      ) + BufferSpec.shortTypeHints
    } +
      new TStructSerializer +
      new TypeSerializer +
      new PTypeSerializer +
      new ETypeSerializer +
      new AvroSchemaSerializer

  def extract(ctx: ExecuteContext, jv: JValue): PartitionReader = {
    (jv \ "name").extract[String] match {
      case "PartitionNativeIntervalReader" =>
        val path = (jv \ "path").extract[String]
        val spec = TableNativeReader.read(ctx.fs, path, None).spec
        PartitionNativeIntervalReader(
          ctx.stateManager,
          path,
          spec,
          (jv \ "uidFieldName").extract[String],
        )
      case "GVCFPartitionReader" =>
        val header = VCFHeaderInfo.fromJSON((jv \ "header"))
        val callFields = (jv \ "callFields").extract[Set[String]]
        val entryFloatType = IRParser.parseType((jv \ "entryFloatType").extract[String])
        val arrayElementsRequired = (jv \ "arrayElementsRequired").extract[Boolean]
        val rg = (jv \ "rg") match {
          case JString(s) => Some(s)
          case JNothing => None
        }
        val contigRecoding = (jv \ "contigRecoding").extract[Map[String, String]]
        val skipInvalidLoci = (jv \ "skipInvalidLoci").extract[Boolean]
        val filterAndReplace = (jv \ "filterAndReplace").extract[TextInputFilterAndReplace]
        val entriesFieldName = (jv \ "entriesFieldName").extract[String]
        val uidFieldName = (jv \ "uidFieldName").extract[String]
        GVCFPartitionReader(header, callFields, entryFloatType, arrayElementsRequired, rg,
          contigRecoding,
          skipInvalidLoci, filterAndReplace, entriesFieldName, uidFieldName)
      case _ => jv.extract[PartitionReader]
    }
  }
}

object PartitionWriter {
  implicit val formats: Formats =
    new DefaultFormats() {
      override val typeHints = ShortTypeHints(
        List(
          classOf[PartitionNativeWriter],
          classOf[TableTextPartitionWriter],
          classOf[VCFPartitionWriter],
          classOf[GenSampleWriter],
          classOf[GenVariantWriter],
          classOf[AbstractTypedCodecSpec],
          classOf[TypedCodecSpec],
        ),
        typeHintFieldName = "name",
      ) + BufferSpec.shortTypeHints
    } +
      new TStructSerializer +
      new TypeSerializer +
      new PTypeSerializer +
      new PStructSerializer +
      new ETypeSerializer
}

object MetadataWriter {
  implicit val formats: Formats =
    new DefaultFormats() {
      override val typeHints = ShortTypeHints(
        List(
          classOf[RVDSpecWriter],
          classOf[TableSpecWriter],
          classOf[RelationalWriter],
          classOf[TableTextFinalizer],
          classOf[VCFExportFinalizer],
          classOf[SimpleMetadataWriter],
          classOf[RVDSpecMaker],
          classOf[AbstractTypedCodecSpec],
          classOf[TypedCodecSpec],
        ),
        typeHintFieldName = "name",
      ) + BufferSpec.shortTypeHints
    } +
      new TStructSerializer +
      new TypeSerializer +
      new PTypeSerializer +
      new ETypeSerializer
}

abstract class PartitionReader {
  assert(fullRowType.hasField(uidFieldName))

  def contextType: Type

  def fullRowType: TStruct

  def uidFieldName: String

  def rowRequiredness(requestedType: TStruct): RStruct

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct,
  ): IEmitCode

  def toJValue: JValue
}

abstract class PartitionWriter {
  def consumeStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    context: EmitCode,
    region: Value[Region],
  ): IEmitCode

  def ctxType: Type
  def returnType: Type

  def unionTypeRequiredness(
    r: TypeWithRequiredness,
    ctxType: TypeWithRequiredness,
    streamType: RIterable,
  ): Unit

  def toJValue: JValue = Extraction.decompose(this)(PartitionWriter.formats)
}

abstract class SimplePartitionWriter extends PartitionWriter {
  def ctxType: Type = TString
  def returnType: Type = TString

  def unionTypeRequiredness(
    r: TypeWithRequiredness,
    ctxType: TypeWithRequiredness,
    streamType: RIterable,
  ): Unit = {
    r.union(ctxType.required)
    r.union(streamType.required)
  }

  def consumeElement(
    cb: EmitCodeBuilder,
    element: EmitCode,
    os: Value[OutputStream],
    region: Value[Region],
  ): Unit

  def preConsume(cb: EmitCodeBuilder, os: Value[OutputStream]): Unit = ()
  def postConsume(cb: EmitCodeBuilder, os: Value[OutputStream]): Unit = ()

  final def consumeStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    context: EmitCode,
    region: Value[Region],
  ): IEmitCode = {
    context.toI(cb).map(cb) { case ctx: SStringValue =>
      val filename = ctx.loadString(cb)
      val os = cb.memoize(cb.emb.create(filename))

      preConsume(cb, os)
      stream.memoryManagedConsume(region, cb) { cb =>
        consumeElement(cb, stream.element, os, stream.elementRegion)
      }
      postConsume(cb, os)

      cb += os.invoke[Unit]("flush")
      cb += os.invoke[Unit]("close")

      SJavaString.construct(cb, filename)
    }
  }
}

abstract class MetadataWriter {
  def annotationType: Type

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit

  def toJValue: JValue = Extraction.decompose(this)(MetadataWriter.formats)
}

final case class SimpleMetadataWriter(val annotationType: Type) extends MetadataWriter {
  def writeMetadata(writeAnnotations: => IEmitCode, cb: EmitCodeBuilder, region: Value[Region])
    : Unit =
    writeAnnotations.consume(cb, {}, _ => ())
}

final case class ReadPartition(context: IR, rowType: TStruct, reader: PartitionReader) extends IR
final case class WritePartition(value: IR, writeCtx: IR, writer: PartitionWriter) extends IR
final case class WriteMetadata(writeAnnotations: IR, writer: MetadataWriter) extends IR

final case class ReadValue(path: IR, reader: ValueReader, requestedType: Type) extends IR

final case class WriteValue(
  value: IR,
  path: IR,
  writer: ValueWriter,
  stagingFile: Option[IR] = None,
) extends IR

class PrimitiveIR(val self: IR) extends AnyVal {
  def +(other: IR): IR = {
    assert(self.typ == other.typ)
    if (self.typ == TString)
      invoke("concat", TString, self, other)
    else
      ApplyBinaryPrimOp(Add(), self, other)
  }

  def -(other: IR): IR = ApplyBinaryPrimOp(Subtract(), self, other)
  def *(other: IR): IR = ApplyBinaryPrimOp(Multiply(), self, other)
  def /(other: IR): IR = ApplyBinaryPrimOp(FloatingPointDivide(), self, other)
  def floorDiv(other: IR): IR = ApplyBinaryPrimOp(RoundToNegInfDivide(), self, other)

  def &&(other: IR): IR = invoke("land", TBoolean, self, other)
  def ||(other: IR): IR = invoke("lor", TBoolean, self, other)

  def toI: IR = Cast(self, TInt32)
  def toL: IR = Cast(self, TInt64)
  def toF: IR = Cast(self, TFloat32)
  def toD: IR = Cast(self, TFloat64)

  def unary_-(): IR = ApplyUnaryPrimOp(Negate, self)
  def unary_!(): IR = ApplyUnaryPrimOp(Bang, self)

  def ceq(other: IR): IR = ApplyComparisonOp(EQWithNA(self.typ, other.typ), self, other)
  def cne(other: IR): IR = ApplyComparisonOp(NEQWithNA(self.typ, other.typ), self, other)
  def <(other: IR): IR = ApplyComparisonOp(LT(self.typ, other.typ), self, other)
  def >(other: IR): IR = ApplyComparisonOp(GT(self.typ, other.typ), self, other)
  def <=(other: IR): IR = ApplyComparisonOp(LTEQ(self.typ, other.typ), self, other)
  def >=(other: IR): IR = ApplyComparisonOp(GTEQ(self.typ, other.typ), self, other)
}

object ErrorIDs {
  val NO_ERROR = -1
}
