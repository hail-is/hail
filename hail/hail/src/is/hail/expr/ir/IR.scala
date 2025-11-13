package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s.Value
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.agg.PhysicalAggSig
import is.hail.expr.ir.functions._
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

import scala.collection.compat._

import java.io.OutputStream

import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}
import org.json4s.JsonAST.{JNothing, JString}

trait IR extends BaseIR {
  private var _typ: Type = null

  override def typ: Type = {
    if (_typ == null) {
      try
        _typ = InferType(this)
      catch {
        case e: Throwable => throw new RuntimeException(s"typ: inference failure:", e)
      }
      assert(_typ != null)
    }
    _typ
  }

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
}

package defs {

  trait TypedIR[T <: Type] extends IR {
    override def typ: T = tcoerce[T](super.typ)
  }

  // Mark Refs and constants as IRs that are safe to duplicate
  trait TrivialIR extends IR

  class WrappedByteArrays(val ba: Array[Array[Byte]]) {
    override def hashCode(): Int =
      ba.foldLeft(31)((h, b) => 37 * h + java.util.Arrays.hashCode(b))

    override def equals(obj: Any): Boolean = {
      this.eq(obj.asInstanceOf[AnyRef]) || {
        if (!obj.isInstanceOf[WrappedByteArrays]) {
          false
        } else {
          val other = obj.asInstanceOf[WrappedByteArrays]
          ba.length == other.ba.length && ba.lazyZip(other.ba).forall(java.util.Arrays.equals)
        }
      }
    }
  }

  object AggLet {
    def apply(name: Name, value: IR, body: IR, isScan: Boolean): IR = {
      val scope = if (isScan) Scope.SCAN else Scope.AGG
      Block(FastSeq(Binding(name, value, scope)), body)
    }
  }

  object Let {
    def apply(bindings: IndexedSeq[(Name, IR)], body: IR): Block =
      Block(
        bindings.map { case (name, value) => Binding(name, value) },
        body,
      )

    def void(bindings: IndexedSeq[(Name, IR)]): IR = {
      if (bindings.isEmpty) {
        Void()
      } else {
        assert(bindings.last._2.typ == TVoid)
        Let(bindings.init, bindings.last._2)
      }
    }
  }

  object Begin {
    def apply(xs: IndexedSeq[IR]): IR =
      if (xs.isEmpty)
        Void()
      else
        Let(xs.init.map(x => (freshName(), x)), xs.last)
  }

  case class Binding(name: Name, value: IR, scope: Int = Scope.EVAL)

  trait BaseRef extends IR with TrivialIR {
    def name: Name
    def _typ: Type
  }

  object ArrayZipBehavior extends Enumeration {
    type ArrayZipBehavior = Value
    val AssumeSameLength: Value = Value(0)
    val AssertSameLength: Value = Value(1)
    val TakeMinLength: Value = Value(2)
    val ExtendNA: Value = Value(3)
  }

  object StreamJoin {
    def apply(
      left: IR,
      right: IR,
      lKey: IndexedSeq[String],
      rKey: IndexedSeq[String],
      l: Name,
      r: Name,
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

        val rElt = Ref(freshName(), tcoerce[TStream](rightGrouped.typ).elementType)
        val lElt = Ref(freshName(), lEltType)
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
        StreamJoinRightDistinct(left, right, lKey, rKey, l, r, joinF, joinType)
      }
    }
  }

  trait NDArrayIR extends TypedIR[TNDArray] {
    def elementTyp: Type = typ.elementType
  }

  object GetFieldByIdx {
    def apply(s: IR, field: Int): IR =
      (s.typ: @unchecked) match {
        case t: TStruct => GetField(s, t.fieldNames(field))
        case _: TTuple => GetTupleElement(s, field)
      }
  }

  trait AbstractApplyNode[F <: JVMFunction] extends IR {
    def function: String

    def args: Seq[IR]

    def returnType: Type

    def typeArgs: Seq[Type]

    def argTypes: Seq[Type] = args.map(_.typ)

    lazy val implementation: F =
      IRFunctionRegistry.lookupFunctionOrFail(function, returnType, typeArgs, argTypes)
        .asInstanceOf[F]
  }

  object PartitionReader {
    implicit val formats: Formats =
      new DefaultFormats() {
        override val typeHints = ShortTypeHints(
          List(
            classOf[PartitionRVDReader],
            classOf[PartitionNativeReader],
            classOf[PartitionNativeReaderIndexed],
            classOf[PartitionNativeIntervalReader],
            classOf[PartitionZippedNativeIntervalReader],
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
        case "PartitionZippedNativeIntervalReader" =>
          val path = (jv \ "path").extract[String]
          val spec = RelationalSpec.read(ctx.fs, path).asInstanceOf[AbstractMatrixTableSpec]
          PartitionZippedNativeIntervalReader(
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
          GVCFPartitionReader(
            header, callFields, entryFloatType, arrayElementsRequired, rg,
            contigRecoding,
            skipInvalidLoci, filterAndReplace, entriesFieldName, uidFieldName,
          )
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
    override def ctxType: Type = TString

    override def returnType: Type = TString

    override def unionTypeRequiredness(
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

    final override def consumeStream(
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
    override def writeMetadata(
      writeAnnotations: => IEmitCode,
      cb: EmitCodeBuilder,
      region: Value[Region],
    ): Unit =
      writeAnnotations.consume(cb, {}, _ => ())
  }

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

    def unary_- : IR = ApplyUnaryPrimOp(Negate, self)

    def unary_! : IR = ApplyUnaryPrimOp(Bang, self)

    def ceq(other: IR): IR = ApplyComparisonOp(EQWithNA, self, other)

    def cne(other: IR): IR = ApplyComparisonOp(NEQWithNA, self, other)

    def <(other: IR): IR = ApplyComparisonOp(LT, self, other)

    def >(other: IR): IR = ApplyComparisonOp(GT, self, other)

    def <=(other: IR): IR = ApplyComparisonOp(LTEQ, self, other)

    def >=(other: IR): IR = ApplyComparisonOp(GTEQ, self, other)

    def log(messages: AnyRef*): IR = logIR(self, messages: _*)
  }

  object ErrorIDs {
    val NO_ERROR = -1
  }

  package exts {

    abstract class UUID4CompanionExt {
      def apply(): UUID4 = UUID4(genUID())
    }

    abstract class MakeArrayCompanionExt {
      def apply(args: IR*): MakeArray = {
        assert(args.nonEmpty)
        MakeArray(args.toFastSeq, TArray(args.head.typ))
      }

      def unify(ctx: ExecuteContext, args: IndexedSeq[IR], requestedType: TArray = null)
        : MakeArray = {
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

    abstract class LiteralCompanionExt {
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

    abstract class EncodedLiteralCompanionExt {
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
            val etype = EType.defaultFromPType(ctx, pt)
            val codec = TypedCodecSpec(etype, pt.virtualType, BufferSpec.wireSpec)
            val bytes = codec.encodeArrays(ctx, pt, addr)
            EncodedLiteral(codec, bytes)
        }
      }
    }

    abstract class BlockCompanionExt {
      object Insert {
        def unapply(bindings: IndexedSeq[Binding])
          : Option[(IndexedSeq[Binding], Binding, IndexedSeq[Binding])] = {
          val idx = bindings.indexWhere(_.value.isInstanceOf[InsertFields])
          if (idx == -1) None else Some((bindings.take(idx), bindings(idx), bindings.drop(idx + 1)))
        }
      }

      object Nested {
        def unapply(bindings: IndexedSeq[Binding]): Option[(Int, IndexedSeq[Binding])] = {
          val idx = bindings.indexWhere(_.value.isInstanceOf[Block])
          if (idx == -1) None else Some((idx, bindings))
        }
      }
    }

    abstract class MakeStreamCompanionExt {
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

    abstract class ArraySortCompanionExt {
      def apply(a: IR, ascending: IR = True(), onKey: Boolean = false): ArraySort = {
        val l = freshName()
        val r = freshName()
        val atyp = tcoerce[TStream](a.typ)
        val compare = if (onKey) {
          val elementType = atyp.elementType.asInstanceOf[TBaseStruct]
          elementType match {
            case _: TStruct =>
              val elt = tcoerce[TStruct](atyp.elementType)
              ApplyComparisonOp(
                Compare,
                GetField(Ref(l, elt), elt.fieldNames(0)),
                GetField(Ref(r, atyp.elementType), elt.fieldNames(0)),
              )
            case _: TTuple =>
              val elt = tcoerce[TTuple](atyp.elementType)
              ApplyComparisonOp(
                Compare,
                GetTupleElement(Ref(l, elt), elt.fields(0).index),
                GetTupleElement(Ref(r, atyp.elementType), elt.fields(0).index),
              )
          }
        } else {
          ApplyComparisonOp(
            Compare,
            Ref(l, atyp.elementType),
            Ref(r, atyp.elementType),
          )
        }

        ArraySort(a, l, r, If(ascending, compare < 0, compare > 0))
      }
    }

    abstract class StreamFold2CompanionExt {
      def apply(a: StreamFold): StreamFold2 =
        StreamFold2(
          a.a,
          FastSeq((a.accumName, a.zero)),
          a.valueName,
          FastSeq(a.body),
          Ref(a.accumName, a.zero.typ),
        )
    }

    trait StreamJoinRightDistinctExt { self: StreamJoinRightDistinct =>
      def isIntervalJoin: Boolean = {
        if (rKey.size != 1) return false
        val lKeyTyp = tcoerce[TStruct](tcoerce[TStream](left.typ).elementType).fieldType(lKey(0))
        val rKeyTyp = tcoerce[TStruct](tcoerce[TStream](right.typ).elementType).fieldType(rKey(0))

        rKeyTyp.isInstanceOf[TInterval] && lKeyTyp != rKeyTyp
      }
    }

    abstract class MakeNDArrayCompanionExt {
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

    abstract class NDArrayQRCompanionExt {
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

    abstract class NDArraySVDCompanionExt {
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

    abstract class NDArrayEighCompanionExt {
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

    abstract class NDArrayInvCompanionExt {
      val pType = PCanonicalNDArray(PFloat64Required, 2)
    }

    abstract class ApplyAggOpCompanionExt {
      def apply(op: AggOp, initOpArgs: IR*)(seqOpArgs: IR*): ApplyAggOp =
        ApplyAggOp(initOpArgs.toIndexedSeq, seqOpArgs.toIndexedSeq, op)
    }

    abstract class AggFoldCompanionExt {
      def min(element: IR, sortFields: IndexedSeq[SortField]): IR = {
        val elementType = element.typ.asInstanceOf[TStruct]
        val keyType = elementType.select(sortFields.map(_.field))._1
        minAndMaxHelper(element, keyType, StructLT(sortFields))
      }

      def max(element: IR, sortFields: IndexedSeq[SortField]): IR = {
        val elementType = element.typ.asInstanceOf[TStruct]
        val keyType = elementType.select(sortFields.map(_.field))._1
        minAndMaxHelper(element, keyType, StructGT(sortFields))
      }

      def all(element: IR): IR =
        aggFoldIR(True()) { accum =>
          ApplySpecial(
            "land",
            Seq.empty[Type],
            FastSeq(accum, element),
            TBoolean,
            ErrorIDs.NO_ERROR,
          )
        } { (accum1, accum2) =>
          ApplySpecial(
            "land",
            Seq.empty[Type],
            FastSeq(accum1, accum2),
            TBoolean,
            ErrorIDs.NO_ERROR,
          )
        }

      private def minAndMaxHelper(element: IR, keyType: TStruct, comp: ComparisonOp[Boolean])
        : IR = {
        val keyFields = keyType.fields.map(_.name)

        val minAndMaxZero = NA(keyType)
        val aggFoldMinAccumName1 = freshName()
        val aggFoldMinAccumName2 = freshName()
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

    abstract class ApplyScanOpCompanionExt {
      def apply(op: AggOp, initOpArgs: IR*)(seqOpArgs: IR*): ApplyScanOp =
        ApplyScanOp(initOpArgs.toIndexedSeq, seqOpArgs.toIndexedSeq, op)
    }

    trait ApplyScanOpExt { self: ApplyScanOp =>
      def nSeqOpArgs = seqOpArgs.length

      def nInitArgs = initOpArgs.length
    }

    abstract class ResultOpCompanionExt {
      def makeTuple(aggs: IndexedSeq[PhysicalAggSig]) =
        MakeTuple.ordered(aggs.zipWithIndex.map { case (aggSig, index) =>
          ResultOp(index, aggSig)
        })
    }

    abstract class MakeTupleCompanionExt {
      def ordered(types: IndexedSeq[IR]): MakeTuple = MakeTuple(types.zipWithIndex.map {
        case (ir, i) =>
          (i, ir)
      })
    }

    abstract class InCompanionExt {
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

    abstract class DieCompanionExt {
      def apply(message: String, typ: Type): Die = Die(Str(message), typ, ErrorIDs.NO_ERROR)

      def apply(message: String, typ: Type, errorId: Int): Die = Die(Str(message), typ, errorId)
    }

    trait ApplyIRExt { self: ApplyIR =>
      lazy val (body, inline): (IR, Boolean) = {
        val ((_, _, _, inline), impl) =
          IRFunctionRegistry.lookupIR(function, typeArgs, args.map(_.typ)).get
        val body = impl(typeArgs, refs, errorID).deepCopy()
        (body, inline)
      }

      lazy val refs: IndexedSeq[Ref] = args.map(a => Ref(freshName(), a.typ)).toFastSeq

      lazy val explicitNode: IR = {
        val ir = Let(refs.map(_.name).zip(args), body)
        assert(ir.typ == returnType)
        ir
      }
    }
  }
}
