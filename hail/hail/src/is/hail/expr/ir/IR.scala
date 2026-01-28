package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s.Value
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.agg.PhysicalAggSig
import is.hail.expr.ir.defs.{ComputableType, ExplicitType, InferableType}
import is.hail.expr.ir.functions._
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, TypedCodecSpec}
import is.hail.io.avro.{AvroPartitionReader, AvroSchemaSerializer}
import is.hail.io.bgen.BgenPartitionReader
import is.hail.io.vcf.{GVCFPartitionReader, VCFHeaderInfo}
import is.hail.rvd.RVDSpecMaker
import is.hail.types.{RIterable, RStruct, TypeWithRequiredness, tcoerce}
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
  protected var _typ: Type = null

  override def typ: Type = {
    assert(_typ != null)
    _typ
  }

  override def mapChildren(f: BaseIR => BaseIR): IR = super.mapChildren(f).asInstanceOf[IR]

  override def mapChildrenWithIndex(f: (BaseIR, Int) => BaseIR): IR =
    super.mapChildrenWithIndex(f).asInstanceOf[IR]

  override def mapChildrenWithIndexUntyped(f: (BaseIR, Int) => UntypedBaseIR[BaseIR])
    : UntypedBaseIR[IR] =
    super.mapChildrenWithIndexUntyped(f).asInstanceOf[UntypedBaseIR[IR]]

  lazy val size: Int = 1 + children.map {
    case x: IR => x.size
    case _ => 0
  }.sum

  override def annotateTypes(ctx: ExecuteContext, env: BindingEnv[Type]): Unit = {
    this match {
      case x: InferableType =>
        super.annotateTypes(ctx, env)
        if (_typ == null) {
          _typ = x.inferType(ctx, env)
          assert(_typ != null)
        }
      case x: ComputableType => if (_typ == null) {
          super.annotateTypes(ctx, env)
          _typ = x.computeTypeFromChildren
          assert(_typ != null)
        }
      case x: ExplicitType =>
        super.annotateTypes(ctx, env)
        x.check()
    }
  }
}

package defs {

  import is.hail.expr.Nat

  // Mark Refs and constants as IRs that are safe to duplicate
  trait TrivialIR extends IR

  trait ComputableType extends IR {
    def computeTypeFromChildren: Type
  }

  trait ExplicitType extends IR {
    def check(): Unit
  }

  trait InferableType extends IR {
    def inferType(ctx: ExecuteContext, env: BindingEnv[Type]): Type
    def check(): Unit
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

    def untyped(name: Name, value: UntypedIR, body: UntypedIR, isScan: Boolean): UntypedIR = {
      val scope = if (isScan) Scope.SCAN else Scope.AGG
      Block.untyped(FastSeq(Binding.untyped(name, value, scope)), body)
    }
  }

  object Let extends ((IndexedSeq[(Name, IR)], IR) => Block) {
    def apply(bindings: IndexedSeq[(Name, IR)], body: IR): Block =
      Block(
        bindings.map { case (name, value) => Binding(name, value) },
        body,
      )

    def untyped(bindings: IndexedSeq[(Name, UntypedIR)], body: UntypedIR): UntypedIR =
      Block.untyped(
        bindings.map { case (name, value) => Binding.untyped(name, value) },
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

    def untyped(xs: IndexedSeq[UntypedIR]): UntypedIR =
      if (xs.isEmpty)
        Void()
      else
        Let.untyped(xs.init.map(x => (freshName(), x)), xs.last)
  }

  object Binding {
    def untyped(name: Name, value: UntypedIR, scope: Int = Scope.EVAL): untyped =
      new untyped(Binding(name, value.get, scope))

    class untyped(val get: Binding) extends AnyVal
  }

  case class Binding(name: Name, value: IR, scope: Int = Scope.EVAL)

  trait BaseRef extends IR with TrivialIR {
    def name: Name
//    def _typ: Type
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

    def typeArgs: Seq[Type]

    def argTypes: Seq[Type] =
      args.map(_.typ)

    lazy val implementation: F =
      IRFunctionRegistry.lookupFunctionOrFail(function, typ, typeArgs, argTypes)
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

    trait I32Ext { self: I32 =>
      def computeTypeFromChildren: TInt32.type = TInt32
    }

    trait I64Ext { self: I64 =>
      def computeTypeFromChildren: TInt64.type = TInt64
    }

    trait F32Ext { self: F32 =>
      def computeTypeFromChildren: TFloat32.type = TFloat32
    }

    trait F64Ext { self: F64 =>
      def computeTypeFromChildren: TFloat64.type = TFloat64
    }

    trait StrExt { self: Str =>
      def computeTypeFromChildren: TString.type = TString
    }

    abstract class UUID4CompanionExt {
      def apply(): UUID4 = UUID4(genUID())
    }

    trait UUID4Ext { self: UUID4 =>
      def computeTypeFromChildren: TString.type = TString
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

    trait LiteralExt { self: Literal =>
      def check(): Unit = {
        require(!CanEmit(typ))
        require(value != null)
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

    trait EncodedLiteralExt { self: EncodedLiteral =>
      def computeTypeFromChildren: Type = codec.encodedVirtualType
    }

    trait TrueExt { self: True =>
      def computeTypeFromChildren: TBoolean.type = TBoolean
    }

    trait FalseExt { self: False =>
      def computeTypeFromChildren: TBoolean.type = TBoolean
    }

    trait VoidExt { self: Void =>
      def computeTypeFromChildren: TVoid.type = TVoid
    }

    trait CastExt { self: Cast =>
      def check(): Unit = if (!Casts.valid(v.typ, typ))
        throw new RuntimeException(s"invalid cast:\n  " +
          s"child type: ${v.typ.parsableString()}\n  " +
          s"cast type:  ${typ.parsableString()}")
    }

    trait CastRenameExt { self: CastRename =>
      def check(): Unit = if (!v.typ.isIsomorphicTo(typ))
        throw new RuntimeException(s"invalid cast:\n  " +
          s"child type: ${v.typ.parsableString()}\n  " +
          s"cast type:  ${typ.parsableString()}")
    }

    trait NAExt { self: NA =>
      def check(): Unit = {}
    }

    trait IsNAExt { self: IsNA =>
      def computeTypeFromChildren: TBoolean.type = {
        assert(value.typ.isRealizable)
        TBoolean
      }
    }

    trait CoalesceExt { self: Coalesce =>
      def computeTypeFromChildren: Type = {
        assert(values.tail.forall(v => v.typ == values.head.typ))
        val t = values.head.typ
        assert(t.isRealizable)
        t
      }
    }

    trait ConsumeExt { self: Consume =>
      def computeTypeFromChildren: TInt64.type = {
        assert(value.typ.isRealizable)
        TInt64
      }
    }

    abstract class RefCompanionExt {
      def untyped(name: Name): UntypedIR = UntypedIR(new Ref(name))
    }

    trait RefExt extends InferableType { self: Ref =>
      override def check(): Unit = {}

      override def inferType(ctx: ExecuteContext, env: BindingEnv[Type]): Type =
        env.eval(name)
    }

    trait RelationalRefExt { self: RelationalRef =>
      def check(): Unit = {}
    }

    trait RelationalLetExt { self: RelationalLet =>
      def computeTypeFromChildren: Type = {
        assert(body.typ.isRealizable)
        body.typ
      }
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

    trait InExt { self: In =>
      def computeTypeFromChildren: Type = {
        assert(paramType != null)
        val t = paramType.virtualType
        t match {
          case stream: TStream => assert(stream.elementType.isRealizable)
          case _ => assert(t.isRealizable)
        }
        t
      }
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

    trait MakeArrayExt extends InferableType { self: MakeArray =>
      override def inferType(ctx: ExecuteContext, env: BindingEnv[Type]): Type = {
        assert(args.nonEmpty)
        TArray(args.head.typ)
      }

      override def check(): Unit = {
        assert(typ.elementType.isRealizable, typ.elementType)
        args.map(_.typ).zipWithIndex.foreach { case (x, i) =>
          assert(
            x == typ.elementType,
            s"at position $i type mismatch: ${typ.parsableString()} ${x.parsableString()}",
          )
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
        )
      }
    }

    trait MakeStreamExt { self: MakeStream =>
      def check(): Unit = {
        assert(typ.elementType.isRealizable, typ.elementType)
        args.map(_.typ).zipWithIndex.foreach { case (x, i) =>
          assert(
            x == typ.elementType,
            s"at position $i type mismatch: ${typ.elementType.parsableString()} ${x.parsableString()}",
          )
        }
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

    trait MakeNDArrayExt { self: MakeNDArray =>
      def computeTypeFromChildren: TNDArray = {
        val shapeTyp = tcoerce[TTuple](shape.typ)
        assert(data.typ.isInstanceOf[TArray] || data.typ.isInstanceOf[TStream])
        assert(shapeTyp.types.forall(t => t == TInt64))
        assert(rowMajor.typ == TBoolean)
        TNDArray(tcoerce[TIterable](data.typ).elementType, Nat(shapeTyp.size))
      }
    }

    trait StreamBufferedAggregateExt { self: StreamBufferedAggregate =>
      def computeTypeFromChildren: TStream = {
        assert(streamChild.typ.isInstanceOf[TStream])
        assert(initAggs.typ == TVoid)
        assert(seqOps.typ == TVoid)
        val tupleFieldTypes = TTuple(aggSignature.map(_ => TBinary): _*)
        TStream(tcoerce[TStruct](newKey.typ).insertFields(IndexedSeq(("agg", tupleFieldTypes))))
      }
    }

    trait ArrayLenExt { self: ArrayLen =>
      def computeTypeFromChildren: TInt32.type = {
        assert(a.typ.isInstanceOf[TArray])
        TInt32
      }
    }

    trait StreamIotaExt { self: StreamIota =>
      def computeTypeFromChildren: TStream = {
        assert(start.typ == TInt32)
        assert(step.typ == TInt32)
        TStream(TInt32)
      }
    }

    trait StreamRangeExt { self: StreamRange =>
      def computeTypeFromChildren: TStream = {
        assert(start.typ == TInt32)
        assert(stop.typ == TInt32)
        assert(step.typ == TInt32)
        TStream(TInt32)
      }
    }

    trait SeqSampleExt { self: SeqSample =>
      def computeTypeFromChildren: TStream = {
        assert(totalRange.typ == TInt32)
        assert(numToSample.typ == TInt32)
        assert(rngState.typ == TRNGState)
        TStream(TInt32)
      }
    }

    trait ArrayZerosExt { self: ArrayZeros =>
      def computeTypeFromChildren: TArray = {
        assert(length.typ == TInt32)
        TArray(TInt32)
      }
    }

    trait LowerBoundOnOrderedCollectionExt { self: LowerBoundOnOrderedCollection =>
      def computeTypeFromChildren: TInt32.type = {
        val elt = tcoerce[TContainer](orderedCollection.typ).elementType
        val elemTyp = if (onKey) elt match {
          case t: TBaseStruct => t.types(0)
          case t: TInterval => t.pointType
        }
        else elt
        assert(elem.typ == elemTyp)
        TInt32
      }
    }

    trait StreamForExt { self: StreamFor =>
      def computeTypeFromChildren: TVoid.type = {
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ == TVoid)
        TVoid
      }
    }

    trait InitOpExt { self: InitOp =>
      def computeTypeFromChildren: TVoid.type = {
        assert(
          args.map(_.typ) == aggSig.initOpTypes,
          s"${args.map(_.typ)} !=  ${aggSig.initOpTypes}",
        )
        TVoid
      }
    }

    trait SeqOpExt { self: SeqOp =>
      def computeTypeFromChildren: TVoid.type = {
        assert(args.map(_.typ) == aggSig.seqOpTypes)
        TVoid
      }
    }

    trait CombOpExt { self: CombOp =>
      def computeTypeFromChildren: TVoid.type = TVoid
    }

    abstract class ResultOpCompanionExt {
      def makeTuple(aggs: IndexedSeq[PhysicalAggSig]) =
        MakeTuple.ordered(aggs.zipWithIndex.map { case (aggSig, index) =>
          ResultOp(index, aggSig)
        })
    }

    trait ResultOpExt { self: ResultOp =>
      def computeTypeFromChildren: Type = aggSig.resultType
    }

    trait AggStateValueExt { self: AggStateValue =>
      def computeTypeFromChildren: TBinary.type = TBinary
    }

    trait CombOpValueExt { self: CombOpValue =>
      def computeTypeFromChildren: TVoid.type = {
        assert(value.typ == TBinary)
        TVoid
      }
    }

    trait InitFromSerializedValueExt { self: InitFromSerializedValue =>
      def computeTypeFromChildren: TVoid.type = {
        assert(value.typ == TBinary)
        TVoid
      }
    }

    trait SerializeAggsExt { self: SerializeAggs =>
      def computeTypeFromChildren: TVoid.type = TVoid
    }

    trait DeserializeAggsExt { self: DeserializeAggs =>
      def computeTypeFromChildren: TVoid.type = TVoid
    }

    abstract class DieCompanionExt {
      def apply(message: String, typ: Type): Die = Die(Str(message), typ, ErrorIDs.NO_ERROR)

      def apply(message: String, typ: Type, errorId: Int): Die = Die(Str(message), typ, errorId)
    }

    trait DieExt { self: Die =>
      def check(): Unit = assert(message.typ == TString)
    }

    trait TrapExt { self: Trap =>
      def computeTypeFromChildren: TTuple =
        TTuple(TTuple(TString, TInt32), child.typ)
    }

    trait ConsoleLogExt { self: ConsoleLog =>
      def computeTypeFromChildren: Type = {
        assert(message.typ == TString)
        result.typ
      }
    }

    trait IfExt { self: If =>
      def computeTypeFromChildren: Type = {
        assert(cond.typ == TBoolean)
        assert(cnsq.typ == altr.typ)
        cnsq.typ match {
          case tstream: TStream => assert(tstream.elementType.isRealizable)
          case _ =>
        }
        cnsq.typ
      }
    }

    trait SwitchExt { self: Switch =>
      def computeTypeFromChildren: Type = {
        assert(x.typ == TInt32)
        assert(cases.forall(_.typ == default.typ))
        default.typ
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

    trait BlockExt { self: Block =>
      def computeTypeFromChildren: Type = {
        assert(bindings.forall(_.value.typ != TVoid) || (body.typ == TVoid))
        body.typ
      }
    }

    trait TailLoopExt { self: TailLoop =>
      def check(): Unit = {}
    }

    trait RecurExt extends InferableType { self: Recur =>
      override def check(): Unit = {}

      override def inferType(ctx: ExecuteContext, env: BindingEnv[Type]): Type = {
        val TTuple(IndexedSeq(_, TupleField(_, rt))) = env.eval.lookup(name)
        rt
      }
    }

    trait ApplyBinaryPrimOpExt { self: ApplyBinaryPrimOp =>
      def computeTypeFromChildren: Type =
        BinaryOp.getReturnType(op, l.typ, r.typ)
    }

    trait ApplyUnaryPrimOpExt { self: ApplyUnaryPrimOp =>
      def computeTypeFromChildren: Type =
        UnaryOp.getReturnType(op, x.typ)
    }

    trait ApplyComparisonOpExt { self: ApplyComparisonOp =>
      def computeTypeFromChildren: Type = {
        ComparisonOp.checkCompatible(l.typ, r.typ)
        op match {
          case Compare => TInt32
          case _ => TBoolean
        }
      }
    }

    trait ApplyIRExt { self: ApplyIR =>
      def check(): Unit = {}

      lazy val (body, inline): (IR, Boolean) = {
        val ((_, _, _, inline), impl) =
          IRFunctionRegistry.lookupIR(function, typeArgs, args.map(_.typ)).get
        val body = impl(typeArgs, refs, errorID).deepCopy()
        (body, inline)
      }

      lazy val refs: IndexedSeq[Ref] = args.map(a => Ref(freshName(), a.typ)).toFastSeq

      lazy val explicitNode: IR = {
        val ir = Let(refs.map(_.name).zip(args), body)
        assert(ir.typ == typ)
        ir
      }
    }

    trait ApplyExt { self: Apply =>
      def check(): Unit = {}
    }

    trait ApplySeededExt { self: ApplySeeded =>
      def check(): Unit = {}
    }

    trait ApplySpecialExt { self: ApplySpecial =>
      def check(): Unit = {}
    }

    trait ArrayRefExt { self: ArrayRef =>
      def computeTypeFromChildren: Type = {
        assert(i.typ == TInt32)
        tcoerce[TArray](a.typ).elementType
      }
    }

    trait ArraySliceExt { self: ArraySlice =>
      def computeTypeFromChildren: TArray = {
        assert(start.typ == TInt32)
        stop.foreach(ir => assert(ir.typ == TInt32))
        assert(step.typ == TInt32)
        tcoerce[TArray](a.typ)
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

    trait ArraySortExt { self: ArraySort =>
      def computeTypeFromChildren: TArray = {
        assert(lessThan.typ == TBoolean)
        val et = tcoerce[TStream](a.typ).elementType
        TArray(et)
      }
    }

    trait ArrayMaximalIndependentSetExt { self: ArrayMaximalIndependentSet =>
      def computeTypeFromChildren: TArray = {
        val edgeType = tcoerce[TBaseStruct](tcoerce[TArray](edges.typ).elementType)
        val Array(leftType, rightType) = edgeType.types
        assert(leftType == rightType)
        tieBreaker.foreach { case (_, _, tb) => assert(tb.typ == TFloat64) }
        TArray(leftType)
      }
    }

    trait ToSetExt { self: ToSet =>
      def computeTypeFromChildren: TSet = {
        val et = tcoerce[TStream](a.typ).elementType
        TSet(et)
      }
    }

    trait ToDictExt { self: ToDict =>
      def computeTypeFromChildren: TDict = {
        val elt = tcoerce[TBaseStruct](tcoerce[TStream](a.typ).elementType)
        assert(elt.size == 2)
        TDict(elt.types(0), elt.types(1))
      }
    }

    trait ToArrayExt { self: ToArray =>
      def computeTypeFromChildren: TArray = {
        val elt = tcoerce[TStream](a.typ).elementType
        TArray(elt)
      }
    }

    trait CastToArrayExt { self: CastToArray =>
      def computeTypeFromChildren: TArray = {
        val elt = tcoerce[TContainer](a.typ).elementType
        TArray(elt)
      }
    }

    trait ToStreamExt { self: ToStream =>
      def computeTypeFromChildren: TStream = {
        val elt = tcoerce[TContainer](a.typ).elementType
        TStream(elt)
      }
    }

    trait RNGStateLiteralExt { self: RNGStateLiteral =>
      def computeTypeFromChildren: TRNGState.type = TRNGState
    }

    trait RNGSplitExt { self: RNGSplit =>
      def computeTypeFromChildren: TRNGState.type = {
        assert(state.typ == TRNGState)
        def isValid: Type => Boolean = {
          case tuple: TTuple => tuple.types.forall(_ == TInt64)
          case t => t == TInt64
        }
        assert(isValid(dynBitstring.typ))
        TRNGState
      }
    }

    trait StreamLenExt { self: StreamLen =>
      def computeTypeFromChildren: TInt32.type = {
        assert(a.typ.isInstanceOf[TStream])
        TInt32
      }
    }

    trait GroupByKeyExt { self: GroupByKey =>
      def computeTypeFromChildren: TDict = {
        val elt = tcoerce[TBaseStruct](tcoerce[TStream](collection.typ).elementType)
        TDict(elt.types(0), TArray(elt.types(1)))
      }
    }

    trait StreamTakeExt { self: StreamTake =>
      def computeTypeFromChildren: TStream = {
        assert(num.typ == TInt32)
        tcoerce[TStream](a.typ)
      }
    }

    trait StreamDropExt { self: StreamDrop =>
      def computeTypeFromChildren: TStream = {
        assert(num.typ == TInt32)
        tcoerce[TStream](a.typ)
      }
    }

    trait StreamGroupedExt { self: StreamGrouped =>
      def computeTypeFromChildren: TStream = {
        assert(a.typ.isInstanceOf[TStream])
        assert(groupSize.typ == TInt32)
        TStream(a.typ)
      }
    }

    trait StreamGroupByKeyExt { self: StreamGroupByKey =>
      def computeTypeFromChildren: TStream = {
        val structType = tcoerce[TStruct](tcoerce[TStream](a.typ).elementType)
        assert(key.forall(structType.hasField))
        TStream(a.typ)
      }
    }

    trait StreamMapExt { self: StreamMap =>
      def computeTypeFromChildren: TStream = {
        assert(a.typ.isInstanceOf[TStream])
        TStream(body.typ)
      }
    }

    trait StreamZipExt { self: StreamZip =>
      def computeTypeFromChildren: TStream = {
        assert(as.length == names.length)
        assert(as.forall(_.typ.isInstanceOf[TStream]))
        TStream(body.typ)
      }
    }

    trait StreamZipJoinExt { self: StreamZipJoin =>
      def computeTypeFromChildren: TStream = {
        val streamType = tcoerce[TStream](as.head.typ)
        assert(as.forall(_.typ == streamType))
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(key.forall(eltType.hasField))
        TStream(joinF.typ)
      }
    }

    trait StreamZipJoinProducersExt { self: StreamZipJoinProducers =>
      def computeTypeFromChildren: TStream = {
        assert(contexts.typ.isInstanceOf[TArray])
        val streamType = tcoerce[TStream](makeProducer.typ)
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(key.forall(eltType.hasField))
        TStream(joinF.typ)
      }
    }

    trait StreamMultiMergeExt { self: StreamMultiMerge =>
      def computeTypeFromChildren: TStream = {
        val streamType = tcoerce[TStream](as.head.typ)
        assert(as.forall(_.typ == streamType))
        val eltType = tcoerce[TStruct](streamType.elementType)
        assert(key.forall(eltType.hasField))
        TStream(streamType.elementType)
      }
    }

    trait StreamFilterExt { self: StreamFilter =>
      def computeTypeFromChildren: TStream = {
        val streamType = tcoerce[TStream](a.typ)
        assert(streamType.elementType.isRealizable)
        assert(cond.typ == TBoolean, cond.typ)
        streamType
      }
    }

    trait StreamTakeWhileExt { self: StreamTakeWhile =>
      def computeTypeFromChildren: TStream = {
        val streamType = tcoerce[TStream](a.typ)
        assert(streamType.elementType.isRealizable)
        assert(body.typ == TBoolean)
        streamType
      }
    }

    trait StreamDropWhileExt { self: StreamDropWhile =>
      def computeTypeFromChildren: TStream = {
        val streamType = tcoerce[TStream](a.typ)
        assert(streamType.elementType.isRealizable)
        assert(body.typ == TBoolean)
        streamType
      }
    }

    trait StreamFlatMapExt { self: StreamFlatMap =>
      def computeTypeFromChildren: TStream = {
        assert(a.typ.isInstanceOf[TStream])
        TStream(tcoerce[TStream](cond.typ).elementType)
      }
    }

    trait StreamFoldExt { self: StreamFold =>
      def computeTypeFromChildren: Type = {
        assert(tcoerce[TStream](a.typ).elementType.isRealizable)
        assert(body.typ == zero.typ)
        zero.typ
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

    trait StreamFold2Ext { self: StreamFold2 =>
      def computeTypeFromChildren: Type = {
        assert(a.typ.isInstanceOf[TStream])
        assert(accum.zip(seq).forall { case ((_, z), s) => s.typ == z.typ })
        result.typ
      }
    }

    trait StreamDistributeExt { self: StreamDistribute =>
      def computeTypeFromChildren: TArray = {
        assert(path.typ == TString)
        assert(child.typ.isInstanceOf[TStream])
        val keyType = tcoerce[TStruct](tcoerce[TArray](pivots.typ).elementType)
        TArray(TStruct(
          ("interval", TInterval(keyType)),
          ("fileName", TString),
          ("numElements", TInt32),
          ("numBytes", TInt64),
        ))
      }
    }

    trait StreamWhitenExt { self: StreamWhiten =>
      def computeTypeFromChildren: TStream = {
        val streamTyp = tcoerce[TStream](stream.typ)
        val structTyp = tcoerce[TStruct](streamTyp.elementType)
        val matTyp = TNDArray(TFloat64, Nat(2))
        assert(structTyp.field(newChunk).typ == matTyp)
        assert(structTyp.field(prevWindow).typ == matTyp)
        assert(windowSize % chunkSize == 0)
        streamTyp
      }
    }

    trait StreamScanExt { self: StreamScan =>
      def computeTypeFromChildren: TStream = {
        assert(a.typ.isInstanceOf[TStream])
        assert(body.typ == zero.typ)
        assert(zero.typ.isRealizable)
        TStream(zero.typ)
      }
    }

    trait StreamAggExt { self: StreamAgg =>
      def computeTypeFromChildren = {
        assert(a.typ.isInstanceOf[TStream])
        query.typ
      }
    }

    trait StreamAggScanExt { self: StreamAggScan =>
      def computeTypeFromChildren = {
        assert(a.typ.isInstanceOf[TStream])
        TStream(query.typ)
      }
    }

    trait StreamLocalLDPruneExt { self: StreamLocalLDPrune =>
      def computeTypeFromChildren = {
        val eltType = tcoerce[TStruct](tcoerce[TStream](child.typ).elementType)
        assert(r2Threshold.typ == TFloat64)
        assert(windowSize.typ == TInt32)
        assert(maxQueueSize.typ == TInt32)
        assert(nSamples.typ == TInt32)
        val allelesType = eltType.fieldType("alleles")
        assert(tcoerce[TArray](allelesType).elementType == TString)
        assert(tcoerce[TArray](eltType.fieldType("genotypes")).elementType == TCall)
        TStream(TStruct(
          "locus" -> tcoerce[TLocus](eltType.fieldType("locus")),
          "alleles" -> allelesType,
          "mean" -> TFloat64,
          "centered_length_rec" -> TFloat64,
        ))
      }
    }

    trait RunAggExt { self: RunAgg =>
      def computeTypeFromChildren = {
        assert(body.typ == TVoid)
        result.typ
      }
    }

    trait RunAggScanExt { self: RunAggScan =>
      def computeTypeFromChildren = {
        assert(array.typ.isInstanceOf[TStream])
        assert(init.typ == TVoid)
        assert(seqs.typ == TVoid)
        TStream(result.typ)
      }
    }

    trait StreamLeftIntervalJoinExt { self: StreamLeftIntervalJoin =>
      def computeTypeFromChildren = {
        val lEltTy = tcoerce[TStruct](tcoerce[TStream](left.typ).elementType)
        val rPointTy = tcoerce[TStruct](tcoerce[TStream](right.typ).elementType)
          .fieldType(rIntervalFieldName)
          .asInstanceOf[TInterval]
          .pointType
        assert(lEltTy.fieldType(lKeyFieldName) == rPointTy)
        assert(body.typ.isInstanceOf[TStruct])
        TStream(body.typ)
      }
    }

    abstract class StreamJoinRightDistinctCompanionExt {
      def isIntervalJoin(
        lTyp: TStruct,
        rTyp: TStruct,
        lKey: IndexedSeq[String],
        rKey: IndexedSeq[String],
      ): Boolean = {
        if (rKey.size != 1) return false
        val lKeyTyp = lTyp.fieldType(lKey(0))
        val rKeyTyp = rTyp.fieldType(rKey(0))

        rKeyTyp.isInstanceOf[TInterval] && lKeyTyp != rKeyTyp
      }
    }

    trait StreamJoinRightDistinctExt { self: StreamJoinRightDistinct =>
      def computeTypeFromChildren = {
        val lEltTyp = tcoerce[TStruct](tcoerce[TStream](left.typ).elementType)
        val rEltTyp = tcoerce[TStruct](tcoerce[TStream](right.typ).elementType)
        assert(lKey.forall(lEltTyp.hasField))
        assert(rKey.forall(rEltTyp.hasField))
        if (defs.StreamJoinRightDistinct.isIntervalJoin(lEltTyp, rEltTyp, lKey, rKey)) {
          val lKeyTyp = lEltTyp.fieldType(lKey(0))
          val rKeyTyp = rEltTyp.fieldType(rKey(0)).asInstanceOf[TInterval]
          assert(lKeyTyp == rKeyTyp.pointType)
          assert((joinType == "left") || (joinType == "inner"))
        } else {
          assert(lKey.lazyZip(rKey).forall { case (lk, rk) =>
            lEltTyp.fieldType(lk) == rEltTyp.fieldType(rk)
          })
        }
        TStream(joinF.typ)
      }

      def isIntervalJoin: Boolean = {
        val lTyp = tcoerce[TStruct](tcoerce[TStream](left.typ).elementType)
        val rTyp = tcoerce[TStruct](tcoerce[TStream](right.typ).elementType)
        StreamJoinRightDistinct.isIntervalJoin(lTyp, rTyp, lKey, rKey)
      }
    }

    trait NDArrayShapeExt { self: NDArrayShape =>
      def computeTypeFromChildren = tcoerce[TNDArray](nd.typ).shapeType
    }

    trait NDArrayReshapeExt { self: NDArrayReshape =>
      def computeTypeFromChildren = {
        val shapeTyp = tcoerce[TTuple](shape.typ)
        assert(shapeTyp.types.forall(t => t == TInt64))
        TNDArray(tcoerce[TNDArray](nd.typ).elementType, Nat(shapeTyp.size))
      }
    }

    trait NDArrayConcatExt { self: NDArrayConcat =>
      def computeTypeFromChildren = {
        val ndType = tcoerce[TNDArray](tcoerce[TArray](nds.typ).elementType)
        assert(axis < ndType.nDims)
        ndType
      }
    }

    trait NDArrayMapExt { self: NDArrayMap =>
      def computeTypeFromChildren =
        TNDArray(body.typ, tcoerce[TNDArray](nd.typ).nDimsBase)
    }

    trait NDArrayMap2Ext { self: NDArrayMap2 =>
      def computeTypeFromChildren = {
        val lTyp = tcoerce[TNDArray](l.typ)
        val rTyp = tcoerce[TNDArray](r.typ)
        assert(lTyp.nDims == rTyp.nDims)
        TNDArray(body.typ, lTyp.nDimsBase)
      }
    }

    trait NDArrayReindexExt { self: NDArrayReindex =>
      def computeTypeFromChildren = {
        val ndTyp = tcoerce[TNDArray](nd.typ)
        val nInputDims = ndTyp.nDims
        val nOutputDims = indexExpr.length
        assert(nInputDims <= nOutputDims)
        assert(indexExpr.forall(i => i < nOutputDims))
        assert((0 until nOutputDims).forall(i => indexExpr.contains(i)))
        TNDArray(ndTyp.elementType, Nat(indexExpr.length))
      }
    }

    trait NDArrayAggExt { self: NDArrayAgg =>
      def computeTypeFromChildren = {
        val childType = tcoerce[TNDArray](nd.typ)
        val nInputDims = childType.nDims
        assert(axes.length <= nInputDims)
        assert(axes.forall(i => i < nInputDims))
        assert(axes.distinct.length == axes.length)
        TNDArray(childType.elementType, Nat(nInputDims - axes.length))
      }
    }

    trait NDArrayRefExt { self: NDArrayRef =>
      def computeTypeFromChildren = {
        val childType = tcoerce[TNDArray](nd.typ)
        assert(childType.nDims == idxs.length)
        assert(idxs.forall(_.typ == TInt64))
        childType.elementType
      }
    }

    trait NDArraySliceExt { self: NDArraySlice =>
      def computeTypeFromChildren = {
        val childTyp = tcoerce[TNDArray](nd.typ)
        val slicesTyp = tcoerce[TTuple](slices.typ)
        assert(slicesTyp.size == childTyp.nDims)
        assert(slicesTyp.types.forall(t => (t == TTuple(TInt64, TInt64, TInt64)) || (t == TInt64)))
        val tuplesOnly = slicesTyp.types.collect {
          case x: TTuple => x
        }
        val remainingDims = Nat(tuplesOnly.length)
        TNDArray(childTyp.elementType, remainingDims)
      }
    }

    trait NDArrayFilterExt { self: NDArrayFilter =>
      def computeTypeFromChildren = {
        val ndtyp = tcoerce[TNDArray](nd.typ)
        assert(ndtyp.nDims == keep.length)
        assert(keep.forall(f => tcoerce[TArray](f.typ).elementType == TInt64))
        ndtyp
      }
    }

    trait NDArrayMatMulExt { self: NDArrayMatMul =>
      def computeTypeFromChildren = {
        val lTyp = tcoerce[TNDArray](l.typ)
        val rTyp = tcoerce[TNDArray](r.typ)
        assert(lTyp.elementType == rTyp.elementType, "element type did not match")
        assert(lTyp.nDims > 0)
        assert(rTyp.nDims > 0)
        assert(lTyp.nDims == 1 || rTyp.nDims == 1 || lTyp.nDims == rTyp.nDims)
        TNDArray(lTyp.elementType, Nat(TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims)))
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

    trait NDArrayQRExt { self: NDArrayQR =>
      def computeTypeFromChildren = {
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
        if (Array("complete", "reduced").contains(mode)) {
          TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)))
        } else if (mode == "raw") {
          TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(1)))
        } else if (mode == "r") {
          TNDArray(TFloat64, Nat(2))
        } else {
          throw new NotImplementedError(s"Cannot infer type for mode $mode")
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

    trait NDArraySVDExt { self: NDArraySVD =>
      def computeTypeFromChildren = {
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
        if (computeUV) {
          TTuple(TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(1)), TNDArray(TFloat64, Nat(2)))
        } else {
          TNDArray(TFloat64, Nat(1))
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

    trait NDArrayEighExt { self: NDArrayEigh =>
      def computeTypeFromChildren = {
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
        if (eigvalsOnly) {
          TNDArray(TFloat64, Nat(1))
        } else {
          TTuple(TNDArray(TFloat64, Nat(1)), TNDArray(TFloat64, Nat(2)))
        }
      }
    }

    abstract class NDArrayInvCompanionExt {
      val pType = PCanonicalNDArray(PFloat64Required, 2)
    }

    trait NDArrayInvExt { self: NDArrayInv =>
      def computeTypeFromChildren = {
        val ndType = nd.typ.asInstanceOf[TNDArray]
        assert(ndType.elementType == TFloat64)
        assert(ndType.nDims == 2)
        TNDArray(TFloat64, Nat(2))
      }
    }

    trait NDArrayWriteExt { self: NDArrayWrite =>
      def computeTypeFromChildren = {
        assert(nd.typ.isInstanceOf[TNDArray])
        assert(path.typ == TString)
        TVoid
      }
    }

    trait AggFilterExt { self: AggFilter =>
      def computeTypeFromChildren = {
        assert(cond.typ == TBoolean)
        aggIR.typ
      }
    }

    trait AggExplodeExt { self: AggExplode =>
      def computeTypeFromChildren = {
        assert(array.typ.isInstanceOf[TStream])
        aggBody.typ
      }
    }

    trait AggGroupByExt { self: AggGroupBy =>
      def computeTypeFromChildren = TDict(key.typ, aggIR.typ)
    }

    trait AggArrayPerElementExt { self: AggArrayPerElement =>
      def computeTypeFromChildren = {
        assert(a.typ.isInstanceOf[TArray])
        assert(knownLength.forall(_.typ == TInt32))
        TArray(aggBody.typ)
      }
    }

    abstract class ApplyAggOpCompanionExt {
      def apply(op: AggOp, initOpArgs: IR*)(seqOpArgs: IR*): ApplyAggOp =
        ApplyAggOp(initOpArgs.toIndexedSeq, seqOpArgs.toIndexedSeq, op)
    }

    trait ApplyAggOpExt { self: ApplyAggOp =>
      def computeTypeFromChildren =
        AggOp.getReturnType(op, seqOpArgs.map(_.typ))
    }

    abstract class ApplyScanOpCompanionExt {
      def apply(op: AggOp, initOpArgs: IR*)(seqOpArgs: IR*): ApplyScanOp =
        ApplyScanOp(initOpArgs.toIndexedSeq, seqOpArgs.toIndexedSeq, op)
    }

    trait ApplyScanOpExt { self: ApplyScanOp =>
      def computeTypeFromChildren =
        AggOp.getReturnType(op, seqOpArgs.map(_.typ))

      def nSeqOpArgs = seqOpArgs.length

      def nInitArgs = initOpArgs.length
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

    trait AggFoldExt { self: AggFold =>
      def computeTypeFromChildren = {
        assert(zero.typ == seqOp.typ)
        assert(zero.typ == combOp.typ)
        zero.typ
      }
    }

    trait MakeStructExt { self: MakeStruct =>
      def computeTypeFromChildren =
        TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*)
    }

    trait SelectFieldsExt { self: SelectFields =>
      def computeTypeFromChildren = {
        val tbs = tcoerce[TStruct](old.typ)
        val oldfields = tbs.fieldNames.toSet
        assert(fields.forall(id => oldfields.contains(id)))
        tbs.select(fields.toFastSeq)._1
      }
    }

    trait InsertFieldsExt { self: InsertFields =>
      def computeTypeFromChildren = {
        val tbs = tcoerce[TStruct](old.typ)
        val s = tbs.insertFields(fields.map(f => (f._1, f._2.typ)))
        fieldOrder.map { fds =>
          assert(fds.areDistinct())
          assert(fds.size == s.size)
          TStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      }
    }

    trait GetFieldExt { self: GetField =>
      def computeTypeFromChildren = {
        val t = tcoerce[TStruct](o.typ)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        t.field(name).typ
      }
    }

    abstract class MakeTupleCompanionExt {
      def ordered(types: IndexedSeq[IR]): MakeTuple = MakeTuple(types.zipWithIndex.map {
        case (ir, i) =>
          (i, ir)
      })
    }

    trait MakeTupleExt { self: MakeTuple =>
      def computeTypeFromChildren = {
        val indices = fields.map(_._1)
        assert(indices.areDistinct())
        assert(indices.isSorted)
        TTuple(fields.map { case (i, value) => TupleField(i, value.typ) }.toFastSeq)
      }
    }

    trait GetTupleElementExt { self: GetTupleElement =>
      def computeTypeFromChildren = {
        val t = tcoerce[TTuple](o.typ)
        val fd = t.fields(t.fieldIndex(idx)).typ
        fd
      }
    }

    trait TableCountExt { self: TableCount =>
      def computeTypeFromChildren = TInt64
    }

    trait MatrixCountExt { self: MatrixCount =>
      def computeTypeFromChildren = TTuple(TInt64, TInt32)
    }

    trait TableAggregateExt { self: TableAggregate =>
      def computeTypeFromChildren = query.typ
    }

    trait MatrixAggregateExt { self: MatrixAggregate =>
      def computeTypeFromChildren = query.typ
    }

    trait TableWriteExt { self: TableWrite =>
      def computeTypeFromChildren = TVoid
    }

    trait TableMultiWriteExt { self: TableMultiWrite =>
      def computeTypeFromChildren = {
        val t = children.head.typ
        assert(children.forall(_.typ == t))
        TVoid
      }
    }

    trait MatrixWriteExt { self: MatrixWrite =>
      def computeTypeFromChildren = TVoid
    }

    trait MatrixMultiWriteExt { self: MatrixMultiWrite =>
      def computeTypeFromChildren = {
        val t = _children.head.typ
        assert(
          !t.rowType.hasField(MatrixReader.rowUIDFieldName) &&
            !t.colType.hasField(MatrixReader.colUIDFieldName),
          t,
        )
        assert(children.forall(_.typ == t))
        TVoid
      }
    }

    trait BlockMatrixCollectExt { self: BlockMatrixCollect =>
      def computeTypeFromChildren = TNDArray(TFloat64, Nat(2))
    }

    trait BlockMatrixWriteExt { self: BlockMatrixWrite =>
      def computeTypeFromChildren = writer.loweredTyp
    }

    trait BlockMatrixMultiWriteExt { self: BlockMatrixMultiWrite =>
      def computeTypeFromChildren = TVoid
    }

    trait TableGetGlobalsExt { self: TableGetGlobals =>
      def computeTypeFromChildren = child.typ.globalType
    }

    trait TableCollectExt { self: TableCollect =>
      def computeTypeFromChildren = {
        assert(child.typ.key.isEmpty)
        TStruct("rows" -> TArray(child.typ.rowType), "global" -> child.typ.globalType)
      }
    }

    trait TableToValueApplyExt { self: TableToValueApply =>
      def computeTypeFromChildren = function.typ(child.typ)
    }

    trait MatrixToValueApplyExt { self: MatrixToValueApply =>
      def computeTypeFromChildren = function.typ(child.typ)
    }

    trait BlockMatrixToValueApplyExt { self: BlockMatrixToValueApply =>
      def computeTypeFromChildren = function.typ(child.typ)
    }

    trait CollectDistributedArrayExt { self: CollectDistributedArray =>
      def computeTypeFromChildren = {
        assert(contexts.typ.isInstanceOf[TStream])
        assert(dynamicID.typ == TString)
        TArray(body.typ)
      }
    }

    trait ReadPartitionExt { self: ReadPartition =>
      def computeTypeFromChildren = {
        assert(rowType.isRealizable)
        assert(context.typ == reader.contextType)
        assert(PruneDeadFields.isSupertype(rowType, reader.fullRowType))
        TStream(rowType)
      }
    }

    trait WritePartitionExt { self: WritePartition =>
      def computeTypeFromChildren = {
        assert(value.typ.isInstanceOf[TStream])
        assert(writeCtx.typ == writer.ctxType)
        writer.returnType
      }
    }

    trait WriteMetadataExt { self: WriteMetadata =>
      def computeTypeFromChildren = {
        assert(writeAnnotations.typ == writer.annotationType)
        TVoid
      }
    }

    trait ReadValueExt { self: ReadValue =>
      def check(): Unit = {
        assert(path.typ == TString)
        reader match {
          case reader: ETypeValueReader =>
            assert(reader.spec.encodedType.decodedPType(typ).virtualType == typ)
          case _ => // do nothing, we can't in general typecheck an arbitrary value reader
        }
      }
    }

    trait WriteValueExt { self: WriteValue =>
      def computeTypeFromChildren = {
        assert(path.typ == TString)
        assert(stagingFile.forall(_.typ == TString))
        TString
      }
    }

    trait LiftMeOutExt { self: LiftMeOut =>
      def computeTypeFromChildren = child.typ
    }
  }
}
