package is.hail.expr.ir

import java.io.PrintWriter

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.Parser
import is.hail.expr.types.Type
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.TaskContext
import org.objectweb.asm.tree.AbstractInsnNode

import scala.collection.generic.Growable
import scala.collection.mutable
import scala.reflect.ClassTag

object EmitFunctionBuilder {
  def apply[R: TypeInfo](): EmitFunctionBuilder[AsmFunction0[R]] =
    new EmitFunctionBuilder[AsmFunction0[R]](Array[MaybeGenericTypeInfo[_]](), GenericTypeInfo[R])

  def apply[A: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction1[A, R]] =
    new EmitFunctionBuilder[AsmFunction1[A, R]](Array(GenericTypeInfo[A]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction2[A, B, R]] =
    new EmitFunctionBuilder[AsmFunction2[A, B, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction3[A, B, C, R]] =
    new EmitFunctionBuilder[AsmFunction3[A, B, C, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction4[A, B, C, D, R]] =
    new EmitFunctionBuilder[AsmFunction4[A, B, C, D, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction5[A, B, C, D, E, R]] =
    new EmitFunctionBuilder[AsmFunction5[A, B, C, D, E, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]] =
    new EmitFunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, G: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]] =
    new EmitFunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F], GenericTypeInfo[G]), GenericTypeInfo[R])
}

trait FunctionWithHadoopConfiguration {
  def addHadoopConfiguration(hConf: SerializableHadoopConfiguration): Unit
}

class EmitMethodBuilder(
  override val fb: EmitFunctionBuilder[_],
  mname: String,
  parameterTypeInfo: Array[TypeInfo[_]],
  returnTypeInfo: TypeInfo[_]
) extends MethodBuilder(fb, mname, parameterTypeInfo, returnTypeInfo) {

  def numReferenceGenomes: Int = fb.numReferenceGenomes

  def getReferenceGenome(rg: ReferenceGenome): Code[ReferenceGenome] =
    fb.getReferenceGenome(rg)

  def numTypes: Int = fb.numTypes

  def getType(t: Type): Code[Type] = fb.getType(t)

  def getCodeOrdering[T](t: Type, op: CodeOrdering.Op, missingGreatest: Boolean): CodeOrdering.F[T] =
    getCodeOrdering[T](t, op, missingGreatest, ignoreMissingness = false)

  def getCodeOrdering[T](t: Type, op: CodeOrdering.Op, missingGreatest: Boolean, ignoreMissingness: Boolean): CodeOrdering.F[T] =
    fb.getCodeOrdering[T](t, op, missingGreatest, ignoreMissingness)

  def getCodeOrdering[T](t1: Type, t2: Type, op: CodeOrdering.Op, missingGreatest: Boolean): CodeOrdering.F[T] =
    fb.getCodeOrdering[T](t1, t2, op, missingGreatest, ignoreMissingness = false)

  def getCodeOrdering[T](t1: Type, t2: Type, op: CodeOrdering.Op, missingGreatest: Boolean, ignoreMissingness: Boolean): CodeOrdering.F[T] =
    fb.getCodeOrdering[T](t1, t2, op, missingGreatest, ignoreMissingness)
}

class DependentEmitFunction[F >: Null <: AnyRef : TypeInfo : ClassTag](
  parentfb: EmitFunctionBuilder[_],
  parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  returnTypeInfo: MaybeGenericTypeInfo[_],
  packageName: String = "is/hail/codegen/generated"
) extends EmitFunctionBuilder[F](parameterTypeInfo, returnTypeInfo, packageName) with DependentFunction[F] {

  private[this] val rgMap: mutable.Map[ReferenceGenome, Code[ReferenceGenome]] =
    mutable.Map[ReferenceGenome, Code[ReferenceGenome]]()

  private[this] val typMap: mutable.Map[Type, Code[Type]] =
    mutable.Map[Type, Code[Type]]()

  override def getReferenceGenome(rg: ReferenceGenome): Code[ReferenceGenome] =
    rgMap.getOrElse(rg, {
      val fromParent = parentfb.getReferenceGenome(rg)
      val field = addField[ReferenceGenome](fromParent)
      field.load()
    })

  override def getType(t: Type): Code[Type] =
    typMap.getOrElse(t, {
      val fromParent = parentfb.getType(t)
      val field = addField[Type](fromParent)
      field.load()
    })

}

class EmitFunctionBuilder[F >: Null](
  parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  returnTypeInfo: MaybeGenericTypeInfo[_],
  packageName: String = "is/hail/codegen/generated"
)(implicit interfaceTi: TypeInfo[F]) extends FunctionBuilder[F](parameterTypeInfo, returnTypeInfo, packageName) {

  private[this] val rgMap: mutable.Map[ReferenceGenome, Code[ReferenceGenome]] =
    mutable.Map[ReferenceGenome, Code[ReferenceGenome]]()

  private[this] val typMap: mutable.Map[Type, Code[Type]] =
    mutable.Map[Type, Code[Type]]()

  private[this] val compareMap: mutable.Map[(Type, Type, CodeOrdering.Op, Boolean, Boolean), CodeOrdering.F[_]] =
    mutable.Map[(Type, Type, CodeOrdering.Op, Boolean, Boolean), CodeOrdering.F[_]]()

  def numReferenceGenomes: Int = rgMap.size

  def getReferenceGenome(rg: ReferenceGenome): Code[ReferenceGenome] =
    rgMap.getOrElseUpdate(rg, newLazyField[ReferenceGenome](rg.codeSetup(this)))

  def numTypes: Int = typMap.size

  private[this] def addReferenceGenome(rg: ReferenceGenome): Code[Unit] = {
    val rgExists = Code.invokeScalaObject[String, Boolean](ReferenceGenome.getClass, "hasReference", const(rg.name))
    val addRG = Code.invokeScalaObject[ReferenceGenome, Unit](ReferenceGenome.getClass, "addReference", getReferenceGenome(rg))
    rgExists.mux(Code._empty, addRG)
  }

  private[this] var _hconf: SerializableHadoopConfiguration = _
  private[this] var _hfield: ClassFieldRef[SerializableHadoopConfiguration] = _

  def addHadoopConfiguration(hConf: SerializableHadoopConfiguration): Unit = {
    assert(hConf != null)
    if (_hconf == null) {
      cn.interfaces.asInstanceOf[java.util.List[String]].add(typeInfo[FunctionWithHadoopConfiguration].iname)
      val confField = newField[SerializableHadoopConfiguration]
      val mb = new EmitMethodBuilder(this, "addHadoopConfiguration", Array(typeInfo[SerializableHadoopConfiguration]), typeInfo[Unit])
      methods.append(mb)
      mb.emit(confField := mb.getArg[SerializableHadoopConfiguration](1))
      _hconf = hConf
      _hfield = confField
    }
    assert(_hconf.value == hConf.value && _hfield != null)
  }

  def getHadoopConfiguration: Code[SerializableHadoopConfiguration] = {
    assert(_hconf != null && _hfield != null, s"${_hfield == null}")
    _hfield.load()
  }

  def getType(t: Type): Code[Type] = {
    val references = ReferenceGenome.getReferences(t).toArray
    val setup = Code(Code(references.map(addReferenceGenome): _*),
      Code.invokeScalaObject[String, Type](
        Parser.getClass, "parseType", const(t.parsableString())))
    typMap.getOrElseUpdate(t,
      newLazyField[Type](setup))
  }

  def getCodeOrdering[T](t1: Type, t2: Type, op: CodeOrdering.Op, missingGreatest: Boolean, ignoreMissingness: Boolean): CodeOrdering.F[T] = {
    val f = compareMap.getOrElseUpdate((t1, t2, op, missingGreatest, ignoreMissingness), {
      val ti = typeToTypeInfo(t1)
      val rt = if (op == CodeOrdering.compare) typeInfo[Int] else typeInfo[Boolean]

      val newMB = if (ignoreMissingness) {
        val newMB = newMethod(Array[TypeInfo[_]](typeInfo[Region], ti, typeInfo[Region], ti), rt)
        val ord = t1.codeOrdering(newMB, t2)
        val r1 = newMB.getArg[Region](1)
        val r2 = newMB.getArg[Region](3)
        val v1 = newMB.getArg(2)(ti)
        val v2 = newMB.getArg(4)(ti)
        val c: Code[_] = op match {
          case CodeOrdering.compare => ord.compareNonnull(r1, coerce[ord.T](v1), r2, coerce[ord.T](v2), missingGreatest)
          case CodeOrdering.equiv => ord.equivNonnull(r1, coerce[ord.T](v1), r2, coerce[ord.T](v2), missingGreatest)
          case CodeOrdering.lt => ord.ltNonnull(r1, coerce[ord.T](v1), r2, coerce[ord.T](v2), missingGreatest)
          case CodeOrdering.lteq => ord.lteqNonnull(r1, coerce[ord.T](v1), r2, coerce[ord.T](v2), missingGreatest)
          case CodeOrdering.gt => ord.gtNonnull(r1, coerce[ord.T](v1), r2, coerce[ord.T](v2), missingGreatest)
          case CodeOrdering.gteq => ord.gteqNonnull(r1, coerce[ord.T](v1), r2, coerce[ord.T](v2), missingGreatest)
          case CodeOrdering.neq => !ord.equivNonnull(r1, coerce[ord.T](v1), r2, coerce[ord.T](v2), missingGreatest)
        }
        newMB.emit(c)
        newMB
      } else {
        val newMB = newMethod(Array[TypeInfo[_]](typeInfo[Region], typeInfo[Boolean], ti, typeInfo[Region], typeInfo[Boolean], ti), rt)
        val ord = t1.codeOrdering(newMB, t2)
        val r1 = newMB.getArg[Region](1)
        val r2 = newMB.getArg[Region](4)
        val m1 = newMB.getArg[Boolean](2)
        val v1 = newMB.getArg(3)(ti)
        val m2 = newMB.getArg[Boolean](5)
        val v2 = newMB.getArg(6)(ti)
        val c: Code[_] = op match {
          case CodeOrdering.compare => ord.compare(r1, (m1, coerce[ord.T](v1)), r2, (m2, coerce[ord.T](v2)), missingGreatest)
          case CodeOrdering.equiv => ord.equiv(r1, (m1, coerce[ord.T](v1)), r2, (m2, coerce[ord.T](v2)), missingGreatest)
          case CodeOrdering.lt => ord.lt(r1, (m1, coerce[ord.T](v1)), r2, (m2, coerce[ord.T](v2)), missingGreatest)
          case CodeOrdering.lteq => ord.lteq(r1, (m1, coerce[ord.T](v1)), r2, (m2, coerce[ord.T](v2)), missingGreatest)
          case CodeOrdering.gt => ord.gt(r1, (m1, coerce[ord.T](v1)), r2, (m2, coerce[ord.T](v2)), missingGreatest)
          case CodeOrdering.gteq => ord.gteq(r1, (m1, coerce[ord.T](v1)), r2, (m2, coerce[ord.T](v2)), missingGreatest)
          case CodeOrdering.neq => !ord.equiv(r1, (m1, coerce[ord.T](v1)), r2, (m2, coerce[ord.T](v2)), missingGreatest)
        }
        newMB.emit(c)
        newMB
      }
      val f = { (rx: Code[Region], x: (Code[Boolean], Code[_]), ry: Code[Region], y: (Code[Boolean], Code[_])) =>
        if (ignoreMissingness)
          newMB.invoke(rx, x._2, ry, y._2)
        else
          newMB.invoke(rx, x._1, x._2, ry, y._1, y._2)
      }
      f
    })
    (r1: Code[Region], v1: (Code[Boolean], Code[_]), r2: Code[Region], v2: (Code[Boolean], Code[_])) => coerce[T](f(r1, v1, r2, v2))
  }

  def getCodeOrdering[T](t: Type, op: CodeOrdering.Op, missingGreatest: Boolean, ignoreMissingness: Boolean): CodeOrdering.F[T] =
    getCodeOrdering[T](t, t, op, missingGreatest, ignoreMissingness)

  override val apply_method: EmitMethodBuilder = {
    val m = new EmitMethodBuilder(this, "apply", parameterTypeInfo.map(_.base), returnTypeInfo.base)
    if (parameterTypeInfo.exists(_.isGeneric) || returnTypeInfo.isGeneric) {
      val generic = new MethodBuilder(this, "apply", parameterTypeInfo.map(_.generic), returnTypeInfo.generic)
      methods.append(generic)
      generic.emit(
        new Code[Unit] {
          def emit(il: Growable[AbstractInsnNode]) {
            returnTypeInfo.castToGeneric(
              m.invoke(parameterTypeInfo.zipWithIndex.map { case (ti, i) =>
                ti.castFromGeneric(generic.getArg(i + 1)(ti.generic))
              }: _*)).emit(il)
          }
        }
      )
    }
    m
  }

  override def newMethod(argsInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_]): EmitMethodBuilder = {
    val mb = new EmitMethodBuilder(this, s"method${ methods.size }", argsInfo, returnInfo)
    methods.append(mb)
    mb
  }

  override def newMethod[R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](), typeInfo[R])

  override def newMethod[A: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A]), typeInfo[R])

  override def newMethod[A: TypeInfo, B: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B]), typeInfo[R])

  override def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C]), typeInfo[R])

  override def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D]), typeInfo[R])

  override def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D], typeInfo[E]), typeInfo[R])

  def newDependentFunction[A1: TypeInfo, A2: TypeInfo, R: TypeInfo]: DependentEmitFunction[AsmFunction2[A1, A2, R]] = {
    val df = new DependentEmitFunction[AsmFunction2[A1, A2, R]](
      this, Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R])
    children += df
    df
  }

  def newDependentFunction[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo]: DependentEmitFunction[AsmFunction3[A1, A2, A3, R]] = {
    val df = new DependentEmitFunction[AsmFunction3[A1, A2, A3, R]](
      this, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R])
    children += df
    df
  }

  override def result(print: Option[PrintWriter] = None): () => F = {
    val childClasses = children.result().map(f => (f.name.replace("/","."), f.classAsBytes(print)))

    val bytes = classAsBytes(print)
    val n = name.replace("/",".")
    val localHConf = _hconf

    assert(TaskContext.get() == null,
      "FunctionBuilder emission should happen on master, but happened on worker")

    new (() => F) with java.io.Serializable {
      @transient
      @volatile private var f: F = null

      def apply(): F = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                childClasses.foreach { case (fn, b) => loadClass(fn, b) }
                f = loadClass(n, bytes).newInstance().asInstanceOf[F]
                if (localHConf != null)
                  f.asInstanceOf[FunctionWithHadoopConfiguration].addHadoopConfiguration(localHConf)
              }
            }
          }

          f
        } catch {
          //  only triggers on classloader
          case e@(_: Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}
