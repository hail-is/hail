package is.hail.asm4s

import java.io._
import java.util

import scala.collection.JavaConverters._
import scala.collection.generic.Growable
import scala.collection.mutable
import scala.language.{higherKinds, implicitConversions}
import is.hail.utils._
import org.apache.spark.TaskContext
import org.objectweb.asm.{ClassReader, ClassWriter, Type}
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._
import org.objectweb.asm.util.{CheckClassAdapter, Textifier, TraceClassVisitor}

import scala.reflect.ClassTag

object FunctionBuilder {
  val stderrAndLoggerErrorOS = getStderrAndLogOutputStream[FunctionBuilder[_]]

  var count = 0

  def newUniqueID(): Int = {
    val id = count
    count += 1
    id
  }

  def bytesToBytecodeString(bytes: Array[Byte], out: OutputStream) {
    val tcv = new TraceClassVisitor(null, new Textifier, new PrintWriter(out))
    new ClassReader(bytes).accept(tcv, 0)
  }

  def functionBuilder[R: TypeInfo]: FunctionBuilder[AsmFunction0[R]] =
    new FunctionBuilder(Array[MaybeGenericTypeInfo[_]](), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction1[A, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction2[A, B, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction3[A, B, C, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction4[A, B, C, D, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction5[A, B, C, D, E, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, G: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F], GenericTypeInfo[G]), GenericTypeInfo[R])
}

class MethodBuilder(val fb: FunctionBuilder[_], val mname: String, val parameterTypeInfo: Array[TypeInfo[_]], val returnTypeInfo: TypeInfo[_]) {

  def descriptor: String = s"(${ parameterTypeInfo.map(_.name).mkString })${ returnTypeInfo.name }"

  val mn = new MethodNode(ACC_PUBLIC, mname, descriptor, null, null)
  fb.cn.methods.asInstanceOf[util.List[MethodNode]].add(mn)

  val start = new LabelNode
  val end = new LabelNode
  val layout: Array[Int] = 0 +: (parameterTypeInfo.scanLeft(1) { case (prev, gti) => prev + gti.slots })
  val argIndex: Array[Int] = layout.init
  var locals: Int = layout.last
  private val localBitSet = new LocalBitSet(this)

  def allocLocal[T](name: String = null)(implicit tti: TypeInfo[T]): Int = {
    val i = locals
    assert(i < (1 << 16))
    locals += tti.slots

    mn.localVariables.asInstanceOf[util.List[LocalVariableNode]]
      .add(new LocalVariableNode(if (name == null) "local" + i else name, tti.name, null, start, end, i))
    i
  }

  def newLocalBit(): SettableBit = localBitSet.newBit()

  def newClassBit(): SettableBit = fb.classBitSet.newBit(this)

  def newLocal[T](implicit tti: TypeInfo[T]): LocalRef[T] =
    newLocal()

  def newLocal[T](name: String = null)(implicit tti: TypeInfo[T]): LocalRef[T] =
    new LocalRef[T](allocLocal[T](name))

  def newField[T: TypeInfo]: ClassFieldRef[T] = newField[T]()

  def newField[T: TypeInfo](name: String = null): ClassFieldRef[T] = fb.newField[T](name)

  def newLazyField[T: TypeInfo](setup: Code[T]): LazyFieldRef[T] = newLazyField("")(setup)

  def newLazyField[T: TypeInfo](name: String)(setup: Code[T]): LazyFieldRef[T] = fb.newLazyField(name)(setup)

  def getArg[T](i: Int)(implicit tti: TypeInfo[T]): LocalRef[T] = {
    assert(i >= 0)
    assert(i < layout.length)
    new LocalRef[T](argIndex(i))
  }

  val l = new mutable.ArrayBuffer[AbstractInsnNode]()

  def emit(c: Code[_]) {
    c.emit(l)
  }

  def emit(insn: AbstractInsnNode) {
    l += insn
  }

  def close() {
    mn.instructions.add(start)
    val dupes = l.groupBy(x => x).map(_._2.toArray).filter(_.length > 1).toArray
    assert(dupes.isEmpty, s"some instructions were repeated in the instruction list: ${ dupes: Seq[Any] }")
    l.foreach(mn.instructions.add _)
    mn.instructions.add(new InsnNode(returnTypeInfo.returnOp))
    mn.instructions.add(end)
  }

  def invoke(args: Code[_]*) = {
    var c: Code[_] = getArg[java.lang.Object](0)
    args.foreach { a => c = Code(c, a) }
    Code(c, new MethodInsnNode(INVOKESPECIAL, fb.name, mname, descriptor, false))
  }
}

trait DependentFunction[F >: Null <: AnyRef] extends FunctionBuilder[F] {
  var definedFields: ArrayBuilder[Growable[AbstractInsnNode] => Unit] = new ArrayBuilder(16)

  def addField[T : TypeInfo](value: Code[T]): ClassFieldRef[T] = {
    val cfr = newField[T]
    val add: (Growable[AbstractInsnNode]) => Unit = { (il: Growable[AbstractInsnNode]) =>
      il += new TypeInsnNode(CHECKCAST, name)
      value.emit(il)
      il += new FieldInsnNode(PUTFIELD, name, cfr.name, typeInfo[T].name)
    }
    definedFields += add
    cfr
  }

  def newInstance()(implicit fct: ClassTag[F]): Code[F] = {
    val instance: Code[F] =
      new Code[F] {
        def emit(il: Growable[AbstractInsnNode]): Unit = {
          il += new TypeInsnNode(NEW, name)
          il += new InsnNode(DUP)
          il += new MethodInsnNode(INVOKESPECIAL, name, "<init>", "()V", false)
          il += new TypeInsnNode(CHECKCAST, classInfo[F].iname)
          definedFields.result().foreach { add =>
            il += new InsnNode(DUP)
            add(il)
          }
        }
      }
    instance
  }

  override def result(pw: Option[PrintWriter]): () => F =
    throw new UnsupportedOperationException("cannot call result() on a dependent function")

}

class DependentFunctionBuilder[F >: Null <: AnyRef : TypeInfo : ClassTag](
  parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  returnTypeInfo: MaybeGenericTypeInfo[_],
  packageName: String = "is/hail/codegen/generated"
) extends FunctionBuilder[F](parameterTypeInfo, returnTypeInfo, packageName) with DependentFunction[F]

class FunctionBuilder[F >: Null](val parameterTypeInfo: Array[MaybeGenericTypeInfo[_]], val returnTypeInfo: MaybeGenericTypeInfo[_],
  val packageName: String = "is/hail/codegen/generated")(implicit val interfaceTi: TypeInfo[F]) {

  import FunctionBuilder._

  val cn = new ClassNode()
  cn.version = V1_8
  cn.access = ACC_PUBLIC

  val name = packageName + "/C" + newUniqueID()
  cn.name = name
  cn.superName = "java/lang/Object"
  cn.interfaces.asInstanceOf[java.util.List[String]].add("java/io/Serializable")

  val methods: mutable.ArrayBuffer[MethodBuilder] = new mutable.ArrayBuffer[MethodBuilder](16)
  val fields: mutable.ArrayBuffer[FieldNode] = new mutable.ArrayBuffer[FieldNode](16)

  val init = new MethodNode(ACC_PUBLIC, "<init>", "()V", null, null)
  // FIXME why is cast necessary?
  cn.methods.asInstanceOf[util.List[MethodNode]].add(init)

  init.instructions.add(new IntInsnNode(ALOAD, 0))
  init.instructions.add(new MethodInsnNode(INVOKESPECIAL, Type.getInternalName(classOf[java.lang.Object]), "<init>", "()V", false))
  init.instructions.add(new InsnNode(RETURN))

  protected[this] val children: mutable.ArrayBuffer[DependentFunction[_]] = new mutable.ArrayBuffer[DependentFunction[_]](16)

  private[this] lazy val _apply_method: MethodBuilder = {
    val m = new MethodBuilder(this, "apply", parameterTypeInfo.map(_.base), returnTypeInfo.base)
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

  def apply_method: MethodBuilder = _apply_method

  val classBitSet = new ClassBitSet(this)

  def newLocalBit(): SettableBit = apply_method.newLocalBit()

  def newDependentFunction[A1 : TypeInfo, R : TypeInfo]: DependentFunction[AsmFunction1[A1, R]] = {
    val df = new DependentFunctionBuilder[AsmFunction1[A1, R]](Array(GenericTypeInfo[A1]), GenericTypeInfo[R])
    children += df
    df
  }

  def newClassBit(): SettableBit = classBitSet.newBit(apply_method)

  def newField[T: TypeInfo]: ClassFieldRef[T] = newField()

  def newField[T: TypeInfo](name: String = null): ClassFieldRef[T] =
    new ClassFieldRef[T](this, s"field${ cn.fields.size() }${ if (name == null) "" else s"_$name" }")

  def newLazyField[T: TypeInfo](setup: Code[T]): LazyFieldRef[T] = newLazyField("")(setup)

  def newLazyField[T: TypeInfo](name: String)(setup: Code[T]): LazyFieldRef[T] =
    new LazyFieldRef[T](this, s"field${ cn.fields.size() }_$name", setup)

  def allocLocal[T](name: String = null)(implicit tti: TypeInfo[T]): Int = apply_method.allocLocal[T](name)

  def newLocal[T](implicit tti: TypeInfo[T]): LocalRef[T] = newLocal()

  def newLocal[T](name: String = null)(implicit tti: TypeInfo[T]): LocalRef[T] = apply_method.newLocal[T](name)

  def getArg[T](i: Int)(implicit tti: TypeInfo[T]): LocalRef[T] = apply_method.getArg[T](i)

  def emit(c: Code[_]) = apply_method.emit(c)

  def emit(insn: AbstractInsnNode) = apply_method.emit(insn)

  def newMethod(argsInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder = {
    val mb = new MethodBuilder(this, s"method${ methods.size }", argsInfo, returnInfo)
    methods.append(mb)
    mb
  }

  def newMethod[R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](), typeInfo[R])

  def newMethod[A: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D], typeInfo[E]), typeInfo[R])

  def classAsBytes(print: Option[PrintWriter] = None): Array[Byte] = {
    apply_method.close()
    methods.toArray.foreach { m => m.close() }

    val cw = new ClassWriter(ClassWriter.COMPUTE_MAXS + ClassWriter.COMPUTE_FRAMES)
    val sw1 = new StringWriter()
    var bytes: Array[Byte] = new Array[Byte](0)
    try {
      for (method <- cn.methods.asInstanceOf[util.List[MethodNode]].asScala) {
        val count = method.instructions.size
        log.info(s"${ cn.name }.${ method.name } instruction count: $count")
        if (count > 8000)
          log.info(s"${ cn.name }.${ method.name } instruction count > 8000")
      }

      cn.accept(cw)
      bytes = cw.toByteArray
//      CheckClassAdapter.verify(new ClassReader(bytes), false, new PrintWriter(sw1))
    } catch {
      case e: Exception =>
        // if we fail with frames, try without frames for better error message
        val cwNoFrames = new ClassWriter(ClassWriter.COMPUTE_MAXS)
        val sw2 = new StringWriter()
        cn.accept(cwNoFrames)
        CheckClassAdapter.verify(new ClassReader(cwNoFrames.toByteArray), false, new PrintWriter(sw2))

        if (sw2.toString().length() != 0) {
          System.err.println("Verify Output 2 for " + name + ":")
          System.err.println(sw2)
          throw new IllegalStateException("Bytecode failed verification 1", e)
        } else {
          if (sw1.toString().length() != 0) {
            System.err.println("Verifiy Output 1 for " + name + ":")
            System.err.println(sw1)
          }
          throw e
        }
    }

    if (sw1.toString.length != 0) {
      System.err.println("Verify Output 1 for " + name + ":")
      System.err.println(sw1)
      throw new IllegalStateException("Bytecode failed verification 2")
    }

    print.foreach { pw =>
      val cr = new ClassReader(bytes)
      val tcv = new TraceClassVisitor(null, new Textifier, pw)
      cr.accept(tcv, 0)
    }
    bytes
  }

  cn.interfaces.asInstanceOf[java.util.List[String]].add(interfaceTi.iname)

  def result(print: Option[PrintWriter] = None): () => F = {
    val childClasses = children.result().map(f => (f.name.replace("/","."), f.classAsBytes(print)))

    val bytes = classAsBytes(print)
    val n = name.replace("/",".")

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

class Function2Builder[A1 : TypeInfo, A2 : TypeInfo, R : TypeInfo]
  extends FunctionBuilder[AsmFunction2[A1, A2, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R]) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)
}
