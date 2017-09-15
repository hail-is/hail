package is.hail.asm4s

import java.util
import java.io._

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._
import org.objectweb.asm.{ClassWriter, Type}
import java.util

import org.objectweb.asm.util.{CheckClassAdapter, Textifier, TraceClassVisitor}
import org.objectweb.asm.{ClassReader, ClassWriter, Type}

import scala.collection.mutable
import scala.collection.generic.Growable
import scala.language.implicitConversions
import scala.language.higherKinds
import scala.reflect.ClassTag
import scala.reflect.classTag

import is.hail.utils._

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

  private implicit def methodNodeToGrowable(mn: MethodNode): Growable[AbstractInsnNode] = new Growable[AbstractInsnNode] {
    def +=(e: AbstractInsnNode) = { mn.instructions.add(e); this }
    def clear() { throw new UnsupportedOperationException() }
  }
}

class FunctionBuilder[F >: Null](parameterTypeInfo: Array[MaybeGenericTypeInfo[_]], returnTypeInfo: MaybeGenericTypeInfo[_],
  packageName: String = "is/hail/codegen/generated")(implicit interfaceTi: TypeInfo[F]) {

  import FunctionBuilder._

  val cn = new ClassNode()
  cn.version = V1_8
  cn.access = ACC_PUBLIC

  val name = packageName + "/C" + newUniqueID()
  cn.name = name
  cn.superName = "java/lang/Object"
  cn.interfaces.asInstanceOf[java.util.List[String]].add("java/io/Serializable")

  def descriptor: String = s"(${ parameterTypeInfo.map(_.name).mkString })${ returnTypeInfo.name }"

  val mn = new MethodNode(ACC_PUBLIC, "apply", descriptor, null, null)
  val init = new MethodNode(ACC_PUBLIC, "<init>", "()V", null, null)
  // FIXME why is cast necessary?
  cn.methods.asInstanceOf[util.List[MethodNode]].add(mn)
  cn.methods.asInstanceOf[util.List[MethodNode]].add(init)

  init.instructions.add(new IntInsnNode(ALOAD, 0))
  init.instructions.add(new MethodInsnNode(INVOKESPECIAL, Type.getInternalName(classOf[java.lang.Object]), "<init>", "()V", false))
  init.instructions.add(new InsnNode(RETURN))

  val start = new LabelNode
  val end = new LabelNode

  val layout: Array[Int] =
    0 +: (parameterTypeInfo.scanLeft(1) { case (prev, ti) => prev + ti.slots })
  val argIndex: Array[Int] = layout.init
  var locals: Int = layout.last

  if (parameterTypeInfo.exists(_.isGeneric) || returnTypeInfo.isGeneric) {
    def genericDescriptor: String = s"(${ parameterTypeInfo.map(_.generic.name).mkString })${ returnTypeInfo.generic.name }"

    val genericMn = new MethodNode(ACC_PUBLIC, "apply", genericDescriptor, null, null)
    cn.methods.asInstanceOf[util.List[MethodNode]].add(genericMn)
    val genericLayout: Array[Int] =
      0 +: (parameterTypeInfo.scanLeft(1) { case (prev, ti) => prev + ti.generic.slots })
    val genericArgIndex: Array[Int] = genericLayout.init

    def getArg[T](i: Int)(implicit tti: TypeInfo[T]): LocalRef[T] =
      new LocalRef[T](genericArgIndex(i))

    val callSpecialized = Code(
      getArg[java.lang.Object](0),
      toCodeFromIndexedSeq(parameterTypeInfo.zipWithIndex.map { case (ti, i) => ti.castFromGeneric(getArg(i+1)(ti.generic)) }),
      Code(new MethodInsnNode(INVOKESPECIAL, name, "apply", descriptor, false)))

    Code(
      returnTypeInfo.castToGeneric(callSpecialized),
      new InsnNode(returnTypeInfo.generic.returnOp)
    ).emit(genericMn)
  }

  def allocLocal[T]()(implicit tti: TypeInfo[T]): Int = {
    val i = locals
    locals += tti.slots

    mn.localVariables.asInstanceOf[util.List[LocalVariableNode]]
      .add(new LocalVariableNode("local" + i, tti.name, null, start, end, i))
    i
  }

  def newLocal[T]()(implicit tti: TypeInfo[T]): LocalRef[T] =
    new LocalRef[T](allocLocal[T]())

  def getStatic[T, S](field: String)(implicit tct: ClassTag[T], sct: ClassTag[S], sti: TypeInfo[S]): Code[S] = {
    val f = FieldRef[T, S](field)
    assert(f.isStatic)
    f.get(null)
  }

  def putStatic[T, S](field: String, rhs: Code[S])(implicit tct: ClassTag[T], sct: ClassTag[S], sti: TypeInfo[S]): Code[Unit] = {
    val f = FieldRef[T, S](field)
    assert(f.isStatic)
    f.put(null, rhs)
  }

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

  def classAsBytes(print: Option[PrintWriter] = None): Array[Byte] = {
    mn.instructions.add(start)
    val dupes = l.groupBy(x => x).map(_._2.toArray).filter(_.length > 1).toArray
    assert(dupes.isEmpty, s"some instructions were repeated in the instruction list: ${dupes: Seq[Any]}")
    l.foreach(mn.instructions.add _)
    mn.instructions.add(new InsnNode(returnTypeInfo.returnOp))
    mn.instructions.add(end)

    val cw = new ClassWriter(ClassWriter.COMPUTE_MAXS + ClassWriter.COMPUTE_FRAMES)
    var bytes: Array[Byte] = new Array[Byte](0)
    try {
      cn.accept(cw)
      bytes = cw.toByteArray

      val sw = new StringWriter()
      CheckClassAdapter.verify(new ClassReader(bytes), false, new PrintWriter(sw))
      if (sw.toString.length != 0) {
        println("Verify Output for " + name + ":")
        println(sw)
        throw new IllegalStateException("Bytecode failed verification 2")
      }
    } catch {
      case e: Exception =>
        // if we fail with frames, try without frames for better error message
        val cwNoFrames = new ClassWriter(ClassWriter.COMPUTE_MAXS)
        cn.accept(cwNoFrames)

        // print the bytecode for debugging purposes
        val cr = new ClassReader(cwNoFrames.toByteArray)
        val tcv = new TraceClassVisitor(null, new Textifier, new PrintWriter(System.out))
        cr.accept(tcv, 0)

        val sw = new StringWriter()
        CheckClassAdapter.verify(cr, false, new PrintWriter(sw))
        if (sw.toString().length() != 0) {
          println("Verify Output for " + name + ":")
          println(sw)
          throw new IllegalStateException("Bytecode failed verification 1", e)
        } else {
          throw e
        }
    }

    print.foreach { pw =>
      val cr = new ClassReader(bytes)
      val tcv = new TraceClassVisitor(null, new Textifier, pw)
      cr.accept(tcv, 0)
    }

    bytes
  }

  cn.interfaces.asInstanceOf[java.util.List[String]].add(interfaceTi.iname)

  def result(): () => F = {
    val bytes = classAsBytes()
    val localName = name.replaceAll("/",".")

    new (() => F) with java.io.Serializable {
      @transient @volatile private var f: F = null
      def apply(): F = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[F]
              }
            }
          }

          f
        } catch {
          //  only triggers on classloader
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

class Function2Builder[A1 >: Null : TypeInfo, A2 >: Null : TypeInfo, R >: Null : TypeInfo]
    extends FunctionBuilder[AsmFunction2[A1, A2, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R]) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)
}
