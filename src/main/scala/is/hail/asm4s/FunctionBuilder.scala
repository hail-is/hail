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
import scala.language.implicitConversions
import scala.reflect.ClassTag

import is.hail.utils._

object FunctionBuilder {
  val stderrAndLoggerErrorOS = getStderrAndLogOutputStream[FunctionBuilder[AnyRef]]

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

}

abstract class FunctionBuilder[R](parameterTypeInfo: Array[TypeInfo[_]], returnTypeInfo: TypeInfo[R],
  packageName: String = "is/hail/codegen/generated") {

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

  def classAsBytes(c: Code[R]): Array[Byte] = {
    mn.instructions.add(start)
    emit(c)
    val dupes = l.groupBy(x => x).map(_._2.toArray).filter(_.length > 1).toArray
    assert(dupes.isEmpty, s"some instructions were repeated in the instruction list: ${dupes: Seq[Any]}")
    l.foreach(mn.instructions.add _)
    mn.instructions.add(new InsnNode(returnTypeInfo.returnOp))
    mn.instructions.add(end)

    // The following block of code can help when the output of Verification 2 is
    // inscrutable; however, it is prone to false rejections (i.e. good code is
    // rejected) so we leave it disabled.

    if (false) {
      // compute without frames first in case frame tester fails miserably
      val cwNoMaxesNoFrames = new ClassWriter(ClassWriter.COMPUTE_MAXS)
      cn.accept(cwNoMaxesNoFrames)
      val cr = new ClassReader(cwNoMaxesNoFrames.toByteArray)
      val tcv = new TraceClassVisitor(null, new Textifier, new PrintWriter(System.out))
      cr.accept(tcv, 0)

      val sw = new StringWriter()
      CheckClassAdapter.verify(cr, false, new PrintWriter(sw))
      if (sw.toString().length() != 0) {
        println("Verify Output for " + name + ":")
        try {
          val out = new BufferedWriter(new FileWriter("ByteCodeOutput.txt"))
          out.write(sw.toString())
          out.close()
        } catch {
          case e: IOException => System.out.println("Exception " + e)
        }
        println(sw)
        println("Bytecode failed verification 1")
      }
    }
    {
      val cw = new ClassWriter(ClassWriter.COMPUTE_MAXS + ClassWriter.COMPUTE_FRAMES)
      cn.accept(cw)

      val sw = new StringWriter()
      CheckClassAdapter.verify(new ClassReader(cw.toByteArray), false, new PrintWriter(sw))
      if (sw.toString.length != 0) {
        println("Verify Output for " + name + ":")
        try {
          val out = new BufferedWriter(new FileWriter("ByteCodeOutput.txt"))
          out.write(sw.toString)
          out.close
        } catch {
          case e: IOException => System.out.println("Exception " + e)
        }
        throw new IllegalStateException("Bytecode failed verification 2")
      }
    }

    val cw = new ClassWriter(ClassWriter.COMPUTE_MAXS + ClassWriter.COMPUTE_FRAMES)
    cn.accept(cw)
    cw.toByteArray
  }
}

class Function0Builder[R >: Null](implicit rti: TypeInfo[R]) extends FunctionBuilder[R](Array[TypeInfo[_]](), rti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("scala/Function0")

  def result(c: Code[R]): () => R = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function0[R] with java.io.Serializable {
      @transient @volatile private var f: () => Any = null
      def apply(): R = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[() => Any]
              }
            }
          }

          f().asInstanceOf[R]
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

trait ToI { def apply(): Int }
class FunctionToIBuilder(implicit rti: TypeInfo[Int]) extends FunctionBuilder[Int](Array[TypeInfo[_]](), rti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("is/hail/asm4s/ToI")

  def result(c: Code[Int]): () => Int = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function0[Int] with java.io.Serializable {
      @transient @volatile private var f: ToI = null
      def apply(): Int = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[ToI]
              }
            }
          }

          f()
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

class Function1Builder[A >: Null, R >: Null](implicit act: ClassTag[A], ati: TypeInfo[A],
  rti: TypeInfo[R]) extends FunctionBuilder[R](Array[TypeInfo[_]](ati), rti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("scala/Function1")

  def arg1 = getArg[A](1)

  def result(c: Code[R]): (A) => R = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function1[A, R] with java.io.Serializable {
      @transient @volatile private var f: (Any) => Any = null
      def apply(a: A): R = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[(Any) => Any]
              }
            }
          }

          f(a).asInstanceOf[R]
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

trait ZToZ { def apply(b: Boolean): Boolean }
class FunctionZToZBuilder(implicit zct: ClassTag[Boolean], zti: TypeInfo[Boolean])
    extends FunctionBuilder[Boolean](Array[TypeInfo[_]](zti), zti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("is/hail/asm4s/ZToZ")

  def arg1 = getArg[Boolean](1)

  def result(c: Code[Boolean]): (Boolean) => Boolean = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function1[Boolean, Boolean] with java.io.Serializable {
      @transient @volatile private var f: ZToZ = null
      def apply(a: Boolean): Boolean = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[ZToZ]
              }
            }
          }

          f(a)
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

trait ZToI { def apply(b: Boolean): Int }
class FunctionZToIBuilder(implicit act: ClassTag[Boolean], ati: TypeInfo[Boolean],
  rti: TypeInfo[Int]) extends FunctionBuilder[Int](Array[TypeInfo[_]](ati), rti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("is/hail/asm4s/ZToI")

  def arg1 = getArg[Boolean](1)

  def result(c: Code[Int]): (Boolean) => Int = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function1[Boolean, Int] with java.io.Serializable {
      @transient @volatile private var f: ZToI = null
      def apply(a: Boolean): Int = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[ZToI]
              }
            }
          }

          f(a)
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

trait IToI { def apply(b: Int): Int }
class FunctionIToIBuilder(implicit act: ClassTag[Int], ati: TypeInfo[Int],
  rti: TypeInfo[Int]) extends FunctionBuilder[Int](Array[TypeInfo[_]](ati), rti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("is/hail/asm4s/IToI")

  def arg1 = getArg[Int](1)

  def result(c: Code[Int]): (Int) => Int = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function1[Int, Int] with java.io.Serializable {
      @transient @volatile private var f: IToI = null
      def apply(a: Int): Int = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[IToI]
              }
            }
          }

          f(a)
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}


trait AToI[A] { def apply(b: A): Int }
class FunctionAToIBuilder[A](implicit act: ClassTag[A], ati: TypeInfo[A],
  rti: TypeInfo[Int]) extends FunctionBuilder[Int](Array[TypeInfo[_]](implicitly[TypeInfo[AnyRef]]), rti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("is/hail/asm4s/AToI")

  def arg1 = Code.checkcast[A](getArg[AnyRef](1))

  def result(c: Code[Int]): (A) => Int = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function1[A, Int] with java.io.Serializable {
      @transient @volatile private var f: AToI[A] = null
      def apply(a: A): Int = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[AToI[A]]
              }
            }
          }

          f(a)
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

class Function2Builder[A1 >: Null, A2 >: Null, R >: Null]
  (implicit a1ct: ClassTag[A1], a1ti: TypeInfo[A1], a2ct: ClassTag[A2], a2ti: TypeInfo[A2], rti: TypeInfo[R])
    extends FunctionBuilder[R](Array[TypeInfo[_]](implicitly[TypeInfo[AnyRef]], implicitly[TypeInfo[AnyRef]]), rti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("scala/Function2")

  def arg1 = Code.checkcast[A1](getArg[AnyRef](1))

  def arg2 = Code.checkcast[A2](getArg[AnyRef](2))

  def result(c: Code[R]): (A1, A2) => R = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function2[A1, A2, R] with java.io.Serializable {
      @transient @volatile private var f: (Any, Any) => Any = null
      def apply(a1: A1, a2: A2): R = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[(Any, Any) => Any]
              }
            }
          }

          f(a1, a2).asInstanceOf[R]
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

trait IAndIToI { def apply(a1: Int, a2: Int): Int }
class FunctionIAndIToIBuilder(implicit ict: ClassTag[Int], iti: TypeInfo[Int])
    extends FunctionBuilder[Int](Array[TypeInfo[_]](iti, iti), iti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("is/hail/asm4s/IAndIToI")

  def arg1 = getArg[Int](1)

  def arg2 = getArg[Int](2)

  def result(c: Code[Int]): (Int, Int) => Int = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function2[Int, Int, Int] with java.io.Serializable {
      @transient @volatile private var f: IAndIToI = null
      def apply(a1: Int, a2: Int): Int = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[IAndIToI]
              }
            }
          }

          f(a1, a2)
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

trait IAndIToZ { def apply(a1: Int, a2: Int): Boolean }
class FunctionIAndIToZBuilder(implicit ict: ClassTag[Int], iti: TypeInfo[Int], rti: TypeInfo[Boolean])
    extends FunctionBuilder[Boolean](Array[TypeInfo[_]](iti, iti), rti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("is/hail/asm4s/IAndIToZ")

  def arg1 = getArg[Int](1)

  def arg2 = getArg[Int](2)

  def result(c: Code[Boolean]): (Int, Int) => Boolean = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function2[Int, Int, Boolean] with java.io.Serializable {
      @transient @volatile private var f: IAndIToZ = null
      def apply(a1: Int, a2: Int): Boolean = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[IAndIToZ]
              }
            }
          }

          f(a1, a2)
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}

trait DAndDToZ { def apply(a1: Double, a2: Double): Boolean }
class FunctionDAndDToZBuilder(implicit ict: ClassTag[Double], iti: TypeInfo[Double], rti: TypeInfo[Boolean])
    extends FunctionBuilder[Boolean](Array[TypeInfo[_]](iti, iti), rti) {

  cn.interfaces.asInstanceOf[java.util.List[String]].add("is/hail/asm4s/DAndDToZ")

  def arg1 = getArg[Double](1)

  def arg2 = getArg[Double](2)

  def result(c: Code[Boolean]): (Double, Double) => Boolean = {
    val bytes = classAsBytes(c)
    val localName = name.replaceAll("/",".")

    new Function2[Double, Double, Boolean] with java.io.Serializable {
      @transient @volatile private var f: DAndDToZ = null
      def apply(a1: Double, a2: Double): Boolean = {
        try {
          if (f == null) {
            this.synchronized {
              if (f == null) {
                f = loadClass(localName, bytes).newInstance().asInstanceOf[DAndDToZ]
              }
            }
          }

          f(a1, a2)
        } catch {
          case e @ (_ : Exception | _: LinkageError) => {
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
          }
        }
      }
    }
  }
}
