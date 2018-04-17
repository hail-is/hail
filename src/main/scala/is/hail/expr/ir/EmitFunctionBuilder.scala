package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.variant.ReferenceGenome
import org.objectweb.asm.tree.AbstractInsnNode

import scala.collection.generic.Growable
import scala.collection.mutable

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

class EmitMethodBuilder(
  fb: EmitFunctionBuilder[_],
  mname: String,
  parameterTypeInfo: Array[TypeInfo[_]],
  returnTypeInfo: TypeInfo[_]
) extends MethodBuilder(fb, mname, parameterTypeInfo, returnTypeInfo) {

  def numReferenceGenomes: Int = fb.numReferenceGenomes

  def getReferenceGenome(rg: ReferenceGenome): Code[ReferenceGenome] =
    fb.getReferenceGenome(rg)
}

class EmitFunctionBuilder[F >: Null](
  parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  returnTypeInfo: MaybeGenericTypeInfo[_],
  packageName: String = "is/hail/codegen/generated"
)(implicit interfaceTi: TypeInfo[F]) extends FunctionBuilder[F](parameterTypeInfo, returnTypeInfo, packageName) {

  private[this] val rgMap: mutable.Map[ReferenceGenome, Code[ReferenceGenome]] =
    mutable.Map[ReferenceGenome, Code[ReferenceGenome]]()

  def numReferenceGenomes: Int = rgMap.size

  def getReferenceGenome(rg: ReferenceGenome): Code[ReferenceGenome] =
    rgMap.getOrElseUpdate(rg, newLazyField[ReferenceGenome](rg.codeSetup))

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
}
