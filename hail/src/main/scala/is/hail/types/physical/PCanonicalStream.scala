package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeOrdering}
import is.hail.asm4s.Code
import is.hail.expr.ir.EmitStream.SizedStream
import is.hail.types.virtual.{TStream, Type}
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, Stream}

final case class PCanonicalStream(elementType: PType, required: Boolean = false) extends PStream {
  override val fundamentalType: PStream = {
    if (elementType == elementType.fundamentalType)
      this
    else
      this.copy(elementType = elementType.fundamentalType)
  }

  override def unsafeOrdering(): UnsafeOrdering = throw new NotImplementedError()

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCStream[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  override def defaultValue: PCanonicalStreamCode =
    PCanonicalStreamCode(this, SizedStream(Code._empty, Stream.empty(EmitCode.missing(elementType)), Some(0)))

  override def deepRename(t: Type) = deepRenameStream(t.asInstanceOf[TStream])

  private def deepRenameStream(t: TStream): PStream =
    PCanonicalStream(this.elementType.deepRename(t.elementType), this.required)

  def setRequired(required: Boolean): PCanonicalStream = if(required == this.required) this else this.copy(required = required)
}

object PCanonicalStreamCode {
  def apply(_pt: PCanonicalStream, stream: SizedStream): PCanonicalStreamCode = new PCanonicalStreamCode {
    val pt = _pt

    def getStream(eltRegion: Code[Region]): SizedStream =
      stream
  }

  def apply(_pt: PCanonicalStream, stream: Code[Region] => SizedStream): PCanonicalStreamCode = new PCanonicalStreamCode {
    val pt = _pt

    def getStream(eltRegion: Code[Region]): SizedStream =
      stream(eltRegion)
  }
}

abstract class PCanonicalStreamCode extends PStreamCode { self =>
  def getStream(eltRegion: Code[Region]): SizedStream

  def memoize(cb: EmitCodeBuilder, name: String): PValue = new PValue {
    val pt = self.pt
    var used: Boolean = false
    def get: PCode = {
      assert(!used)
      used = true
      self
    }
  }
}
